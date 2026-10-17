"""Compare reconstruction with static calibration without downloading a model."""

from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Self, assert_never

import onnx
from onnx import ModelProto
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict
from rich.console import Console

from examples._shared.plans import save_plans
from examples.projection.model import example_model, sample_source
from quantsmith import Quantizer, Replay
from quantsmith.backends.onnx.reconstruction import reconstruct
from quantsmith.calibration import Samples
from quantsmith.config import PercentileConfig, QuantizationConfig
from quantsmith.pipeline import QuantizationResult
from quantsmith.reconstruction import Reconstructor
from quantsmith.reconstruction.gptqv2 import GPTQv2
from quantsmith.reconstruction.qdrop import QDrop
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.rules import Rules
from quantsmith.runtime import Evaluation, OnnxEvaluator, evaluate


class Method(StrEnum):
    """Name independently calibrated candidates in the comparison report."""

    MINMAX = "minmax"
    PERCENTILE = "percentile"
    GPTQV2 = "gptqv2"
    QDROP = "qdrop"


class Config(BaseSettings, frozen=True):
    """Expose calibration, optimization, and held-out evaluation budgets."""

    model_config = SettingsConfigDict(cli_kebab_case=True, populate_by_name=True)
    output: Path = Path("work_dir/artifacts/reconstruction")
    calibration_batches: int = Field(default=8, ge=1)
    evaluation_batches: int = Field(default=8, ge=1)
    calibration_seed: int = Field(default=11, ge=0)
    evaluation_seed: int = Field(default=19, ge=0)
    qdrop_steps: int = Field(default=2000, ge=1)
    percentile: float = Field(default=99.99, gt=0, le=100)
    quiet: CliImplicitFlag[bool] = False
    json_output: CliImplicitFlag[bool] = Field(default=False, alias="json")

    @model_validator(mode="after")
    def separate_evaluation(self) -> Self:
        """Prevent accidental reuse of calibration data for held-out evaluation."""
        if self.calibration_seed == self.evaluation_seed:
            raise ValueError("calibration and evaluation seeds must differ")
        return self


class Measurement(BaseModel, frozen=True):
    """Store held-out error and calibration cost alongside the exported artifact."""

    method: Method
    calibration_seconds: float
    metrics: Evaluation
    artifact: Path


class Report(BaseModel, frozen=True):
    """Preserve reproducible settings without implying task-accuracy evidence."""

    created_at: datetime
    quantsmith: str
    config: Config
    measurements: tuple[Measurement, ...]


def candidate(
    model: ModelProto, samples: Samples, method: Method, config: Config
) -> Result[QuantizationResult, QuantSmithError]:
    """Apply each method to the same original graph and replay factory."""
    match method:
        case Method.MINMAX:
            return (
                Quantizer.from_config(QuantizationConfig())
                .quantize(model)
                .calibrate(Replay(samples))
            )
        case Method.PERCENTILE:
            settings = QuantizationConfig(
                calibration=PercentileConfig(percentile=config.percentile)
            )
            return (
                Quantizer.from_config(settings)
                .quantize(model)
                .calibrate(Replay(samples))
            )
        case Method.GPTQV2:
            optimizer: Reconstructor = GPTQv2()
        case Method.QDROP:
            optimizer = QDrop(steps=config.qdrop_steps)
        case _:
            assert_never(method)
    rules: Rules[Reconstructor] = Rules(default=optimizer)
    return reconstruct(model, samples, rules)


def measure(
    model: ModelProto, method: Method, config: Config
) -> Result[Measurement, QuantSmithError]:
    """Export exact plans and evaluate on independently seeded streamed samples."""
    started = perf_counter()
    samples = sample_source(config.calibration_seed, config.calibration_batches)
    match candidate(model, samples, method, config):
        case Err() as error:
            return error
        case Ok(result):
            pass
    elapsed = perf_counter() - started
    held_out = sample_source(config.evaluation_seed, config.evaluation_batches)
    match evaluate(OnnxEvaluator(model), OnnxEvaluator(result.model), held_out, ("y",)):
        case Err() as error:
            return error
        case Ok(metrics):
            pass
    artifact = Path(f"{method.value}.onnx")
    onnx.save(result.model, config.output / artifact)
    save_plans(config.output / f"{method.value}.plan.json", result.plans)
    return Ok(
        Measurement(
            method=method,
            calibration_seconds=elapsed,
            metrics=metrics,
            artifact=artifact,
        )
    )


def run(config: Config) -> Result[Report, QuantSmithError]:
    """Run the comparison with visible progress and provision the artifact folder."""
    config.output.mkdir(parents=True, exist_ok=True)
    model = example_model()
    onnx.save(model, config.output / "fp32.onnx")
    console = Console(stderr=True, quiet=config.quiet)
    measurements: list[Measurement] = []
    for method in Method:
        with console.status(f"{method.value}: calibration and held-out evaluation"):
            match measure(model, method, config):
                case Err() as error:
                    return error
                case Ok(measurement):
                    measurements.append(measurement)
        console.print(
            f"{method.value}: MSE={measurement.metrics.mean_squared_error:.6g}"
        )
    report = Report(
        created_at=datetime.now(UTC),
        quantsmith=version("quantsmith"),
        config=config,
        measurements=tuple(measurements),
    )
    (config.output / "report.json").write_text(report.model_dump_json(indent=2))
    return Ok(report)


def main() -> None:
    """Expose a typed CLI and machine-readable success or failure payloads."""
    config = CliApp.run(Config)
    match run(config):
        case Err(error):
            print(error.model_dump_json())
            raise SystemExit(1)
        case Ok(report):
            if config.json_output:
                print(report.model_dump_json())
            elif not config.quiet:
                print(config.output / "report.json")


if __name__ == "__main__":
    main()
