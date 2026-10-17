"""Compare selected quantization recipes on a self-contained FP32 projection."""

from datetime import UTC, datetime
from importlib.metadata import version

import onnx
from onnx import ModelProto
from pydantic_settings import CliApp
from rich.console import Console
from rich.progress import Progress

from examples._shared.cli import run_and_report
from examples._shared.plans import coverage, save_plans
from examples._shared.recipes import Recipes
from examples._shared.schema import QuantizationMethod
from examples.projection.model import example_model, sample_source
from examples.projection.reporting import save
from examples.projection.schema import Config, Report, Variant
from quantsmith import Quantizer, Replay
from quantsmith.backends.onnx import describe
from quantsmith.config import PercentileConfig, QuantizationConfig, SmoothQuantConfig
from quantsmith.pipeline import RuntimeScaling
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.runtime import OnnxEvaluator, evaluate


def run(config: Config) -> Result[Report, QuantSmithError]:
    """Persist the original graph and compare only the explicitly selected recipes."""
    config.output.mkdir(parents=True, exist_ok=True)
    model = example_model()
    onnx.save(model, config.output / "onnx_fp32.onnx")
    variants: list[Variant] = []
    console = Console(stderr=True, quiet=config.quiet)
    with Progress(console=console, disable=config.quiet) as progress:
        for recipe in config.methods:
            match run_variant(model, recipe, config, progress):
                case Err() as error:
                    return error
                case Ok(variant):
                    variants.append(variant)
    report = Report(
        created_at=datetime.now(UTC),
        quantsmith=version("quantsmith"),
        config=config,
        baseline=config.output / "onnx_fp32.onnx",
        variants=tuple(variants),
    )
    save(report)
    return Ok(report)


def recipes(config: Config) -> Recipes[QuantizationConfig]:
    """Expose the minimal QuantSmith configuration for each matrix column."""
    return Recipes(
        minmax=QuantizationConfig(
            histogram_bins=config.histogram_bins, outputs=config.outputs
        ),
        percentile=QuantizationConfig(
            calibration=PercentileConfig(percentile=config.percentile),
            histogram_bins=config.histogram_bins,
            outputs=config.outputs,
        ),
        smoothquant=QuantizationConfig(
            smoothing=SmoothQuantConfig(
                alpha=config.smoothquant_alpha, folding=RuntimeScaling()
            ),
            histogram_bins=config.histogram_bins,
            outputs=config.outputs,
        ),
    )


def run_variant(
    model: ModelProto,
    recipe: QuantizationMethod,
    config: Config,
    progress: Progress,
) -> Result[Variant, QuantSmithError]:
    """Quantize from the FP32 source, evaluate held-out batches, and save coverage."""
    method = recipe.variant()
    calibration = sample_source(config.calibration_seed, config.calibration_samples)
    quantization = recipes(config).select(recipe)
    with progress.console.status(f"{recipe.value}: calibrating and lowering"):
        outcome = (
            Quantizer.from_config(quantization)
            .quantize(model)
            .calibrate(Replay(calibration))
        )
    match outcome:
        case Err() as error:
            return error
        case Ok(result):
            pass
    graph = result.model
    match describe(model):
        case Err() as error:
            return error
        case Ok(source_graph):
            pass
    scope = coverage(source_graph, result.plans)
    progress.console.print(f"{recipe.value}: measuring held-out projection error")
    evaluation = sample_source(config.evaluation_seed, config.evaluation_samples)
    match evaluate(OnnxEvaluator(model), OnnxEvaluator(graph), evaluation, ("y",)):
        case Err() as error:
            return error
        case Ok(metrics):
            pass
    artifact = config.output / f"{method.value}.onnx"
    onnx.save(graph, artifact)
    (config.output / f"{method.value}.coverage.json").write_text(
        scope.model_dump_json(indent=2)
    )
    save_plans(config.output / f"{method.value}.plan.json", result.plans)
    return Ok(
        Variant(
            method=method,
            artifact=artifact.relative_to(config.output),
            metrics=metrics,
            coverage=scope,
        )
    )


def main() -> None:
    """Expose method selection, progress controls, and human or JSON output."""
    config = CliApp.run(Config)
    run_and_report(config, run, output=config.output, json_output=config.json_output)


if __name__ == "__main__":
    main()
