"""Compare QuantSmith and native ORT quantization on identical pretrained exports."""

import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import override

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict
from rich.console import Console

from examples._shared.environment import environment
from examples._shared.schema import Environment
from examples._shared.vision.data import (
    calibration_source,
    download_dataset,
    read_input,
    select_records,
)
from examples._shared.vision.models import LoadedModel, OrtRunner, TorchRunner
from examples._shared.vision.profile_runtime import Report as ProfileReport
from examples._shared.vision.schema import ImageRecord
from examples.mobilenet_v2.main import export_onnx as export_mobile
from examples.mobilenet_v2.main import load_model as load_mobile
from examples.mobilenet_v2.schema import Config as MobileConfig
from examples.resnet18.main import export_onnx as export_resnet
from examples.resnet18.main import load_model as load_resnet
from examples.resnet18.schema import Config as ResnetConfig
from quantsmith import Quantizer, Replay
from quantsmith.calibration import Samples
from quantsmith.config import QuantizationConfig
from quantsmith.domain import FloatArray
from quantsmith.pipeline.policies import FunctionOutputs, ProducerOutputs
from quantsmith.result import Err, Ok


class Model(StrEnum):
    """Architectures with existing pretrained export walkthroughs."""

    RESNET18 = "resnet18"
    MOBILENET_V2 = "mobilenet_v2"


class Variant(StrEnum):
    """Explicit graph-generation and activation-format controls."""

    FP32 = "fp32"
    QS_PRODUCER = "qs_producer_u8s8"
    QS_FUNCTION = "qs_function_u8s8"
    ORT_QDQ = "ort_qdq_u8s8"
    ORT_SIGNED = "ort_qdq_s8s8"
    ORT_OPERATOR = "ort_qoperator_u8s8"
    ORT_TENSOR = "ort_qdq_tensor_u8s8"
    ORT_TENSOR_SIGNED = "ort_qdq_tensor_s8s8"


class Config(BaseSettings):
    """Keep sample budgets, paths, and sequential measurement controls explicit."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_parse_args=False)
    models: list[Model] = Field(default=list(Model), min_length=1)
    variants: list[Variant] = Field(default=list(Variant), min_length=1)
    fp32_root: Path = Path("work_dir/artifacts/integer_execution")
    cache: Path = Path("work_dir/artifacts/cache")
    output: Path = Path("work_dir/artifacts/quantization_performance")
    calibration_samples: int = Field(default=128, ge=1)
    evaluation_samples: int = Field(default=256, ge=1)
    seed: int = Field(default=42, ge=0)
    threads: int = Field(default=1, ge=1)
    warmup: int = Field(default=50, ge=1)
    runs: int = Field(default=300, ge=1)
    repetitions: int = Field(default=3, ge=1)
    quiet: CliImplicitFlag[bool] = False


class Score(BaseModel):
    """Measure held-out quality against the exact FP32 export."""

    samples: int
    top1: float
    agreement: float
    logit_mse: float


class Measurement(BaseModel):
    """Retain every fresh-process result instead of selecting the fastest run."""

    variant: Variant
    model_sha256: str
    score: Score
    profiles: tuple[ProfileReport, ...]


class ModelReport(BaseModel):
    """Bind results to checkpoint preprocessing and exact graph/input identities."""

    model: Model
    weights: str
    preprocessing: str
    fp32_sha256: str
    input_sha256: str
    calibration: tuple[ImageRecord, ...]
    held_out: tuple[ImageRecord, ...]
    measurements: tuple[Measurement, ...]


class Report(BaseModel):
    """Save reproducible configuration and bounded experiment evidence."""

    created_at: datetime
    config: Config
    environment: Environment
    models: tuple[ModelReport, ...]


class Reader(CalibrationDataReader):
    """Adapt streamed replay to ORT's external end-of-data sentinel contract."""

    def __init__(self, records: tuple[ImageRecord, ...], loaded: LoadedModel) -> None:
        """Retain file metadata and preprocessing, never decoded calibration images."""
        self.records = records
        self.loaded = loaded
        self.position = 0
        self.end = len(records)

    @override
    def get_next(self) -> dict[str, FloatArray]:
        """Return a batch; ORT calibration accepts an empty mapping as exhaustion."""
        if self.position == self.end:
            return {}
        record = self.records[self.position]
        self.position += 1
        return {"images": read_input(record, self.loaded.preprocess)}

    @override
    def __len__(self) -> int:
        """Expose the sample count for ORT's bounded strided MinMax calibration."""
        return len(self.records)

    @override
    def __next__(self) -> dict[str, FloatArray]:
        """Preserve the parent reader's iterator exhaustion behavior."""
        sample = self.get_next()
        if not sample:
            raise StopIteration
        return sample

    @override
    def set_range(self, start_index: int, end_index: int) -> None:
        """Select a metadata slice without reading skipped calibration images."""
        if not 0 <= start_index <= end_index <= len(self.records):
            raise ValueError("invalid calibration range")
        self.position = start_index
        self.end = end_index


def file_sha256(path: Path) -> str:
    """Hash artifacts without retaining their contents or leaving files open."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_model(name: Model, config: Config, output: Path) -> tuple[LoadedModel, Path]:
    """Reuse a saved export or automate its creation with the existing example."""
    match name:
        case Model.RESNET18:
            loaded = load_resnet()
        case Model.MOBILENET_V2:
            loaded = load_mobile()
    path = config.fp32_root / name.value / "onnx_fp32.onnx"
    if path.is_file():
        return loaded, path
    example = np.zeros((1, 3, 224, 224), dtype=np.float32)
    match name:
        case Model.RESNET18:
            export_resnet(loaded, example, ResnetConfig(output=output))
        case Model.MOBILENET_V2:
            export_mobile(loaded, example, MobileConfig(output=output))
    return loaded, output / "onnx_fp32.onnx"


def build_variant(
    variant: Variant,
    fp32: Path,
    samples: Samples,
    reader: Reader,
    output: Path,
) -> Path:
    """Generate each candidate independently from the same FP32 graph."""
    if variant == Variant.FP32:
        return fp32
    destination = output / f"{variant.value}.onnx"
    if variant in (Variant.QS_PRODUCER, Variant.QS_FUNCTION):
        outputs = (
            FunctionOutputs() if variant == Variant.QS_FUNCTION else ProducerOutputs()
        )
        result = (
            Quantizer.from_config(QuantizationConfig(outputs=outputs))
            .quantize(onnx.load(fp32))
            .calibrate(Replay(samples))
        )
        match result:
            case Err(error):
                raise RuntimeError(error.model_dump_json())
            case Ok(value):
                onnx.save(value.model, destination)
        return destination
    quantize_static(
        fp32,
        destination,
        reader,
        quant_format=(
            QuantFormat.QOperator
            if variant == Variant.ORT_OPERATOR
            else QuantFormat.QDQ
        ),
        per_channel=variant not in (Variant.ORT_TENSOR, Variant.ORT_TENSOR_SIGNED),
        activation_type=(
            QuantType.QInt8
            if variant in (Variant.ORT_SIGNED, Variant.ORT_TENSOR_SIGNED)
            else QuantType.QUInt8
        ),
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={
            "ActivationSymmetric": variant
            in (Variant.ORT_SIGNED, Variant.ORT_TENSOR_SIGNED),
            "WeightSymmetric": True,
            "CalibStridedMinMax": 1,
        },
    )
    return destination


def evaluate(
    path: Path,
    reference: Path,
    loaded: LoadedModel,
    held_out: tuple[ImageRecord, ...],
    config: Config,
    console: Console,
) -> Score:
    """Stream the same held-out images without including evaluation in timings."""
    runner = OrtRunner(path, config.threads)
    baseline = OrtRunner(reference, config.threads)
    correct, agreement, squared_error = 0, 0, 0.0
    for index, record in enumerate(held_out):
        inputs = read_input(record, loaded.preprocess)
        actual, expected = runner(inputs), baseline(inputs)
        predicted = int(np.argmax(actual))
        correct += predicted == record.label
        agreement += predicted == int(np.argmax(expected))
        squared_error += float(np.mean((actual.astype(np.float64) - expected) ** 2))
        if (index + 1) % 64 == 0:
            console.print(f"Held-out evaluation: {index + 1}/{len(held_out)}")
    count = len(held_out)
    return Score(
        samples=count,
        top1=correct / count,
        agreement=agreement / count,
        logit_mse=squared_error / count,
    )


def profile(path: Path, inputs: Path, output: Path, config: Config) -> ProfileReport:
    """Run one isolated deployment timing followed by a separate profiling session."""
    command = [
        sys.executable,
        "-m",
        "examples._shared.vision.profile_runtime",
        "--model",
        str(path.resolve()),
        "--input",
        str(inputs.resolve()),
        "--output",
        str(output.resolve()),
        "--threads",
        str(config.threads),
        "--warmup",
        str(config.warmup),
        "--runs",
        str(config.runs),
        "--quiet",
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stdout.json").open("w") as stdout:
        subprocess.run(command, check=True, stdout=stdout)
    return ProfileReport.model_validate_json((output / "report.json").read_text())


def run_model(
    name: Model, dataset: Path, config: Config, console: Console
) -> ModelReport:
    """Build and evaluate candidates, then alternate sequential fresh-process timing."""
    output = config.output / name.value
    output.mkdir(parents=True, exist_ok=True)
    loaded, fp32 = source_model(name, config, output)
    calibration = select_records(
        dataset / "train", loaded.categories, config.calibration_samples, config.seed
    )
    held_out = select_records(
        dataset / "val", loaded.categories, config.evaluation_samples, config.seed
    )
    samples = calibration_source(calibration, loaded.preprocess)
    inputs = output / "input.npy"
    example = read_input(held_out[0], loaded.preprocess)
    np.testing.assert_allclose(
        OrtRunner(fp32, config.threads)(example),
        TorchRunner(loaded.model)(example),
        rtol=1e-3,
        atol=1e-4,
    )
    np.save(inputs, example, allow_pickle=False)
    paths: dict[Variant, Path] = {}
    scores: dict[Variant, Score] = {}
    profiles: dict[Variant, list[ProfileReport]] = {}
    for variant in config.variants:
        console.print(f"{name}: building {variant}")
        path = build_variant(
            variant, fp32, samples, Reader(calibration, loaded), output
        )
        paths[variant] = path
        scores[variant] = evaluate(path, fp32, loaded, held_out, config, console)
        profiles[variant] = []
    for repetition in range(config.repetitions):
        order = config.variants if repetition % 2 == 0 else config.variants[::-1]
        for variant in order:
            console.print(f"{name}: timing {variant}, repetition {repetition + 1}")
            measurement = profile(
                paths[variant],
                inputs,
                output / f"repeat_{repetition}_{variant}",
                config,
            )
            profiles[variant].append(measurement)
            console.print(f"{measurement.median_ms:.3f} ms")
    report = ModelReport(
        model=name,
        weights=loaded.weights,
        preprocessing=repr(loaded.transform),
        fp32_sha256=file_sha256(fp32),
        input_sha256=file_sha256(inputs),
        calibration=calibration,
        held_out=held_out,
        measurements=tuple(
            Measurement(
                variant=variant,
                model_sha256=file_sha256(paths[variant]),
                score=scores[variant],
                profiles=tuple(profiles[variant]),
            )
            for variant in config.variants
        ),
    )
    (output / "comparison.json").write_text(report.model_dump_json(indent=2) + "\n")
    return report


def main() -> None:
    """Provision dataset/export inputs, run comparisons, and emit JSON evidence."""
    config = CliApp.run(Config)
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / "configuration.json").write_text(config.model_dump_json(indent=2))
    torch.hub.set_dir(str(config.cache / "torch"))
    torch.set_num_threads(config.threads)
    console = Console(stderr=True, quiet=config.quiet)
    dataset = download_dataset(config.cache)
    report = Report(
        created_at=datetime.now(UTC),
        config=config,
        environment=environment(),
        models=tuple(
            run_model(name, dataset, config, console) for name in config.models
        ),
    )
    content = report.model_dump_json(indent=2)
    (config.output / "report.json").write_text(content + "\n")
    print(content)


if __name__ == "__main__":
    main()
