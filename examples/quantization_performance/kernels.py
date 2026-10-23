"""Measure actual source shapes, broadcasting, and opsets with FP32 controls."""

import subprocess
import sys
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Literal, override

import numpy as np
import onnx
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from onnx import ModelProto, helper, numpy_helper, version_converter
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict
from rich.console import Console

from examples._shared.environment import environment
from examples._shared.schema import Environment
from examples._shared.vision.data import download_dataset, read_input
from examples._shared.vision.models import LoadedModel
from examples._shared.vision.schema import ImageRecord
from examples.quantization_performance.main import (
    Model,
    ModelReport,
    Reader,
    Report,
    Variant,
    build_variant,
    file_sha256,
    source_model,
)
from examples.quantization_performance.regions import (
    Region,
    capture_graph,
    extract_region,
    profile_costs,
    select_sources,
)
from quantsmith.domain import FloatArray

_ARRAYS = TypeAdapter(
    list[NDArray[np.generic]], config=ConfigDict(arbitrary_types_allowed=True)
)
SCHEMES = (Variant.FP32, Variant.QS_FUNCTION, Variant.ORT_QDQ, Variant.ORT_SIGNED)
type Context = Literal["core", "region", "scalar", "channel", "full"]


def ensure_report(
    path: Path,
    module: Literal[
        "examples.quantization_performance.main",
        "examples.quantization_performance.kernels",
    ],
) -> None:
    """Generate missing prerequisite evidence automatically."""
    if path.is_file():
        return
    if path.name != "report.json":
        raise ValueError("automatic setup requires an output named report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "setup_stdout.json").open("w") as stdout:
        subprocess.run(
            [sys.executable, "-m", module, "--output", str(path.parent)],
            check=True,
            stdout=stdout,
        )


class Config(BaseSettings):
    """Bound the shape experiment and expose all timing/search choices."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_parse_args=False)
    report: Path = Path(
        "work_dir/artifacts/quantization_performance_verified/report.json"
    )
    output: Path = Path("work_dir/artifacts/quantization_kernels")
    top_k: int = Field(default=2, ge=1)
    batches: list[int] = Field(default=[1, 4], min_length=1)
    opsets: list[int] = Field(default=[18, 21], min_length=1)
    region_hops: int = Field(default=2, ge=1)
    calibration_samples: int = Field(default=128, ge=1)
    warmup: int = Field(default=10, ge=1)
    runs: int = Field(default=100, ge=1)
    repetitions: int = Field(default=3, ge=1)
    quiet: CliImplicitFlag[bool] = False


class Timing(BaseModel):
    """Retain order-alternated warm session timings and numerical error."""

    scheme: Variant
    opset: int
    path: Path
    sha256: str
    median_ms: tuple[float, ...]
    p95_ms: tuple[float, ...]
    relative_mse: float
    optimized_operators: dict[str, int]


class Case(BaseModel):
    """Bind timings to an original source identity and exact boundary geometry."""

    model: Model
    source: Region
    context: Context
    batch: int
    input_shapes: dict[str, tuple[int, ...]]
    timings: tuple[Timing, ...]


class KernelReport(BaseModel):
    """Keep local timings diagnostic; model-level acceptance is a separate step."""

    config: Config
    environment: Environment
    source_report_sha256: str
    cases: tuple[Case, ...]


def session(
    model: ModelProto | Path, threads: int, optimized: Path
) -> ort.InferenceSession:
    """Use the same optimized single-provider settings for every candidate."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.optimized_model_filepath = str(optimized)
    source = model.SerializeToString() if isinstance(model, ModelProto) else str(model)
    return ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])


def run_arrays(
    runtime: ort.InferenceSession, feed: Mapping[str, FloatArray]
) -> list[FloatArray]:
    """Validate native outputs once at the runtime boundary without dtype coercion."""
    values = _ARRAYS.validate_python(runtime.run(None, dict(feed)))
    outputs: list[FloatArray] = []
    for value in values:
        if value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise ValueError("region output must be finite FP32")
        outputs.append(np.asarray(value, dtype=np.float32))
    return outputs


class Capture:
    """Replay one real prefix and retain only this region's current input tensors."""

    def __init__(
        self,
        runtime: ort.InferenceSession,
        loaded: LoadedModel,
        region: Region,
        batch: int,
    ) -> None:
        """Bind exact source boundaries to their original preprocessing and batch."""
        self.runtime, self.loaded, self.region, self.batch = (
            runtime,
            loaded,
            region,
            batch,
        )

    def __call__(self, record: ImageRecord) -> dict[str, FloatArray]:
        """Repeat the current image for batch sweeps without caching dataset images."""
        pixels = read_input(record, self.loaded.preprocess)
        values = run_arrays(self.runtime, {"images": pixels})
        feed: dict[str, FloatArray] = {}
        for spec, value in zip(self.region.inputs, values, strict=True):
            if value.shape != spec.dimensions:
                raise ValueError(f"captured shape mismatch for {spec.name}")
            feed[spec.name] = (
                np.repeat(value, self.batch, axis=0)
                if value.ndim and value.shape[0] == 1 and self.batch != 1
                else value
            )
        return feed


class CapturedReader(Reader):
    """Reuse metadata slicing while supplying measured internal activations."""

    def __init__(
        self, records: tuple[ImageRecord, ...], loaded: LoadedModel, capture: Capture
    ) -> None:
        """Keep only replay metadata and an owned prefix session."""
        super().__init__(records, loaded)
        self.capture = capture

    @override
    def get_next(self) -> dict[str, FloatArray]:
        """Return one freshly captured region input, with bounded strided replay."""
        if self.position == self.end:
            return {}
        record = self.records[self.position]
        self.position += 1
        return self.capture(record)


def with_batch(model: ModelProto, batch: int) -> ModelProto:
    """Change only singleton batch dimensions, then recompute output shapes."""
    if batch < 1:
        raise ValueError("batch must be positive")
    changed = deepcopy(model)
    for value in changed.graph.input:
        dimensions = value.type.tensor_type.shape.dim
        if dimensions and dimensions[0].dim_value == 1:
            dimensions[0].dim_value = batch
    del changed.graph.value_info[:]
    for value in changed.graph.output:
        value.type.tensor_type.ClearField("shape")
    return onnx.shape_inference.infer_shapes(changed)


def with_broadcast(model: ModelProto, context: Context) -> ModelProto:
    """Use numerically identical scaling with scalar/channel/full stored operands."""
    if context in ("core", "region"):
        return model
    changed = deepcopy(model)
    output = changed.graph.output[0]
    shape = tuple(value.dim_value for value in output.type.tensor_type.shape.dim)
    if len(shape) != 4:
        raise ValueError("broadcast controls require a concrete NCHW output")
    match context:
        case "scalar":
            operand_shape = ()
        case "channel":
            operand_shape = (1, shape[1], 1, 1)
        case "full":
            operand_shape = shape
    operand = numpy_helper.from_array(
        np.full(operand_shape, 1.125, dtype=np.float32), "benchmark_coefficient"
    )
    changed.graph.initializer.append(operand)
    changed.graph.node.append(
        helper.make_node(
            "Mul",
            [output.name, operand.name],
            ["benchmark_output"],
            name="benchmark_mul",
        )
    )
    output.name = "benchmark_output"
    return changed


def converted(model: ModelProto, opset: int) -> ModelProto:
    """Apply ONNX's version adapters rather than rewriting the opset declaration."""
    changed = version_converter.convert_version(deepcopy(model), opset)
    onnx.checker.check_model(changed)
    return changed


def measure(
    runtime: ort.InferenceSession, feed: Mapping[str, FloatArray], config: Config
) -> tuple[float, float]:
    """Validate once, then time warmed ORT calls without scanning output tensors."""
    run_arrays(runtime, feed)
    inputs = dict(feed)
    for _ in range(config.warmup):
        runtime.run(None, inputs)
    timings: list[float] = []
    for _ in range(config.runs):
        start = perf_counter()
        runtime.run(None, inputs)
        timings.append((perf_counter() - start) * 1000)
    return median(timings), float(np.percentile(timings, 95))


def output_error(actual: list[FloatArray], expected: list[FloatArray]) -> float:
    """Normalize aggregate output MSE; zero references use a finite denominator."""
    error, norm = 0.0, 0.0
    for value, reference in zip(actual, expected, strict=True):
        if value.shape != reference.shape:
            raise ValueError("candidate output shape differs from FP32")
        error += float(np.sum((value.astype(np.float64) - reference) ** 2))
        norm += float(np.sum(reference.astype(np.float64) ** 2))
    return error / max(norm, np.finfo(np.float64).eps)


def benchmark_case(
    model: ModelProto,
    region: Region,
    context: Context,
    batch: int,
    capture: Capture,
    loaded: LoadedModel,
    source: ModelReport,
    output: Path,
    config: Config,
    threads: int,
) -> Case:
    """Build calibrated controls, verify opset parity, and alternate timing order."""
    output.mkdir(parents=True, exist_ok=True)
    fp32 = output / "fp32.onnx"
    onnx.save(with_broadcast(with_batch(model, batch), context), fp32)
    feed = capture(source.held_out[0])
    expected = run_arrays(session(fp32, threads, output / "reference.onnx"), feed)
    records = source.calibration[: config.calibration_samples]
    if len(records) != config.calibration_samples:
        raise ValueError("source report has fewer calibration images than requested")

    def samples() -> Iterable[Mapping[str, FloatArray]]:
        """Replay the same source calibration metadata for each graph candidate."""
        for record in records:
            yield capture(record)

    paths: dict[tuple[Variant, int], Path] = {}
    errors: dict[tuple[Variant, int], float] = {}
    for scheme in SCHEMES:
        reader = CapturedReader(records, loaded, capture)
        path = build_variant(scheme, fp32, samples, reader, output)
        scheme_reference = run_arrays(
            session(path, threads, output / "scheme_reference.onnx"), feed
        )
        for opset in config.opsets:
            target = output / f"{scheme}_opset{opset}.onnx"
            onnx.save(converted(onnx.load(path), opset), target)
            runtime = session(target, threads, target.with_suffix(".optimized.onnx"))
            actual = run_arrays(runtime, feed)
            for value, reference in zip(actual, scheme_reference, strict=True):
                np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-6)
            key = (scheme, opset)
            paths[key], errors[key] = target, output_error(actual, expected)
    times: dict[tuple[Variant, int], list[tuple[float, float]]] = {
        key: [] for key in paths
    }
    for repetition in range(config.repetitions):
        order = tuple(paths) if repetition % 2 == 0 else tuple(reversed(paths))
        for key in order:
            path = paths[key]
            runtime = session(path, threads, path.with_suffix(".optimized.onnx"))
            times[key].append(measure(runtime, feed, config))
    return Case(
        model=source.model,
        source=region,
        context=context,
        batch=batch,
        input_shapes={name: value.shape for name, value in feed.items()},
        timings=tuple(
            Timing(
                scheme=scheme,
                opset=opset,
                path=path,
                sha256=file_sha256(path),
                median_ms=tuple(pair[0] for pair in times[scheme, opset]),
                p95_ms=tuple(pair[1] for pair in times[scheme, opset]),
                relative_mse=errors[scheme, opset],
                optimized_operators=operator_counts(
                    path.with_suffix(".optimized.onnx")
                ),
            )
            for (scheme, opset), path in paths.items()
        ),
    )


def operator_counts(path: Path) -> dict[str, int]:
    """Count compiled operations without treating integer coverage as an objective."""
    counts: dict[str, int] = {}
    for node in onnx.load(path).graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def run_model(
    source: ModelReport, original: Report, config: Config, console: Console
) -> tuple[Case, ...]:
    """Rank real source costs and measure isolated/expanded regions and broadcasts."""
    root = config.output / source.model.value
    root.mkdir(parents=True, exist_ok=True)
    loaded, fp32 = source_model(source.model, original.config, root)
    if file_sha256(fp32) != source.fp32_sha256:
        raise ValueError("source FP32 graph differs from the recorded benchmark")
    model = onnx.load(fp32)
    baseline = next(
        value for value in source.measurements if value.variant == Variant.FP32
    )
    profile_root = config.report.parent / source.model.value / "repeat_0_fp32"
    costs = profile_costs(model, sorted(profile_root.glob("ort_*.json"))[-1])
    if not costs:
        raise ValueError("trace contains no recognized source Conv/MatMul/Gemm events")
    sources = select_sources(model, costs, config.top_k)
    results: list[Case] = []
    for index, name in enumerate(sources):
        contexts: tuple[Context, ...] = ("core", "region")
        _, core_region = extract_region(
            model, name, 0, costs[name] / baseline.profiles[0].runs
        )
        if index == 0 and core_region.operator == "Conv":
            contexts += ("scalar", "channel", "full")
        for context in contexts:
            hops = config.region_hops if context == "region" else 0
            region_model, region = extract_region(
                model, name, hops, core_region.profiled_us
            )
            if context == "region" and region.nodes == core_region.nodes:
                continue
            prefix = session(
                capture_graph(model, region.inputs),
                original.config.threads,
                root / "prefix.onnx",
            )
            for batch in config.batches:
                console.print(f"{source.model}: {name}, {context}, batch {batch}")
                capture = Capture(prefix, loaded, region, batch)
                result = benchmark_case(
                    region_model,
                    region,
                    context,
                    batch,
                    capture,
                    loaded,
                    source,
                    root / name / f"{context}_batch{batch}",
                    config,
                    original.config.threads,
                )
                results.append(result)
                (root / "cases.json").write_text(
                    TypeAdapter(list[Case]).dump_json(results, indent=2).decode() + "\n"
                )
    return tuple(results)


def main() -> None:
    """Read verified provenance, replay bounded activations, and save JSON timings."""
    config = CliApp.run(Config)
    if any(batch < 1 for batch in config.batches):
        raise ValueError("batches must be positive")
    ensure_report(config.report, "examples.quantization_performance.main")
    original = Report.model_validate_json(config.report.read_text())
    config.output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(original.config.cache / "torch"))
    torch.set_num_threads(original.config.threads)
    console = Console(stderr=True, quiet=config.quiet)
    download_dataset(original.config.cache)
    report = KernelReport(
        config=config,
        environment=environment(),
        source_report_sha256=file_sha256(config.report),
        cases=tuple(
            case
            for source in original.models
            for case in run_model(source, original, config, console)
        ),
    )
    content = report.model_dump_json(indent=2)
    (config.output / "report.json").write_text(content + "\n")
    print(content)


if __name__ == "__main__":
    main()
