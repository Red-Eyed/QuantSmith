"""Export PyTorch fragments and benchmark calibration-free ORT integer kernels."""

import contextlib
import io
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Literal, TypeAlias

import numpy as np
import onnx
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    InvalidGraph,
    NotImplemented,
    RuntimeException,
)
from onnxruntime.quantization import QuantType, quantize_dynamic
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict
from rich.console import Console
from torch import Tensor, nn

from examples._shared.environment import environment
from examples._shared.schema import Environment
from examples._shared.vision.profile_runtime import Event
from examples.quantization_performance.transformer_blocks import (
    FRAGMENTS,
    Fragment,
    Geometry,
    build_fragment,
    dynamic_export_inputs,
    fixed_onnx,
    fragment_output,
    synthetic_inputs,
)
from quantsmith.domain import FloatArray

_ARRAYS = TypeAdapter(
    list[NDArray[np.generic]], config=ConfigDict(arbitrary_types_allowed=True)
)
Scheme: TypeAlias = Literal[
    "fp32",
    "dynamic",
    "fixed_quantized_u8s8",
    "fixed_float_u8s8",
    "fixed_quantized_s8s8",
    "fixed_float_s8s8",
]


def default_geometries() -> list[Geometry]:
    """Cover small controls, token-at-a-time projections, prefill and batched work."""
    return [
        Geometry(tokens=32, context=32, width=64),
        Geometry(tokens=1, context=128, width=256),
        Geometry(tokens=128, context=128, width=256),
        Geometry(batch=4, tokens=32, context=32, width=256, heads=8),
        Geometry(tokens=1, context=128, width=768, heads=12),
        Geometry(tokens=128, context=128, width=768, heads=12),
    ]


class Config(BaseSettings):
    """Declare the synthetic workload and timing budgets without any dataset options."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_parse_args=False)
    output: Path = Path("work_dir/artifacts/transformer_ort_dynamic")
    fragments: list[Fragment] = Field(default=list(FRAGMENTS), min_length=1)
    geometries: list[Geometry] = Field(default_factory=default_geometries, min_length=1)
    opsets: list[Literal[18, 21]] = Field(default=[18, 21], min_length=1)
    opset_mode: Literal["convert", "native"] = "convert"
    activation_range: float = Field(default=4, gt=0, allow_inf_nan=False)
    scope: Literal["weights", "all"] = "all"
    seed: int = Field(default=42, ge=0)
    threads: int = Field(default=1, ge=1)
    warmup: int = Field(default=10, ge=1)
    runs: int = Field(default=100, ge=1)
    repetitions: int = Field(default=3, ge=1)
    profile_runs: int = Field(default=10, ge=1)
    quiet: CliImplicitFlag[bool] = False


class Measurement(BaseModel):
    """Retain repetitions, numerical error, graph identity and executed ops."""

    scheme: Scheme
    opset: int
    median_us: list[float]
    p95_us: list[float]
    relative_mse: float
    maximum_absolute_error: float
    integer_matmuls: int
    scope: Literal["none", "weights", "all"]
    operators: dict[str, int]
    operator_us: dict[str, float]
    graph_sha256: str
    model_bytes: int


class Unsupported(BaseModel):
    """Preserve native backend rejection instead of treating a fallback as INT8."""

    scheme: Scheme
    opset: int
    reason: str


class Case(BaseModel):
    """Bind each measurement to exact inputs and a reproducible random model."""

    fragment: Fragment
    geometry: Geometry
    input_shapes: tuple[tuple[int, ...], ...]
    input_strides: tuple[tuple[int, ...], ...]
    input_sha256: str
    torch_linear_modules: int
    measurements: list[Measurement]
    unsupported: list[Unsupported]


class Report(BaseModel):
    """Persist environment, explicit search space and every result as JSON."""

    config: Config
    environment: Environment
    dynamic_sequence: bool
    cases: list[Case]


@dataclass
class Probe:
    """Keep backend execution out of report serialization and measurement policy."""

    measurement: Measurement
    run: Callable[[], None]


def errors(actual: FloatArray, expected: FloatArray) -> tuple[float, float]:
    """Reject malformed outputs and measure error without changing acceptance policy."""
    if (
        actual.dtype != np.float32
        or actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
    ):
        raise ValueError("backend output must be finite FP32 with the reference shape")
    difference = actual.astype(np.float64) - expected
    power = float(np.mean(expected.astype(np.float64) ** 2))
    return float(np.mean(difference**2)) / max(power, 1e-30), float(
        np.max(np.abs(difference))
    )


def export_fragment(
    module: nn.Module, inputs: tuple[Tensor, Tensor], path: Path, opset: int = 18
) -> None:
    """Export the requested opset and preserve the exporter diagnostics."""
    representative, dimensions = dynamic_export_inputs(module, inputs)
    with (
        path.with_suffix(".export.log").open("w") as log,
        contextlib.redirect_stdout(log),
    ):
        torch.onnx.export(
            module,
            representative,
            str(path),
            input_names=["inputs", "memory"],
            output_names=["output"],
            opset_version=opset,
            dynamo=True,
            dynamic_shapes=dimensions,
            external_data=False,
            optimize=True,
        )
    model = onnx.load(path)
    onnx.checker.check_model(model)
    verify_dynamic_sequence(model)


def verify_dynamic_sequence(model: onnx.ModelProto) -> None:
    """Reject exports or rewrites that specialize sequence dimensions."""
    for value in model.graph.input:
        dimensions = value.type.tensor_type.shape.dim
        axes = (2, 3) if value.name == "inputs" and len(dimensions) == 4 else (1,)
        for axis in axes:
            if not dimensions[axis].dim_param:
                raise ValueError(
                    f"{value.name} axis {axis} must be a symbolic sequence dimension"
                )


def ort_probe(
    path: Path,
    inputs: tuple[Tensor, Tensor],
    expected: FloatArray,
    scheme: Scheme,
    opset: int,
    config: Config,
) -> Probe:
    """Audit the optimized graph and profile separately from warmed inference."""
    verify_dynamic_sequence(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = config.threads, 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    optimized = path.with_suffix(".optimized.onnx")
    options.optimized_model_filepath = str(optimized)
    runtime = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    supplied = {"inputs": inputs[0].numpy(), "memory": inputs[1].numpy()}
    feed = {item.name: supplied[item.name] for item in runtime.get_inputs()}
    arrays = _ARRAYS.validate_python(runtime.run(None, feed))
    actual = arrays[0]
    if actual.dtype != np.float32:
        raise ValueError("ORT output has wrong dtype")
    actual_float = np.asarray(actual, dtype=np.float32)
    mse, maximum = errors(actual_float, expected)
    if scheme == "fp32":
        np.testing.assert_allclose(actual_float, expected, rtol=1e-3, atol=1e-4)
    operators = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}"
        for node in onnx.load(optimized).graph.node
    )

    def run() -> None:
        """Execute the same feed without output copying or profiling."""
        runtime.run(None, feed)

    options.optimized_model_filepath = ""
    options.enable_profiling = True
    options.profile_file_prefix = str(path.with_suffix(".trace"))
    profiled = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    for _ in range(config.profile_runs):
        profiled.run(None, feed)
    totals: dict[str, float] = {}
    for event in TypeAdapter(list[Event]).validate_json(
        Path(profiled.end_profiling()).read_text()
    ):
        if event.cat == "Node" and event.args.op_name:
            totals[event.args.op_name] = (
                totals.get(event.args.op_name, 0) + event.dur / config.profile_runs
            )
    return Probe(
        Measurement(
            scheme=scheme,
            opset=opset,
            median_us=[],
            p95_us=[],
            relative_mse=mse,
            maximum_absolute_error=maximum,
            integer_matmuls=sum(
                count
                for name, count in operators.items()
                if name.split("::")[-1]
                in {
                    "MatMulInteger",
                    "MatMulIntegerToFloat",
                    "DynamicQuantizeMatMul",
                    "QLinearMatMul",
                    "QGemm",
                }
            ),
            scope=(
                "none"
                if scheme == "fp32"
                else "weights"
                if scheme == "dynamic"
                else config.scope
            ),
            operators=dict(operators),
            operator_us=totals,
            graph_sha256=sha256(path.read_bytes()).hexdigest(),
            model_bytes=path.stat().st_size,
        ),
        run,
    )


def prepare_ort(
    module: nn.Module,
    inputs: tuple[Tensor, Tensor],
    expected: FloatArray,
    case: Case,
    output: Path,
    config: Config,
) -> list[Probe]:
    """Create dynamic and fixed-grid graphs without running any calibration samples."""
    original = output / "source.onnx"
    if config.opset_mode == "convert":
        export_fragment(module, inputs, original)
    probes: list[Probe] = []
    schemes: tuple[Scheme, ...] = (
        "fp32",
        "dynamic",
        "fixed_float_u8s8",
        "fixed_quantized_u8s8",
        "fixed_float_s8s8",
        "fixed_quantized_s8s8",
    )
    for opset in config.opsets:
        if config.opset_mode == "native":
            native = output / f"source_{opset}.onnx"
            export_fragment(module, inputs, native, opset)
            model = onnx.load(native)
        else:
            model = onnx.version_converter.convert_version(onnx.load(original), opset)
        for scheme in schemes:
            path = output / f"opset_{opset}_{scheme}.onnx"
            try:
                if scheme == "dynamic":
                    with contextlib.redirect_stdout(io.StringIO()):
                        quantize_dynamic(
                            model,
                            path,
                            weight_type=QuantType.QInt8,
                            op_types_to_quantize=["MatMul"],
                        )
                else:
                    candidate = (
                        model
                        if scheme == "fp32"
                        else fixed_onnx(
                            model,
                            config.activation_range,
                            "float" if "float" in scheme else "quantized",
                            signed="s8s8" in scheme,
                            scope=config.scope,
                        )
                    )
                    onnx.save(candidate, path)
                probes.append(ort_probe(path, inputs, expected, scheme, opset, config))
            except (
                Fail,
                InvalidArgument,
                InvalidGraph,
                NotImplemented,
                RuntimeException,
            ) as error:
                case.unsupported.append(
                    Unsupported(scheme=scheme, opset=opset, reason=str(error))
                )
    return probes


def time_probes(probes: list[Probe], config: Config) -> None:
    """Alternate candidate order while retaining all medians and tail measurements."""
    for probe in probes:
        for _ in range(config.warmup):
            probe.run()
    for repetition in range(config.repetitions):
        for probe in probes if repetition % 2 == 0 else probes[::-1]:
            timings: list[float] = []
            for _ in range(config.runs):
                start = perf_counter_ns()
                probe.run()
                timings.append((perf_counter_ns() - start) / 1000)
            probe.measurement.median_us.append(median(timings))
            probe.measurement.p95_us.append(float(np.percentile(timings, 95)))


def measure_case(
    kind: Fragment, geometry: Geometry, output: Path, config: Config
) -> Case:
    """Own only the current case's model, inputs and prepared runtimes."""
    output.mkdir(parents=True, exist_ok=True)
    module = build_fragment(kind, geometry, config.seed)
    inputs = synthetic_inputs(kind, geometry, config.seed + 1)
    expected = fragment_output(module, inputs).detach().numpy()
    case = Case(
        fragment=kind,
        geometry=geometry,
        input_shapes=tuple(tuple(value.shape) for value in inputs),
        input_strides=tuple(value.stride() for value in inputs),
        input_sha256=sha256(
            b"".join(value.numpy().tobytes() for value in inputs)
        ).hexdigest(),
        torch_linear_modules=sum(
            isinstance(child, nn.Linear) for child in module.modules()
        ),
        measurements=[],
        unsupported=[],
    )
    probes = prepare_ort(module, inputs, expected, case, output, config)
    time_probes(probes, config)
    case.measurements.extend(probe.measurement for probe in probes)
    (output / "report.json").write_text(case.model_dump_json(indent=2) + "\n")
    return case


def main(config: Config) -> Report:
    """Run without calibration and checkpoint each completed case."""
    config.output.mkdir(parents=True, exist_ok=True)
    console = Console(stderr=True, quiet=config.quiet)
    report = Report(
        config=config, environment=environment(), dynamic_sequence=True, cases=[]
    )
    torch.set_num_threads(config.threads)
    total = len(config.geometries) * len(config.fragments)
    with torch.inference_mode():
        for geometry_index, geometry in enumerate(config.geometries):
            for kind in config.fragments:
                console.print(
                    f"[{len(report.cases) + 1}/{total}] {kind}: {geometry.model_dump()}"
                )
                case = measure_case(
                    kind,
                    geometry,
                    config.output / f"shape_{geometry_index}_{kind}",
                    config,
                )
                report.cases.append(case)
                (config.output / "report.json").write_text(
                    report.model_dump_json(indent=2) + "\n"
                )
    return report


if __name__ == "__main__":
    settings = CliApp.run(Config)
    print(main(settings).model_dump_json(indent=2))
