"""Quantize the saved ORT_ENABLE_ALL graph itself, with fresh calibration."""

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import ModelProto
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict
from rich.console import Console

from examples._shared.environment import environment
from examples._shared.plans import save_plans
from examples._shared.schema import Environment
from examples._shared.vision.data import (
    calibration_source,
    download_dataset,
    read_input,
)
from examples._shared.vision.models import LoadedModel
from examples._shared.vision.profile_runtime import Report as ProfileReport
from examples._shared.vision.schema import ImageRecord
from examples.quantization_performance.kernels import ensure_report, run_arrays
from examples.quantization_performance.main import (
    Config as TimingConfig,
)
from examples.quantization_performance.main import (
    Model,
    ModelReport,
    Report,
    Score,
    Variant,
    evaluate,
    file_sha256,
    source_model,
)
from examples.quantization_performance.selection import Candidate, timing_profiles
from quantsmith import Quantizer, Replay
from quantsmith.backends.onnx.integer_execution.audit import ExecutionReport
from quantsmith.calibration import Samples
from quantsmith.config import QuantizationConfig
from quantsmith.domain import Absent
from quantsmith.pipeline import QuantizationResult
from quantsmith.pipeline.policies import FunctionOutputs
from quantsmith.result import Err, Ok, QuantSmithError, Result

type RevisionName = Literal["original", "optimized"]


class Config(BaseSettings):
    """Control matched sample budgets and sequential fresh-process measurements."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_parse_args=False)
    report: Path = Path(
        "work_dir/artifacts/quantization_performance_verified/report.json"
    )
    output: Path = Path("work_dir/artifacts/quantization_preoptimization")
    calibration_samples: int = Field(default=128, ge=1)
    evaluation_samples: int = Field(default=256, ge=1)
    parity_rtol: float = Field(default=1e-3, ge=0, allow_inf_nan=False)
    parity_atol: float = Field(default=1e-4, ge=0, allow_inf_nan=False)
    warmup: int = Field(default=50, ge=1)
    runs: int = Field(default=300, ge=1)
    repetitions: int = Field(default=3, ge=1)
    quiet: CliImplicitFlag[bool] = False


class SourceLink(BaseModel):
    """Preserve only observable name/output lineage, rather than guessing fusions."""

    original_node: str
    original_operator: str
    optimized_nodes: tuple[str, ...] | Absent


class GraphRecord(BaseModel):
    """Identify the exact serialized graph supplied to a quantizer or runtime."""

    path: Path
    sha256: str
    model_bytes: int
    operators: dict[str, int]
    opsets: dict[str, int]
    ir_version: int


class Parity(BaseModel):
    """Measure full held-out FP32 parity before either revision is quantized."""

    samples: int
    maximum_absolute_error: float
    logit_mse: float


class QuantizedRevision(BaseModel):
    """Bind a fresh calibration/audit result to its actual input graph identity."""

    revision: RevisionName
    input_graph: GraphRecord
    output_graph: GraphRecord
    execution: ExecutionReport


class QuantizationFailure(BaseModel):
    """Keep documented pipeline failures visible without inventing a fallback pass."""

    revision: RevisionName
    input_graph: GraphRecord
    error: QuantSmithError


class Measurement(BaseModel):
    """Pair task quality with every unprofiled timing and separate execution trace."""

    candidate: Candidate
    graph: GraphRecord
    quality: Score
    profiles: tuple[ProfileReport, ...]


class ModelExperiment(BaseModel):
    """Record graph differences, provenance, unsupported fusions, and outcomes."""

    model: Model
    original: GraphRecord
    optimized: GraphRecord
    provider: Literal["CPUExecutionProvider"] = "CPUExecutionProvider"
    optimization: Literal["ORT_ENABLE_ALL"] = "ORT_ENABLE_ALL"
    parity: Parity
    provenance: tuple[SourceLink, ...]
    calibration: tuple[ImageRecord, ...]
    held_out: tuple[ImageRecord, ...]
    quantized: tuple[QuantizedRevision, ...]
    failures: tuple[QuantizationFailure, ...]
    measurements: tuple[Measurement, ...]


class ExperimentReport(BaseModel):
    """Keep hardware/runtime provenance separate from any deployment recommendation."""

    config: Config
    environment: Environment
    source_report_sha256: str
    models: tuple[ModelExperiment, ...]


def operator_name(domain: str, operator: str) -> str:
    """Distinguish ORT-specific fused operators from standard ONNX operators."""
    return f"{domain}::{operator}" if domain else operator


def graph_record(path: Path) -> GraphRecord:
    """Inspect the on-disk snapshot rather than an earlier in-memory graph."""
    model = onnx.load(path)
    return GraphRecord(
        path=path,
        sha256=file_sha256(path),
        model_bytes=path.stat().st_size,
        operators=dict(
            Counter(
                operator_name(node.domain, node.op_type) for node in model.graph.node
            )
        ),
        opsets={item.domain or "ai.onnx": item.version for item in model.opset_import},
        ir_version=model.ir_version,
    )


def optimize_graph(model: ModelProto, destination: Path, threads: int) -> ModelProto:
    """Save ORT's full optimized FP32 snapshot and reload it without mutating input."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(destination.resolve())
    options.log_severity_level = 3
    runtime = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    del runtime
    optimized = onnx.load(destination)
    onnx.checker.check_model(optimized)
    return optimized


def provenance(original: ModelProto, optimized: ModelProto) -> tuple[SourceLink, ...]:
    """Record exact/token names and surviving outputs; mark erased values explicitly."""
    links: list[SourceLink] = []
    for source in original.graph.node:
        matches = tuple(
            node.name
            for node in optimized.graph.node
            if node.name == source.name
            or node.name.startswith(source.name + "_token_")
            or set(source.output).intersection(node.output)
        )
        links.append(
            SourceLink(
                original_node=source.name,
                original_operator=operator_name(source.domain, source.op_type),
                optimized_nodes=matches
                if matches
                else Absent(
                    reason="source identity was absorbed or removed; original retained"
                ),
            )
        )
    return tuple(links)


def fp32_parity(
    original: Path,
    optimized: Path,
    loaded: LoadedModel,
    records: tuple[ImageRecord, ...],
    threads: int,
    config: Config,
) -> Parity:
    """Compare unfused FP32 execution with the saved graph before calibration."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    before = ort.InferenceSession(
        str(original), options, providers=["CPUExecutionProvider"]
    )
    after = ort.InferenceSession(
        str(optimized), options, providers=["CPUExecutionProvider"]
    )
    maximum, squared_error = 0.0, 0.0
    for record in records:
        inputs = read_input(record, loaded.preprocess)
        expected = run_arrays(before, {"images": inputs})[0]
        actual = run_arrays(after, {"images": inputs})[0]
        np.testing.assert_allclose(
            actual, expected, rtol=config.parity_rtol, atol=config.parity_atol
        )
        difference = actual.astype(np.float64) - expected
        maximum = max(maximum, float(np.max(np.abs(difference))))
        squared_error += float(np.mean(difference**2))
    return Parity(
        samples=len(records),
        maximum_absolute_error=maximum,
        logit_mse=squared_error / len(records),
    )


def quantize_revision(
    path: Path, samples: Samples
) -> Result[QuantizationResult, QuantSmithError]:
    """Load the saved revision as the actual description/calibration/lowering input."""
    return (
        Quantizer.from_config(QuantizationConfig(outputs=FunctionOutputs()))
        .quantize(onnx.load(path))
        .calibrate(Replay(samples))
    )


def save_quantized(
    revision: RevisionName, input_path: Path, output: Path, result: QuantizationResult
) -> QuantizedRevision:
    """Preserve source-oriented plans and the final integer-execution audit."""
    match result.execution:
        case ExecutionReport() as execution:
            pass
        case Absent(reason=reason):
            raise RuntimeError(
                f"verified static pipeline has no execution audit: {reason}"
            )
    onnx.save(result.model, output)
    save_plans(output.with_suffix(".plan.json"), result.plans)
    output.with_suffix(".execution.json").write_text(
        execution.model_dump_json(indent=2) + "\n"
    )
    return QuantizedRevision(
        revision=revision,
        input_graph=graph_record(input_path),
        output_graph=graph_record(output),
        execution=execution,
    )


def candidate(name: str, path: Path, execution: ExecutionReport | Absent) -> Candidate:
    """Keep selected producers diagnostic; their count never ranks candidates."""
    match execution:
        case ExecutionReport(integer_nodes=nodes):
            scheme = Variant.QS_FUNCTION
            selected = tuple(
                node.node
                for node in nodes
                if node.source_operator in ("Conv", "MatMul", "Gemm")
            )
        case Absent():
            scheme, selected = Variant.FP32, ()
    return Candidate(
        name=name,
        scope=name,
        scheme=scheme,
        opset=graph_record(path).opsets["ai.onnx"],
        path=path,
        selected=selected,
    )


def quantized_candidates(
    original: Path,
    optimized: Path,
    samples: Samples,
    output: Path,
    console: Console,
) -> tuple[
    tuple[Candidate, ...],
    tuple[QuantizedRevision, ...],
    tuple[QuantizationFailure, ...],
]:
    """Calibrate each revision independently and retain expected failures explicitly."""
    candidates: list[Candidate] = []
    quantized: list[QuantizedRevision] = []
    failures: list[QuantizationFailure] = []
    revisions: tuple[tuple[RevisionName, Path], ...] = (
        ("original", original),
        ("optimized", optimized),
    )
    for revision, path in revisions:
        console.print(f"Fresh calibration of saved {revision} graph")
        match quantize_revision(path, samples):
            case Err(error):
                failures.append(
                    QuantizationFailure(
                        revision=revision, input_graph=graph_record(path), error=error
                    )
                )
                console.print(error.model_dump_json())
            case Ok(result):
                record = save_quantized(
                    revision, path, output / f"{revision}_quantized.onnx", result
                )
                quantized.append(record)
                candidates.append(
                    candidate(
                        f"{revision}_quantized",
                        record.output_graph.path,
                        record.execution,
                    )
                )
    return tuple(candidates), tuple(quantized), tuple(failures)


def run_model(
    source: ModelReport, original: Report, config: Config, console: Console
) -> ModelExperiment:
    """Compare actual graph revisions with matched inputs, quality, and execution."""
    output = config.output / source.model.value
    output.mkdir(parents=True, exist_ok=True)
    loaded, fp32 = source_model(source.model, original.config, output)
    if file_sha256(fp32) != source.fp32_sha256:
        raise ValueError("source FP32 graph differs from recorded provenance")
    calibration = source.calibration[: config.calibration_samples]
    held_out = source.held_out[: config.evaluation_samples]
    if (
        len(calibration) != config.calibration_samples
        or len(held_out) != config.evaluation_samples
    ):
        raise ValueError("source report has fewer samples than requested")
    before = onnx.load(fp32)
    optimized_path = output / "optimized_fp32.onnx"
    after = optimize_graph(before, optimized_path, original.config.threads)
    console.print(f"{source.model}: checking FP32 parity before calibration")
    parity = fp32_parity(
        fp32, optimized_path, loaded, held_out, original.config.threads, config
    )
    samples = calibration_source(calibration, loaded.preprocess)
    quantized, revisions, failures = quantized_candidates(
        fp32, optimized_path, samples, output, console
    )
    choices = (
        candidate(
            "original_fp32", fp32, Absent(reason="FP32 control has no integer audit")
        ),
        candidate(
            "optimized_fp32",
            optimized_path,
            Absent(reason="FP32 control has no integer audit"),
        ),
        *quantized,
    )
    settings = TimingConfig(
        threads=original.config.threads,
        warmup=config.warmup,
        runs=config.runs,
        repetitions=config.repetitions,
        quiet=config.quiet,
    )
    measurements = measure_candidates(
        choices, fp32, loaded, held_out, settings, output, console
    )
    report = ModelExperiment(
        model=source.model,
        original=graph_record(fp32),
        optimized=graph_record(optimized_path),
        parity=parity,
        provenance=provenance(before, after),
        calibration=calibration,
        held_out=held_out,
        quantized=revisions,
        failures=failures,
        measurements=measurements,
    )
    (output / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    return report


def measure_candidates(
    choices: tuple[Candidate, ...],
    reference: Path,
    loaded: LoadedModel,
    held_out: tuple[ImageRecord, ...],
    settings: TimingConfig,
    output: Path,
    console: Console,
) -> tuple[Measurement, ...]:
    """Keep quality evaluation outside fresh-process latency and RSS measurements."""
    scores: dict[str, Score] = {}
    for choice in choices:
        console.print(f"Held-out quality: {choice.name}")
        scores[choice.name] = evaluate(
            choice.path, reference, loaded, held_out, settings, console
        )
    inputs = output / "input.npy"
    np.save(inputs, read_input(held_out[0], loaded.preprocess), allow_pickle=False)
    profiles = timing_profiles(choices, inputs, output, settings, console)
    return tuple(
        Measurement(
            candidate=choice,
            graph=graph_record(choice.path),
            quality=scores[choice.name],
            profiles=tuple(profiles[choice.name]),
        )
        for choice in choices
    )


def main() -> None:
    """Automate prerequisites and emit an inspectable matched preoptimization report."""
    config = CliApp.run(Config)
    ensure_report(config.report, "examples.quantization_performance.main")
    original = Report.model_validate_json(config.report.read_text())
    config.output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(original.config.cache / "torch"))
    torch.set_num_threads(original.config.threads)
    download_dataset(original.config.cache)
    console = Console(stderr=True, quiet=config.quiet)
    report = ExperimentReport(
        config=config,
        environment=environment(),
        source_report_sha256=file_sha256(config.report),
        models=tuple(
            run_model(source, original, config, console) for source in original.models
        ),
    )
    content = report.model_dump_json(indent=2)
    (config.output / "report.json").write_text(content + "\n")
    print(content)


if __name__ == "__main__":
    main()
