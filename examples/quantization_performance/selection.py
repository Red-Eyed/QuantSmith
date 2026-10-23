"""Validate locally promising mixed precision using complete models and task quality."""

from pathlib import Path
from statistics import median

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
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
from examples._shared.vision.models import LoadedModel, OrtRunner
from examples._shared.vision.profile_runtime import Report as ProfileReport
from examples._shared.vision.schema import ImageRecord
from examples.quantization_performance.kernels import (
    Case,
    KernelReport,
    converted,
    ensure_report,
)
from examples.quantization_performance.main import (
    Config as TimingConfig,
)
from examples.quantization_performance.main import (
    Model,
    ModelReport,
    Reader,
    Report,
    Score,
    Variant,
    evaluate,
    file_sha256,
    profile,
    source_model,
)
from quantsmith import Quantizer, Replay
from quantsmith.algorithms import Algorithm
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx.stages import QuantizationStage, VerifyStage
from quantsmith.domain import Absent, Graph, Node
from quantsmith.pipeline import Pipeline
from quantsmith.pipeline.policies import FunctionOutputs
from quantsmith.result import Err, Ok
from quantsmith.rules import Exclude, Rule, Rules


class Config(BaseSettings):
    """Keep local speed margins separate from user-controlled task-accuracy policy."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_parse_args=False)
    kernels: Path = Path("work_dir/artifacts/quantization_kernels/report.json")
    output: Path = Path("work_dir/artifacts/quantization_selection")
    minimum_speedup: float = Field(default=0.05, ge=0, lt=1)
    tuning_samples: int = Field(default=128, ge=1)
    warmup: int = Field(default=50, ge=1)
    runs: int = Field(default=300, ge=1)
    repetitions: int = Field(default=3, ge=1)
    quiet: CliImplicitFlag[bool] = False


class NamedNodes(BaseModel, frozen=True):
    """Select exact source identities without interpreting names as glob patterns."""

    names: frozenset[str]

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Leave every unmeasured or unprofitable source operation floating point."""
        return node.name in self.names


class Decision(BaseModel):
    """Record a latency-based proposal; the number of selected nodes is not a score."""

    scheme: Variant
    opset: int
    selected: tuple[str, ...]


class Candidate(BaseModel):
    """Identify a complete model's graph-generation choice and selected producers."""

    name: str
    scope: str
    scheme: Variant
    opset: int
    path: Path
    selected: tuple[str, ...]


class Outcome(BaseModel):
    """Keep tuning decisions distinct from final held-out evaluation and runtime."""

    candidate: Candidate
    sha256: str
    tuning: Score
    held_out: Score
    profiles: tuple[ProfileReport, ...]


class ModelSelection(BaseModel):
    """Report quality/latency tradeoffs without inventing an accuracy budget."""

    model: Model
    calibration: tuple[ImageRecord, ...]
    tuning: tuple[ImageRecord, ...]
    held_out: tuple[ImageRecord, ...]
    decisions: tuple[Decision, ...]
    outcomes: tuple[Outcome, ...]
    tuning_frontier: tuple[str, ...]
    deployment_choice: Absent = Field(
        default_factory=lambda: Absent(reason="a task-accuracy budget has not been set")
    )


class SelectionReport(BaseModel):
    """Preserve input identities, environment, and every candidate including FP32."""

    config: Config
    environment: Environment
    kernels_sha256: str
    models: tuple[ModelSelection, ...]


def select_nodes(
    cases: tuple[Case, ...], model: Model, scheme: Variant, opset: int, margin: float
) -> tuple[str, ...]:
    """Prefer measured surrounding regions; never lift artificial broadcast controls."""
    contexts: dict[str, Case] = {}
    for case in cases:
        if (
            case.model != model
            or case.batch != 1
            or case.context not in ("core", "region")
        ):
            continue
        if case.source.source not in contexts or case.context == "region":
            contexts[case.source.source] = case
    selected: list[str] = []
    for name, case in contexts.items():
        timings = {
            item.scheme: median(item.median_ms)
            for item in case.timings
            if item.opset == opset
        }
        if timings[scheme] < timings[Variant.FP32] * (1 - margin):
            selected.append(name)
    return tuple(sorted(selected))


def frontier(outcomes: tuple[Outcome, ...]) -> tuple[str, ...]:
    """Use full-model latency and tuning accuracy, excluding held-out labels."""
    result: list[str] = []
    for candidate in outcomes:
        latency = median(value.median_ms for value in candidate.profiles)
        accuracy = candidate.tuning.top1
        dominated = any(
            median(value.median_ms for value in other.profiles) <= latency
            and other.tuning.top1 >= accuracy
            and (
                median(value.median_ms for value in other.profiles) < latency
                or other.tuning.top1 > accuracy
            )
            for other in outcomes
        )
        if not dominated:
            result.append(candidate.candidate.name)
    return tuple(result)


def mixed_model(
    source: Path,
    selected: tuple[str, ...],
    scheme: Variant,
    records: tuple[ImageRecord, ...],
    loaded: LoadedModel,
    destination: Path,
) -> None:
    """Calibrate only proposed producers on the unchanged FP32 source graph."""
    if not selected:
        raise ValueError("empty selection must reuse the FP32 control")
    if scheme == Variant.QS_FUNCTION:
        rules: Rules[Algorithm] = Rules(
            default=Exclude(reason="unmeasured or not locally faster than FP32"),
            overrides=(
                Rule(
                    selector=NamedNodes(names=frozenset(selected)),
                    decision=StaticW8A8(),
                ),
            ),
        )
        pipeline = Pipeline(QuantizationStage(rules, FunctionOutputs())).then(
            VerifyStage()
        )
        result = (
            Quantizer.from_pipeline(pipeline)
            .quantize(onnx.load(source))
            .calibrate(Replay(calibration_source(records, loaded.preprocess)))
        )
        match result:
            case Err(error):
                raise RuntimeError(error.model_dump_json())
            case Ok(value):
                onnx.save(value.model, destination)
        return
    quantize_static(
        source,
        destination,
        Reader(records, loaded),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8
        if scheme == Variant.ORT_SIGNED
        else QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=list(selected),
        calibration_providers=["CPUExecutionProvider"],
        extra_options={
            "ActivationSymmetric": scheme == Variant.ORT_SIGNED,
            "WeightSymmetric": True,
            "CalibStridedMinMax": 1,
        },
    )


def candidates(
    source: ModelReport,
    kernels: KernelReport,
    loaded: LoadedModel,
    fp32: Path,
    output: Path,
    config: Config,
) -> tuple[tuple[Candidate, ...], tuple[Decision, ...]]:
    """Include full-precision/all-quantized controls and nonempty mixed proposals."""
    schemes = (Variant.QS_FUNCTION, Variant.ORT_QDQ, Variant.ORT_SIGNED)
    items: list[Candidate] = []
    decisions: list[Decision] = []
    all_nodes = tuple(
        node.name
        for node in onnx.load(fp32).graph.node
        if node.op_type in ("Conv", "Gemm", "MatMul")
    )
    for scheme in schemes:
        selected_by_opset = {
            opset: select_nodes(
                kernels.cases, source.model, scheme, opset, config.minimum_speedup
            )
            for opset in kernels.config.opsets
        }
        for opset, selected in selected_by_opset.items():
            decisions.append(Decision(scheme=scheme, opset=opset, selected=selected))
        selections = tuple(dict.fromkeys(selected_by_opset.values()))
        for index, selected in enumerate(selections):
            if not selected:
                continue
            mixed = output / f"mixed_{scheme}_{index}.onnx"
            mixed_model(fp32, selected, scheme, source.calibration, loaded, mixed)
            for opset, names in selected_by_opset.items():
                if names == selected:
                    items.append(
                        save_candidate(mixed, names, scheme, opset, "mixed", output)
                    )
        baseline = next(item for item in source.measurements if item.variant == scheme)
        path = baseline.profiles[0].model
        if file_sha256(path) != baseline.model_sha256:
            raise ValueError("recorded all-quantized candidate changed")
        for opset in kernels.config.opsets:
            items.append(save_candidate(path, all_nodes, scheme, opset, "all", output))
    for opset in kernels.config.opsets:
        items.append(save_candidate(fp32, (), Variant.FP32, opset, "fp32", output))
    return tuple(items), tuple(decisions)


def save_candidate(
    source: Path,
    selected: tuple[str, ...],
    scheme: Variant,
    opset: int,
    scope: str,
    output: Path,
) -> Candidate:
    """Convert valid operator schemas and preserve candidate identity on disk."""
    name = f"{scope}_{scheme}_opset{opset}"
    path = output / f"{name}.onnx"
    onnx.save(converted(onnx.load(source), opset), path)
    return Candidate(
        name=name, scope=scope, scheme=scheme, opset=opset, path=path, selected=selected
    )


def tuning_records(
    source: ModelReport, dataset: Path, loaded: LoadedModel, seed: int, count: int
) -> tuple[ImageRecord, ...]:
    """Choose disjoint train images without using final validation labels for search."""
    pool = select_records(
        dataset / "train", loaded.categories, len(source.calibration) + count, seed
    )
    if tuple(value.path for value in pool[: len(source.calibration)]) != tuple(
        value.path for value in source.calibration
    ):
        raise ValueError(
            "calibration provenance does not match deterministic train selection"
        )
    return pool[len(source.calibration) :]


def run_model(
    source: ModelReport,
    original: Report,
    kernels: KernelReport,
    config: Config,
    dataset: Path,
    console: Console,
) -> ModelSelection:
    """Separate tuning/validation and measure complete models in fresh processes."""
    output = config.output / source.model.value
    output.mkdir(parents=True, exist_ok=True)
    loaded, fp32 = source_model(source.model, original.config, output)
    if file_sha256(fp32) != source.fp32_sha256:
        raise ValueError("source FP32 graph changed")
    tuning = tuning_records(
        source, dataset, loaded, original.config.seed, config.tuning_samples
    )
    choices, decisions = candidates(source, kernels, loaded, fp32, output, config)
    settings = TimingConfig(
        threads=original.config.threads,
        warmup=config.warmup,
        runs=config.runs,
        repetitions=config.repetitions,
        quiet=config.quiet,
    )
    inputs = output / "input.npy"
    np.save(
        inputs, read_input(source.held_out[0], loaded.preprocess), allow_pickle=False
    )
    tune_scores, final_scores = quality_scores(
        choices, fp32, loaded, tuning, source.held_out, settings, console
    )
    profiles = timing_profiles(choices, inputs, output, settings, console)
    outcomes = tuple(
        Outcome(
            candidate=choice,
            sha256=file_sha256(choice.path),
            tuning=tune_scores[choice.name],
            held_out=final_scores[choice.name],
            profiles=tuple(profiles[choice.name]),
        )
        for choice in choices
    )
    report = ModelSelection(
        model=source.model,
        calibration=source.calibration,
        tuning=tuning,
        held_out=source.held_out,
        decisions=decisions,
        outcomes=outcomes,
        tuning_frontier=frontier(outcomes),
    )
    (output / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    return report


def quality_scores(
    choices: tuple[Candidate, ...],
    fp32: Path,
    loaded: LoadedModel,
    tuning: tuple[ImageRecord, ...],
    held_out: tuple[ImageRecord, ...],
    settings: TimingConfig,
    console: Console,
) -> tuple[dict[str, Score], dict[str, Score]]:
    """Keep task measurements and numerical opset checks outside latency intervals."""
    tune_scores: dict[str, Score] = {}
    final_scores: dict[str, Score] = {}
    for choice in choices:
        console.print(
            f"Quality: {choice.name}; {len(choice.selected)} selected producers"
        )
        tune_scores[choice.name] = evaluate(
            choice.path, fp32, loaded, tuning, settings, console
        )
        final_scores[choice.name] = evaluate(
            choice.path, fp32, loaded, held_out, settings, console
        )
        if choice.scheme == Variant.FP32:
            example = read_input(held_out[0], loaded.preprocess)
            np.testing.assert_allclose(
                OrtRunner(choice.path, settings.threads)(example),
                OrtRunner(fp32, settings.threads)(example),
                rtol=1e-6,
                atol=1e-6,
            )
    return tune_scores, final_scores


def timing_profiles(
    choices: tuple[Candidate, ...],
    inputs: Path,
    output: Path,
    settings: TimingConfig,
    console: Console,
) -> dict[str, list[ProfileReport]]:
    """Alternate fresh-process timing order without consulting any task labels."""
    profiles: dict[str, list[ProfileReport]] = {choice.name: [] for choice in choices}
    for repetition in range(settings.repetitions):
        order = choices if repetition % 2 == 0 else choices[::-1]
        for choice in order:
            console.print(f"Timing: {choice.name}, repetition {repetition + 1}")
            profiles[choice.name].append(
                profile(
                    choice.path,
                    inputs,
                    output / f"repeat_{repetition}_{choice.name}",
                    settings,
                )
            )
    return profiles


def main() -> None:
    """Build a latency/quality frontier while preserving FP32 as a valid outcome."""
    config = CliApp.run(Config)
    ensure_report(config.kernels, "examples.quantization_performance.kernels")
    kernels = KernelReport.model_validate_json(config.kernels.read_text())
    original = Report.model_validate_json(kernels.config.report.read_text())
    if file_sha256(kernels.config.report) != kernels.source_report_sha256:
        raise ValueError("kernel experiment's source report changed")
    config.output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(original.config.cache / "torch"))
    torch.set_num_threads(original.config.threads)
    console = Console(stderr=True, quiet=config.quiet)
    dataset = download_dataset(original.config.cache)
    report = SelectionReport(
        config=config,
        environment=environment(),
        kernels_sha256=file_sha256(config.kernels),
        models=tuple(
            run_model(source, original, kernels, config, dataset, console)
            for source in original.models
        ),
    )
    content = report.model_dump_json(indent=2)
    (config.output / "report.json").write_text(content + "\n")
    print(content)


if __name__ == "__main__":
    main()
