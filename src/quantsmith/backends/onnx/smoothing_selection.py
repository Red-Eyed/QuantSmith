"""Build, audit and score declared smoothing graphs through streamed ORT replay."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import assert_never

import onnxruntime as ort
from onnx import TensorProto

from quantsmith.backends.onnx.integer_execution.audit import ExecutionReport
from quantsmith.backends.onnx.stages import ReuseQuantizersStage, VerifyStage
from quantsmith.domain import Absent
from quantsmith.optimization.smoothing import (
    CandidateScore,
    ContractionOutputs,
    ModelOutputs,
    Observations,
    ObservedTensors,
    prefer,
    select,
)
from quantsmith.pipeline.contracts import CalibrationContext, Pipeline, Step
from quantsmith.pipeline.state import FloatGraph, Folded, QuantizedGraph
from quantsmith.plan import QuantizeOutput
from quantsmith.result import Err, FailureKind, Ok, QuantSmithError, Result, failure
from quantsmith.runtime import OnnxEvaluator, evaluate


@dataclass(frozen=True)
class SmoothingCandidate:
    """Bind an explicit identity to a typed floating-to-quantized program."""

    name: str
    program: Step[FloatGraph, QuantizedGraph]

    def __post_init__(self) -> None:
        """Require a stable nonempty identity for reporting."""
        if not self.name:
            raise ValueError("candidate name must be nonempty")


def observed_outputs(
    source: FloatGraph, baseline: QuantizedGraph, observations: Observations
) -> tuple[str, ...]:
    """Resolve explicit tensor identities or baseline-quantized contraction outputs."""
    match observations:
        case ObservedTensors(names=names):
            return names
        case ModelOutputs():
            return tuple(output.name for output in source.revision.model.graph.output)
        case ContractionOutputs():
            selected = frozenset(
                operation.node
                for plan in baseline.revision.plans
                for operation in plan.operations
                if isinstance(operation, QuantizeOutput)
            )
            return tuple(
                node.output[0]
                for node in source.revision.model.graph.node
                if node.name in selected and node.op_type in {"Conv", "MatMul", "Gemm"}
            )
        case _:
            assert_never(observations)


def score_candidate(
    name: str,
    graph: QuantizedGraph,
    reference: OnnxEvaluator,
    outputs: tuple[str, ...],
    context: CalibrationContext,
) -> Result[CandidateScore, QuantSmithError]:
    """Audit the complete graph before instrumenting optimized output replay."""
    match VerifyStage().run(graph, context):
        case Err() as error:
            return error
        case Ok(verified):
            pass
    match verified.execution:
        case Absent():
            return failure(
                FailureKind.CONFLICT, name, "candidate lacks integer execution evidence"
            )
        case ExecutionReport() as execution:
            integer_nodes = len(execution.integer_nodes)
    candidate = OnnxEvaluator(
        graph.revision.model, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    match evaluate(reference, candidate, context.samples, outputs):
        case Err() as error:
            return error
        case Ok(error):
            return Ok(
                CandidateScore(
                    candidate=name,
                    mean_squared_error=error.mean_squared_error,
                    maximum_absolute_error=error.maximum_absolute_error,
                    batches=error.batches,
                    integer_nodes=integer_nodes,
                    folded_nodes=tuple(
                        outcome.node
                        for outcome in graph.revision.folding
                        if isinstance(outcome, Folded)
                    ),
                )
            )


def protect_observations(
    source: FloatGraph, outputs: tuple[str, ...]
) -> Result[FloatGraph, QuantSmithError]:
    """Expose scoring cuts so folding cannot change their coordinate systems."""
    value = deepcopy(source)
    model = value.revision.model
    metadata = {
        entry.name: entry
        for entry in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    public = frozenset(entry.name for entry in model.graph.output)
    for name in outputs:
        if (
            name not in metadata
            or metadata[name].type.tensor_type.elem_type != TensorProto.FLOAT
        ):
            return failure(
                FailureKind.INVALID_DATA,
                "smoothing observations",
                f"requested tensor lacks FP32 metadata: {name}",
            )
        if name not in public:
            model.graph.output.append(metadata[name])
    return Ok(value)


def restore_outputs(candidate: QuantizedGraph, source: FloatGraph) -> QuantizedGraph:
    """Keep diagnostic observations out of the deployed model's public interface."""
    model = candidate.revision.model
    del model.graph.output[:]
    model.graph.output.extend(source.revision.model.graph.output)
    return candidate


@dataclass(frozen=True)
class SelectSmoothingStage:
    """Retain the best declared graph using paired, streaming output replay."""

    candidates: tuple[SmoothingCandidate, ...]
    observations: Observations = field(default_factory=ModelOutputs)

    def __post_init__(self) -> None:
        """Require an ordered baseline and unique candidate identities."""
        names = tuple(candidate.name for candidate in self.candidates)
        if not names or names[0] != "baseline" or len(set(names)) != len(names):
            raise ValueError(
                "candidate sequence requires baseline first and unique names"
            )

    def run(
        self, value: FloatGraph, context: CalibrationContext
    ) -> Result[QuantizedGraph, QuantSmithError]:
        """Rebuild each candidate from the same revision and prefer baseline on ties."""
        reference = OnnxEvaluator(
            value.revision.model, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        baseline = self.candidates[0]
        match baseline.program.run(deepcopy(value), context):
            case Err() as error:
                return error
            case Ok(best):
                pass
        outputs = observed_outputs(value, best, self.observations)
        if not outputs:
            return failure(
                FailureKind.EMPTY,
                "smoothing selection",
                "no quantized contraction outputs to observe",
            )
        match protect_observations(value, outputs):
            case Err() as error:
                return error
            case Ok(protected):
                pass
        if len(protected.revision.model.graph.output) != len(
            value.revision.model.graph.output
        ):
            match baseline.program.run(deepcopy(protected), context):
                case Err() as error:
                    return error
                case Ok(best):
                    best = restore_outputs(best, value)
        match score_candidate(baseline.name, best, reference, outputs, context):
            case Err() as error:
                return error
            case Ok(best_score):
                scores = [best_score]
        for specification in self.candidates[1:]:
            match specification.program.run(deepcopy(protected), context):
                case Err() as error:
                    return error
                case Ok(candidate):
                    candidate = restore_outputs(candidate, value)
            match score_candidate(
                specification.name, candidate, reference, outputs, context
            ):
                case Err() as error:
                    return error
                case Ok(score):
                    scores.append(score)
            if prefer(score, best_score):
                best, best_score = candidate, score
            del candidate
        match select(tuple(scores), outputs):
            case Err() as error:
                return error
            case Ok(report):
                return Ok(replace(best, selection=report))


def selection_stage(
    quantization: Step[FloatGraph, QuantizedGraph],
    transforms: tuple[Step[FloatGraph, FloatGraph], ...],
    observations: Observations,
) -> SelectSmoothingStage:
    """Declare an unsmoothed baseline and transformed alternatives with equal reuse."""
    baseline = Pipeline(quantization).then(ReuseQuantizersStage())
    alternatives = tuple(
        SmoothingCandidate(
            f"joint-{index}",
            Pipeline(transform).then(quantization).then(ReuseQuantizersStage()),
        )
        for index, transform in enumerate(transforms)
    )
    return SelectSmoothingStage(
        (SmoothingCandidate("baseline", baseline), *alternatives), observations
    )
