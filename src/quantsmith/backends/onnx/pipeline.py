"""Revision-separated ORT quantization with explicit stage transformations."""

from dataclasses import dataclass, replace
from typing import assert_never

from onnx import ModelProto
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms.contracts import Algorithm
from quantsmith.algorithms.shell import build_plan
from quantsmith.backends.onnx import admit, describe, lower, normalize
from quantsmith.backends.onnx.optimization import optimize
from quantsmith.backends.onnx.ort_scaling import lower_integer_scaling
from quantsmith.backends.onnx.ort_scaling.boundaries import with_boundaries
from quantsmith.calibration import Samples
from quantsmith.plan import (
    QuantizationPlan,
    QuantizeConstant,
    QuantizeInput,
    RescaleInput,
)
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.rules import Rules
from quantsmith.runtime import OnnxEvaluator


class PipelineResult(BaseModel):
    """Expose the final ORT graph and each revision's inspectable decisions."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    model: ModelProto = Field()
    plans: tuple[QuantizationPlan, ...] = Field()


@dataclass(frozen=True)
class StageState:
    """Bind the current revision, completed plans, and pending smoothing outputs."""

    model: ModelProto
    plans: tuple[QuantizationPlan, ...] = ()
    scaling: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SmoothingStage:
    """Identify a rescaling plan by the consumers whose inputs it transforms."""

    nodes: frozenset[str]


@dataclass(frozen=True)
class QuantizationStage:
    """Identify a stage that consumes any pending smoothing boundaries."""


def _stage_kind(plan: QuantizationPlan) -> SmoothingStage | QuantizationStage:
    """Classify the already validated plan without mixing transform revisions."""
    nodes: set[str] = set()
    for operation in plan.operations:
        match operation:
            case RescaleInput(node=name):
                nodes.add(name)
            case QuantizeInput() | QuantizeConstant():
                pass
            case _:
                assert_never(operation)
    return SmoothingStage(frozenset(nodes)) if nodes else QuantizationStage()


def _finish_smoothing(
    state: StageState, stage: SmoothingStage
) -> Result[StageState, QuantSmithError]:
    """Optimize rescaling before the next stage collects fresh statistics."""
    introduced = frozenset(
        node.input[0] for node in state.model.graph.node if node.name in stage.nodes
    )
    pending = state.scaling | introduced
    match optimize(state.model):
        case Err() as error:
            return error
        case Ok(model):
            remaining = frozenset(
                node.output[0]
                for node in model.graph.node
                if node.op_type == "Div" and node.output[0] in pending
            )
            return Ok(replace(state, model=model, scaling=remaining))


def _finish_quantization(state: StageState) -> Result[StageState, QuantSmithError]:
    """Lower pending channel scales after their integer boundaries are calibrated."""
    if not state.scaling:
        return Ok(state)
    match lower_integer_scaling(state.model, state.scaling):
        case Err() as error:
            return error
        case Ok(model):
            return Ok(replace(state, model=model, scaling=frozenset()))


def _lower_stage(
    state: StageState, plan: QuantizationPlan
) -> Result[StageState, QuantSmithError]:
    """Lower one revision, then dispatch to its concrete completion function."""
    match lower(state.model, plan):
        case Err() as error:
            return error
        case Ok(model):
            lowered = replace(state, model=model, plans=state.plans + (plan,))
    match _stage_kind(plan):
        case SmoothingStage() as stage:
            return _finish_smoothing(lowered, stage)
        case QuantizationStage():
            return _finish_quantization(lowered)


def _advance(
    state: StageState, rules: Rules[Algorithm], samples: Samples, histogram_bins: int
) -> Result[StageState, QuantSmithError]:
    """Describe, calibrate, plan, and lower one graph revision."""
    match describe(state.model):
        case Err() as error:
            return error
        case Ok(graph):
            configured = with_boundaries(state.model, graph, rules, state.scaling)
    match build_plan(
        graph,
        OnnxEvaluator(state.model),
        samples,
        configured,
        histogram_bins=histogram_bins,
    ):
        case Err() as error:
            return error
        case Ok(plan):
            return _lower_stage(state, plan)


def quantize(
    model: ModelProto,
    samples: Samples,
    stages: tuple[Rules[Algorithm], ...],
    *,
    histogram_bins: int = 2048,
) -> Result[PipelineResult, QuantSmithError]:
    """Build an ORT graph with fresh calibration at every transformed revision.

    SmoothQuant followed by built-in UINT8 static quantization automatically adds
    calibrated integer scaling boundaries. This changes rounding and can introduce
    ORT-specific operators; evaluate accuracy on the returned model. Unsupported
    layouts retain their existing execution paths. Expected failures use Result.
    """
    match admit("ONNX validation", lambda: normalize(model)):
        case Err() as error:
            return error
        case Ok(current):
            state = StageState(current)
    for rules in stages:
        match _advance(state, rules, samples, histogram_bins):
            case Err() as error:
                return error
            case Ok(updated):
                state = updated
    return Ok(PipelineResult(model=state.model, plans=state.plans))
