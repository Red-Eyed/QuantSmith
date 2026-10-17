"""Application wiring for revision-separated quantization with explicit outcomes."""

from onnx import ModelProto
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms.contracts import Algorithm
from quantsmith.algorithms.shell import build_plan
from quantsmith.backends.onnx import admit, describe, lower, normalize
from quantsmith.backends.onnx.optimization import optimize
from quantsmith.calibration import Samples
from quantsmith.plan import QuantizationPlan, RescaleInput
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.rules import Rules
from quantsmith.runtime import OnnxEvaluator


class PipelineResult(BaseModel):
    """Expose the final model and each revision's inspectable decisions."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    model: ModelProto = Field()
    plans: tuple[QuantizationPlan, ...] = Field()


def quantize(
    model: ModelProto,
    samples: Samples,
    stages: tuple[Rules[Algorithm], ...],
    *,
    histogram_bins: int = 2048,
    ort_integer_scaling: bool = False,
) -> Result[PipelineResult, QuantSmithError]:
    """Plan each revision and return expected admission or planning failures.

    ``ort_integer_scaling`` adds MinMax UINT8 boundaries in smoothed static regions
    and lowers supported scales to ORT integer operations. This opt-in deployment
    mode changes rounding and requires accuracy validation on representative data.
    """
    match admit("ONNX validation", lambda: normalize(model)):
        case Err() as error:
            return error
        case Ok(current):
            pass
    plans: list[QuantizationPlan] = []
    pending_scaling: frozenset[str] = frozenset()
    for rules in stages:
        match describe(current):
            case Err() as error:
                return error
            case Ok(graph):
                pass
        if ort_integer_scaling and pending_scaling:
            from quantsmith.backends.onnx.ort_scaling.boundaries import with_boundaries

            rules = with_boundaries(current, graph, rules, pending_scaling)
        match build_plan(
            graph, OnnxEvaluator(current), samples, rules, histogram_bins=histogram_bins
        ):
            case Err() as error:
                return error
            case Ok(plan):
                pass
        match lower(current, plan):
            case Err() as error:
                return error
            case Ok(current):
                plans.append(plan)
        smoothing = {
            operation.node
            for operation in plan.operations
            if isinstance(operation, RescaleInput)
        }
        if ort_integer_scaling and pending_scaling and not smoothing:
            from quantsmith.backends.onnx.ort_scaling import lower_integer_scaling

            match lower_integer_scaling(current, pending_scaling):
                case Err() as error:
                    return error
                case Ok(current):
                    pending_scaling = frozenset()
        if smoothing:
            if ort_integer_scaling:
                pending_scaling = pending_scaling | frozenset(
                    node.input[0]
                    for node in current.graph.node
                    if node.name in smoothing
                )
            match optimize(current):
                case Err() as error:
                    return error
                case Ok(current):
                    pass
    return Ok(PipelineResult(model=current, plans=tuple(plans)))
