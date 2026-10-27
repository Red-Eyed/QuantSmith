"""Calibration-free dynamic W8A8 projection planning; see README.md."""

from pydantic import BaseModel, ConfigDict

from quantsmith.algorithms.contracts import Needs, Statistics
from quantsmith.algorithms.encoding import encode
from quantsmith.calibration import extrema
from quantsmith.domain import Graph, IntegerType, Node, PerChannel
from quantsmith.plan import DynamicQuantizeMatMul, QuantizationPlan
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)


class DynamicW8A8(BaseModel):
    """Plan constant-weight MatMul without observing activation samples."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Reject unsupported projections and request no calibration statistics."""
        if (
            node.op != "MatMul"
            or len(node.inputs) != 2
            or len(node.outputs) != 1
            or node.attributes
        ):
            return failure(
                FailureKind.UNSUPPORTED, node.name, "dynamic W8A8 requires a MatMul"
            )
        if node.inputs[1] not in graph.weights:
            return failure(
                FailureKind.UNSUPPORTED, node.name, "constant FP32 weights required"
            )
        if graph.weights[node.inputs[1]].ndim != 2:
            return failure(
                FailureKind.UNSUPPORTED, node.name, "rank-two weights required"
            )
        return Ok(Needs())

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Encode owned weights offline and leave activation ranges unspecified."""
        match self.requirements(node, graph):
            case Err() as error:
                return error
            case Ok():
                pass
        match extrema(graph.weights[node.inputs[1]], (1,)):
            case Err() as error:
                return error
            case Ok(weight_stats):
                pass
        return validate(
            node.name,
            lambda: QuantizationPlan(
                operations=(
                    DynamicQuantizeMatMul(
                        node=node.name,
                        weight_encoding=encode(
                            weight_stats, IntegerType.INT8, True, PerChannel(axis=1)
                        ),
                    ),
                )
            ),
        )
