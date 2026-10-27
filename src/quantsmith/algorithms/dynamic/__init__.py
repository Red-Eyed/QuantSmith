"""Calibration-free dynamic W8A8 projection planning; see README.md."""

from pydantic import BaseModel, ConfigDict

from quantsmith.algorithms.contracts import Needs, Statistics
from quantsmith.algorithms.encoding import encode
from quantsmith.calibration import extrema
from quantsmith.domain import Graph, IntegerType, Node, PerChannel
from quantsmith.domain.layout import weight_axis
from quantsmith.plan import DynamicQuantizeGemm, DynamicQuantizeMatMul, QuantizationPlan
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
    """Plan constant-weight matrix projections without activation samples."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Reject unsupported projections and request no calibration statistics."""
        if len(node.outputs) != 1:
            return failure(
                FailureKind.UNSUPPORTED, node.name, "dynamic W8A8 requires one output"
            )
        match node.op:
            case "MatMul" if len(node.inputs) == 2 and not node.attributes:
                pass
            case "Gemm" if len(node.inputs) in (2, 3):
                if (
                    not node.attributes.keys() <= {"alpha", "beta", "transA", "transB"}
                    or node.attributes.get("alpha", 1) != 1
                    or node.attributes.get("beta", 1) != 1
                    or node.attributes.get("transA", 0) not in (0, 1)
                    or node.attributes.get("transB", 0) not in (0, 1)
                ):
                    return failure(
                        FailureKind.UNSUPPORTED,
                        node.name,
                        "dynamic Gemm requires unit alpha/beta "
                        "and valid transpose flags",
                    )
            case _:
                return failure(
                    FailureKind.UNSUPPORTED,
                    node.name,
                    "dynamic W8A8 requires a MatMul or Gemm",
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
        match weight_axis(node, graph):
            case Err() as error:
                return error
            case Ok(axis):
                pass
        operation_type = (
            DynamicQuantizeGemm if node.op == "Gemm" else DynamicQuantizeMatMul
        )
        match extrema(graph.weights[node.inputs[1]], (axis,)):
            case Err() as error:
                return error
            case Ok(weight_stats):
                pass
        return validate(
            node.name,
            lambda: QuantizationPlan(
                operations=(
                    operation_type(
                        node=node.name,
                        weight_encoding=encode(
                            weight_stats, IntegerType.INT8, True, PerChannel(axis=axis)
                        ),
                    ),
                )
            ),
        )
