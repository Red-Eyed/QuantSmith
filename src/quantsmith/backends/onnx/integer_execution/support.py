"""Conservative CPU eligibility checks before numerical calibration or lowering."""

from onnx import ModelProto

from quantsmith.algorithms.static import StaticW8A8
from quantsmith.domain import Absent, Graph, IntegerType, Node


def restriction(
    model: ModelProto, graph: Graph, node: Node, method: StaticW8A8
) -> str | Absent:
    """Explain unsupported layouts so they retain their original FP32 computation."""
    proto = next(value for value in model.graph.node if value.name == node.name)
    if proto.domain or node.op not in {"Conv", "MatMul", "Gemm"}:
        return "operator is outside the ORT CPU integer contract"
    if len(node.inputs) < 2 or node.inputs[1] not in graph.weights:
        return "integer execution requires constant weights"
    weight = graph.weights[node.inputs[1]]
    if len(node.outputs) != 1:
        return "integer execution requires one output"
    if (
        method.activation_type == IntegerType.INT8
        and method.weight_type != IntegerType.INT8
    ):
        return "signed activations require signed weights in the CPU contract"
    if not method.weight_symmetric:
        return "asymmetric weights are outside the verified CPU contract"
    match node.op:
        case "Conv" if weight.ndim not in (3, 4, 5):
            return "integer Conv requires spatial rank one, two or three"
        case "MatMul" | "Gemm" if weight.ndim != 2:
            return "integer matrix operations require a rank-two constant weight"
        case "Gemm" if (
            node.attributes.get("alpha", 1.0) != 1
            or node.attributes.get("beta", 1.0) != 1
        ):
            return "integer Gemm currently requires alpha=beta=1"
    if node.op in {"Conv", "Gemm"} and len(node.inputs) > 2 and node.inputs[2]:
        bias = graph.weights.get(node.inputs[2])
        if bias is None or bias.ndim != 1:
            return "integer execution requires a constant vector bias"
        axis = 0 if node.op == "Conv" or node.attributes.get("transB", 0) else 1
        if bias.shape != (weight.shape[axis],):
            return "integer execution requires one bias value per output channel"
    return Absent(reason="layout satisfies the verified CPU contract")
