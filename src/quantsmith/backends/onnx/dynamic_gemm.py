"""Lower selected Gemm projections through the shared dynamic integer contraction."""

from collections.abc import Callable

from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx import attributes
from quantsmith.backends.onnx.dynamic import (
    _require_float_activation,
    lower_dynamic_matmul,
)
from quantsmith.domain import Encoding, PerChannel
from quantsmith.plan import DynamicQuantizeGemm, DynamicQuantizeMatMul


def _matmul_weight(
    model: ModelProto,
    node: NodeProto,
    operation: DynamicQuantizeGemm,
    names: Callable[[str], str],
) -> tuple[str, Encoding]:
    """Normalize weight storage locally, preserving shared FP32 initializers."""
    transposed = attributes(node).get("transB", 0) == 1
    expected_axis = 0 if transposed else 1
    encoding = operation.weight_encoding
    if encoding.granularity != PerChannel(axis=expected_axis):
        raise ValueError(
            "dynamic Gemm encoding must follow the source output-channel axis"
        )
    if not transposed:
        return node.input[1], encoding
    constants = {value.name: value for value in model.graph.initializer}
    if node.input[1] not in constants:
        raise ValueError("dynamic Gemm requires constant FP32 weights")
    weight = constants[node.input[1]]
    weight_name = names(node.name + "_matmul_weight")
    model.graph.initializer.append(
        numpy_helper.from_array(numpy_helper.to_array(weight).T.copy(), weight_name)
    )
    return weight_name, Encoding(
        scale=encoding.scale,
        zero_point=encoding.zero_point,
        granularity=PerChannel(axis=1),
    )


def lower_dynamic_gemm(
    model: ModelProto,
    node: NodeProto,
    operation: DynamicQuantizeGemm,
    names: Callable[[str], str],
) -> list[NodeProto]:
    """Keep source identity, transpose semantics and optional FP32 bias broadcasting.

    Only unit alpha/beta Gemm is supported. The shared MatMul lowering owns the
    integer arithmetic; this adapter handles Gemm's layout and additive term.
    """
    layout = attributes(node)
    if (
        node.domain
        or node.op_type != "Gemm"
        or len(node.input) not in (2, 3)
        or len(node.output) != 1
        or not layout.keys() <= {"alpha", "beta", "transA", "transB"}
        or layout.get("alpha", 1) != 1
        or layout.get("beta", 1) != 1
        or layout.get("transA", 0) not in (0, 1)
        or layout.get("transB", 0) not in (0, 1)
    ):
        raise ValueError("dynamic Gemm requires standard layout and unit alpha/beta")
    _require_float_activation(model, node.input[0])
    activation = node.input[0]
    preceding: list[NodeProto] = []
    if layout.get("transA", 0) == 1:
        activation = names(node.name + "_activation")
        preceding.append(
            helper.make_node(
                "Transpose",
                [node.input[0]],
                [activation],
                name=names(node.name + "_transpose"),
                perm=[1, 0],
            )
        )
        model.graph.value_info.append(
            helper.make_tensor_value_info(activation, TensorProto.FLOAT, None)
        )
    weight, encoding = _matmul_weight(model, node, operation, names)
    bias = node.input[2] if len(node.input) == 3 else ""
    output = names(node.name + "_matmul_output") if bias else node.output[0]
    matmul = helper.make_node("MatMul", [activation, weight], [output], name=node.name)
    lowered = lower_dynamic_matmul(
        model,
        matmul,
        DynamicQuantizeMatMul(node=node.name, weight_encoding=encoding),
        names,
    )
    if bias:
        _require_float_activation(model, bias)
        lowered.append(
            helper.make_node(
                "Add", [output, bias], node.output, name=names(node.name + "_bias")
            )
        )
    return preceding + lowered
