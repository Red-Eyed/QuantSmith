"""Explicit integer MatMul with FP32 output and unchanged input encodings."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from onnx import ModelProto, NodeProto, helper

from quantsmith.algorithms.contracts import Algorithm, Needs, Statistics
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.domain import Graph, IntegerType, Node, PerChannel, PerTensor
from quantsmith.plan import FloatOutput, Operation, QuantizationPlan, QuantizeInput
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.rules import Exclude


def float_matmul_algorithm(method: StaticW8A8) -> Algorithm | Exclude:
    """Limit the initial policy to the measured UINT8/INT8 execution contract."""
    if (
        method.activation_type != IntegerType.UINT8
        or method.weight_type != IntegerType.INT8
    ):
        return Exclude(
            reason="float-output MatMul requires verified "
            "UINT8 activations and INT8 weights"
        )
    return FloatMatMulQuantizer(method)


@dataclass(frozen=True)
class FloatMatMulQuantizer:
    """Request only input statistics; output requantization is intentionally absent."""

    algorithm: StaticW8A8

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Reuse static input requirements without collecting unused output ranges."""
        return self.algorithm.requirements(node, graph)

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Expose floating-output semantics alongside the original input encodings."""
        match self.algorithm.plan(node, graph, stats):
            case Err() as error:
                return error
            case Ok(plan):
                return Ok(
                    QuantizationPlan(
                        operations=plan.operations
                        + (FloatOutput(node=node.name, index=0),),
                        excluded=plan.excluded,
                    )
                )


def lower_float_matmul(
    model: ModelProto,
    node: NodeProto,
    operations: Sequence[Operation],
    preceding: list[NodeProto],
) -> list[NodeProto]:
    """Replace private input decoders with a direct ORT integer contraction."""
    if (
        node.domain
        or node.op_type != "MatMul"
        or len(node.input) != 2
        or len(node.output) != 1
    ):
        raise ValueError("float integer output requires a standard two-input MatMul")
    encodings = {
        operation.index: operation.encoding
        for operation in operations
        if isinstance(operation, QuantizeInput)
    }
    outputs = tuple(
        operation for operation in operations if isinstance(operation, FloatOutput)
    )
    if (
        len(operations) != 3
        or set(encodings) != {0, 1}
        or len(outputs) != 1
        or outputs[0].index != 0
    ):
        raise ValueError(
            "float MatMul requires two quantized inputs and one float output"
        )
    activation, weight = encodings[0], encodings[1]
    if activation.zero_point.dtype != np.uint8 or weight.zero_point.dtype != np.int8:
        raise ValueError(
            "float MatMul requires verified UINT8 activations and INT8 weights"
        )
    if not isinstance(
        activation.granularity, PerTensor
    ) or weight.granularity != PerChannel(axis=1):
        raise ValueError(
            "float MatMul requires scalar activation and output-column weight scales"
        )
    decoders = {
        value.output[0]: value
        for value in preceding
        if value.op_type == "DequantizeLinear" and not value.domain
    }
    if any(name not in decoders for name in node.input):
        raise ValueError("float MatMul requires private input decoders")
    left, right = (decoders[name] for name in node.input)
    if len(left.input) != 3 or len(right.input) != 3:
        raise ValueError("float MatMul requires explicit zero points")
    constants = {value.name: value for value in model.graph.initializer}
    if (
        right.input[0] not in constants
        or len(constants[right.input[0]].dims) != 2
        or np.any(weight.zero_point != 0)
    ):
        raise ValueError("float MatMul requires rank-two constant symmetric weights")
    replacement = helper.make_node(
        "MatMulIntegerToFloat",
        [
            left.input[0],
            right.input[0],
            left.input[1],
            right.input[1],
            left.input[2],
            right.input[2],
        ],
        list(node.output),
        name=node.name,
        domain="com.microsoft",
    )
    node.CopyFrom(replacement)
    if not any(item.domain == "com.microsoft" for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    removed = {left.name, right.name}
    return [value for value in preceding if value.name not in removed]
