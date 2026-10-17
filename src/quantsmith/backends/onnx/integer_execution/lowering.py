"""Producer QDQ boundaries and constant INT32 bias preparation for ORT."""

from collections.abc import Callable, Sequence

import numpy as np
from onnx import ModelProto, NodeProto, helper, numpy_helper

from quantsmith.backends.onnx import validate_encoding
from quantsmith.domain import Encoding, PerChannel, PerTensor
from quantsmith.plan import Operation, QuantizeInput, QuantizeOutput


def output_qdq(
    model: ModelProto,
    node: NodeProto,
    operation: QuantizeOutput,
    names: Callable[[str], str],
) -> list[NodeProto]:
    """Retain graph-output and shared-consumer names after producer quantization."""
    if operation.index >= len(node.output):
        raise ValueError("plan refers to a missing output")
    source = node.output[operation.index]
    validate_encoding(model, source, operation.encoding)
    match operation.encoding.granularity:
        case PerTensor():
            pass
        case _:
            raise ValueError("integer execution requires per-tensor output encoding")
    raw, codes, scale, zero = (
        names(source + suffix) for suffix in ("_float", "_integer", "_scale", "_zero")
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(operation.encoding.scale, scale),
            numpy_helper.from_array(operation.encoding.zero_point, zero),
        ]
    )
    node.output[operation.index] = raw
    return [
        helper.make_node(
            "QuantizeLinear",
            [raw, scale, zero],
            [codes],
            name=names(node.name + "_output_Q"),
        ),
        helper.make_node(
            "DequantizeLinear",
            [codes, scale, zero],
            [source],
            name=names(node.name + "_output_DQ"),
        ),
    ]


def _input_encodings(operations: Sequence[Operation]) -> dict[int, Encoding]:
    """Extract only activation and weight input encodings from one node's plan."""
    return {
        operation.index: operation.encoding
        for operation in operations
        if isinstance(operation, QuantizeInput)
    }


def bias_qdq(
    model: ModelProto,
    node: NodeProto,
    operations: Sequence[Operation],
    names: Callable[[str], str],
) -> list[NodeProto]:
    """Bind constant biases to accumulator scales; reject unsafe conversions."""
    if node.op_type not in ("Conv", "Gemm") or len(node.input) < 3 or not node.input[2]:
        return []
    if not any(isinstance(operation, QuantizeOutput) for operation in operations):
        return []
    encodings = _input_encodings(operations)
    if 0 not in encodings or 1 not in encodings:
        raise ValueError("integer bias requires activation and weight encodings")
    constants = {tensor.name: tensor for tensor in model.graph.initializer}
    if node.input[2] not in constants:
        raise ValueError("integer execution requires constant bias")
    bias = numpy_helper.to_array(constants[node.input[2]])
    if bias.dtype != np.float32 or bias.ndim != 1:
        raise ValueError("integer execution requires a one-dimensional FP32 bias")
    weight = encodings[1]
    match weight.granularity:
        case PerChannel():
            if bias.shape != weight.scale.shape:
                raise ValueError("bias and output-channel counts differ")
        case _:
            raise ValueError("integer execution requires per-channel weight encoding")
    scale = encodings[0].scale * weight.scale
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("bias accumulator scales are not representable in FP32")
    rounded = np.rint(bias.astype(np.float64) / scale.astype(np.float64))
    bounds = np.iinfo(np.int32)
    if (
        not np.all(np.isfinite(rounded))
        or np.any(rounded < bounds.min)
        or np.any(rounded > bounds.max)
    ):
        raise ValueError("bias cannot be represented in INT32")
    integer, scale_name, zero, restored = (
        names(node.name + suffix)
        for suffix in ("_bias_integer", "_bias_scale", "_bias_zero", "_bias_float")
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(rounded.astype(np.int32), integer),
            numpy_helper.from_array(scale, scale_name),
            numpy_helper.from_array(np.zeros(scale.shape, dtype=np.int32), zero),
        ]
    )
    node.input[2] = restored
    return [
        helper.make_node(
            "DequantizeLinear",
            [integer, scale_name, zero],
            [restored],
            axis=0,
            name=names(node.name + "_bias_DQ"),
        )
    ]
