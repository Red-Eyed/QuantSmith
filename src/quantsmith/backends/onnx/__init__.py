"""ONNX parsing and edge-local QDQ lowering, isolated from algorithms."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import TypedDict, assert_never

import numpy as np
import onnx
from onnx import (
    AttributeProto,
    ModelProto,
    NodeProto,
    TensorProto,
    helper,
    numpy_helper,
)
from typing_extensions import TypeVar

from quantsmith.domain import (
    Encoding,
    FloatArray,
    Graph,
    Node,
    PerChannel,
    PerTensor,
)
from quantsmith.plan import (
    DynamicQuantizeGemm,
    DynamicQuantizeMatMul,
    FloatOutput,
    QuantizationPlan,
    QuantizeConstant,
    QuantizeInput,
    QuantizeOutput,
    RescaleInput,
)
from quantsmith.reconstruction.affine import codes
from quantsmith.result import FailureKind, QuantSmithError, Result, failure, validate

T = TypeVar("T", infer_variance=True)


def admit(operation: str, call: Callable[[], T]) -> Result[T, QuantSmithError]:
    """Translate ONNX checker/inference errors as well as validation and I/O errors."""
    try:
        return validate(operation, call)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as error:
        return failure(FailureKind.INVALID_DATA, operation, str(error))


class AxisAttribute(TypedDict, total=False):
    """Describe the optional ONNX per-channel axis keyword."""

    axis: int


def attributes(node: NodeProto) -> Mapping[str, int | float | tuple[int, ...]]:
    """Extract numeric attributes required by graph algorithms."""
    values: dict[str, int | float | tuple[int, ...]] = {}
    for attribute in node.attribute:
        match attribute.type:
            case AttributeProto.INT:
                values[attribute.name] = attribute.i
            case AttributeProto.FLOAT:
                values[attribute.name] = attribute.f
            case AttributeProto.INTS:
                values[attribute.name] = tuple(attribute.ints)
    return MappingProxyType(values)


def normalize(model: ModelProto) -> ModelProto:
    """Copy and validate a flat FP32 graph; assign unique stable node names."""
    result = deepcopy(model)
    onnx.checker.check_model(result)
    versions = [item.version for item in result.opset_import if item.domain == ""]
    if len(versions) != 1 or versions[0] < 13:
        raise ValueError("QDQ lowering requires ONNX opset >= 13")
    names: set[str] = {node.name for node in result.graph.node if node.name}
    if len(names) != sum(bool(node.name) for node in result.graph.node):
        raise ValueError("node names must be unique")
    for index, node in enumerate(result.graph.node):
        if any(
            attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
            for attr in node.attribute
        ):
            raise ValueError("control-flow subgraphs are not supported yet")
        if not node.name:
            candidate = f"quantsmith_node_{index}"
            while candidate in names:
                candidate += "_"
            node.name = candidate
            names.add(candidate)
    inputs = {value.name for value in result.graph.input}
    if inputs.intersection(value.name for value in result.graph.initializer):
        raise ValueError("overridable initializers must be frozen before quantization")
    return onnx.shape_inference.infer_shapes(result, strict_mode=True)


def _describe(model: ModelProto) -> Graph:
    """Validate and detach a graph into backend-independent immutable records."""
    normalized = normalize(model)
    weights: dict[str, FloatArray] = {}
    for tensor in normalized.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            weights[tensor.name] = np.asarray(
                numpy_helper.to_array(tensor), dtype=np.float32
            )
    nodes = tuple(
        Node(
            name=node.name,
            op=node.op_type
            if node.domain in ("", "ai.onnx")
            else f"{node.domain}::{node.op_type}",
            inputs=tuple(node.input),
            outputs=tuple(node.output),
            attributes=attributes(node),
        )
        for node in normalized.graph.node
    )
    return Graph(nodes=nodes, weights=MappingProxyType(weights))


def _load(path: Path) -> ModelProto:
    """Load external data and validate an ONNX graph; propagate I/O failures."""
    return normalize(onnx.load(path))


class Names:
    """Allocate collision-free names against existing graph identifiers."""

    def __init__(self, model: ModelProto) -> None:
        """Reserve both tensor and node names in the model."""
        self.used = {value.name for value in model.graph.initializer}
        self.used.update(value.name for value in model.graph.input)
        self.used.update(value.name for value in model.graph.output)
        self.used.update(value.name for value in model.graph.value_info)
        for node in model.graph.node:
            self.used.add(node.name)
            self.used.update(node.input)
            self.used.update(node.output)

    def new(self, stem: str) -> str:
        """Reserve a unique identifier derived from a readable stem."""
        candidate = stem
        while candidate in self.used:
            candidate += "_"
        self.used.add(candidate)
        return candidate


def qdq(
    model: ModelProto, node: NodeProto, operation: QuantizeInput, names: Names
) -> list[NodeProto]:
    """Store constant edges as integers; insert QDQ for runtime activations."""
    if operation.index >= len(node.input) or not node.input[operation.index]:
        raise ValueError("plan refers to a missing input")
    source = node.input[operation.index]
    encoding = operation.encoding
    validate_encoding(model, source, encoding)
    constant = next(
        (tensor for tensor in model.graph.initializer if tensor.name == source), None
    )
    if constant is not None:
        weight = np.asarray(numpy_helper.to_array(constant), dtype=np.float32)
        if not np.all(np.isfinite(weight)):
            raise ValueError("quantized constants must be finite")
        operation_codes = QuantizeConstant(
            node=operation.node,
            index=operation.index,
            values=codes(weight, encoding),
            encoding=encoding,
        )
        return quantized_constant(model, node, operation_codes, names)
    stem = names.new(f"{node.name}_input{operation.index}")
    scale_name, zero_name = (names.new(stem + "_scale"), names.new(stem + "_zero"))
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(encoding.scale, scale_name),
            numpy_helper.from_array(encoding.zero_point, zero_name),
        ]
    )
    quantized, restored = (names.new(stem + "_q"), names.new(stem + "_dq"))
    kwargs: AxisAttribute = {}
    match encoding.granularity:
        case PerChannel(axis=axis):
            kwargs["axis"] = axis
        case PerTensor():
            pass
        case _:
            assert_never(encoding.granularity)
    node.input[operation.index] = restored
    return [
        helper.make_node(
            "QuantizeLinear",
            [source, scale_name, zero_name],
            [quantized],
            name=names.new(stem + "_Q"),
            **kwargs,
        ),
        helper.make_node(
            "DequantizeLinear",
            [quantized, scale_name, zero_name],
            [restored],
            name=names.new(stem + "_DQ"),
            **kwargs,
        ),
    ]


def quantized_constant(
    model: ModelProto, node: NodeProto, operation: QuantizeConstant, names: Names
) -> list[NodeProto]:
    """Lower exact reconstructed codes without rewriting shared initializers."""
    if operation.index >= len(node.input):
        raise ValueError("constant plan refers to a missing input")
    source = node.input[operation.index]
    constants = {value.name: value for value in model.graph.initializer}
    if source not in constants:
        raise ValueError("reconstructed edge must have a constant initializer")
    if tuple(constants[source].dims) != operation.values.shape:
        raise ValueError("reconstructed constant shape differs from original")
    validate_encoding(model, source, operation.encoding)
    stem = names.new(f"{node.name}_input{operation.index}_reconstructed")
    integer, scale, zero, output = (
        names.new(stem + suffix) for suffix in ("_q", "_scale", "_zero", "_dq")
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(operation.values, integer),
            numpy_helper.from_array(operation.encoding.scale, scale),
            numpy_helper.from_array(operation.encoding.zero_point, zero),
        ]
    )
    kwargs: AxisAttribute = {}
    match operation.encoding.granularity:
        case PerChannel(axis=axis):
            kwargs["axis"] = axis
    node.input[operation.index] = output
    return [
        helper.make_node(
            "DequantizeLinear",
            [integer, scale, zero],
            [output],
            name=names.new(stem + "_DQ"),
            **kwargs,
        )
    ]


def validate_encoding(model: ModelProto, tensor: str, encoding: Encoding) -> None:
    """Require FP32 and check a per-channel encoding against known dimensions."""
    metadata = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    constants = {value.name: value for value in model.graph.initializer}
    if tensor in constants:
        constant = constants[tensor]
        dtype, shape = (constant.data_type, tuple(constant.dims))
    elif tensor in metadata:
        info = metadata[tensor].type.tensor_type
        dtype, shape = (info.elem_type, tuple(d.dim_value for d in info.shape.dim))
    else:
        raise ValueError(f"missing tensor metadata: {tensor}")
    if dtype != TensorProto.FLOAT:
        raise ValueError(f"quantization currently requires FP32: {tensor}")
    match encoding.granularity:
        case PerChannel(axis=axis):
            if not -len(shape) <= axis < len(shape):
                raise ValueError("quantization axis out of range")
            if shape[axis] != encoding.scale.size:
                raise ValueError("encoding size differs from channel dimension")
        case PerTensor():
            pass
        case _:
            assert_never(encoding.granularity)


def rescaled_weight(
    model: ModelProto, node: NodeProto, operation: RescaleInput
) -> FloatArray:
    """Validate contraction channels and calculate the replacement initializer."""
    constants = {value.name: value for value in model.graph.initializer}
    if len(node.input) < 2 or node.input[1] not in constants:
        raise ValueError("rescaling requires constant weights")
    tensor = constants[node.input[1]]
    if tensor.data_type != TensorProto.FLOAT:
        raise ValueError("rescaling requires FP32 weights")
    weight = np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
    weight_shape = [1] * weight.ndim
    if not -weight.ndim <= operation.weight_axis < weight.ndim:
        raise ValueError("weight rescaling axis out of range")
    if weight.shape[operation.weight_axis] != operation.scale.size:
        raise ValueError("weight rescaling channel mismatch")
    weight_shape[operation.weight_axis] = operation.scale.size
    transformed = weight * operation.scale.reshape(weight_shape)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("rescaling overflowed weights")
    return transformed


def activation_scale(
    model: ModelProto, node: NodeProto, operation: RescaleInput
) -> FloatArray:
    """Bind an activation scale to its validated broadcast layout."""
    metadata = {
        value.name: value
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    if node.input[0] not in metadata:
        raise ValueError("missing activation shape for rescaling")
    info = metadata[node.input[0]].type.tensor_type
    rank = len(info.shape.dim)
    if (
        info.elem_type != TensorProto.FLOAT
        or not -rank <= operation.activation_axis < rank
    ):
        raise ValueError("rescaling requires FP32 activations with a known rank")
    dimension = info.shape.dim[operation.activation_axis].dim_value
    if dimension and dimension != operation.scale.size:
        raise ValueError("activation rescaling channel mismatch")
    activation_shape = [1] * rank
    activation_shape[operation.activation_axis] = operation.scale.size
    return operation.scale.reshape(activation_shape)


def rescale(
    model: ModelProto, node: NodeProto, operation: RescaleInput, names: Names
) -> list[NodeProto]:
    """Lower an edge-local equivalent transform without changing shared weights."""
    transformed = rescaled_weight(model, node, operation)
    broadcast_scale = activation_scale(model, node, operation)
    weight_name = names.new(node.name + "_smooth_weight")
    scale_name = names.new(node.name + "_smooth_scale")
    activation_name = names.new(node.name + "_smooth_activation")
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(transformed, weight_name),
            numpy_helper.from_array(broadcast_scale, scale_name),
        ]
    )
    division = helper.make_node(
        "Div",
        [node.input[0], scale_name],
        [activation_name],
        name=names.new(node.name + "_smooth"),
    )
    node.input[0], node.input[1] = (activation_name, weight_name)
    return [division]


def _lower(model: ModelProto, plan: QuantizationPlan) -> ModelProto:
    """Apply a plan to a copy and check the resulting ordinary ONNX graph."""
    from quantsmith.backends.onnx.integer_execution.lowering import bias_qdq, output_qdq

    result = normalize(model)
    names = Names(result)
    known = {node.name for node in result.graph.node}
    if any(operation.node not in known for operation in plan.operations):
        raise ValueError("plan refers to an unknown node")
    if not set(plan.excluded) <= known:
        raise ValueError("plan excludes an unknown node")
    operations = {name: [] for name in known}
    for operation in plan.operations:
        operations[operation.node].append(operation)
    rewritten: list[NodeProto] = []
    for node in result.graph.node:
        dynamic = [
            operation
            for operation in operations[node.name]
            if isinstance(operation, DynamicQuantizeMatMul | DynamicQuantizeGemm)
        ]
        if dynamic:
            from quantsmith.backends.onnx.dynamic import lower_dynamic_matmul
            from quantsmith.backends.onnx.dynamic_gemm import lower_dynamic_gemm

            match dynamic[0]:
                case DynamicQuantizeMatMul() as operation:
                    rewritten.extend(
                        lower_dynamic_matmul(result, node, operation, names.new)
                    )
                case DynamicQuantizeGemm() as operation:
                    rewritten.extend(
                        lower_dynamic_gemm(result, node, operation, names.new)
                    )
            continue
        preceding = bias_qdq(result, node, operations[node.name], names.new)
        following: list[NodeProto] = []
        for operation in operations[node.name]:
            match operation:
                case QuantizeOutput():
                    following.extend(output_qdq(result, node, operation, names.new))
                case QuantizeInput():
                    preceding.extend(qdq(result, node, operation, names))
                case QuantizeConstant():
                    preceding.extend(quantized_constant(result, node, operation, names))
                case RescaleInput():
                    preceding.extend(rescale(result, node, operation, names))
                case FloatOutput():
                    pass
                case DynamicQuantizeMatMul() | DynamicQuantizeGemm():
                    raise AssertionError(
                        "dynamic operations are lowered as complete nodes"
                    )
                case _:
                    assert_never(operation)
        if any(
            isinstance(operation, FloatOutput) for operation in operations[node.name]
        ):
            from quantsmith.backends.onnx.integer_execution.float_output import (
                lower_float_matmul,
            )

            preceding = lower_float_matmul(
                result, node, operations[node.name], preceding
            )
        rewritten.extend(preceding)
        rewritten.append(node)
        rewritten.extend(following)
    del result.graph.node[:]
    result.graph.node.extend(rewritten)
    remove_unused_initializers(result)
    onnx.checker.check_model(result)
    return onnx.shape_inference.infer_shapes(result, strict_mode=True)


def remove_unused_initializers(model: ModelProto) -> None:
    """Discard replaced weights while preserving graph outputs and shared uses.

    Normalization rejects subgraphs, so every use is visible in this flat graph.
    """
    used = {name for node in model.graph.node for name in node.input}
    used.update(value.name for value in model.graph.output)
    retained = [tensor for tensor in model.graph.initializer if tensor.name in used]
    removed = {tensor.name for tensor in model.graph.initializer} - used
    metadata = [value for value in model.graph.value_info if value.name not in removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    del model.graph.value_info[:]
    model.graph.value_info.extend(metadata)


def describe(model: ModelProto) -> Result[Graph, QuantSmithError]:
    """Validate an external graph, returning typed admission failure details."""
    return admit("ONNX graph analysis", lambda: _describe(model))


def load(path: Path) -> Result[ModelProto, QuantSmithError]:
    """Load a file and expose expected I/O and graph validation failures."""
    return admit("ONNX loading", lambda: _load(path))


def lower(
    model: ModelProto, plan: QuantizationPlan
) -> Result[ModelProto, QuantSmithError]:
    """Lower a plan on a copy, returning expected graph validation failures."""
    return admit("ONNX lowering", lambda: _lower(model, plan))
