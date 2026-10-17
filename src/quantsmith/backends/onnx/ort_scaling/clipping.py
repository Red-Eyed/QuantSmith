"""Pure recognition, validation, and planning of scaled integer clipping."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from onnx import NodeProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx.ort_scaling.graph import (
    GraphIndex,
    NodeReplacement,
    RewritePatch,
    ScalarEncoding,
    ScalingBoundary,
    fresh_names,
    producer_chain,
)
from quantsmith.domain import Absent, FloatArray


@dataclass(frozen=True)
class ClipMatch:
    """Name each node in a recognized exclusive Conv/Clip/Q/DQ scaling path."""

    boundary: ScalingBoundary
    convolution: NodeProto
    clip: NodeProto
    source_quantizer: NodeProto
    weight_dequantizer: NodeProto


@dataclass(frozen=True)
class WeightParameters:
    """Bind channel layout and optional bias to validated per-channel weights."""

    shape: tuple[int, ...]
    scale: FloatArray
    bias: FloatArray | Absent


@dataclass(frozen=True)
class ClipLimits:
    """Bind positive scalar clipping bound and zero-based UINT8 output encoding."""

    upper: FloatArray
    scale: FloatArray


@dataclass(frozen=True)
class ScaledParameters:
    """Hold calculated constants before any graph patch is constructed."""

    weight_scale: FloatArray
    bias: FloatArray | Absent
    bound: NDArray[np.uint8]


def match_clipping(index: GraphIndex, boundary: ScalingBoundary) -> ClipMatch | Absent:
    """Recognize topology and exclusive ownership independently of arithmetic."""
    match producer_chain(
        index, boundary.dequantizer.input[0], ("QuantizeLinear", "Clip", "Conv")
    ):
        case Absent() as absent:
            return absent
        case tuple() as nodes:
            quantizer, clip, convolution = nodes
    match index.producer(convolution.input[1], "DequantizeLinear"):
        case Absent() as absent:
            return absent
        case NodeProto() as weight:
            chain = (convolution, clip, quantizer, boundary.dequantizer, weight)
    if (
        len(clip.input) != 3
        or len(weight.input) != 3
        or not all(index.exclusive(node) for node in chain)
    ):
        return Absent(reason="clipping requires an exclusive Conv/Clip/QDQ chain")
    match index.scalar_encoding(quantizer), index.scalar_encoding(boundary.dequantizer):
        case ScalarEncoding() as encoded, ScalarEncoding() as decoded:
            if np.array_equal(encoded.scale, decoded.scale) and np.array_equal(
                encoded.zero, decoded.zero
            ):
                return ClipMatch(boundary, convolution, clip, quantizer, weight)
            return Absent(reason="source quantizer and dequantizer encodings differ")
        case _:
            return Absent(reason="source QDQ encodings are unsupported")


def _weight_parameters(
    index: GraphIndex, matched: ClipMatch
) -> WeightParameters | Absent:
    """Admit channel-axis integer weights and a shape-matched constant bias."""
    dequantizer = matched.weight_dequantizer
    weight = index.constants.get(dequantizer.input[0])
    axis = next((attr.i for attr in dequantizer.attribute if attr.name == "axis"), 1)
    match weight, index.float_constant(dequantizer.input[1]):
        case TensorProto() as tensor, np.ndarray() as scale:
            shape = tuple(tensor.dims)
        case _:
            return Absent(reason="weight codes and FP32 scales must be constant")
    if (
        tensor.data_type not in (TensorProto.INT8, TensorProto.UINT8)
        or len(shape) < 3
        or axis not in (0, -len(shape))
    ):
        return Absent(reason="weight layout is not channel-axis integer convolution")
    if scale.shape != (shape[0],) or np.any(scale <= 0):
        return Absent(reason="weight scales must match positive output channels")
    bias: FloatArray | Absent = Absent(reason="convolution has no bias")
    inputs = matched.convolution.input
    if len(inputs) > 2 and inputs[2]:
        match index.float_constant(inputs[2]):
            case Absent() as absent:
                return absent
            case np.ndarray() as values:
                if values.shape != scale.shape:
                    return Absent(reason="bias shape differs from output channels")
                bias = values
    return WeightParameters(shape, scale, bias)


def _clip_limits(index: GraphIndex, matched: ClipMatch) -> ClipLimits | Absent:
    """Validate ReLU-style scalar bounds and the target quantization origin."""
    clip = matched.clip
    match (
        index.float_constant(clip.input[1]),
        index.float_constant(clip.input[2]),
        index.scalar_encoding(matched.boundary.quantizer),
    ):
        case np.ndarray() as low, np.ndarray() as high, ScalarEncoding() as encoding:
            valid = (
                low.ndim == high.ndim == 0
                and low == 0
                and high > 0
                and encoding.zero == 0
            )
            if valid:
                return ClipLimits(high, encoding.scale)
            return Absent(reason="clipping requires zero lower bound and zero point")
        case _:
            return Absent(reason="clip limits or output encoding are unsupported")


def _scaled_parameters(
    weight: WeightParameters, limits: ClipLimits, divisor: FloatArray
) -> ScaledParameters | Absent:
    """Calculate private scales, bias, and integer bounds in the smoothed domain."""
    shape = (1, weight.shape[0], *((1,) * (len(weight.shape) - 2)))
    if divisor.shape != shape or np.any(divisor <= 0):
        return Absent(reason="divisor does not match positive output channels")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = weight.scale / divisor.reshape(-1)
        upper = limits.upper.astype(np.float64) / divisor / limits.scale
        match weight.bias:
            case Absent() as absent:
                bias: FloatArray | Absent = absent
            case np.ndarray() as values:
                bias = values / divisor.reshape(-1)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        return Absent(reason="scaled weight encoding is not finite and positive")
    match bias:
        case Absent():
            pass
        case np.ndarray() as values:
            if not np.all(np.isfinite(values)):
                return Absent(reason="scaled bias is not finite")
    bound = np.clip(np.rint(upper), 0, 255).astype(np.uint8)
    return ScaledParameters(scale, bias, bound)


def plan_clipping(
    index: GraphIndex, boundary: ScalingBoundary
) -> RewritePatch | Absent:
    """Compose topology recognition, numerical validation, and pure patch creation."""
    match match_clipping(index, boundary):
        case Absent() as absent:
            return absent
        case ClipMatch() as matched:
            pass
    match _weight_parameters(index, matched):
        case Absent() as absent:
            return absent
        case WeightParameters() as weight:
            pass
    match _clip_limits(index, matched):
        case Absent() as absent:
            return absent
        case ClipLimits() as limits:
            pass
    match index.float_constant(boundary.division.input[1]):
        case Absent() as absent:
            return absent
        case np.ndarray() as divisor:
            pass
    match _scaled_parameters(weight, limits, divisor):
        case Absent() as absent:
            return absent
        case ScaledParameters() as parameters:
            return _patch(index, matched, parameters)


def _patch(
    index: GraphIndex, matched: ClipMatch, parameters: ScaledParameters
) -> RewritePatch:
    """Construct new nodes/constants without editing the matched protobuf views."""
    conv = deepcopy(matched.convolution)
    weight = deepcopy(matched.weight_dequantizer)
    stem = conv.name
    scale, bias, bound, quantized, quantizer_name, minimum_name = fresh_names(
        index,
        stem + "_output_scale",
        stem + "_output_bias",
        stem + "_clip_bound",
        stem + "_before_clip",
        stem + "_output_Q",
        stem + "_int8_clip",
    )
    weight.input[1] = scale
    constants = [numpy_helper.from_array(parameters.weight_scale, scale)]
    match parameters.bias:
        case Absent():
            pass
        case np.ndarray() as values:
            conv.input[2] = bias
            constants.append(numpy_helper.from_array(values, bias))
    constants.append(numpy_helper.from_array(parameters.bound, bound))
    target = matched.boundary.quantizer
    quantizer = helper.make_node(
        "QuantizeLinear",
        [conv.output[0], *target.input[1:3]],
        [quantized],
        name=quantizer_name,
    )
    minimum = helper.make_node(
        "Min", [quantized, bound], list(target.output), name=minimum_name
    )
    removed = (
        matched.clip,
        matched.source_quantizer,
        matched.boundary.dequantizer,
        matched.boundary.division,
        target,
    )
    return RewritePatch(
        replacements=(
            NodeReplacement(weight.name, (weight,)),
            NodeReplacement(conv.name, (conv, quantizer, minimum)),
        ),
        removed=frozenset(node.name for node in removed),
        constants=tuple(constants),
    )
