"""Absorb channel smoothing into Conv output encodings and integer ReLU6 bounds."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from onnx import NodeProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx.ort_scaling.graph import (
    GraphEditor,
    ScalingBoundary,
    standard,
)
from quantsmith.domain import Absent, FloatArray


@dataclass(frozen=True)
class ClipRewrite:
    """Hold a fully checked rewrite before any graph parameters are modified."""

    boundary: ScalingBoundary
    convolution: NodeProto
    clip: NodeProto
    source_quantizer: NodeProto
    weight_dequantizer: NodeProto
    weight_scale: FloatArray
    bound: NDArray[np.uint8]
    bias: FloatArray | Absent


def _chain(
    editor: GraphEditor, boundary: ScalingBoundary
) -> tuple[NodeProto, NodeProto, NodeProto, NodeProto] | None:
    """Require exclusive Conv/Clip/Q/DQ edges and matching source QDQ encodings."""
    dq = boundary.dequantizer
    q = editor.producers.get(dq.input[0])
    if q is None or not standard(q, "QuantizeLinear"):
        return None
    encoding = editor.scalar_encoding(q)
    restored = editor.scalar_encoding(dq)
    if encoding is None or restored is None:
        return None
    if not all(
        np.array_equal(left, right)
        for left, right in zip(encoding, restored, strict=True)
    ):
        return None
    clip = editor.producers.get(q.input[0])
    if clip is None or not standard(clip, "Clip") or len(clip.input) != 3:
        return None
    conv = editor.producers.get(clip.input[0])
    if conv is None or not standard(conv, "Conv"):
        return None
    wdq = editor.producers.get(conv.input[1])
    if wdq is None or not standard(wdq, "DequantizeLinear") or len(wdq.input) != 3:
        return None
    if any(not editor.exclusive(node) for node in (conv, clip, q, dq, wdq)):
        return None
    return conv, clip, q, wdq


def _rewrite(editor: GraphEditor, boundary: ScalingBoundary) -> ClipRewrite | None:
    """Calculate a safe positive channel rewrite with a zero-based UINT8 output."""
    chain = _chain(editor, boundary)
    if chain is None:
        return None
    conv, clip, q, wdq = chain
    divisor = editor.float_constant(boundary.division.input[1])
    weight_scale = editor.float_constant(wdq.input[1])
    weight = editor.constants.get(wdq.input[0])
    low = editor.float_constant(clip.input[1])
    high = editor.float_constant(clip.input[2])
    encoding = editor.scalar_encoding(boundary.quantizer)
    axis = next((attr.i for attr in wdq.attribute if attr.name == "axis"), 1)
    if divisor is None or weight_scale is None or weight is None or encoding is None:
        return None
    if (
        weight.data_type not in (TensorProto.INT8, TensorProto.UINT8)
        or len(weight.dims) < 3
        or axis not in (0, -len(weight.dims))
    ):
        return None
    if divisor.shape != (1, weight.dims[0], *((1,) * (len(weight.dims) - 2))):
        return None
    if (
        np.any(divisor <= 0)
        or weight_scale.shape != (weight.dims[0],)
        or np.any(weight_scale <= 0)
    ):
        return None
    scale, zero = encoding
    if (
        low is None
        or high is None
        or low.ndim != 0
        or high.ndim != 0
        or low != 0
        or high <= 0
        or zero != 0
    ):
        return None
    bias: FloatArray | Absent = Absent(reason="convolution has no bias")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = weight_scale / divisor.reshape(-1)
        if len(conv.input) > 2 and conv.input[2]:
            original_bias = editor.float_constant(conv.input[2])
            if original_bias is None or original_bias.shape != weight_scale.shape:
                return None
            bias = original_bias / divisor.reshape(-1)
        upper = high.astype(np.float64) / divisor / scale
    if not np.all(np.isfinite(transformed)) or np.any(transformed <= 0):
        return None
    if not isinstance(bias, Absent) and not np.all(np.isfinite(bias)):
        return None
    bound = np.clip(np.rint(upper), 0, 255).astype(np.uint8)
    return ClipRewrite(boundary, conv, clip, q, wdq, transformed, bound, bias)


def fold_clipping(editor: GraphEditor, boundary: ScalingBoundary) -> bool:
    """Commit a checked rewrite; rounding changes are part of the opt-in mode."""
    rewrite = _rewrite(editor, boundary)
    if rewrite is None:
        return False
    conv = rewrite.convolution
    rewrite.weight_dequantizer.input[1] = editor.add_float(
        conv.name + "_output_scale", rewrite.weight_scale
    )
    if not isinstance(rewrite.bias, Absent):
        conv.input[2] = editor.add_float(conv.name + "_output_bias", rewrite.bias)
    bound_name = editor.names.new(conv.name + "_clip_bound")
    editor.model.graph.initializer.append(
        numpy_helper.from_array(rewrite.bound, bound_name)
    )
    quantized = editor.names.new(conv.name + "_before_clip")
    target = boundary.quantizer
    editor.insertions[conv.name] = [
        helper.make_node(
            "QuantizeLinear",
            [conv.output[0], *target.input[1:3]],
            [quantized],
            name=editor.names.new(conv.name + "_output_Q"),
        ),
        helper.make_node(
            "Min",
            [quantized, bound_name],
            list(target.output),
            name=editor.names.new(conv.name + "_int8_clip"),
        ),
    ]
    editor.removed.update(
        node.name
        for node in (
            rewrite.clip,
            rewrite.source_quantizer,
            boundary.dequantizer,
            boundary.division,
            target,
        )
    )
    return True
