"""Pure planning of integer scaling with ORT's quantized multiplication kernel."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from onnx import helper, numpy_helper

from quantsmith.backends.onnx.ort_scaling.graph import (
    GraphIndex,
    NodeReplacement,
    RewritePatch,
    ScalingBoundary,
    fresh_names,
)
from quantsmith.domain import Absent, FloatArray


@dataclass(frozen=True)
class ReciprocalEncoding:
    """Bind representable positive reciprocal codes to their scalar scale."""

    codes: NDArray[np.uint8]
    scale: FloatArray


def encode_reciprocal(divisor: FloatArray) -> ReciprocalEncoding | Absent:
    """Calculate coefficients without modifying the source or accepting zero codes."""
    if not divisor.size or np.any(divisor <= 0):
        return Absent(reason="scaling coefficients must be positive and nonempty")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        reciprocal = np.reciprocal(divisor)
        scale = np.array(np.max(reciprocal) / 255, dtype=np.float32)
    if (
        not np.all(np.isfinite(reciprocal))
        or not np.all(reciprocal > 0)
        or not np.isfinite(scale)
        or scale <= 0
    ):
        return Absent(reason="reciprocal encoding is not finite and positive")
    codes = np.clip(np.rint(reciprocal / scale), 0, 255).astype(np.uint8)
    if np.any(codes == 0):
        return Absent(reason="reciprocal range would erase positive coefficients")
    return ReciprocalEncoding(codes, scale)


def plan_multiplication(
    index: GraphIndex, boundary: ScalingBoundary
) -> RewritePatch | Absent:
    """Return a complete QLinearMul patch or a reason to preserve the float path."""
    if index.ort_versions not in ((), (1,)):
        return Absent(reason="unsupported com.microsoft opset")
    match index.float_constant(boundary.division.input[1]):
        case Absent() as absent:
            return absent
        case np.ndarray() as divisor:
            pass
    match encode_reciprocal(divisor):
        case Absent() as absent:
            return absent
        case ReciprocalEncoding() as encoding:
            return _patch(index, boundary, encoding)


def _patch(
    index: GraphIndex, boundary: ScalingBoundary, encoding: ReciprocalEncoding
) -> RewritePatch:
    """Materialize only new protobufs; existing nodes remain read-only inputs."""
    division, source, target = (
        boundary.division,
        boundary.dequantizer,
        boundary.quantizer,
    )
    code, zero, scale, operator = fresh_names(
        index,
        division.name + "_codes",
        division.name + "_zero",
        division.name + "_coefficient_scale",
        division.name + "_int8",
    )
    node = helper.make_node(
        "QLinearMul",
        [*source.input[:3], code, scale, zero, *target.input[1:3]],
        list(target.output),
        name=operator,
        domain="com.microsoft",
    )
    removed = {target.name}
    if index.exclusive(source):
        removed.add(source.name)
    return RewritePatch(
        replacements=(NodeReplacement(division.name, (node,)),),
        removed=frozenset(removed),
        constants=(
            numpy_helper.from_array(encoding.scale, scale),
            numpy_helper.from_array(encoding.codes, code),
            numpy_helper.from_array(np.array(0, dtype=np.uint8), zero),
        ),
        requires_ort=True,
    )
