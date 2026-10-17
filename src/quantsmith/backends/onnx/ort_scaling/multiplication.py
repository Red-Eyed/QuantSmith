"""Replace calibrated scaling boundaries with ORT's integer multiplication kernel."""

import numpy as np
from onnx import helper, numpy_helper

from quantsmith.backends.onnx.ort_scaling.graph import GraphEditor, ScalingBoundary


def fold_multiplication(editor: GraphEditor, boundary: ScalingBoundary) -> bool:
    """Quantize positive reciprocal coefficients, skipping unrepresentable cases."""
    divisor = editor.float_constant(boundary.division.input[1])
    if divisor is None or not divisor.size or np.any(divisor <= 0):
        return False
    versions = [
        value.version
        for value in editor.model.opset_import
        if value.domain == "com.microsoft"
    ]
    if versions and versions != [1]:
        return False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        reciprocal = np.reciprocal(divisor)
        scale = np.array(np.max(reciprocal) / 255, dtype=np.float32)
    if (
        not np.all(np.isfinite(reciprocal))
        or not np.all(reciprocal > 0)
        or not np.isfinite(scale)
        or scale <= 0
    ):
        return False
    codes = np.clip(np.rint(reciprocal / scale), 0, 255).astype(np.uint8)
    if np.any(codes == 0):
        return False
    division, source, target = (
        boundary.division,
        boundary.dequantizer,
        boundary.quantizer,
    )
    code_name = editor.names.new(division.name + "_codes")
    zero_name = editor.names.new(division.name + "_zero")
    scale_name = editor.add_float(division.name + "_coefficient_scale", scale)
    editor.model.graph.initializer.extend(
        [
            numpy_helper.from_array(codes, code_name),
            numpy_helper.from_array(np.array(0, dtype=np.uint8), zero_name),
        ]
    )
    editor.replacements[division.name] = [
        helper.make_node(
            "QLinearMul",
            [*source.input[:3], code_name, scale_name, zero_name, *target.input[1:3]],
            list(target.output),
            name=editor.names.new(division.name + "_int8"),
            domain="com.microsoft",
        )
    ]
    editor.removed.add(target.name)
    if editor.exclusive(source):
        editor.removed.add(source.name)
    if not versions:
        editor.model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return True
