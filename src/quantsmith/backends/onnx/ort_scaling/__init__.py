"""Optional ORT integer scaling with calibrated, inspectable rounding boundaries."""

from onnx import ModelProto

from quantsmith.backends.onnx import admit
from quantsmith.backends.onnx.ort_scaling.clipping import fold_clipping
from quantsmith.backends.onnx.ort_scaling.graph import GraphEditor
from quantsmith.backends.onnx.ort_scaling.multiplication import fold_multiplication
from quantsmith.result import QuantSmithError, Result


def _lower(model: ModelProto, divisions: frozenset[str]) -> ModelProto:
    """Try clipping fusion first, then integer multiplication on remaining scales."""
    editor = GraphEditor(model)
    for node in editor.model.graph.node:
        if not node.output or node.output[0] not in divisions:
            continue
        boundary = editor.boundary(node)
        if boundary is None:
            continue
        if not fold_clipping(editor, boundary):
            fold_multiplication(editor, boundary)
    return editor.finish()


def lower_integer_scaling(
    model: ModelProto, divisions: frozenset[str]
) -> Result[ModelProto, QuantSmithError]:
    """Own the output and preserve unsupported graph patterns unchanged.

    UINT8 activation boundaries and positive FP32 coefficients are required.
    Clipping rewrites change rounding; QLinearMul also quantizes reciprocals.
    The resulting graph requires ONNX Runtime's com.microsoft operator domain.
    """
    return admit("ORT integer scaling", lambda: _lower(model, divisions))
