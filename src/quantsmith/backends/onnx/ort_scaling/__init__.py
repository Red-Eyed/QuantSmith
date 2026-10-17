"""Plan ORT integer rewrites before applying a single owned graph revision."""

from onnx import ModelProto, NodeProto

from quantsmith.backends.onnx import admit, normalize
from quantsmith.backends.onnx.ort_scaling.clipping import plan_clipping
from quantsmith.backends.onnx.ort_scaling.graph import (
    GraphIndex,
    RewritePatch,
    ScalingBoundary,
    apply_patches,
    index_graph,
    scaling_boundary,
)
from quantsmith.backends.onnx.ort_scaling.multiplication import plan_multiplication
from quantsmith.domain import Absent
from quantsmith.result import QuantSmithError, Result


def plan_boundary(
    index: GraphIndex, boundary: ScalingBoundary
) -> RewritePatch | Absent:
    """Choose the first applicable concrete rule without mutating graph state."""
    for rule in (plan_clipping, plan_multiplication):
        match rule(index, boundary):
            case RewritePatch() as patch:
                return patch
            case Absent():
                continue
    return Absent(reason="no supported integer rewrite for this boundary")


def _plan_node(index: GraphIndex, node: NodeProto) -> RewritePatch | Absent:
    """Recognize the quantization boundary before selecting a concrete rewrite."""
    match scaling_boundary(index, node):
        case Absent() as absent:
            return absent
        case ScalingBoundary() as boundary:
            return plan_boundary(index, boundary)


def _lower(model: ModelProto, divisions: frozenset[str]) -> ModelProto:
    """Recognize and plan every boundary, then transfer ownership to application."""
    owned = normalize(model)
    index = index_graph(owned)
    patches: list[RewritePatch] = []
    for node in owned.graph.node:
        if not node.output or node.output[0] not in divisions:
            continue
        match _plan_node(index, node):
            case Absent():
                continue
            case RewritePatch() as patch:
                patches.append(patch)
                index = index.reserve(patch)
    return apply_patches(owned, tuple(patches))


def lower_integer_scaling(
    model: ModelProto, divisions: frozenset[str]
) -> Result[ModelProto, QuantSmithError]:
    """Return an owned ORT graph while preserving unsupported patterns.

    Clipping rewrites change rounding; integer multiplication also quantizes
    reciprocals. Unsupported layouts and encodings retain their existing paths.
    """
    return admit("ORT integer scaling", lambda: _lower(model, divisions))
