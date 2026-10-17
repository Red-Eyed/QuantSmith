"""Pure exact-reuse planning and owned application for private activation quantizers."""

from copy import deepcopy
from dataclasses import dataclass

import onnx
from onnx import ModelProto, TensorProto

from quantsmith.backends.onnx import admit, normalize, remove_unused_initializers
from quantsmith.result import QuantSmithError, Result


@dataclass(frozen=True)
class QuantizerKey:
    """Require identical source, encoding representations and attributes."""

    source: str
    scale: bytes
    zero_point: bytes
    attributes: tuple[bytes, ...]


@dataclass(frozen=True)
class QuantizerReuse:
    """Replace one private quantized value with an earlier identical computation."""

    node: str
    original: str
    replacement: str


@dataclass(frozen=True)
class ReusePlan:
    """Own an immutable set of exact graph aliases without numerical changes."""

    aliases: tuple[QuantizerReuse, ...]


def fingerprint(tensor: TensorProto) -> bytes:
    """Ignore constant identity and retain its complete representation."""
    value = deepcopy(tensor)
    value.ClearField("name")
    return value.SerializeToString()


def plan_reuse(model: ModelProto) -> ReusePlan:
    """Select identical private quantizers without mutating a topological graph.

    Dynamic encodings, omitted zero points, public outputs and custom domains remain
    untouched. Semantically equivalent but differently serialized grids are not merged.
    """
    encodings = frozenset(
        name
        for node in model.graph.node
        if not node.domain and node.op_type == "QuantizeLinear" and len(node.input) == 3
        for name in node.input[1:]
    )
    overridable = {value.name for value in model.graph.input}
    constants = {
        tensor.name: fingerprint(tensor)
        for tensor in model.graph.initializer
        if tensor.name in encodings and tensor.name not in overridable
    }
    public = {value.name for value in model.graph.output}
    seen: dict[QuantizerKey, str] = {}
    aliases: list[QuantizerReuse] = []
    for node in model.graph.node:
        if (
            node.domain
            or node.op_type != "QuantizeLinear"
            or len(node.input) != 3
            or len(node.output) != 1
            or node.output[0] in public
        ):
            continue
        if node.input[1] not in constants or node.input[2] not in constants:
            continue
        key = QuantizerKey(
            node.input[0],
            constants[node.input[1]],
            constants[node.input[2]],
            tuple(attribute.SerializeToString() for attribute in node.attribute),
        )
        if key in seen:
            aliases.append(QuantizerReuse(node.name, node.output[0], seen[key]))
        else:
            seen[key] = node.output[0]
    return ReusePlan(tuple(aliases))


def _apply_owned(result: ModelProto, plan: ReusePlan) -> ModelProto:
    """Edit a transferred owned revision while preserving public tensor names."""
    replacements = {alias.original: alias.replacement for alias in plan.aliases}
    removed = {alias.node for alias in plan.aliases}
    for node in result.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = replacements.get(name, name)
    kept = [node for node in result.graph.node if node.name not in removed]
    metadata = [
        value for value in result.graph.value_info if value.name not in replacements
    ]
    del result.graph.node[:]
    result.graph.node.extend(kept)
    del result.graph.value_info[:]
    result.graph.value_info.extend(metadata)
    remove_unused_initializers(result)
    onnx.checker.check_model(result)
    return result


def apply_reuse(model: ModelProto, plan: ReusePlan) -> ModelProto:
    """Own a copy before applying a plan bound to the same normalized graph revision."""
    return _apply_owned(normalize(model), plan)


def reuse_quantizers(model: ModelProto) -> Result[ModelProto, QuantSmithError]:
    """Own and rewrite the graph, translating documented ONNX failures only."""

    def rewrite() -> ModelProto:
        """Bind the plan and application to the same owned normalized revision."""
        current = normalize(model)
        return _apply_owned(current, plan_reuse(current))

    return admit("reuse activation quantizers", rewrite)
