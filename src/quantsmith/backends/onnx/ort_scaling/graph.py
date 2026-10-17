"""Read-only graph queries and explicit application of planned ORT rewrites."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Self

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx import remove_unused_initializers
from quantsmith.domain import Absent, FloatArray


@dataclass(frozen=True)
class ScalarEncoding:
    """Bind validated scalar UINT8 quantization parameters."""

    scale: FloatArray
    zero: NDArray[np.uint8]


@dataclass(frozen=True)
class ScalingBoundary:
    """Bind one exclusively consumed division to its input/output quantizers."""

    division: NodeProto
    dequantizer: NodeProto
    quantizer: NodeProto


@dataclass(frozen=True)
class NodeReplacement:
    """Replace one node in place, preserving topological insertion order."""

    anchor: str
    nodes: tuple[NodeProto, ...]


@dataclass(frozen=True)
class RewritePatch:
    """Describe a complete rewrite without modifying the indexed graph."""

    replacements: tuple[NodeReplacement, ...]
    removed: frozenset[str]
    constants: tuple[TensorProto, ...]
    requires_ort: bool = False


@dataclass(frozen=True)
class GraphIndex:
    """Borrow read-only protobuf views from one privately owned graph revision."""

    constants: Mapping[str, TensorProto]
    producers: Mapping[str, NodeProto]
    consumers: Mapping[str, tuple[NodeProto, ...]]
    outputs: frozenset[str]
    names: frozenset[str]
    ort_versions: tuple[int, ...]

    def producer(self, value: str, operator: str) -> NodeProto | Absent:
        """Find a standard-domain producer without accepting a foreign namesake."""
        match self.producers.get(value):
            case NodeProto(domain="", op_type=kind) as node if kind == operator:
                return node
            case _:
                return Absent(reason=f"no standard {operator} producer for {value}")

    def float_constant(self, name: str) -> FloatArray | Absent:
        """Borrow finite FP32 initializer data without coercing other dtypes."""
        match self.constants.get(name):
            case TensorProto(data_type=TensorProto.FLOAT) as tensor:
                value = np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
                if np.all(np.isfinite(value)):
                    return value
                return Absent(reason=f"nonfinite constant: {name}")
            case _:
                return Absent(reason=f"no FP32 constant: {name}")

    def scalar_encoding(self, node: NodeProto) -> ScalarEncoding | Absent:
        """Validate positive scalar FP32 scale and scalar UINT8 zero point."""
        if len(node.input) != 3 or any(attr.name != "axis" for attr in node.attribute):
            return Absent(reason="unsupported quantizer parameters")
        match self.float_constant(node.input[1]), self.constants.get(node.input[2]):
            case np.ndarray() as scale, TensorProto(
                data_type=TensorProto.UINT8
            ) as tensor:
                zero = np.asarray(numpy_helper.to_array(tensor), dtype=np.uint8)
                if scale.ndim == 0 and scale > 0 and zero.ndim == 0:
                    return ScalarEncoding(scale, zero)
                return Absent(reason="encoding is not positive scalar UINT8")
            case _:
                return Absent(
                    reason="encoding requires FP32 scale and UINT8 zero point"
                )

    def exclusive(self, node: NodeProto) -> bool:
        """Require an internal output with exactly one consuming edge."""
        return (
            len(node.output) == 1
            and node.output[0] not in self.outputs
            and len(self.consumers.get(node.output[0], ())) == 1
        )

    def reserve(self, patch: RewritePatch) -> Self:
        """Return a naming revision without changing connectivity or graph storage."""
        names = {value.name for value in patch.constants}
        for replacement in patch.replacements:
            for node in replacement.nodes:
                names.add(node.name)
                names.update(node.output)
        return replace(self, names=self.names | names)


def index_graph(model: ModelProto) -> GraphIndex:
    """Index one owned normalized model; query functions never mutate its protobufs."""
    consumers: dict[str, list[NodeProto]] = defaultdict(list)
    names = {value.name for value in model.graph.initializer}
    names.update(
        value.name
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    )
    for node in model.graph.node:
        names.add(node.name)
        names.update(node.output)
        for name in node.input:
            consumers[name].append(node)
    return GraphIndex(
        constants=MappingProxyType(
            {value.name: value for value in model.graph.initializer}
        ),
        producers=MappingProxyType(
            {output: node for node in model.graph.node for output in node.output}
        ),
        consumers=MappingProxyType(
            {name: tuple(nodes) for name, nodes in consumers.items()}
        ),
        outputs=frozenset(value.name for value in model.graph.output),
        names=frozenset(names),
        ort_versions=tuple(
            value.version
            for value in model.opset_import
            if value.domain == "com.microsoft"
        ),
    )


def fresh_names(index: GraphIndex, *stems: str) -> tuple[str, ...]:
    """Allocate deterministic names locally, returning no mutation of the index."""
    reserved = set(index.names)
    allocated: list[str] = []
    for stem in stems:
        while stem in reserved:
            stem += "_"
        reserved.add(stem)
        allocated.append(stem)
    return tuple(allocated)


def producer_chain(
    index: GraphIndex, value: str, operators: tuple[str, ...]
) -> tuple[NodeProto, ...] | Absent:
    """Recognize a linear chain by following each producer's first input."""
    nodes: list[NodeProto] = []
    for operator in operators:
        match index.producer(value, operator):
            case Absent() as absent:
                return absent
            case NodeProto() as node:
                nodes.append(node)
                value = node.input[0]
    return tuple(nodes)


def scaling_boundary(
    index: GraphIndex, division: NodeProto
) -> ScalingBoundary | Absent:
    """Recognize an exclusive scalar-UINT8 DQ/Div/Q boundary."""
    if division.domain or division.op_type != "Div" or not index.exclusive(division):
        return Absent(reason="division is shared, exposed, or not standard ONNX")
    match index.producer(division.input[0], "DequantizeLinear"):
        case Absent() as absent:
            return absent
        case NodeProto() as source:
            target = index.consumers[division.output[0]][0]
    if target.domain or target.op_type != "QuantizeLinear":
        return Absent(reason="division does not feed a standard quantizer")
    match index.scalar_encoding(source), index.scalar_encoding(target):
        case ScalarEncoding(), ScalarEncoding():
            return ScalingBoundary(division, source, target)
        case _:
            return Absent(reason="boundary requires scalar UINT8 encodings")


def apply_patches(model: ModelProto, patches: tuple[RewritePatch, ...]) -> ModelProto:
    """Apply completed patches to a privately owned model and validate the revision.

    The caller transfers exclusive ownership here. Planning has already finished;
    no graph query or numerical calculation runs while protobufs are being edited.
    """
    replacements = {
        item.anchor: item.nodes for patch in patches for item in patch.replacements
    }
    removed = frozenset(name for patch in patches for name in patch.removed)
    nodes = tuple(
        replacement
        for node in model.graph.node
        if node.name not in removed
        for replacement in replacements.get(node.name, (node,))
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for patch in patches:
        model.graph.initializer.extend(patch.constants)
    if any(patch.requires_ort for patch in patches) and not any(
        value.domain == "com.microsoft" for value in model.opset_import
    ):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    live = {output for node in nodes for output in node.output}
    live.update(value.name for value in (*model.graph.input, *model.graph.initializer))
    metadata = [value for value in model.graph.value_info if value.name in live]
    del model.graph.value_info[:]
    model.graph.value_info.extend(metadata)
    remove_unused_initializers(model)
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)
