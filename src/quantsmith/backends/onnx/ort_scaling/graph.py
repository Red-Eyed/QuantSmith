"""Owned graph editing and validated constants for optional ORT scaling rewrites."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

from quantsmith.backends.onnx import Names, normalize, remove_unused_initializers
from quantsmith.domain import FloatArray


def standard(node: NodeProto, operator: str) -> bool:
    """Accept only standard-domain operators with the expected identity."""
    return node.domain == "" and node.op_type == operator


@dataclass(frozen=True)
class ScalingBoundary:
    """Bind one exclusively consumed division to its input/output quantizers."""

    division: NodeProto
    dequantizer: NodeProto
    quantizer: NodeProto


class GraphEditor:
    """Own a normalized graph and apply validated rewrites without caller mutation."""

    def __init__(self, model: ModelProto) -> None:
        """Index flat connectivity once; stage removals and insertions until finish."""
        self.model = normalize(model)
        self.names = Names(self.model)
        self.constants = {value.name: value for value in self.model.graph.initializer}
        self.producers = {
            output: node for node in self.model.graph.node for output in node.output
        }
        self.consumers: dict[str, list[NodeProto]] = defaultdict(list)
        for node in self.model.graph.node:
            for name in node.input:
                self.consumers[name].append(node)
        self.outputs = {value.name for value in self.model.graph.output}
        self.removed: set[str] = set()
        self.insertions: dict[str, list[NodeProto]] = {}
        self.replacements: dict[str, list[NodeProto]] = {}

    def float_constant(self, name: str) -> FloatArray | None:
        """Borrow finite FP32 data from this editor's private graph."""
        tensor = self.constants.get(name)
        if tensor is None or tensor.data_type != TensorProto.FLOAT:
            return None
        value = np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
        return value if np.all(np.isfinite(value)) else None

    def scalar_encoding(
        self, node: NodeProto
    ) -> tuple[FloatArray, NDArray[np.uint8]] | None:
        """Admit finite positive scalar FP32 scale and scalar UINT8 zero point."""
        if len(node.input) != 3 or any(attr.name != "axis" for attr in node.attribute):
            return None
        scale = self.float_constant(node.input[1])
        zero_tensor = self.constants.get(node.input[2])
        if scale is None or scale.ndim != 0 or scale <= 0:
            return None
        if zero_tensor is None or zero_tensor.data_type != TensorProto.UINT8:
            return None
        zero = np.asarray(numpy_helper.to_array(zero_tensor), dtype=np.uint8)
        if zero.ndim != 0:
            return None
        return scale, zero

    def exclusive(self, node: NodeProto) -> bool:
        """Require an internal output with exactly one edge consuming its value."""
        return (
            len(node.output) == 1
            and node.output[0] not in self.outputs
            and len(self.consumers[node.output[0]]) == 1
        )

    def boundary(self, division: NodeProto) -> ScalingBoundary | None:
        """Recognize a UINT8 DQ/Div/Q boundary without hiding exposed outputs."""
        if not standard(division, "Div") or not self.exclusive(division):
            return None
        source = self.producers.get(division.input[0])
        target = self.consumers[division.output[0]][0]
        if source is None or not standard(source, "DequantizeLinear"):
            return None
        if not standard(target, "QuantizeLinear"):
            return None
        if self.scalar_encoding(source) is None or self.scalar_encoding(target) is None:
            return None
        return ScalingBoundary(division, source, target)

    def add_float(self, stem: str, value: FloatArray) -> str:
        """Append a private initializer after all rewrite checks have passed."""
        name = self.names.new(stem)
        tensor = numpy_helper.from_array(value, name)
        self.model.graph.initializer.append(tensor)
        self.constants[name] = tensor
        return name

    def finish(self) -> ModelProto:
        """Prune obsolete metadata/constants and validate the finished revision."""
        nodes: list[NodeProto] = []
        for node in self.model.graph.node:
            if node.name in self.replacements:
                nodes.extend(self.replacements[node.name])
            elif node.name not in self.removed:
                nodes.append(node)
                nodes.extend(self.insertions.get(node.name, ()))
        del self.model.graph.node[:]
        self.model.graph.node.extend(nodes)
        live = {output for node in nodes for output in node.output}
        live.update(value.name for value in self.model.graph.input)
        live.update(self.constants)
        metadata = [
            value for value in self.model.graph.value_info if value.name in live
        ]
        del self.model.graph.value_info[:]
        self.model.graph.value_info.extend(metadata)
        remove_unused_initializers(self.model)
        onnx.checker.check_model(self.model)
        return onnx.shape_inference.infer_shapes(self.model, strict_mode=True)
