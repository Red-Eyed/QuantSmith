"""Owned graph rewriting shared by compact floating constant encoders."""

from collections.abc import Callable
from copy import deepcopy

import numpy as np
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx import Names
from quantsmith.domain import Absent, FloatArray

type Encoder = Callable[[FloatArray, str], TensorProto | Absent]


def store_constants(
    model: ModelProto, minimum_elements: int, encode: Encoder, suffix: str
) -> ModelProto:
    """Restore FP32 consumer names while leaving unsafe/overridable data intact."""
    if minimum_elements < 1:
        raise ValueError("minimum_elements must be positive")
    result = deepcopy(model)
    names = Names(result)
    names.used.update(value.name for value in result.graph.value_info)
    overridable = {value.name for value in result.graph.input}
    casts: list[NodeProto] = []
    for constant in result.graph.initializer:
        if constant.data_type != TensorProto.FLOAT or constant.name in overridable:
            continue
        values = np.asarray(numpy_helper.to_array(constant), dtype=np.float32)
        if values.size < minimum_elements or not np.all(np.isfinite(values)):
            continue
        original = constant.name
        stored = names.new(f"{original}_storage_{suffix}")
        match encode(values, stored):
            case Absent():
                continue
            case TensorProto() as encoded:
                constant.CopyFrom(encoded)
        casts.append(
            helper.make_node(
                "Cast",
                [stored],
                [original],
                name=names.new(f"{original}_restore_fp32"),
                to=TensorProto.FLOAT,
            )
        )
    nodes = tuple(result.graph.node)
    del result.graph.node[:]
    result.graph.node.extend((*casts, *nodes))
    return result
