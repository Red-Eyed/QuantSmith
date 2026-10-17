"""Compact constant storage whose restoration ORT folds before FP32 execution."""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx import Names, admit
from quantsmith.pipeline import CalibrationContext, FloatGraph
from quantsmith.result import Err, Ok, QuantSmithError, Result


def store_float16(model: ModelProto, minimum_elements: int) -> ModelProto:
    """Own the result and preserve FP32 consumer names and overridable inputs.

    Nonfinite constants and values outside FP16's finite range remain unchanged.
    Representable values round to FP16, including possible underflow to zero.
    """
    result = deepcopy(model)
    names = Names(result)
    overridable = {value.name for value in result.graph.input}
    casts: list[NodeProto] = []
    for constant in result.graph.initializer:
        if constant.data_type != TensorProto.FLOAT or constant.name in overridable:
            continue
        values = numpy_helper.to_array(constant)
        if values.size < minimum_elements or not np.all(np.isfinite(values)):
            continue
        if np.any(np.abs(values) > np.finfo(np.float16).max):
            continue
        original = constant.name
        stored = names.new(f"{original}_storage_fp16")
        constant.CopyFrom(numpy_helper.from_array(values.astype(np.float16), stored))
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


@dataclass(frozen=True)
class Float16StorageStage:
    """Change serialized precision without collecting activation statistics."""

    minimum_elements: int

    def __post_init__(self) -> None:
        """Reject thresholds that would admit empty constants."""
        if self.minimum_elements < 1:
            raise ValueError("minimum_elements must be positive")

    def run(
        self, value: FloatGraph, context: CalibrationContext
    ) -> Result[FloatGraph, QuantSmithError]:
        """Retain source provenance without reading the supplied samples."""
        match admit(
            "FP16 constant storage",
            lambda: store_float16(value.revision.model, self.minimum_elements),
        ):
            case Err() as error:
                return error
            case Ok(model):
                return Ok(replace(value, revision=replace(value.revision, model=model)))
