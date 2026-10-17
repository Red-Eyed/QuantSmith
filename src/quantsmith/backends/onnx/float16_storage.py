"""Compact constant storage whose restoration ORT folds before FP32 execution."""

from dataclasses import dataclass, replace

import numpy as np
from onnx import ModelProto, TensorProto, numpy_helper

from quantsmith.backends.onnx import admit
from quantsmith.backends.onnx.constant_storage import store_constants
from quantsmith.domain import Absent, FloatArray
from quantsmith.pipeline import CalibrationContext, FloatGraph
from quantsmith.result import Err, Ok, QuantSmithError, Result


def encode_float16(values: FloatArray, name: str) -> TensorProto | Absent:
    """Retain tensors outside FP16's finite range instead of overflowing storage."""
    if np.any(np.abs(values) > np.finfo(np.float16).max):
        return Absent(reason="constant exceeds finite FP16 range")
    return numpy_helper.from_array(values.astype(np.float16), name)


def store_float16(model: ModelProto, minimum_elements: int) -> ModelProto:
    """Own the result and preserve FP32 consumer names and overridable inputs.

    Nonfinite constants and values outside FP16's finite range remain unchanged.
    Representable values round to FP16, including possible underflow to zero.
    """
    return store_constants(model, minimum_elements, encode_float16, "fp16")


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
