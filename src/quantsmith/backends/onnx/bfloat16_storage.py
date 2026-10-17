"""BF16 constant storage restored to FP32 by ORT initialization folding."""

from dataclasses import dataclass, replace

import numpy as np
from onnx import ModelProto, TensorProto

from quantsmith.backends.onnx import admit
from quantsmith.backends.onnx.constant_storage import store_constants
from quantsmith.domain import Absent, FloatArray
from quantsmith.pipeline import CalibrationContext, FloatGraph
from quantsmith.result import Err, Ok, QuantSmithError, Result


def encode_bfloat16(values: FloatArray, name: str) -> TensorProto | Absent:
    """Round finite FP32 to BF16 ties-to-even; retain tensors outside finite range."""
    bits = values.view(np.uint32)
    if np.any((bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F7F0000)):
        return Absent(reason="constant exceeds finite BF16 range")
    # The retained low bit selects even codes when the discarded half is exactly tied.
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    codes = (rounded >> np.uint32(16)).astype("<u2")
    return TensorProto(
        name=name,
        data_type=TensorProto.BFLOAT16,
        dims=values.shape,
        raw_data=codes.tobytes(),
    )


def store_bfloat16(model: ModelProto, minimum_elements: int) -> ModelProto:
    """Own compact BF16 constants while keeping computation and public names FP32."""
    return store_constants(model, minimum_elements, encode_bfloat16, "bf16")


@dataclass(frozen=True)
class BFloat16StorageStage:
    """Apply BF16 storage without collecting activation statistics."""

    minimum_elements: int

    def __post_init__(self) -> None:
        """Reject thresholds that would admit empty constants."""
        if self.minimum_elements < 1:
            raise ValueError("minimum_elements must be positive")

    def run(
        self, value: FloatGraph, context: CalibrationContext
    ) -> Result[FloatGraph, QuantSmithError]:
        """Retain source provenance without requesting calibration samples."""
        match admit(
            "BF16 constant storage",
            lambda: store_bfloat16(value.revision.model, self.minimum_elements),
        ):
            case Err() as error:
                return error
            case Ok(model):
                return Ok(replace(value, revision=replace(value.revision, model=model)))
