"""Runtime activation quantization with an offline weight encoding."""

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantsmith.domain import Encoding, PerChannel


class DynamicQuantizeMatMul(BaseModel):
    """Replace a projection with runtime UINT8 inputs and FP32 outputs.

    Activation scales and zero points are computed from each invocation's entire
    input tensor. Only the symmetric INT8 output-column weight encoding is fixed.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    node: str = Field(min_length=1)
    weight_encoding: Encoding = Field()

    @model_validator(mode="after")
    def validate_weight_encoding(self) -> Self:
        """Limit lowering to the verified rank-two symmetric INT8 contract."""
        encoding = self.weight_encoding
        if (
            encoding.granularity != PerChannel(axis=1)
            or encoding.zero_point.dtype != np.int8
            or np.any(encoding.zero_point != 0)
        ):
            raise ValueError(
                "dynamic MatMul requires symmetric INT8 output-column weights"
            )
        return self


class DynamicQuantizeGemm(BaseModel):
    """Own a Gemm's runtime UINT8 contraction and floating bias addition.

    Weight scales follow the source weight's output axis; the backend adapts
    transposed storage without changing other consumers of those weights.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    node: str = Field(min_length=1)
    weight_encoding: Encoding = Field()

    @model_validator(mode="after")
    def validate_weight_encoding(self) -> Self:
        """Require symmetric INT8 encoding on a rank-two output-channel axis."""
        encoding = self.weight_encoding
        if (
            encoding.granularity not in (PerChannel(axis=0), PerChannel(axis=1))
            or encoding.zero_point.dtype != np.int8
            or np.any(encoding.zero_point != 0)
        ):
            raise ValueError(
                "dynamic Gemm requires symmetric INT8 output-channel weights"
            )
        return self
