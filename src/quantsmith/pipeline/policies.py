"""Closed built-in alternatives with fields belonging only to their own variant."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeScaling(BaseModel):
    """Permit residual smoothing scales to execute at inference time."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["runtime"] = Field(default="runtime")


class FoldOnly(BaseModel):
    """Accept smoothing only when every introduced runtime scale can be absorbed."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["fold_only"] = Field(default="fold_only")


class JointFoldOnly(BaseModel):
    """Absorb a common scale only when its complete consumer group participates."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["joint_fold_only"] = Field(default="joint_fold_only")


type Folding = RuntimeScaling | FoldOnly | JointFoldOnly


class ProducerOutputs(BaseModel):
    """Observe the producer output for static output encodings."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["producer"] = Field(default="producer")


class FunctionOutputs(BaseModel):
    """Observe eligible private Relu/Clip outputs, changing clipping and rounding."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["function"] = Field(default="function")


type Outputs = ProducerOutputs | FunctionOutputs
