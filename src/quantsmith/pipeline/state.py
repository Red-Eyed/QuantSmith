"""Distinct graph phases and explicit transformation outcomes for the ORT shell."""

from dataclasses import dataclass

from onnx import ModelProto
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.backends.onnx.integer_execution.audit import (
    ExecutionReport,
    UnverifiedNode,
)
from quantsmith.domain import Absent
from quantsmith.plan import QuantizationPlan


@dataclass(frozen=True)
class Folded:
    """Identify a smoothing operation whose introduced scale was absorbed."""

    node: str


@dataclass(frozen=True)
class Unfoldable:
    """Explain why an attempted smoothing operation left its original region intact."""

    node: str
    reason: str


type FoldingOutcome = Folded | Unfoldable


@dataclass(frozen=True)
class Revision:
    """Own model snapshots and plans; do not expose mutable caller-owned graphs."""

    source: ModelProto
    model: ModelProto
    plans: tuple[QuantizationPlan, ...] = ()
    folding: tuple[FoldingOutcome, ...] = ()


@dataclass(frozen=True)
class FloatGraph:
    """Permit floating transformations and fresh static calibration."""

    revision: Revision
    scaling: frozenset[str] = frozenset()


@dataclass(frozen=True)
class QuantizedGraph:
    """Carry a lowered graph that may be verified, but not smoothed again."""

    revision: Revision
    exclusions: tuple[UnverifiedNode, ...]


class QuantizationResult(BaseModel):
    """Expose the owned executable, source-oriented plans and verification evidence."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    model: ModelProto = Field()
    plans: tuple[QuantizationPlan, ...] = Field()
    execution: ExecutionReport | Absent = Field()
    folding: tuple[FoldingOutcome, ...] = Field(default=())
