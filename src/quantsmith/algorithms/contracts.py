"""Backend-independent inputs and outputs for quantization algorithms."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantsmith.calibration import (
    ChannelHistogramStats,
    ChannelRequirement,
    HistogramStats,
    MinMaxStats,
    Requirement,
)
from quantsmith.domain import Graph, Node
from quantsmith.plan import QuantizationPlan
from quantsmith.result import QuantSmithError, Result


class Needs(BaseModel):
    """Declare reusable range and histogram requirements for an algorithm."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    ranges: tuple[Requirement, ...] = Field(default=())
    histograms: tuple[Requirement, ...] = Field(default=())
    channel_histograms: tuple[ChannelRequirement, ...] = Field(default=())


class Statistics(BaseModel):
    """Own read-only statistics mappings for exactly one graph revision."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    ranges: Mapping[Requirement, MinMaxStats] = Field()
    histograms: Mapping[Requirement, HistogramStats] = Field()
    channel_histograms: Mapping[ChannelRequirement, ChannelHistogramStats] = Field(
        default_factory=dict, validate_default=True
    )

    @field_validator("ranges")
    @classmethod
    def freeze_ranges(
        cls, value: Mapping[Requirement, MinMaxStats]
    ) -> Mapping[Requirement, MinMaxStats]:
        """Detach the range mapping from the caller's mutable collection."""
        return MappingProxyType(dict(value))

    @field_validator("histograms")
    @classmethod
    def freeze_histograms(
        cls, value: Mapping[Requirement, HistogramStats]
    ) -> Mapping[Requirement, HistogramStats]:
        """Detach the histogram mapping from the caller's mutable collection."""
        return MappingProxyType(dict(value))

    @field_validator("channel_histograms")
    @classmethod
    def freeze_channel_histograms(
        cls, value: Mapping[ChannelRequirement, ChannelHistogramStats]
    ) -> Mapping[ChannelRequirement, ChannelHistogramStats]:
        """Detach the channel distribution mapping from caller-owned mutable storage."""
        return MappingProxyType(dict(value))


@runtime_checkable
class Algorithm(Protocol):
    """Produce decisions without execution, I/O, or mutation of input records.

    Implementations return expected failures through Result. Unexpected plugin
    exceptions propagate; the protocol cannot enforce numerical correctness or purity.
    """

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Declare statistics needed for planning or reject an unsupported layout."""
        ...

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Return a plan from supplied statistics without executing the graph."""
        ...
