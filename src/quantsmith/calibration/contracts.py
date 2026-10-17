"""Shared calibration records and protocols, independent of interval policies."""

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, Self, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quantsmith.domain import Absent, FloatArray, InputArray, frozen_array
from quantsmith.result import QuantSmithError, Result

Samples = Callable[[], Iterable[Mapping[str, InputArray]]]

DEFAULT_CHANNEL_HISTOGRAM_BYTES = 512 * 1024 * 1024


class Evaluator(Protocol):
    """Execute a graph for explicitly requested floating-point tensors."""

    def run(
        self, sample: Mapping[str, InputArray], outputs: tuple[str, ...]
    ) -> Result[Mapping[str, FloatArray], QuantSmithError]:
        """Return finite tensors or execution failure without retaining samples."""
        ...


class Requirement(BaseModel):
    """Request extrema reduced over all dimensions except retained axes."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    tensor: str = Field(min_length=1)
    axes: tuple[int, ...] = Field(default=())


class MinMaxStats(BaseModel):
    """Own finite elementwise lower and upper bounds."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    minimum: FloatArray = Field()
    maximum: FloatArray = Field()

    @field_validator("minimum", "maximum")
    @classmethod
    def freeze_extrema(cls, value: FloatArray) -> FloatArray:
        """Own finite, nonempty extrema independently of caller storage."""
        return frozen_array(value)

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Require compatible shapes and ordered extrema."""
        low, high = self.minimum, self.maximum
        if low.shape != high.shape or np.any(low > high):
            raise ValueError("invalid extrema")
        return self


class HistogramStats(BaseModel):
    """Store fixed edges and bounded bin counts independently of sample count."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    edges: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = Field()
    counts: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = Field()

    @field_validator("edges")
    @classmethod
    def freeze_edges(
        cls, value: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Own finite, increasing float64 bin boundaries."""
        if value.dtype != np.float64:
            raise ValueError("histograms require float64 edges")
        if (
            value.ndim != 1
            or not np.all(np.isfinite(value))
            or np.any(np.diff(value) <= 0)
        ):
            raise ValueError("histogram edges must be a finite increasing vector")
        edges = value.copy()
        edges.flags.writeable = False
        return edges

    @field_validator("counts")
    @classmethod
    def freeze_counts(
        cls, value: np.ndarray[tuple[int, ...], np.dtype[np.int64]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
        """Own a populated vector of nonnegative int64 bin counts."""
        if value.dtype != np.int64:
            raise ValueError("histograms require int64 counts")
        if value.ndim != 1 or np.any(value < 0) or value.sum() <= 0:
            raise ValueError("calibration replay yielded no values or invalid counts")
        counts = value.copy()
        counts.flags.writeable = False
        return counts

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Require one more bin boundary than count."""
        if self.edges.size != self.counts.size + 1:
            raise ValueError("invalid histogram dimensions")
        return self


@runtime_checkable
class Calibration(Protocol):
    """Estimate a tensor range from a shared collection of statistics."""

    @property
    def needs_histogram(self) -> bool:
        """Declare whether range estimation needs a replay pass."""
        ...

    def interval(
        self, stats: MinMaxStats, histogram: HistogramStats | Absent
    ) -> Result[MinMaxStats, QuantSmithError]:
        """Return a finite ordered interval or a typed failure."""
        ...


class ChannelRequirement(BaseModel):
    """Request one histogram per coordinate on a single retained tensor axis."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    tensor: str = Field(min_length=1)
    axis: int = Field()

    @property
    def range(self) -> Requirement:
        """Identify the channel extrema needed to establish replay bin boundaries."""
        return Requirement(tensor=self.tensor, axes=(self.axis,))


class ChannelHistogramStats(BaseModel):
    """Own channel-by-bin distributions without retaining calibration activations."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    edges: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = Field()
    counts: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = Field()

    @field_validator("edges")
    @classmethod
    def freeze_edges(
        cls, value: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Own finite, strictly increasing boundaries for each nonempty channel."""
        if value.dtype != np.float64:
            raise ValueError("channel histograms require float64 edges")
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] < 3:
            raise ValueError("channel histogram edges require finite increasing rows")
        for row in value:
            if not np.all(np.isfinite(row)) or np.any(row[1:] <= row[:-1]):
                raise ValueError(
                    "channel histogram edges require finite increasing rows"
                )
        edges = value.copy()
        edges.flags.writeable = False
        return edges

    @field_validator("counts")
    @classmethod
    def freeze_counts(
        cls, value: np.ndarray[tuple[int, ...], np.dtype[np.int64]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
        """Own nonnegative counts and require observations in every channel."""
        if value.dtype != np.int64:
            raise ValueError("channel histograms require int64 counts")
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] < 2:
            raise ValueError("channel histogram replay yielded empty or invalid counts")
        for row in value:
            if np.any(row < 0) or row.sum() <= 0:
                raise ValueError(
                    "channel histogram replay yielded empty or invalid counts"
                )
        counts = value.copy()
        counts.flags.writeable = False
        return counts

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Require matching channels and one additional edge per channel."""
        channels, bins = self.counts.shape
        if self.edges.shape != (channels, bins + 1):
            raise ValueError("invalid channel histogram dimensions")
        return self
