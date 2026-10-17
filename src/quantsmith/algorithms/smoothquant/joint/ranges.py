"""Typed per-channel range policies for histogram-informed offline smoothing."""

from typing import Literal, assert_never

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms.encoding import encode
from quantsmith.calibration import ChannelHistogramStats, MinMaxStats
from quantsmith.domain import Absent, IntegerType, PerTensor
from quantsmith.result import (
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)


class ChannelExtrema(BaseModel):
    """Use each channel's complete observed range without histogram replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["extrema"] = Field(default="extrema")


class ChannelPercentile(BaseModel):
    """Balance channels using central histogram intervals instead of rare extrema."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["percentile"] = Field(default="percentile")
    percentile: float = Field(default=99.99, gt=0, le=100, allow_inf_nan=False)


class ChannelMSE(BaseModel):
    """Minimize a histogram-weighted per-channel asymmetric eight-bit error proxy."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["mse"] = Field(default="mse")
    min_percentile: float = Field(default=99.0, gt=0, le=100, allow_inf_nan=False)
    candidates: int = Field(default=16, ge=2, strict=True)


type ChannelSelection = ChannelExtrema | ChannelPercentile | ChannelMSE


def interval_at(
    observed: MinMaxStats,
    histogram: ChannelHistogramStats,
    channel: int,
    percentile: float,
) -> tuple[float, float]:
    """Choose enclosing bin edges with the established equal-tail percentile rule."""
    low, high = float(observed.minimum[channel]), float(observed.maximum[channel])
    if percentile == 100:
        return low, high
    cumulative = np.cumsum(histogram.counts[channel])
    tail = (100 - percentile) / 200
    first = int(np.searchsorted(cumulative, cumulative[-1] * tail, side="right"))
    last = int(np.searchsorted(cumulative, cumulative[-1] * (1 - tail)))
    return max(low, float(histogram.edges[channel, first])), min(
        high, float(histogram.edges[channel, last + 1])
    )


def error_at(
    points: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    counts: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    interval: tuple[float, float],
) -> float:
    """Estimate clipping plus rounding error on the full original channel histogram."""
    low, high = interval
    encoding = encode(
        MinMaxStats(
            minimum=np.asarray(low, dtype=np.float32),
            maximum=np.asarray(high, dtype=np.float32),
        ),
        IntegerType.UINT8,
        False,
        PerTensor(),
    )
    scale, zero = float(encoding.scale), int(encoding.zero_point)
    quantized = np.clip(np.rint(points / scale) + zero, 0, 255)
    residual = (quantized - zero) * scale - points
    return float(np.dot(counts.astype(np.float64), residual * residual) / counts.sum())


def mse_interval(
    observed: MinMaxStats,
    histogram: ChannelHistogramStats,
    channel: int,
    policy: ChannelMSE,
) -> tuple[float, float]:
    """Include the untrimmed interval and prefer it when proxy errors tie."""
    best = interval_at(observed, histogram, channel, 100)
    edges = histogram.edges[channel]
    points = np.clip(
        (edges[:-1] + edges[1:]) / 2,
        observed.minimum[channel],
        observed.maximum[channel],
    )
    counts = histogram.counts[channel]
    best_error = error_at(points, counts, best)
    for percentile in np.linspace(policy.min_percentile, 100, policy.candidates):
        candidate = interval_at(observed, histogram, channel, float(percentile))
        error = error_at(points, counts, candidate)
        if error < best_error:
            best, best_error = candidate, error
    return best


def selected_intervals(
    policy: ChannelPercentile | ChannelMSE,
    observed: MinMaxStats,
    histogram: ChannelHistogramStats,
) -> MinMaxStats:
    """Select independent channel ranges with only one channel's search temporaries."""
    minimum = np.empty_like(observed.minimum)
    maximum = np.empty_like(observed.maximum)
    for channel in range(minimum.size):
        match policy:
            case ChannelPercentile(percentile=percentile):
                low, high = interval_at(observed, histogram, channel, percentile)
            case ChannelMSE():
                low, high = mse_interval(observed, histogram, channel, policy)
            case _:
                assert_never(policy)
        minimum[channel], maximum[channel] = low, high
    return MinMaxStats(minimum=minimum, maximum=maximum)


def select_ranges(
    policy: ChannelSelection,
    observed: MinMaxStats,
    histogram: ChannelHistogramStats | Absent,
) -> Result[MinMaxStats, QuantSmithError]:
    """Choose balancing ranges without adding clipping to the executable graph."""
    match policy:
        case ChannelExtrema():
            return Ok(observed)
        case ChannelPercentile() | ChannelMSE():
            match histogram:
                case Absent(reason=reason):
                    return failure(FailureKind.INVALID_DATA, "channel ranges", reason)
                case ChannelHistogramStats():
                    pass
            if (
                observed.minimum.ndim != 1
                or observed.minimum.size != histogram.counts.shape[0]
            ):
                return failure(
                    FailureKind.INVALID_DATA,
                    "channel ranges",
                    "channel histogram dimensions differ from observed ranges",
                )
            if np.any(histogram.edges[:, 0] > observed.minimum) or np.any(
                histogram.edges[:, -1] < observed.maximum
            ):
                return failure(
                    FailureKind.INVALID_DATA,
                    "channel ranges",
                    "histogram edges do not contain observed channel ranges",
                )
            return validate(
                "channel ranges",
                lambda: selected_intervals(policy, observed, histogram),
            )
        case _:
            assert_never(policy)
