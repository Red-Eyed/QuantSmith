"""Streaming per-channel histogram replay with an explicit bounded storage budget."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from quantsmith.calibration.contracts import (
    DEFAULT_CHANNEL_HISTOGRAM_BYTES,
    ChannelHistogramStats,
    ChannelRequirement,
    Evaluator,
    MinMaxStats,
    Samples,
)
from quantsmith.domain import FloatArray
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)


@dataclass(frozen=True)
class ChannelBins:
    """Privately own mutable replay counters until immutable statistics are returned."""

    edges: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    counts: np.ndarray[tuple[int, ...], np.dtype[np.int64]]


def channel_edges(
    bounds: MinMaxStats, bins: int
) -> Result[np.ndarray[tuple[int, ...], np.dtype[np.float64]], QuantSmithError]:
    """Create independent fixed boundaries, padding constant channels in float64."""
    if bounds.minimum.ndim != 1 or bins < 2:
        return failure(
            FailureKind.INVALID_DATA,
            "channel histogram",
            "channel ranges require vectors and at least two bins",
        )
    low = bounds.minimum.astype(np.float64)
    high = bounds.maximum.astype(np.float64)
    constant = low == high
    padding = np.maximum(np.abs(low) * 1e-6, 1e-6)
    low = np.where(constant, low - padding, low)
    high = np.where(constant, high + padding, high)
    return Ok(np.linspace(low, high, bins + 1, axis=-1, dtype=np.float64))


def accumulate(
    target: ChannelBins, values: FloatArray, axis: int
) -> Result[int, QuantSmithError]:
    """Count one channel view at a time; reject shape drift and out-of-range replay."""
    if values.dtype != np.float32 or not values.size:
        return failure(
            FailureKind.INVALID_DATA,
            "channel histogram",
            "calibration tensors must be nonempty finite float32 arrays",
        )
    if not -values.ndim <= axis < values.ndim:
        return failure(
            FailureKind.INVALID_DATA, "channel histogram", "channel axis out of range"
        )
    retained = axis % values.ndim
    if values.shape[retained] != target.counts.shape[0]:
        return failure(
            FailureKind.INVALID_DATA,
            "channel histogram",
            "retained calibration dimensions changed",
        )
    for channel in range(target.counts.shape[0]):
        selection = (slice(None),) * retained + (slice(channel, channel + 1),)
        view = values[selection]
        if not np.all(np.isfinite(view)):
            return failure(
                FailureKind.INVALID_DATA,
                "channel histogram",
                "calibration tensors must be nonempty finite float32 arrays",
            )
        counts, _ = np.histogram(view, bins=target.edges[channel])
        if int(counts.sum()) != view.size:
            return failure(
                FailureKind.INVALID_DATA,
                "channel histogram",
                "replay data exceeded channel calibration range",
            )
        target.counts[channel] += counts
    return Ok(values.size)


def channel_histograms(
    evaluator: Evaluator,
    samples: Samples,
    ranges: Mapping[ChannelRequirement, MinMaxStats],
    bins: int,
    *,
    max_bytes: int = DEFAULT_CHANNEL_HISTOGRAM_BYTES,
) -> Result[dict[ChannelRequirement, ChannelHistogramStats], QuantSmithError]:
    """Replay fixed channel bins without caching batches or exceeding histogram storage.

    The budget includes accumulator storage and the owned immutable result copies.
    Evaluator activations and per-channel counting temporaries are separate.
    """
    if bins < 2 or max_bytes < 1:
        return failure(
            FailureKind.INVALID_DATA,
            "channel histogram",
            "at least two bins and a positive memory budget are required",
        )
    if any(bounds.minimum.ndim != 1 for bounds in ranges.values()):
        return failure(
            FailureKind.INVALID_DATA,
            "channel histogram",
            "channel ranges require vectors",
        )
    channels = sum(bounds.minimum.size for bounds in ranges.values())
    required = 2 * channels * (bins * 8 + (bins + 1) * 8)
    if required > max_bytes:
        return failure(
            FailureKind.INVALID_DATA,
            "channel histogram",
            f"histogram workspace requires {required} bytes; budget is {max_bytes}",
        )
    accumulators: dict[ChannelRequirement, ChannelBins] = {}
    for request, bounds in ranges.items():
        match channel_edges(bounds, bins):
            case Err() as error:
                return error
            case Ok(edges):
                accumulators[request] = ChannelBins(
                    edges, np.zeros((bounds.minimum.size, bins), dtype=np.int64)
                )
    if not accumulators:
        return Ok({})
    outputs = tuple(dict.fromkeys(request.tensor for request in ranges))
    for sample in samples():
        match evaluator.run(sample, outputs):
            case Err() as error:
                return error
            case Ok(observed):
                pass
        for request, target in accumulators.items():
            if request.tensor not in observed:
                return failure(
                    FailureKind.INVALID_DATA,
                    "channel histogram",
                    f"missing output {request.tensor}",
                )
            match accumulate(target, observed[request.tensor], request.axis):
                case Err() as error:
                    return error
                case Ok():
                    pass
    return validate(
        "channel histogram",
        lambda: {
            request: ChannelHistogramStats(edges=target.edges, counts=target.counts)
            for request, target in accumulators.items()
        },
    )
