"""Replayable sample sources and bounded reusable statistics collection."""

from collections.abc import Mapping

import numpy as np

from quantsmith.calibration.channel_histogram import channel_histograms
from quantsmith.calibration.contracts import (
    Calibration,
    ChannelHistogramStats,
    ChannelRequirement,
    Evaluator,
    HistogramStats,
    MinMaxStats,
    Requirement,
    Samples,
)
from quantsmith.calibration.minmax import MinMax
from quantsmith.calibration.percentile import Percentile
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

__all__ = [
    "Calibration",
    "ChannelHistogramStats",
    "ChannelRequirement",
    "Evaluator",
    "HistogramStats",
    "MinMax",
    "MinMaxStats",
    "Percentile",
    "Requirement",
    "Samples",
    "channel_histograms",
    "collect",
    "count_histogram",
    "extrema",
    "histogram_edges",
    "histograms",
    "merge_extrema",
]


def extrema(
    values: FloatArray, axes: tuple[int, ...]
) -> Result[MinMaxStats, QuantSmithError]:
    """Reduce finite nonempty tensors or explain an invalid numerical observation."""
    if not values.size or not np.all(np.isfinite(values)):
        return failure(
            FailureKind.INVALID_DATA,
            "extrema",
            "calibration tensors must be nonempty and finite",
        )
    if any(axis < -values.ndim or axis >= values.ndim for axis in axes):
        return failure(
            FailureKind.INVALID_DATA, "extrema", "calibration axis out of range"
        )
    retained = {axis % values.ndim for axis in axes}
    reduced = tuple(axis for axis in range(values.ndim) if axis not in retained)
    return validate(
        "extrema",
        lambda: MinMaxStats(
            minimum=np.asarray(np.min(values, axis=reduced), dtype=np.float32),
            maximum=np.asarray(np.max(values, axis=reduced), dtype=np.float32),
        ),
    )


def merge_extrema(
    previous: MinMaxStats, current: MinMaxStats
) -> Result[MinMaxStats, QuantSmithError]:
    """Merge compatible channel statistics without hiding dimension drift."""
    if previous.minimum.shape != current.minimum.shape:
        return failure(
            FailureKind.INVALID_DATA,
            "calibration",
            "retained calibration dimensions changed",
        )
    return validate(
        "calibration",
        lambda: MinMaxStats(
            minimum=np.asarray(
                np.minimum(previous.minimum, current.minimum), dtype=np.float32
            ),
            maximum=np.asarray(
                np.maximum(previous.maximum, current.maximum), dtype=np.float32
            ),
        ),
    )


def collect(
    evaluator: Evaluator, samples: Samples, requirements: tuple[Requirement, ...]
) -> Result[dict[Requirement, MinMaxStats], QuantSmithError]:
    """Stream deduplicated observations and return expected data failures."""
    unique = tuple(dict.fromkeys(requirements))
    outputs = tuple(dict.fromkeys(item.tensor for item in unique))
    stats: dict[Requirement, MinMaxStats] = {}
    if not unique:
        return Ok(stats)
    for sample in samples():
        match evaluator.run(sample, outputs):
            case Err() as error:
                return error
            case Ok(observed):
                pass
        for item in unique:
            if item.tensor not in observed:
                return failure(
                    FailureKind.INVALID_DATA,
                    "calibration",
                    f"missing output {item.tensor}",
                )
            match extrema(observed[item.tensor], item.axes):
                case Err() as error:
                    return error
                case Ok(current):
                    pass
            if item in stats:
                match merge_extrema(stats[item], current):
                    case Err() as error:
                        return error
                    case Ok(current):
                        pass
            stats[item] = current
    if not stats:
        return failure(
            FailureKind.EMPTY, "calibration", "calibration source yielded no samples"
        )
    return Ok(stats)


def histogram_edges(
    bounds: MinMaxStats, bins: int
) -> Result[np.ndarray[tuple[int, ...], np.dtype[np.float64]], QuantSmithError]:
    """Create scalar fixed bins or report unsupported range dimensions."""
    if bounds.minimum.ndim != 0 or bins < 2:
        return failure(
            FailureKind.INVALID_DATA,
            "histogram",
            "histograms require scalar ranges and at least two bins",
        )
    low, high = float(bounds.minimum), float(bounds.maximum)
    if low == high:
        padding = max(abs(low) * 1e-6, 1e-6)
        low, high = low - padding, high + padding
    return Ok(np.linspace(low, high, bins + 1, dtype=np.float64))


def histograms(
    evaluator: Evaluator,
    samples: Samples,
    ranges: Mapping[Requirement, MinMaxStats],
    bins: int,
) -> Result[dict[Requirement, HistogramStats], QuantSmithError]:
    """Replay into fixed bins; expose invalid replay instead of throwing."""
    if bins < 2 or any(item.axes for item in ranges):
        return failure(
            FailureKind.INVALID_DATA,
            "histogram",
            "histograms require per-tensor ranges and at least two bins",
        )
    edges: dict[Requirement, np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = {}
    for item, bounds in ranges.items():
        match histogram_edges(bounds, bins):
            case Err() as error:
                return error
            case Ok(value):
                edges[item] = value
    counts = {item: np.zeros(bins, dtype=np.int64) for item in ranges}
    if not ranges:
        return Ok({})
    outputs = tuple(dict.fromkeys(item.tensor for item in ranges))
    for sample in samples():
        match evaluator.run(sample, outputs):
            case Err() as error:
                return error
            case Ok(observed):
                pass
        for item in ranges:
            if item.tensor not in observed:
                return failure(
                    FailureKind.INVALID_DATA,
                    "histogram",
                    f"missing output {item.tensor}",
                )
            match count_histogram(observed[item.tensor], edges[item]):
                case Err() as error:
                    return error
                case Ok(count):
                    counts[item] += count
    return validate(
        "histogram",
        lambda: {
            item: HistogramStats(edges=edges[item], counts=counts[item])
            for item in ranges
        },
    )


def count_histogram(
    values: FloatArray,
    edges: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> Result[np.ndarray[tuple[int, ...], np.dtype[np.int64]], QuantSmithError]:
    """Count one batch, preserving why a replay cannot reuse its first-pass range."""
    match extrema(values, ()):
        case Err() as error:
            return error
        case Ok():
            pass
    counts, _ = np.histogram(values, bins=edges)
    if int(counts.sum()) != values.size:
        return failure(
            FailureKind.INVALID_DATA,
            "histogram",
            "replay data exceeded calibration range",
        )
    return Ok(np.asarray(counts, dtype=np.int64))
