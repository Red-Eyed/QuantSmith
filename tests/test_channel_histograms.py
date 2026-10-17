"""Per-channel histogram counts, replay failures, ownership and bounded storage."""

import weakref
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import numpy as np
import pytest
from pydantic import ValidationError

from quantsmith.algorithms import Statistics
from quantsmith.calibration import (
    ChannelHistogramStats,
    ChannelRequirement,
    MinMaxStats,
    channel_histograms,
    collect,
)
from quantsmith.domain import FloatArray
from tests.outcomes import expect_error, expect_ok
from tests.test_calibration import IdentityEvaluator


@pytest.fixture
def channel_evaluator() -> IdentityEvaluator:
    """Provide an isolated observer that does not retain batch arrays."""
    return IdentityEvaluator()


@pytest.mark.parametrize(
    ("shape", "axis"), [((3,), 0), ((2, 3), -1), ((2, 3, 4), 1), ((2, 3, 4, 5), -3)]
)
def test_channel_counts(
    channel_evaluator: IdentityEvaluator, shape: tuple[int, ...], axis: int
) -> None:
    """Every retained coordinate agrees with an independent NumPy histogram."""
    batch = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    def source() -> Iterable[Mapping[str, FloatArray]]:
        """Replay two distinct finite batches with identical channel shapes."""
        yield {"x": batch}
        yield {"x": batch + 1}

    request = ChannelRequirement(tensor="x", axis=axis)
    bounds = expect_ok(collect(channel_evaluator, source, (request.range,)))
    histogram = expect_ok(
        channel_histograms(
            channel_evaluator, source, {request: bounds[request.range]}, 16
        )
    )[request]
    for channel in range(shape[axis]):
        values = np.concatenate(
            (
                np.take(batch, channel, axis=axis).reshape(-1),
                np.take(batch + 1, channel, axis=axis).reshape(-1),
            )
        )
        expected, _ = np.histogram(values, bins=histogram.edges[channel])
        np.testing.assert_array_equal(histogram.counts[channel], expected)
    assert histogram.counts.shape == (shape[axis], 16)
    assert not histogram.counts.flags.writeable
    assert not histogram.edges.flags.writeable


def test_constant_channels(channel_evaluator: IdentityEvaluator) -> None:
    """Constant channels retain finite independent bins and populated counts."""
    data = np.broadcast_to(np.array([0.0, 1.0, -3.0], dtype=np.float32), (4, 3))
    request = ChannelRequirement(tensor="x", axis=1)

    def source() -> Iterable[Mapping[str, FloatArray]]:
        """Replay constants without changing their caller-owned storage."""
        yield {"x": data}

    bounds = expect_ok(collect(channel_evaluator, source, (request.range,)))
    histogram = expect_ok(
        channel_histograms(
            channel_evaluator, source, {request: bounds[request.range]}, 32
        )
    )[request]
    np.testing.assert_array_equal(histogram.counts.sum(axis=1), [4, 4, 4])
    assert np.all(np.diff(histogram.edges, axis=1) > 0)


def test_streaming_does_not_retain_batches(
    channel_evaluator: IdentityEvaluator,
) -> None:
    """Dataset length does not grow retained activations or histogram dimensions."""
    references: list[weakref.ReferenceType[FloatArray]] = []
    request = ChannelRequirement(tensor="x", axis=1)
    bounds = MinMaxStats(
        minimum=np.zeros(3, dtype=np.float32),
        maximum=np.full(3, 99.0, dtype=np.float32),
    )

    def source() -> Iterable[Mapping[str, FloatArray]]:
        """Check old batches are released while many new batches are generated."""
        for index in range(100):
            assert sum(ref() is not None for ref in references) <= 2
            value = np.full((2, 3), index, dtype=np.float32)
            references.append(weakref.ref(value))
            yield {"x": value}

    histogram = expect_ok(
        channel_histograms(channel_evaluator, source, {request: bounds}, 32)
    )[request]
    np.testing.assert_array_equal(histogram.counts.sum(axis=1), [200, 200, 200])
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("limit_delta", [-1, 0])
def test_workspace_limit_precedes_replay(
    channel_evaluator: IdentityEvaluator, limit_delta: int
) -> None:
    """Include owned result copies in the budget and reject before consuming replay."""
    request = ChannelRequirement(tensor="x", axis=1)
    bounds = MinMaxStats(
        minimum=np.zeros(3, dtype=np.float32), maximum=np.ones(3, dtype=np.float32)
    )
    passes = 0

    def source() -> Iterable[Mapping[str, FloatArray]]:
        """Track whether histogram replay started."""
        nonlocal passes
        passes += 1
        yield {"x": np.ones((2, 3), dtype=np.float32)}

    required = 2 * 3 * (16 * 8 + 17 * 8)
    result = channel_histograms(
        channel_evaluator,
        source,
        {request: bounds},
        16,
        max_bytes=required + limit_delta,
    )
    if limit_delta < 0:
        expect_error(result, "budget")
        assert passes == 0
    else:
        expect_ok(result)
        assert passes == 1


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ("axis", "axis"),
        ("shape", "dimensions"),
        ("nonfinite", "finite"),
        ("range", "exceeded"),
        ("empty", "empty"),
    ],
)
def test_invalid_channel_replay(
    channel_evaluator: IdentityEvaluator, problem: str, expected: str
) -> None:
    """Reject malformed replay instead of silently dropping observations."""
    request = ChannelRequirement(tensor="x", axis=3 if problem == "axis" else 1)
    bounds = MinMaxStats(
        minimum=np.zeros(3, dtype=np.float32), maximum=np.ones(3, dtype=np.float32)
    )

    def source() -> Iterable[Mapping[str, FloatArray]]:
        """Produce one explicit invalid observation or an empty replay."""
        if problem == "empty":
            return
        values = np.ones((2, 4 if problem == "shape" else 3), dtype=np.float32)
        if problem == "nonfinite":
            values[0, 0] = np.nan
        if problem == "range":
            values[0, 0] = 2
        yield {"x": values}

    outcome = channel_histograms(channel_evaluator, source, {request: bounds}, 16)
    expect_error(outcome, expected)


def test_channel_record_ownership() -> None:
    """Returned distributions and their statistics mapping own immutable storage."""
    edges = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    counts = np.array([[1, 2]], dtype=np.int64)
    histogram = ChannelHistogramStats(edges=edges, counts=counts)
    request = ChannelRequirement(tensor="x", axis=1)
    mapping = {request: histogram}
    stats = Statistics(ranges={}, histograms={}, channel_histograms=mapping)
    mapping.clear()
    edges[0, 0], counts[0, 0] = -9.0, 99
    np.testing.assert_array_equal(histogram.edges, [[0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(histogram.counts, [[1, 2]])
    assert request in stats.channel_histograms
    assert isinstance(stats.channel_histograms, MappingProxyType)
    assert isinstance(
        Statistics(ranges={}, histograms={}).channel_histograms, MappingProxyType
    )


@pytest.mark.parametrize(
    "malformation",
    [
        "edges_dtype",
        "counts_dtype",
        "edges_shape",
        "counts_shape",
        "negative",
        "empty",
        "nonfinite",
        "order",
        "channels",
    ],
)
def test_channel_record_validation(malformation: str) -> None:
    """Native array-instance validation cannot replace numerical boundary checks."""
    edges = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    counts = np.array([[1, 2]], dtype=np.int64)
    match malformation:
        case "edges_dtype":
            with pytest.raises(ValidationError):
                ChannelHistogramStats.model_validate(
                    {"edges": edges.astype(np.float32), "counts": counts}
                )
            return
        case "counts_dtype":
            with pytest.raises(ValidationError):
                ChannelHistogramStats.model_validate(
                    {"edges": edges, "counts": counts.astype(np.int32)}
                )
            return
        case "edges_shape":
            edges = edges.reshape(-1)
        case "counts_shape":
            counts = counts.reshape(-1)
        case "negative":
            counts[0, 0] = -1
        case "empty":
            counts[:] = 0
        case "nonfinite":
            edges[0, 0] = np.inf
        case "order":
            edges[0, 1] = 0
        case "channels":
            counts = np.ones((2, 2), dtype=np.int64)
    with pytest.raises(ValidationError):
        ChannelHistogramStats(edges=edges, counts=counts)


def test_wrong_native_dtype_is_rejected() -> None:
    """An array-instance schema cannot silently authorize a wrong runtime dtype."""
    from pydantic import ConfigDict, TypeAdapter

    from quantsmith.calibration.channel_histogram import ChannelBins, accumulate

    values = TypeAdapter(
        FloatArray, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(np.ones((2, 3), dtype=np.float64))
    target = ChannelBins(
        np.broadcast_to(np.array([0.0, 0.5, 1.0], dtype=np.float64), (3, 3)),
        np.zeros((3, 2), dtype=np.int64),
    )
    expect_error(accumulate(target, values, 1), "float32")
