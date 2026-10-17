"""Histogram-informed channel balancing, proxy error and typed calibration needs."""

from collections.abc import Iterable, Mapping

import numpy as np
import pytest

from quantsmith.algorithms import Algorithm, Statistics, build_plan, select_algorithms
from quantsmith.algorithms.smoothquant.joint import JointSmoothQuant
from quantsmith.algorithms.smoothquant.joint.ranges import (
    ChannelExtrema,
    ChannelMSE,
    ChannelPercentile,
    ChannelSelection,
    error_at,
    select_ranges,
)
from quantsmith.calibration import (
    ChannelHistogramStats,
    ChannelRequirement,
    MinMaxStats,
    Requirement,
)
from quantsmith.domain import Absent, FloatArray, Graph, Node
from quantsmith.plan import RescaleInput
from quantsmith.rules import Rules
from tests.outcomes import expect_error, expect_ok
from tests.test_calibration import IdentityEvaluator


@pytest.fixture
def distributions() -> ChannelHistogramStats:
    """Give equal extrema to rare-outlier and broadly populated channels."""
    edges = np.broadcast_to(np.linspace(-100, 100, 2049), (2, 2049)).copy()
    counts = np.ones((2, 2048), dtype=np.int64)
    counts[0, :] = 0
    counts[0, 0] = counts[0, -1] = 1
    counts[0, 1014:1034] = 1_000_000
    counts[1, :] = 1_000_000
    return ChannelHistogramStats(edges=edges, counts=counts)


@pytest.fixture
def observed_channels() -> MinMaxStats:
    """Both channels share the same extrema despite different probability mass."""
    return MinMaxStats(
        minimum=np.full(2, -100, dtype=np.float32),
        maximum=np.full(2, 100, dtype=np.float32),
    )


@pytest.fixture
def shared_channels() -> Graph:
    """Two projections consume the same two-channel activation with unit weights."""
    return Graph(
        nodes=tuple(
            Node(
                name=name,
                op="MatMul",
                inputs=("x", "w"),
                outputs=(name + "_y",),
                attributes={},
            )
            for name in ("q", "k")
        ),
        weights={"w": np.ones((2, 2), dtype=np.float32)},
    )


@pytest.mark.parametrize(
    "policy",
    [ChannelPercentile(percentile=99), ChannelMSE(min_percentile=99, candidates=8)],
)
def test_histogram_distinguishes_equal_extrema(
    distributions: ChannelHistogramStats,
    observed_channels: MinMaxStats,
    policy: ChannelSelection,
) -> None:
    """Distribution-aware balancing treats rare and common extremes differently."""
    selected = expect_ok(select_ranges(policy, observed_channels, distributions))
    assert float(selected.maximum[0]) < 2
    assert float(selected.minimum[0]) > -2
    assert float(selected.maximum[1]) > 90
    assert float(selected.minimum[1]) < -90
    np.testing.assert_array_equal(observed_channels.maximum, [100, 100])


def test_mse_proxy_does_not_increase(
    distributions: ChannelHistogramStats, observed_channels: MinMaxStats
) -> None:
    """The selected interval never worsens the histogram error proxy."""
    selected = expect_ok(select_ranges(ChannelMSE(), observed_channels, distributions))
    for channel in range(2):
        edges = distributions.edges[channel]
        points = (edges[:-1] + edges[1:]) / 2
        chosen = error_at(
            points,
            distributions.counts[channel],
            (float(selected.minimum[channel]), float(selected.maximum[channel])),
        )
        baseline = error_at(points, distributions.counts[channel], (-100.0, 100.0))
        assert chosen <= baseline
    assert float(selected.maximum[1]) > 99


@pytest.mark.parametrize(
    "policy",
    [
        ChannelPercentile(percentile=100),
        ChannelMSE(min_percentile=100),
        ChannelExtrema(),
    ],
)
def test_full_range_identity(
    distributions: ChannelHistogramStats,
    observed_channels: MinMaxStats,
    policy: ChannelSelection,
) -> None:
    """Histogram endpoint policies reproduce the extrema baseline exactly."""
    selected = expect_ok(select_ranges(policy, observed_channels, distributions))
    np.testing.assert_array_equal(selected.minimum, observed_channels.minimum)
    np.testing.assert_array_equal(selected.maximum, observed_channels.maximum)


@pytest.mark.parametrize("policy", [ChannelPercentile(percentile=99), ChannelMSE()])
def test_joint_plans_share_histogram_scales(
    shared_channels: Graph,
    distributions: ChannelHistogramStats,
    observed_channels: MinMaxStats,
    policy: ChannelSelection,
) -> None:
    """Every sibling derives the same vector from channel distributions."""
    algorithm = JointSmoothQuant(channels=policy)
    request = ChannelRequirement(tensor="x", axis=-1)
    stats = Statistics(
        ranges={request.range: observed_channels},
        histograms={},
        channel_histograms={request: distributions},
    )
    for node in shared_channels.nodes:
        plan = expect_ok(algorithm.plan(node, shared_channels, stats))
        match plan.operations:
            case (RescaleInput(scale=scale),):
                assert float(scale[0]) < 2
                assert float(scale[1]) > 9
            case _:
                pytest.fail("expected a channel rescaling")
    selection = expect_ok(
        select_algorithms(shared_channels, Rules[Algorithm](default=algorithm))
    )
    assert selection.needs.channel_histograms == (request,)
    assert selection.needs.ranges == (request.range,)
    assert not selection.needs.histograms


def test_missing_channel_histogram(
    shared_channels: Graph, observed_channels: MinMaxStats
) -> None:
    """A per-tensor histogram cannot silently substitute for a channel distribution."""
    stats = Statistics(
        ranges={Requirement(tensor="x", axes=(-1,)): observed_channels}, histograms={}
    )
    expect_error(
        JointSmoothQuant(channels=ChannelMSE()).plan(
            shared_channels.nodes[0], shared_channels, stats
        ),
        "not collected",
    )


def test_shell_replays_once_for_shared_channels(shared_channels: Graph) -> None:
    """Shared requests deduplicate to one bounded histogram pass after extrema."""
    passes = 0

    def source() -> Iterable[Mapping[str, FloatArray]]:
        """Recreate observations identically for both calibration passes."""
        nonlocal passes
        passes += 1
        yield {"x": np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)}

    evaluator = IdentityEvaluator()
    algorithm = JointSmoothQuant(channels=ChannelPercentile(percentile=99))
    expect_ok(
        build_plan(
            shared_channels,
            evaluator,
            source,
            Rules[Algorithm](default=algorithm),
            histogram_bins=32,
        )
    )
    assert passes == 2
    assert evaluator.calls == [("x",), ("x",)]


def test_histogram_shape_mismatch(distributions: ChannelHistogramStats) -> None:
    """Reject a distribution from another channel shape instead of broadcasting it."""
    observed = MinMaxStats(
        minimum=np.zeros(3, dtype=np.float32), maximum=np.ones(3, dtype=np.float32)
    )
    expect_error(
        select_ranges(ChannelMSE(), observed, distributions), "dimensions differ"
    )
    expect_error(
        select_ranges(
            ChannelPercentile(), observed, Absent(reason="missing distribution")
        ),
        "missing distribution",
    )
