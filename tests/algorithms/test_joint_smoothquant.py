"""Joint SmoothQuant arithmetic and graph/statistic ownership without execution."""

import numpy as np
import pytest

from quantsmith.algorithms import Statistics
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.smoothquant.joint import JointSmoothQuant
from quantsmith.calibration import MinMaxStats, Requirement
from quantsmith.domain import FloatArray, Graph, Node
from quantsmith.plan import QuantizationPlan, RescaleInput
from tests.outcomes import expect_error, expect_ok


def scale_of(plan: QuantizationPlan) -> FloatArray:
    """Require one rescaling and preserve its precisely typed native array."""
    match plan.operations:
        case (RescaleInput(scale=scale),):
            return scale
        case _:
            pytest.fail("expected one rescaling")


@pytest.fixture
def siblings() -> Graph:
    """Provide complementary outlier channels in two sibling projections."""
    return Graph(
        nodes=tuple(
            Node(
                name=name,
                op="MatMul",
                inputs=("h", name + "_w"),
                outputs=(name + "_y",),
                attributes={},
            )
            for name in ("q", "k")
        ),
        weights={
            "q_w": np.array([[1.0], [4.0]], dtype=np.float32),
            "k_w": np.array([[4.0], [1.0]], dtype=np.float32),
        },
    )


@pytest.fixture
def channel_stats() -> Statistics:
    """Observe activation-channel maxima four and nine."""
    return Statistics(
        ranges={
            Requirement(tensor="h", axes=(-1,)): MinMaxStats(
                minimum=np.array([-4.0, -9.0], dtype=np.float32),
                maximum=np.array([2.0, 3.0], dtype=np.float32),
            )
        },
        histograms={},
    )


@pytest.mark.parametrize(
    ("alpha", "expected"), [(0.0, [0.25, 0.25]), (0.5, [1.0, 1.5]), (1.0, [4.0, 9.0])]
)
def test_joint_scales(
    siblings: Graph, channel_stats: Statistics, alpha: float, expected: list[float]
) -> None:
    """Every sibling uses the strongest weight magnitude in each contraction channel."""
    for node in siblings.nodes:
        plan = expect_ok(
            JointSmoothQuant(alpha=alpha).plan(node, siblings, channel_stats)
        )
        match plan.operations:
            case (RescaleInput(scale=scale),):
                np.testing.assert_allclose(scale, expected)
                assert not scale.flags.writeable
            case _:
                pytest.fail("expected one rescaling per consumer")
        assert expect_ok(JointSmoothQuant().requirements(node, siblings)) == expect_ok(
            SmoothQuant().requirements(node, siblings)
        )


def test_order_invariance(siblings: Graph, channel_stats: Statistics) -> None:
    """Declaration order cannot select a privileged sibling's weight range."""
    reverse = Graph(nodes=siblings.nodes[::-1], weights=siblings.weights)
    node = siblings.nodes[0]
    first = expect_ok(JointSmoothQuant().plan(node, siblings, channel_stats))
    second = expect_ok(JointSmoothQuant().plan(node, reverse, channel_stats))
    np.testing.assert_array_equal(scale_of(first), scale_of(second))


def test_missing_statistics(siblings: Graph) -> None:
    """Return the established explicit missing-statistics diagnostic."""
    expect_error(
        JointSmoothQuant().plan(
            siblings.nodes[0], siblings, Statistics(ranges={}, histograms={})
        ),
        "not collected",
    )


def test_single_consumer_matches_independent(
    siblings: Graph, channel_stats: Statistics
) -> None:
    """Joint planning reduces to ordinary SmoothQuant when no sibling exists."""
    graph = Graph(nodes=(siblings.nodes[0],), weights=siblings.weights)
    node = graph.nodes[0]
    joint = expect_ok(JointSmoothQuant().plan(node, graph, channel_stats))
    independent = expect_ok(SmoothQuant().plan(node, graph, channel_stats))
    np.testing.assert_array_equal(scale_of(joint), scale_of(independent))


def test_dead_channels_use_joint_weights(channel_stats: Statistics) -> None:
    """A dead channel in one sibling must use another sibling's live magnitude."""
    nodes = tuple(
        Node(
            name=name,
            op="MatMul",
            inputs=("h", name + "_w"),
            outputs=(name + "_y",),
            attributes={},
        )
        for name in ("q", "k")
    )
    graph = Graph(
        nodes=nodes,
        weights={
            "q_w": np.array([[0.0], [0.0]], dtype=np.float32),
            "k_w": np.array([[1.0], [0.0]], dtype=np.float32),
        },
    )
    for node in nodes:
        plan = expect_ok(JointSmoothQuant().plan(node, graph, channel_stats))
        np.testing.assert_array_equal(scale_of(plan), [2.0, 1.0])


def test_mismatched_sibling_channels(
    siblings: Graph, channel_stats: Statistics
) -> None:
    """Malformed shared contraction dimensions fail before a partial plan escapes."""
    graph = Graph(
        nodes=siblings.nodes,
        weights={
            "q_w": siblings.weights["q_w"],
            "k_w": np.ones((3, 1), dtype=np.float32),
        },
    )
    expect_error(
        JointSmoothQuant().plan(graph.nodes[0], graph, channel_stats),
        "dimensions differ",
    )
