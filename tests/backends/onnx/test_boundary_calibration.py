"""Explicit interval policies at residual and smoothing boundaries."""

import numpy as np
import pytest

from quantsmith.algorithms import Statistics
from quantsmith.backends.onnx.ort_scaling.boundaries import BoundaryQuantizer
from quantsmith.backends.onnx.stages import QuantizationStage
from quantsmith.calibration import (
    HistogramStats,
    MinMax,
    MinMaxStats,
    Percentile,
    Requirement,
)
from quantsmith.config import (
    FollowCalibration,
    MinMaxConfig,
    PercentileConfig,
    QuantizationConfig,
)
from quantsmith.domain import Graph, Node
from quantsmith.pipeline.contracts import Sequence
from quantsmith.plan import QuantizeInput, QuantizeOutput
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def boundary_stats() -> Statistics:
    """Provide populated central bins and a rare distant upper tail."""
    bounds = MinMaxStats(
        minimum=np.array(0, dtype=np.float32), maximum=np.array(100, dtype=np.float32)
    )
    histogram = HistogramStats(
        edges=np.linspace(0, 100, 101),
        counts=np.array([1000] + [0] * 98 + [1], dtype=np.int64),
    )
    return Statistics(
        ranges={Requirement(tensor=name): bounds for name in ("x", "r", "y")},
        histograms={Requirement(tensor=name): histogram for name in ("x", "r", "y")},
    )


@pytest.mark.parametrize("op", ["Add", "Div"])
def test_percentile_changes_all_boundary_encodings(
    op: str, boundary_stats: Statistics
) -> None:
    """Clip boundary inputs and Add outputs using the requested histogram policy."""
    node = Node(
        name="boundary", op=op, inputs=("x", "r"), outputs=("y",), attributes={}
    )
    graph = Graph(nodes=(node,), weights={})
    baseline = expect_ok(BoundaryQuantizer().plan(node, graph, boundary_stats))
    method = BoundaryQuantizer(calibration=Percentile(percentile=99))
    candidate = expect_ok(method.plan(node, graph, boundary_stats))
    needs = expect_ok(method.requirements(node, graph))
    assert len(needs.histograms) == (3 if op == "Add" else 1)
    for original, changed in zip(
        baseline.operations, candidate.operations, strict=True
    ):
        assert isinstance(original, QuantizeInput | QuantizeOutput)
        assert isinstance(changed, QuantizeInput | QuantizeOutput)
        assert float(changed.encoding.scale) < float(original.encoding.scale)
    missing = Statistics(ranges=boundary_stats.ranges, histograms={})
    expect_error(method.plan(node, graph, missing), "histogram")
    expect_error(
        method.plan(node, graph, Statistics(ranges={}, histograms={})), "not collected"
    )


@pytest.mark.parametrize("follow", [False, True])
def test_config_boundary_policy(follow: bool) -> None:
    """Keep legacy defaults and permit explicit inheritance without string dispatch."""
    config = QuantizationConfig(
        calibration=PercentileConfig(percentile=99),
        boundary_calibration=FollowCalibration() if follow else MinMaxConfig(),
    )
    match config.pipeline().step:
        case Sequence(
            first=QuantizationStage(boundary_calibration=Percentile(percentile=value))
        ):
            assert follow and value == 99
        case Sequence(first=QuantizationStage(boundary_calibration=MinMax())):
            assert not follow
        case _:
            pytest.fail("unexpected interval method")
