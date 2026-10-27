"""Pure dynamic planning, owned encodings, and mutually exclusive node writes."""

from collections.abc import Mapping

import numpy as np
import pytest
from pydantic import ValidationError

from quantsmith.algorithms import Needs, Statistics
from quantsmith.algorithms.dynamic import DynamicW8A8
from quantsmith.domain import Encoding, FloatArray, Graph, Node, PerChannel
from quantsmith.plan import (
    DynamicQuantizeGemm,
    DynamicQuantizeMatMul,
    FloatOutput,
    Operation,
    QuantizationPlan,
    QuantizeInput,
    QuantizeOutput,
    RescaleInput,
)
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def graph() -> Graph:
    """Supply exact column scales and a dead output channel without activation data."""
    return Graph(
        nodes=(
            Node(
                name="projection",
                op="MatMul",
                inputs=("x", "w"),
                outputs=("y",),
                attributes={},
            ),
        ),
        weights={"w": np.array([[-127, 254, 0], [127, -254, 0]], dtype=np.float32)},
    )


@pytest.fixture
def operation(graph: Graph) -> DynamicQuantizeMatMul:
    """Require the single complete-node operation from calibration-free planning."""
    plan = expect_ok(
        DynamicW8A8().plan(graph.nodes[0], graph, Statistics(ranges={}, histograms={}))
    )
    match plan.operations:
        case (DynamicQuantizeMatMul() as operation,):
            return operation
        case _:
            pytest.fail("expected one dynamic projection operation")


def test_offline_weights_and_empty_requirements(
    graph: Graph, operation: DynamicQuantizeMatMul
) -> None:
    """Weights have exact per-column scales, with no invented activation encoding."""
    assert expect_ok(DynamicW8A8().requirements(graph.nodes[0], graph)) == Needs()
    encoding = operation.weight_encoding
    assert encoding.granularity == PerChannel(axis=1)
    np.testing.assert_array_equal(encoding.scale, [1, 2, 1])
    np.testing.assert_array_equal(encoding.zero_point, np.zeros(3, dtype=np.int8))
    assert not encoding.scale.flags.writeable
    assert not encoding.zero_point.flags.writeable
    assert not np.shares_memory(encoding.scale, graph.weights["w"])


@pytest.mark.parametrize(
    ("layout", "detail"),
    [
        ("Conv", "requires"),
        ("runtime", "required"),
        ("batched", "rank-two"),
    ],
)
def test_unsupported_layouts(graph: Graph, layout: str, detail: str) -> None:
    """The pure method returns UNSUPPORTED before looking for activation statistics."""
    node = graph.nodes[0]
    weights: Mapping[str, FloatArray] = graph.weights
    match layout:
        case "Conv":
            node = Node(
                name=node.name,
                op=layout,
                inputs=node.inputs,
                outputs=node.outputs,
                attributes={},
            )
        case "runtime":
            weights = {}
        case "batched":
            weights = {"w": graph.weights["w"][None, ...]}
    unsupported = Graph(nodes=(node,), weights=weights)
    expect_error(
        DynamicW8A8().requirements(node, unsupported),
        detail,
    )


@pytest.mark.parametrize(
    "conflict",
    ["input", "output", "float_output", "transform", "duplicate", "excluded"],
)
def test_complete_node_conflicts(
    operation: DynamicQuantizeMatMul, conflict: str
) -> None:
    """Runtime quantization owns the entire node and cannot mix graph revisions."""
    other: Operation
    match conflict:
        case "input":
            other = QuantizeInput(
                node=operation.node, index=0, encoding=operation.weight_encoding
            )
        case "output":
            other = QuantizeOutput(
                node=operation.node, index=0, encoding=operation.weight_encoding
            )
        case "float_output":
            other = FloatOutput(node=operation.node, index=0)
        case "transform":
            other = RescaleInput(
                node=operation.node,
                scale=np.ones(2, dtype=np.float32),
                activation_axis=-1,
                weight_axis=0,
            )
        case "duplicate":
            other = operation
        case _:
            with pytest.raises(ValidationError, match="excluded"):
                QuantizationPlan(operations=(operation,), excluded=(operation.node,))
            return
    with pytest.raises(ValidationError):
        QuantizationPlan(operations=(operation, other))


@pytest.mark.parametrize("invalid", ["unsigned", "asymmetric", "axis"])
def test_reject_invalid_weight_contract(invalid: str) -> None:
    """Direct operation construction cannot bypass the supported integer layout."""
    zero = (
        np.zeros(3, dtype=np.uint8)
        if invalid == "unsigned"
        else np.full(3, int(invalid == "asymmetric"), dtype=np.int8)
    )
    encoding = Encoding(
        scale=np.ones(3, dtype=np.float32),
        zero_point=zero,
        granularity=PerChannel(axis=0 if invalid == "axis" else 1),
    )
    with pytest.raises(ValidationError, match="symmetric INT8 output-column"):
        DynamicQuantizeMatMul(node="projection", weight_encoding=encoding)


@pytest.mark.parametrize("transposed", [False, True])
def test_gemm_uses_source_output_axis(graph: Graph, transposed: bool) -> None:
    """Gemm plans retain the source weight layout and require no activation data."""
    weight = graph.weights["w"].T if transposed else graph.weights["w"]
    node = Node(
        name="gemm",
        op="Gemm",
        inputs=("x", "w", "bias"),
        outputs=("y",),
        attributes={"transB": int(transposed)},
    )
    gemm = Graph(nodes=(node,), weights={"w": weight})
    assert expect_ok(DynamicW8A8().requirements(node, gemm)) == Needs()
    plan = expect_ok(
        DynamicW8A8().plan(node, gemm, Statistics(ranges={}, histograms={}))
    )
    match plan.operations:
        case (DynamicQuantizeGemm(weight_encoding=encoding),):
            assert encoding.granularity == PerChannel(axis=0 if transposed else 1)
            np.testing.assert_array_equal(encoding.scale, [1, 2, 1])
            assert not encoding.scale.flags.writeable
        case _:
            pytest.fail("expected one dynamic Gemm operation")


@pytest.mark.parametrize("attribute", ["alpha", "beta", "transA", "transB"])
def test_gemm_rejects_unsupported_attributes(graph: Graph, attribute: str) -> None:
    """Unsupported Gemm semantics are explicit before lowering."""
    node = Node(
        name="gemm",
        op="Gemm",
        inputs=("x", "w"),
        outputs=("y",),
        attributes={attribute: 2},
    )
    gemm = Graph(nodes=(node,), weights=graph.weights)
    expect_error(DynamicW8A8().requirements(node, gemm), "unit alpha/beta")


def test_gemm_plan_rejects_conflicting_writes(graph: Graph) -> None:
    """Gemm owns a complete node under the same conflict rules as MatMul."""
    node = Node(
        name="projection", op="Gemm", inputs=("x", "w"), outputs=("y",), attributes={}
    )
    gemm = Graph(nodes=(node,), weights=graph.weights)
    plan = expect_ok(
        DynamicW8A8().plan(node, gemm, Statistics(ranges={}, histograms={}))
    )
    with pytest.raises(ValidationError, match="duplicate dynamic"):
        QuantizationPlan(operations=plan.operations + plan.operations)
    with pytest.raises(ValidationError, match="excluded"):
        QuantizationPlan(operations=plan.operations, excluded=("projection",))
