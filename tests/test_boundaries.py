"""Admission failures and ownership invariants around the numerical core."""

from copy import deepcopy

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper
from pydantic import ValidationError

from quantsmith.algorithms import Algorithm
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe, lower, normalize
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.backends.onnx.stages import (
    FinishTransform,
    QuantizationStage,
    TransformStage,
    VerifyStage,
)
from quantsmith.calibration import (
    HistogramStats,
    Requirement,
    Samples,
    collect,
    histograms,
)
from quantsmith.config import supported
from quantsmith.domain import Encoding, Graph, PerChannel, PerTensor
from quantsmith.pipeline import Pipeline
from quantsmith.plan import QuantizationPlan, QuantizeInput, RescaleInput
from quantsmith.rules import ByName, Exclude, Rule, Rules
from quantsmith.runtime import OnnxEvaluator, evaluate
from tests.conftest import OperatorCase
from tests.outcomes import expect_error, expect_ok
from tests.test_calibration import IdentityEvaluator


def test_graph_owns_arrays() -> None:
    """External array mutation cannot alter a trusted graph snapshot."""
    original = np.ones((2, 3), dtype=np.float32)
    graph = Graph(nodes=(), weights={"w": original})
    original[:] = 9
    np.testing.assert_array_equal(graph.weights["w"], 1)
    with pytest.raises(ValueError):
        graph.weights["w"][0, 0] = 3


@pytest.mark.parametrize("counts", [[0, 0], [-1, 3]])
def test_invalid_histogram_counts(counts: list[int]) -> None:
    """Reject empty replay and invalid plugin statistics at construction."""
    with pytest.raises(ValidationError):
        HistogramStats(
            edges=np.asarray([0, 1, 2], dtype=np.float64),
            counts=np.asarray(counts, dtype=np.int64),
        )


def test_histogram_replay_range() -> None:
    """Detect a replay source whose values exceed its first-pass extrema."""
    evaluator = IdentityEvaluator()
    request = Requirement(tensor="x")
    ranges = expect_ok(
        collect(
            evaluator,
            lambda: iter(({"x": np.asarray([0, 1], dtype=np.float32)},)),
            (request,),
        )
    )
    expect_error(
        histograms(
            evaluator,
            lambda: iter(({"x": np.asarray([2], dtype=np.float32)},)),
            ranges,
            8,
        ),
        "exceeded",
    )


def test_plan_rejects_mixed_revisions() -> None:
    """A transform cannot silently invalidate the encodings in its own stage."""
    encoding = Encoding(
        scale=np.asarray(1, dtype=np.float32),
        zero_point=np.asarray(0, dtype=np.int8),
        granularity=PerTensor(),
    )
    with pytest.raises(ValueError, match="recalibrate"):
        QuantizationPlan(
            operations=(
                QuantizeInput(node="node", index=0, encoding=encoding),
                RescaleInput(
                    node="other",
                    scale=np.ones(3, dtype=np.float32),
                    activation_axis=1,
                    weight_axis=0,
                ),
            )
        )


def test_bad_plan_axis(operator_case: OperatorCase) -> None:
    """A valid numerical encoding must also fit its graph tensor dimensions."""
    encoding = Encoding(
        scale=np.ones(7, dtype=np.float32),
        zero_point=np.zeros(7, dtype=np.int8),
        granularity=PerChannel(axis=99),
    )
    plan = QuantizationPlan(
        operations=(QuantizeInput(node="projection", index=1, encoding=encoding),)
    )
    expect_error(lower(operator_case.model, plan), "axis out of range")


def test_unsupported_opset(operator_case: OperatorCase) -> None:
    """Reject models whose opset cannot encode the promised per-channel QDQ."""
    model = deepcopy(operator_case.model)
    model.opset_import[0].version = 12
    with pytest.raises(ValueError, match="opset"):
        normalize(model)


def test_unknown_plan_node(operator_case: OperatorCase) -> None:
    """Never silently drop a plan exclusion referencing a stale graph."""
    expect_error(
        lower(operator_case.model, QuantizationPlan(excluded=("missing",))),
        "unknown node",
    )


def test_duplicate_node_names(operator_case: OperatorCase) -> None:
    """Reject ambiguous node identities before resolving selectors."""
    model = deepcopy(operator_case.model)
    model.graph.node.append(
        helper.make_node("Identity", ["y"], ["unused"], name="projection")
    )
    expect_error(describe(model), "unique")


def test_static_with_bias(operator_case: OperatorCase, samples: Samples) -> None:
    """Prepare INT32 accumulator biases without mutating the original constant."""
    model = deepcopy(operator_case.model)
    if model.graph.node[0].op_type == "MatMul":
        return
    bias = np.linspace(-1, 1, 5, dtype=np.float32)
    model.graph.initializer.append(numpy_helper.from_array(bias, "bias"))
    model.graph.node[0].input.append("bias")
    result = expect_ok(
        quantize(
            model,
            samples,
            Pipeline(QuantizationStage(supported(StaticW8A8()))).then(VerifyStage()),
        )
    )
    projection = next(
        node for node in result.model.graph.node if node.name == "projection"
    )
    assert projection.input[2] != "bias"
    np.testing.assert_array_equal(
        numpy_helper.to_array(model.graph.initializer[-1]), bias
    )
    assert any(
        tensor.data_type == TensorProto.INT32
        for tensor in result.model.graph.initializer
    )
    report = expect_ok(
        evaluate(OnnxEvaluator(model), OnnxEvaluator(result.model), samples, ("y",))
    )
    assert report.mean_squared_error < 0.01


def test_smoothquant_shared_weight(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Transform one consumer without altering a sibling or a shared initializer."""
    model = deepcopy(operator_case.model)
    sibling = deepcopy(model.graph.node[0])
    sibling.name, sibling.output[0] = "untouched", "other"
    model.graph.node.append(sibling)
    output = deepcopy(model.graph.output[0])
    output.name = "other"
    model.graph.output.append(output)
    rules: Rules[Algorithm] = Rules(
        default=SmoothQuant(),
        overrides=(
            Rule(
                selector=ByName(pattern="untouched"),
                decision=Exclude(reason="sensitive"),
            ),
        ),
    )
    result = expect_ok(
        quantize(
            model, samples, Pipeline(TransformStage(rules)).then(FinishTransform())
        )
    )
    report = expect_ok(
        evaluate(
            OnnxEvaluator(model), OnnxEvaluator(result.model), samples, ("y", "other")
        )
    )
    assert report.maximum_absolute_error < 1e-5
    original = next(
        tensor for tensor in result.model.graph.initializer if tensor.name == "w"
    )
    assert original == model.graph.initializer[0]
