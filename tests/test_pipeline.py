"""Numerical and graph-level guarantees through real ONNX Runtime execution."""

from copy import deepcopy

import numpy as np
import onnx
import pytest
from onnx import helper

from quantsmith.algorithms import Algorithm, build_plan
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe, lower
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.backends.onnx.stages import (
    FinishTransform,
    QuantizationStage,
    TransformStage,
    VerifyStage,
)
from quantsmith.calibration import Percentile, Samples
from quantsmith.config import supported
from quantsmith.domain import IntegerType, PerChannel
from quantsmith.pipeline import Pipeline
from quantsmith.plan import QuantizeInput
from quantsmith.rules import ByName, Exclude, Rule, Rules
from quantsmith.runtime import OnnxEvaluator, evaluate
from tests.conftest import OperatorCase
from tests.outcomes import expect_error, expect_ok


@pytest.mark.parametrize("dtype", list(IntegerType))
@pytest.mark.parametrize("symmetric", [False, True])
def test_static_qdq(
    operator_case: OperatorCase, samples: Samples, dtype: IntegerType, symmetric: bool
) -> None:
    """Verify storage variants, channel axes, valid graphs, and numerical error."""
    before = operator_case.model.SerializeToString()
    algorithm = StaticW8A8(activation_type=dtype, activation_symmetric=symmetric)
    result = expect_ok(
        quantize(
            operator_case.model,
            samples,
            Pipeline(QuantizationStage(supported(algorithm))).then(VerifyStage()),
        )
    )
    assert operator_case.model.SerializeToString() == before
    onnx.checker.check_model(result.model)
    operations = result.plans[0].operations
    assert len(operations) == 3
    match operations[1]:
        case QuantizeInput(encoding=encoding):
            assert encoding.granularity == PerChannel(axis=operator_case.weight_axis)
        case _:
            pytest.fail("expected weight quantization")
    report = expect_ok(
        evaluate(
            OnnxEvaluator(operator_case.model),
            OnnxEvaluator(result.model),
            samples,
            ("y",),
        )
    )
    assert report.mean_squared_error < 0.01
    assert report.maximum_absolute_error < 0.2
    assert report.batches == 4


def test_smoothquant_equivalence(operator_case: OperatorCase, samples: Samples) -> None:
    """Prove the floating transform preserves outputs before quantization."""
    result = expect_ok(
        quantize(
            operator_case.model,
            samples,
            Pipeline(TransformStage(supported(SmoothQuant()))).then(FinishTransform()),
        )
    )
    report = expect_ok(
        evaluate(
            OnnxEvaluator(operator_case.model),
            OnnxEvaluator(result.model),
            samples,
            ("y",),
        )
    )
    assert report.maximum_absolute_error < 1e-05


def test_smoothquant_then_percentile(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Recalibrate transformed activations before constructing static encodings."""
    stages = (
        Pipeline(TransformStage(supported(SmoothQuant())))
        .then(
            QuantizationStage(
                supported(StaticW8A8(calibration=Percentile(percentile=99.9)))
            )
        )
        .then(VerifyStage())
    )
    result = expect_ok(
        quantize(operator_case.model, samples, stages, histogram_bins=256)
    )
    assert len(result.plans) == 2
    report = expect_ok(
        evaluate(
            OnnxEvaluator(operator_case.model),
            OnnxEvaluator(result.model),
            samples,
            ("y",),
        )
    )
    assert report.mean_squared_error < 0.02


def test_shared_edges_preserve_excluded_consumer(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Quantizing one node must not modify a sibling using the same inputs."""
    model = deepcopy(operator_case.model)
    sibling = deepcopy(model.graph.node[0])
    sibling.name, sibling.output[0] = ("untouched", "other")
    model.graph.node.append(sibling)
    output = deepcopy(model.graph.output[0])
    output.name = "other"
    model.graph.output.append(output)
    rules: Rules[Algorithm] = Rules(
        default=StaticW8A8(),
        overrides=(
            Rule(
                selector=ByName(pattern="untouched"),
                decision=Exclude(reason="sensitive"),
            ),
        ),
    )
    result = expect_ok(
        quantize(model, samples, Pipeline(QuantizationStage(rules)).then(VerifyStage()))
    )
    untouched = next(
        node for node in result.model.graph.node if node.name == "untouched"
    )
    assert tuple(untouched.input) == ("x", "w")
    report = expect_ok(
        evaluate(OnnxEvaluator(model), OnnxEvaluator(result.model), samples, ("other",))
    )
    assert report.maximum_absolute_error == 0


def test_conflicting_plan_rejected(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Reject duplicate quantization operations before mutating the graph."""
    plan = expect_ok(
        build_plan(
            expect_ok(describe(operator_case.model)),
            OnnxEvaluator(operator_case.model),
            samples,
            supported(StaticW8A8()),
        )
    )
    expect_error(plan.then(plan), "duplicate")


def test_names_do_not_collide(operator_case: OperatorCase, samples: Samples) -> None:
    """Handle a model whose original name resembles a generated QDQ name."""
    model = deepcopy(operator_case.model)
    model.graph.node.append(
        helper.make_node(
            "Identity", ["y"], ["projection_input0_scale"], name="projection_input0_Q"
        )
    )
    plan = expect_ok(
        build_plan(
            expect_ok(describe(model)),
            OnnxEvaluator(model),
            samples,
            supported(StaticW8A8()),
        )
    )
    lowered = expect_ok(lower(model, plan))
    onnx.checker.check_model(lowered)
    names = [node.name for node in lowered.graph.node]
    assert len(names) == len(set(names))


def test_input_data_unchanged(operator_case: OperatorCase, samples: Samples) -> None:
    """Ensure the runtime never mutates calibration arrays."""
    sample = next(iter(samples()))
    original = sample["x"].copy()
    OnnxEvaluator(operator_case.model).run(sample, ("y",))
    np.testing.assert_array_equal(sample["x"], original)
