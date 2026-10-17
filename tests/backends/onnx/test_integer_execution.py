"""Compiled integer coverage, FP32 boundaries and accumulator safety regressions."""

from collections.abc import Iterator, Mapping
from copy import deepcopy

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper
from pydantic import ValidationError

from quantsmith.algorithms.shell import build_plan
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe, lower
from quantsmith.backends.onnx.integer_execution.audit import (
    ExecutionReport,
    inspect_execution,
)
from quantsmith.backends.onnx.integer_execution.planning import (
    accumulator_safe_plan,
    execution_rules,
)
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.calibration import Samples
from quantsmith.config import supported
from quantsmith.domain import FloatArray, InputArray, IntegerType
from quantsmith.plan import QuantizationPlan, QuantizeOutput
from quantsmith.runtime import OnnxEvaluator, evaluate
from tests.conftest import OperatorCase
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def region_model() -> ModelProto:
    """Connect three convolutions through activation and residual FP32 boundaries."""
    constants = [
        numpy_helper.from_array(
            np.array([[[[0.2]], [[-0.3]]], [[[0.4]], [[0.1]]]], dtype=np.float32), "w"
        )
    ]
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["x", "w"], ["a"], name="first"),
                helper.make_node("Relu", ["a"], ["b"], name="activation"),
                helper.make_node("Conv", ["b", "w"], ["c"], name="second"),
                helper.make_node("Add", ["c", "x"], ["d"], name="residual"),
                helper.make_node("Conv", ["d", "w"], ["e"], name="third"),
                helper.make_node("Sigmoid", ["e"], ["y"], name="unsupported"),
            ],
            "region",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])],
            constants,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


@pytest.fixture
def region_samples() -> Samples:
    """Replay bounded, deterministic inputs spanning positive and negative values."""

    def replay() -> Iterator[Mapping[str, FloatArray]]:
        """Own only the current batch."""
        rng = np.random.default_rng(17)
        for _ in range(4):
            yield {"x": rng.normal(size=(1, 2, 4, 4)).astype(np.float32)}

    return replay


def test_integer_regions_and_float_boundary(
    region_model: ModelProto, region_samples: Samples
) -> None:
    """Verify CPU integer Conv/Add execution while unsupported Sigmoid remains valid."""
    before = region_model.SerializeToString()
    result = expect_ok(
        quantize(region_model, region_samples, (supported(StaticW8A8()),))
    )
    assert region_model.SerializeToString() == before
    assert isinstance(result.execution, ExecutionReport)
    kernels = {
        item.node: item.executed_operator for item in result.execution.integer_nodes
    }
    assert kernels == {
        "first": "QLinearConv",
        "second": "QLinearConv",
        "third": "QLinearConv",
        "residual": "QLinearAdd",
    }
    assert any(item.node == "unsupported" for item in result.execution.unverified_nodes)
    error = expect_ok(
        evaluate(
            OnnxEvaluator(region_model),
            OnnxEvaluator(result.model),
            region_samples,
            ("y",),
        )
    )
    assert error.maximum_absolute_error < 0.03


def test_audit_rejects_float_fallback(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Unrelated integer kernels must never satisfy a planned source's contract."""
    result = expect_ok(
        quantize(operator_case.model, samples, (supported(StaticW8A8()),))
    )
    expect_error(
        inspect_execution(operator_case.model, operator_case.model, result.plans),
        "expected integer",
    )


def test_dynamic_weight_stays_float(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """A selected runtime weight retains its original computation and a reason."""
    model = deepcopy(operator_case.model)
    weight = model.graph.initializer[0]
    model.graph.input.append(
        helper.make_tensor_value_info("w", TensorProto.FLOAT, list(weight.dims))
    )
    values = np.array(numpy_helper.to_array(weight), dtype=np.float32)
    del model.graph.initializer[:]

    def replay() -> Iterator[Mapping[str, InputArray]]:
        """Supply the same runtime weight alongside replayed activations."""
        for batch in samples():
            yield {**batch, "w": values}

    result = expect_ok(quantize(model, replay, (supported(StaticW8A8()),)))
    assert isinstance(result.execution, ExecutionReport)
    assert not result.execution.integer_nodes
    assert "constant weights" in result.execution.unverified_nodes[0].reason
    assert tuple(result.model.graph.node[0].input) == ("x", "w")
    error = expect_ok(
        evaluate(OnnxEvaluator(model), OnnxEvaluator(result.model), replay, ("y",))
    )
    assert error.maximum_absolute_error == 0


def test_output_plan_conflicts(operator_case: OperatorCase, samples: Samples) -> None:
    """Output writes conflict with each other, while input/output indices coexist."""
    result = expect_ok(
        quantize(operator_case.model, samples, (supported(StaticW8A8()),))
    )
    outputs = tuple(
        operation
        for operation in result.plans[0].operations
        if isinstance(operation, QuantizeOutput)
    )
    with pytest.raises(ValidationError, match="duplicate output"):
        QuantizationPlan(operations=outputs + outputs)
    invalid = QuantizeOutput(node="projection", index=1, encoding=outputs[0].encoding)
    expect_error(
        lower(operator_case.model, QuantizationPlan(operations=(invalid,))),
        "missing output",
    )


def test_zero_weight_bias_safety(
    region_model: ModelProto, region_samples: Samples
) -> None:
    """Zero-weight channels with nonzero bias need representable accumulator scales."""
    model = deepcopy(region_model)
    model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.zeros((2, 2, 1, 1), dtype=np.float32), "w")
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([0.3, -0.4], dtype=np.float32), "bias")
    )
    model.graph.node[0].input.append("bias")
    graph = expect_ok(describe(model))
    rules = execution_rules(model, graph, supported(StaticW8A8()))
    plan = expect_ok(build_plan(graph, OnnxEvaluator(model), region_samples, rules))
    assert plan.operations
    result = expect_ok(quantize(model, region_samples, (supported(StaticW8A8()),)))
    assert isinstance(result.execution, ExecutionReport)
    assert len(result.execution.integer_nodes) == 4
    error = expect_ok(
        evaluate(
            OnnxEvaluator(model), OnnxEvaluator(result.model), region_samples, ("y",)
        )
    )
    assert error.maximum_absolute_error < 0.01


def test_bias_widening_preserves_unbiased_plan(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Bias-free planning does not alter numerical algorithm decisions."""
    graph = expect_ok(describe(operator_case.model))
    plan = expect_ok(
        build_plan(
            graph, OnnxEvaluator(operator_case.model), samples, supported(StaticW8A8())
        )
    )
    assert accumulator_safe_plan(graph.nodes[0], graph, plan) is plan


@pytest.mark.parametrize("spatial_rank", [1, 2, 3])
@pytest.mark.parametrize("groups", [1, 2])
def test_spatial_and_grouped_convolutions(spatial_rank: int, groups: int) -> None:
    """Cover ordinary and depthwise integer kernels for every admitted Conv rank."""
    shape = (1, 2) + (5,) * spatial_rank
    weight = np.full((2, 2 // groups) + (3,) * spatial_rank, 0.1, dtype=np.float32)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Conv", ["x", "w"], ["y"], name="conv", group=groups)],
            "spatial",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, (1, 2) + (3,) * spatial_rank
                )
            ],
            [numpy_helper.from_array(weight, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )

    def replay() -> Iterator[Mapping[str, FloatArray]]:
        """Replay a bounded ramp including negative and positive activations."""
        yield {
            "x": np.linspace(-1, 1, num=int(np.prod(shape)), dtype=np.float32).reshape(
                shape
            )
        }

    result = expect_ok(quantize(model, replay, (supported(StaticW8A8()),)))
    assert isinstance(result.execution, ExecutionReport)
    assert result.execution.integer_nodes[0].executed_operator == "QLinearConv"
    error = expect_ok(
        evaluate(OnnxEvaluator(model), OnnxEvaluator(result.model), replay, ("y",))
    )
    assert error.maximum_absolute_error < 0.03


@pytest.mark.parametrize("weight_type", list(IntegerType))
def test_unsigned_activation_weight_formats(
    operator_case: OperatorCase, samples: Samples, weight_type: IntegerType
) -> None:
    """Both admitted weight signedness choices must produce an integer kernel."""
    result = expect_ok(
        quantize(
            operator_case.model,
            samples,
            (supported(StaticW8A8(weight_type=weight_type)),),
        )
    )
    assert isinstance(result.execution, ExecutionReport)
    assert len(result.execution.integer_nodes) == 1


@pytest.mark.parametrize("kind", ["foreign_domain", "source_name_collision"])
def test_audit_does_not_accept_unrelated_kernel(
    region_model: ModelProto, region_samples: Samples, kind: str
) -> None:
    """A namesake or another source's integer node cannot hide a float fallback."""
    result = expect_ok(
        quantize(region_model, region_samples, (supported(StaticW8A8()),))
    )
    source = deepcopy(region_model)
    compiled = deepcopy(region_model)
    compiled.graph.node[0].op_type = "QLinearConv"
    match kind:
        case "foreign_domain":
            compiled.graph.node[0].domain = "foreign"
        case "source_name_collision":
            sibling = deepcopy(source.graph.node[0])
            sibling.name = "first_token_1"
            sibling.output[0] = "sibling"
            source.graph.node.append(sibling)
            compiled.graph.node[0].name = "first_token_1"
    expect_error(
        inspect_execution(source, compiled, result.plans), "expected integer Conv"
    )


def test_unnamed_sources_are_audited(
    region_model: ModelProto, region_samples: Samples
) -> None:
    """Normalized source names must participate in verification and reporting."""
    model = deepcopy(region_model)
    for node in model.graph.node:
        node.name = ""
    result = expect_ok(quantize(model, region_samples, (supported(StaticW8A8()),)))
    assert isinstance(result.execution, ExecutionReport)
    assert len(result.execution.integer_nodes) == 4
    assert all(item.node for item in result.execution.integer_nodes)
