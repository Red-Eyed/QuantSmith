"""Eligibility, clipping and compiled conversion elimination regressions."""

from collections import Counter
from collections.abc import Iterator, Mapping
from copy import deepcopy

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from quantsmith.algorithms.contracts import Statistics
from quantsmith.algorithms.shell import build_plan
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe
from quantsmith.backends.onnx.integer_execution.activation_function_ranges import (
    ActivationFunctionRanges,
    activation_function_outputs,
)
from quantsmith.backends.onnx.integer_execution.audit import (
    ExecutionReport,
    compile_cpu,
)
from quantsmith.backends.onnx.integer_execution.planning import (
    ExecutionQuantizer,
    execution_rules,
)
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.calibration import Percentile, Requirement, Samples
from quantsmith.config import supported
from quantsmith.domain import FloatArray
from quantsmith.plan import QuantizeOutput
from quantsmith.rules import Exclude, Rules
from quantsmith.runtime import OnnxEvaluator, evaluate
from tests.outcomes import expect_error, expect_ok


@pytest.fixture(params=["Relu", "Clip"])
def activation_model(request: pytest.FixtureRequest) -> ModelProto:
    """Connect two unit convolutions with a private activation function."""
    constants = [
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "w"),
        numpy_helper.from_array(np.array(0, dtype=np.float32), "low"),
        numpy_helper.from_array(np.array(6, dtype=np.float32), "high"),
    ]
    activation_inputs = ["a", "low", "high"] if request.param == "Clip" else ["a"]
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["x", "w"], ["a"], name="first"),
                helper.make_node(request.param, activation_inputs, ["b"], name="act"),
                helper.make_node("Conv", ["b", "w"], ["y"], name="second"),
            ],
            "activation-function",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 8])],
            constants,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


@pytest.fixture
def activation_samples() -> Samples:
    """Replay negatives, endpoints, and values beyond Clip's upper bound."""

    def replay() -> Iterator[Mapping[str, FloatArray]]:
        """Recreate the same batch for histogram replay."""
        yield {
            "x": np.array([-12, -4, -1, 0, 1, 3, 6, 12], dtype=np.float32).reshape(
                1, 1, 1, 8
            )
        }

    return replay


@pytest.mark.parametrize("histogram", [False, True])
def test_calibrates_function_output_and_fuses(
    activation_model: ModelProto, activation_samples: Samples, histogram: bool
) -> None:
    """Keep both integer convolutions and eliminate the intermediate QDQ pair."""
    before = activation_model.SerializeToString()
    algorithm = (
        StaticW8A8(calibration=Percentile(percentile=100))
        if histogram
        else StaticW8A8()
    )
    result = expect_ok(
        quantize(
            activation_model,
            activation_samples,
            (supported(algorithm),),
            activation_function_aware=True,
        )
    )
    assert activation_model.SerializeToString() == before
    assert isinstance(result.execution, ExecutionReport)
    assert len(result.execution.integer_nodes) == 2
    compiled = compile_cpu(result.model)
    counts = Counter(node.op_type for node in compiled.graph.node)
    assert counts["QLinearConv"] == 2
    assert counts["QuantizeLinear"] == counts["DequantizeLinear"] == 1
    encoding = next(
        operation.encoding
        for operation in result.plans[0].operations
        if isinstance(operation, QuantizeOutput) and operation.node == "first"
    )
    assert int(encoding.zero_point) == 0
    upper = 6 if activation_model.graph.node[1].op_type == "Clip" else 12
    assert float(encoding.scale) == pytest.approx(upper / 255, rel=0.001)
    error = expect_ok(
        evaluate(
            OnnxEvaluator(activation_model),
            OnnxEvaluator(result.model),
            activation_samples,
            ("y",),
        )
    )
    assert error.maximum_absolute_error < 0.11


@pytest.mark.parametrize(
    "obstacle", ["shared", "public", "unsupported", "foreign", "dynamic_clip"]
)
def test_ineligible_outputs_preserve_original_ranges(
    activation_model: ModelProto, obstacle: str
) -> None:
    """Never borrow a function's range for other observable pre-function uses."""
    model = deepcopy(activation_model)
    match obstacle:
        case "shared":
            model.graph.node.append(helper.make_node("Identity", ["a"], ["extra"]))
        case "public":
            model.graph.output.append(
                helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 1, 1, 8])
            )
        case "unsupported":
            model.graph.node[1].op_type = "Sigmoid"
        case "foreign":
            model.graph.node[1].domain = "custom"
        case "dynamic_clip":
            model.graph.node[1].op_type = "Clip"
            del model.graph.node[1].input[:]
            model.graph.node[1].input.extend(["a", "dynamic"])
    assert not activation_function_outputs(model)


def test_default_and_explicit_disable_match(
    activation_model: ModelProto, activation_samples: Samples
) -> None:
    """Disabled policy requests the original producer range without a wrapper."""
    graph = expect_ok(describe(activation_model))
    rules = supported(StaticW8A8())
    default = execution_rules(activation_model, graph, rules)
    disabled = execution_rules(
        activation_model, graph, rules, activation_function_aware=False
    )
    assert isinstance(default.resolve(graph.nodes[0], graph), ExecutionQuantizer)
    assert isinstance(disabled.resolve(graph.nodes[0], graph), ExecutionQuantizer)
    evaluator = OnnxEvaluator(activation_model)
    first = expect_ok(build_plan(graph, evaluator, activation_samples, default))
    second = expect_ok(build_plan(graph, evaluator, activation_samples, disabled))
    np.testing.assert_array_equal(
        next(
            op.encoding.scale
            for op in first.operations
            if isinstance(op, QuantizeOutput)
        ),
        next(
            op.encoding.scale
            for op in second.operations
            if isinstance(op, QuantizeOutput)
        ),
    )


def test_output_requirements_and_missing_statistics(
    activation_model: ModelProto,
) -> None:
    """Replay function extrema/histograms and preserve expected missing-data failure."""
    graph = expect_ok(describe(activation_model))
    adapter = ActivationFunctionRanges(
        ExecutionQuantizer(StaticW8A8(calibration=Percentile())), "b"
    )
    needs = expect_ok(adapter.requirements(graph.nodes[0], graph))
    assert Requirement(tensor="a") not in needs.ranges
    assert Requirement(tensor="b") in needs.ranges
    assert Requirement(tensor="b") in needs.histograms
    assert Requirement(tensor="x") in needs.histograms
    expect_error(
        adapter.plan(graph.nodes[0], graph, Statistics(ranges={}, histograms={})),
        "activation-function output statistics",
    )


def test_non_static_decisions_are_preserved(activation_model: ModelProto) -> None:
    """Only built-in execution quantizers participate in output-range adaptation."""
    graph = expect_ok(describe(activation_model))
    for decision in (SmoothQuant(), Exclude(reason="user excluded")):
        rules = execution_rules(
            activation_model,
            graph,
            Rules(default=decision),
            activation_function_aware=True,
        )
        assert rules.resolve(graph.nodes[0], graph) is decision


@pytest.mark.parametrize("value", [-12.0, 0.0])
def test_degenerate_function_output(activation_model: ModelProto, value: float) -> None:
    """A completely clipped range remains finite and executes as integer Conv."""

    def replay() -> Iterator[Mapping[str, FloatArray]]:
        """Calibrate an all-negative or all-zero image without storing activations."""
        yield {"x": np.full((1, 1, 1, 8), value, dtype=np.float32)}

    result = expect_ok(
        quantize(
            activation_model,
            replay,
            (supported(StaticW8A8()),),
            activation_function_aware=True,
        )
    )
    error = expect_ok(
        evaluate(
            OnnxEvaluator(activation_model),
            OnnxEvaluator(result.model),
            replay,
            ("y",),
        )
    )
    assert error.maximum_absolute_error == 0


def test_planning_does_not_mutate_statistics(
    activation_model: ModelProto, activation_samples: Samples
) -> None:
    """Remapping an output range must not rewrite the caller's revision statistics."""
    from quantsmith.calibration import collect

    graph = expect_ok(describe(activation_model))
    adapter = ActivationFunctionRanges(ExecutionQuantizer(StaticW8A8()), "b")
    needs = expect_ok(adapter.requirements(graph.nodes[0], graph))
    ranges = expect_ok(
        collect(OnnxEvaluator(activation_model), activation_samples, needs.ranges)
    )
    stats = Statistics(ranges=ranges, histograms={})
    before = tuple(stats.ranges)
    expect_ok(adapter.plan(graph.nodes[0], graph, stats))
    assert tuple(stats.ranges) == before
    assert Requirement(tensor="a") not in stats.ranges
    for value in stats.ranges.values():
        assert not value.minimum.flags.writeable
        assert not value.maximum.flags.writeable
