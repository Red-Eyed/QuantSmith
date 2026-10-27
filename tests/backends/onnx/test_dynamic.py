"""Dynamic projection arithmetic, runtime scale changes, and execution evidence."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from examples._shared.plans import operation_record
from quantsmith import Quantizer, Replay
from quantsmith.algorithms.contracts import Algorithm, Needs, Statistics
from quantsmith.algorithms.dynamic import DynamicW8A8
from quantsmith.backends.onnx import describe, lower
from quantsmith.backends.onnx.dynamic import DynamicQuantizationStage
from quantsmith.backends.onnx.integer_execution.audit import (
    compile_cpu,
    inspect_execution,
)
from quantsmith.backends.onnx.stages import VerifyStage
from quantsmith.calibration import Requirement
from quantsmith.config import DynamicQuantizationConfig, QuantizationConfig
from quantsmith.domain import Absent, FloatArray, Graph, Node
from quantsmith.pipeline import Pipeline
from quantsmith.plan import DynamicQuantizeMatMul, QuantizationPlan, QuantizeInput
from quantsmith.result import FailureKind, Ok, QuantSmithError, Result, failure
from quantsmith.rules import ByName, Exclude, Rule, Rules
from quantsmith.runtime import OnnxEvaluator
from tests.outcomes import expect_error, expect_ok


@dataclass(frozen=True)
class PluginAlgorithm:
    """Exercise the open Algorithm contract without inheriting the built-in method."""

    patch: QuantizationPlan
    behavior: Literal["dynamic", "needs", "static", "failure", "panic"]

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Request calibration only for the deliberately incompatible method."""
        if self.behavior == "needs":
            return Ok(Needs(ranges=(Requirement(tensor=node.inputs[0]),)))
        return Ok(Needs())

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Preserve the distinction between recoverable errors and plugin bugs."""
        match self.behavior:
            case "failure":
                return failure(FailureKind.UNSUPPORTED, node.name, "plugin failure")
            case "panic":
                raise RuntimeError("plugin bug")
            case _:
                return Ok(self.patch)


@pytest.fixture(params=[(3,), ("tokens", 3), ("batch", "tokens", 3)])
def projection_model(request: pytest.FixtureRequest) -> ModelProto:
    """Cover vector and symbolic matrix/batched inputs with distinct column ranges."""
    shape: tuple[int | str, ...] = request.param
    weights = np.array(
        [[0.25, -2, 0.003, 0], [1.25, 0.5, -0.001, 0], [-0.75, 1, 0.007, 0]],
        dtype=np.float32,
    )
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")],
            "dynamic-projection",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, (*shape[:-1], 4))],
            [numpy_helper.from_array(weights, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


@pytest.fixture
def dynamic_plan(projection_model: ModelProto) -> QuantizationPlan:
    """Plan from weights alone, independently of the facade and ONNX lowering."""
    graph = expect_ok(describe(projection_model))
    return expect_ok(
        DynamicW8A8().plan(graph.nodes[0], graph, Statistics(ranges={}, histograms={}))
    )


def reference_output(value: FloatArray, weight: FloatArray) -> FloatArray:
    """Implement ONNX's per-tensor dynamic affine arithmetic independently."""
    low = np.minimum(np.min(value), np.float32(0))
    high = np.maximum(np.max(value), np.float32(0))
    scale = (high - low) / np.float32(255) if high != low else np.float32(1)
    zero = np.clip(np.rint(-low / scale), 0, 255)
    activation = np.clip(np.rint(value / scale) + zero, 0, 255).astype(np.int32)
    bound = np.max(np.abs(weight), axis=0)
    weight_scale = np.where(bound == 0, np.float32(1), bound / np.float32(127))
    integer_weight = np.clip(np.rint(weight / weight_scale), -128, 127).astype(np.int32)
    accumulator = (activation - np.int32(zero)) @ integer_weight
    return np.asarray(
        accumulator.astype(np.float32) * (scale * weight_scale), dtype=np.float32
    )


@pytest.mark.parametrize("magnitude", [0, 0.001, 1, 1000])
def test_runtime_ranges_and_symbolic_lengths(
    projection_model: ModelProto, magnitude: float
) -> None:
    """One graph/session handles prefill, repeated decode, and changing ranges."""
    original = projection_model.SerializeToString()
    result = expect_ok(
        Quantizer.from_config(DynamicQuantizationConfig())
        .quantize(projection_model)
        .execute()
    )
    assert projection_model.SerializeToString() == original
    assert not isinstance(result.execution, Absent)
    assert len(result.execution.integer_nodes) == 1
    assert result.execution.integer_nodes[0].node == "projection"
    assert result.execution.integer_nodes[0].executed_operator in {
        "MatMulInteger",
        "MatMulIntegerToFloat",
        "DynamicQuantizeMatMul",
    }
    assert all(not item.domain for item in result.model.graph.node)
    assert [item.name for item in result.model.graph.input] == ["x"]
    assert [item.name for item in result.model.graph.output] == ["y"]
    assert result.model.graph.input == projection_model.graph.input
    assert result.model.graph.output == projection_model.graph.output
    onnx.checker.check_model(result.model)
    weight = np.asarray(numpy_helper.to_array(projection_model.graph.initializer[0]))
    evaluator = OnnxEvaluator(result.model)
    optimized = OnnxEvaluator(
        result.model, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    rank = len(projection_model.graph.input[0].type.tensor_type.shape.dim)
    for tokens in (8, 1, 3, 1):
        shape = (3,) if rank == 1 else (tokens, 3) if rank == 2 else (2, tokens, 3)
        value = np.random.default_rng(tokens).uniform(-2, 5, shape).astype(np.float32)
        value *= magnitude
        actual = expect_ok(evaluator.run({"x": value}, ("y",)))["y"]
        np.testing.assert_allclose(
            actual, reference_output(value, weight), rtol=1e-5, atol=1e-6
        )
        fused = expect_ok(optimized.run({"x": value}, ("y",)))["y"]
        # ORT's ARM fused kernel can select a different dynamic activation grid.
        for output in (actual, fused):
            error = np.linalg.norm(output - value @ weight)
            assert error <= 0.025 * max(float(np.linalg.norm(value @ weight)), 1e-8)


def test_no_calibration_source_is_read(projection_model: ModelProto) -> None:
    """Dynamic lowering must never open a dataset even when one is supplied."""

    def unread() -> Iterable[Mapping[str, FloatArray]]:
        """Fail on source access rather than providing cached observations."""
        pytest.fail("dynamic quantization read calibration samples")
        yield {}

    result = expect_ok(
        Quantizer.from_config(DynamicQuantizationConfig())
        .quantize(projection_model)
        .calibrate(Replay(unread))
    )
    assert len(result.plans[0].operations) == 1
    assert isinstance(result.plans[0].operations[0], DynamicQuantizeMatMul)
    record = operation_record(result.plans[0].operations[0])
    assert record.get("target") == "dynamic_matmul"


def test_lowering_preserves_owned_graph_and_shared_weights(
    projection_model: ModelProto, dynamic_plan: QuantizationPlan
) -> None:
    """Unselected consumers and public floating constants survive quantization."""
    projection_model.graph.node.append(
        helper.make_node("MatMul", ["x", "w"], ["other"], name="unselected")
    )
    projection_model.graph.output.append(
        helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 4])
    )
    before = projection_model.SerializeToString()
    lowered = expect_ok(lower(projection_model, dynamic_plan))
    assert projection_model.SerializeToString() == before
    assert "w" in {tensor.name for tensor in lowered.graph.initializer}
    unselected = next(node for node in lowered.graph.node if node.name == "unselected")
    assert tuple(unselected.input) == ("x", "w")
    assert tuple(node.name for node in lowered.graph.output) == ("y", "w")


def test_dynamic_attention_is_excluded(projection_model: ModelProto) -> None:
    """Runtime-weight attention products stay float with an explicit reason."""
    projection_model.graph.node.append(
        helper.make_node("MatMul", ["y", "kv"], ["scores"], name="attention")
    )
    projection_model.graph.input.append(
        helper.make_tensor_value_info("kv", TensorProto.FLOAT, [4, "kv_tokens"])
    )
    shape = [
        dim.dim_param or dim.dim_value
        for dim in projection_model.graph.output[0].type.tensor_type.shape.dim
    ]
    projection_model.graph.output.append(
        helper.make_tensor_value_info(
            "scores", TensorProto.FLOAT, [*shape[:-1], "kv_tokens"]
        )
    )
    result = expect_ok(
        Quantizer.from_config(DynamicQuantizationConfig())
        .quantize(projection_model)
        .execute()
    )
    assert not isinstance(result.execution, Absent)
    assert result.plans[0].excluded == ("attention",)
    assert result.execution.unverified_nodes[0].node == "attention"
    assert "constant" in result.execution.unverified_nodes[0].reason
    compiled = compile_cpu(result.model)
    assert any(
        node.name == "attention" and node.op_type == "MatMul"
        for node in compiled.graph.node
    )


def test_dynamic_audit_rejects_floating_graph(
    projection_model: ModelProto, dynamic_plan: QuantizationPlan
) -> None:
    """A dynamic plan cannot claim integer execution for the unchanged graph."""
    expect_error(
        inspect_execution(projection_model, projection_model, (dynamic_plan,)),
        "expected integer",
    )


@pytest.mark.parametrize("obstacle", ["fixed", "wrong_zero", "foreign"])
def test_dynamic_audit_requires_runtime_quantization_and_source(
    projection_model: ModelProto, dynamic_plan: QuantizationPlan, obstacle: str
) -> None:
    """An unrelated or fixed-scale integer node cannot satisfy a dynamic plan."""
    lowered = expect_ok(lower(projection_model, dynamic_plan))
    integer = next(node for node in lowered.graph.node if node.name == "projection")
    quantizer = next(
        node for node in lowered.graph.node if node.op_type == "DynamicQuantizeLinear"
    )
    match obstacle:
        case "fixed":
            quantizer.op_type = "QuantizeLinear"
        case "wrong_zero":
            integer.input[2] = integer.input[3]
        case "foreign":
            integer.name = "unrelated_projection"
    expect_error(
        inspect_execution(projection_model, lowered, (dynamic_plan,)),
        "expected integer",
    )


def test_rule_exclusions_remain_visible(projection_model: ModelProto) -> None:
    """Custom selection keeps its stated reason even for an eligible projection."""
    rules = Rules(
        default=DynamicW8A8(),
        overrides=(
            Rule(
                selector=ByName(pattern="projection"),
                decision=Exclude(reason="sensitive projection"),
            ),
        ),
    )
    program = Pipeline(DynamicQuantizationStage(rules)).then(VerifyStage())
    result = expect_ok(
        Quantizer.from_pipeline(program).quantize(projection_model).execute()
    )
    assert not result.plans[0].operations
    assert result.plans[0].excluded == ("projection",)
    assert not isinstance(result.execution, Absent)
    assert not result.execution.integer_nodes
    assert result.execution.unverified_nodes[0].reason == "sensitive projection"


@pytest.mark.parametrize("behavior", ["dynamic", "needs", "static", "failure", "panic"])
def test_plugin_contract(
    projection_model: ModelProto,
    dynamic_plan: QuantizationPlan,
    behavior: Literal["dynamic", "needs", "static", "failure", "panic"],
) -> None:
    """Admit compatible extensions and preserve errors without weakening stage needs."""
    patch = dynamic_plan
    if behavior == "static":
        match dynamic_plan.operations[0]:
            case DynamicQuantizeMatMul(weight_encoding=encoding):
                patch = QuantizationPlan(
                    operations=(
                        QuantizeInput(node="projection", index=1, encoding=encoding),
                    )
                )
    rules = Rules[Algorithm](default=PluginAlgorithm(patch, behavior))
    program = Pipeline(DynamicQuantizationStage(rules)).then(VerifyStage())
    job = Quantizer.from_pipeline(program).quantize(projection_model)
    match behavior:
        case "dynamic":
            expect_ok(job.execute())
        case "panic":
            with pytest.raises(RuntimeError, match="plugin bug"):
                job.execute()
        case "failure":
            error = expect_error(job.execute(), "plugin failure")
            assert error.kind == FailureKind.UNSUPPORTED
        case "needs" | "static":
            error = expect_error(job.execute(), "dynamic stage requires")
            assert error.kind == FailureKind.CONFLICT


def test_lowering_rejects_mismatched_weight_channels(
    projection_model: ModelProto, dynamic_plan: QuantizationPlan
) -> None:
    """Plans from another weight revision cannot silently broadcast to this graph."""
    projection_model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.ones((3, 5), dtype=np.float32), "w")
    )
    projection_model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = 5
    expect_error(lower(projection_model, dynamic_plan), "encoding size")


@pytest.mark.parametrize("constant", [-3, 0, 3])
def test_constant_activation_intervals(
    projection_model: ModelProto, constant: float
) -> None:
    """Zero and one-sided activation ranges still include real zero."""
    result = expect_ok(
        Quantizer.from_config(DynamicQuantizationConfig())
        .quantize(projection_model)
        .execute()
    )
    rank = len(projection_model.graph.input[0].type.tensor_type.shape.dim)
    shape = (3,) if rank == 1 else (2, 3) if rank == 2 else (1, 2, 3)
    value = np.full(shape, constant, dtype=np.float32)
    weight = np.asarray(numpy_helper.to_array(projection_model.graph.initializer[0]))
    actual = expect_ok(OnnxEvaluator(result.model).run({"x": value}, ("y",)))["y"]
    np.testing.assert_allclose(
        actual, reference_output(value, weight), rtol=1e-5, atol=1e-6
    )


def test_static_pipeline_still_requires_samples(projection_model: ModelProto) -> None:
    """The sample-free entry point cannot silently skip required calibration."""
    expect_error(
        Quantizer.from_config(QuantizationConfig())
        .quantize(projection_model)
        .execute(),
        "no samples",
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_weights_fail(projection_model: ModelProto, invalid: float) -> None:
    """Malformed selected data returns a numerical failure before runtime startup."""
    weight = np.asarray(
        numpy_helper.to_array(projection_model.graph.initializer[0])
    ).copy()
    weight[0, 0] = invalid
    projection_model.graph.initializer[0].CopyFrom(numpy_helper.from_array(weight, "w"))
    expect_error(
        Quantizer.from_config(DynamicQuantizationConfig())
        .quantize(projection_model)
        .execute(),
        "finite",
    )


def test_invalid_model_fails_at_execution(projection_model: ModelProto) -> None:
    """Model binding owns its snapshot and admission errors use Result."""
    job = Quantizer.from_config(DynamicQuantizationConfig()).quantize(projection_model)
    projection_model.graph.node[0].input[0] = "missing"
    expect_ok(job.execute())
    invalid_job = Quantizer.from_config(DynamicQuantizationConfig()).quantize(
        deepcopy(projection_model)
    )
    expect_error(invalid_job.execute(), "missing")


@pytest.fixture
def cache_model() -> ModelProto:
    """Expose a floating cache with symbolic past, current and total lengths."""
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w"], ["new_kv"], name="projection"),
                helper.make_node(
                    "Concat",
                    ["past", "new_kv"],
                    ["present"],
                    name="cache_append",
                    axis=1,
                ),
            ],
            "cache",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "tokens", 3]),
                helper.make_tensor_value_info(
                    "past", TensorProto.FLOAT, [1, "past_tokens", 4]
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "present", TensorProto.FLOAT, [1, "total_tokens", 4]
                )
            ],
            [
                numpy_helper.from_array(
                    np.arange(12, dtype=np.float32).reshape(3, 4) / 12, "w"
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


def test_empty_cache_prefill_and_repeated_decode(cache_model: ModelProto) -> None:
    """Preserve all prior cache entries through prefill and growing decode state."""
    result = expect_ok(
        Quantizer.from_config(DynamicQuantizationConfig())
        .quantize(cache_model)
        .execute()
    )
    assert result.model.graph.input == cache_model.graph.input
    assert result.model.graph.output == cache_model.graph.output
    runtime = OnnxEvaluator(
        result.model, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    past = np.empty((1, 0, 4), dtype=np.float32)
    weight = np.asarray(numpy_helper.to_array(cache_model.graph.initializer[0]))
    for tokens, magnitude in ((3, 1), (1, 0.01), (1, 100)):
        value = (
            np.random.default_rng(tokens).normal(size=(1, tokens, 3)).astype(np.float32)
            * magnitude
        )
        present = expect_ok(runtime.run({"x": value, "past": past}, ("present",)))[
            "present"
        ]
        assert present.dtype == np.float32
        assert present.shape == (1, past.shape[1] + tokens, 4)
        np.testing.assert_array_equal(present[:, : past.shape[1]], past)
        expected = value @ weight
        error = np.linalg.norm(present[:, past.shape[1] :] - expected)
        assert error / np.linalg.norm(expected) < 0.025
        past = present
