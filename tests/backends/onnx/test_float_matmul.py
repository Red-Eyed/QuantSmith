"""Integer float-output contracts, exact reference arithmetic and source audits."""

from collections.abc import Iterable, Mapping
from copy import deepcopy

import numpy as np
import onnx
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper
from pydantic import ValidationError

from examples._shared.plans import operation_record
from quantsmith import Quantizer, Replay
from quantsmith.algorithms.shell import build_plan
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe, lower
from quantsmith.backends.onnx.integer_execution.audit import (
    compile_cpu,
    inspect_execution,
)
from quantsmith.backends.onnx.integer_execution.float_output import FloatMatMulQuantizer
from quantsmith.calibration import Requirement, Samples
from quantsmith.config import QuantizationConfig
from quantsmith.domain import (
    Absent,
    Encoding,
    FloatArray,
    IntegerType,
    PerChannel,
    PerTensor,
)
from quantsmith.pipeline import FloatMatMulOutputs
from quantsmith.plan import FloatOutput, QuantizationPlan, QuantizeInput, QuantizeOutput
from quantsmith.rules import Rules
from quantsmith.runtime import OnnxEvaluator
from tests.conftest import OperatorCase
from tests.outcomes import expect_error, expect_ok


@pytest.fixture(params=[(2,), (4, 2), (2, 4, 2)])
def matrix_model(request: pytest.FixtureRequest) -> ModelProto:
    """Cover vector, matrix and batched activations with rank-two shared weights."""
    shape: tuple[int, ...] = request.param
    weights = np.array([[0.25, -2, 0.003], [1.25, 0.5, -0.001]], dtype=np.float32)
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")],
            "float-output",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, (*shape[:-1], 3))],
            [numpy_helper.from_array(weights, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


@pytest.fixture
def matrix_samples(matrix_model: ModelProto) -> Samples:
    """Replay signed values, nonzero zero points and unequal output-channel ranges."""
    shape = tuple(
        dim.dim_value for dim in matrix_model.graph.input[0].type.tensor_type.shape.dim
    )

    def replay() -> Iterable[Mapping[str, FloatArray]]:
        """Regenerate each batch without caching runtime observations."""
        rng = np.random.default_rng(42)
        for _ in range(3):
            yield {"x": rng.uniform(-2, 5, size=shape).astype(np.float32)}

    return replay


@pytest.fixture
def float_plan(matrix_model: ModelProto, matrix_samples: Samples) -> QuantizationPlan:
    """Build the explicit plan independently of configuration orchestration."""
    graph = expect_ok(describe(matrix_model))
    return expect_ok(
        build_plan(
            graph,
            OnnxEvaluator(matrix_model),
            matrix_samples,
            Rules(default=FloatMatMulQuantizer(StaticW8A8())),
        )
    )


def input_encoding(plan: QuantizationPlan, index: int) -> Encoding:
    """Require the numerical input encoding needed by the independent reference."""
    for operation in plan.operations:
        if isinstance(operation, QuantizeInput) and operation.index == index:
            return operation.encoding
    pytest.fail("missing input encoding")


def test_integer_reference_and_public_outputs(
    matrix_model: ModelProto, matrix_samples: Samples, float_plan: QuantizationPlan
) -> None:
    """Check raw integer accumulation with per-column scales and no output rounding."""
    before = matrix_model.SerializeToString()
    result = expect_ok(
        Quantizer.from_config(QuantizationConfig(matmul_outputs=FloatMatMulOutputs()))
        .quantize(matrix_model)
        .calibrate(Replay(matrix_samples))
    )
    assert matrix_model.SerializeToString() == before
    assert not isinstance(result.execution, Absent)
    assert result.execution.integer_nodes[0].executed_operator == "MatMulIntegerToFloat"
    assert any(isinstance(op, FloatOutput) for op in result.plans[0].operations)
    assert not any(isinstance(op, QuantizeOutput) for op in result.plans[0].operations)
    assert tuple(value.name for value in result.model.graph.output) == ("y",)
    assert all(node.op_type != "DequantizeLinear" for node in result.model.graph.node)
    onnx.checker.check_model(result.model)
    activation, weight = input_encoding(float_plan, 0), input_encoding(float_plan, 1)
    assert int(activation.zero_point) != 0
    original_weight = np.asarray(
        numpy_helper.to_array(matrix_model.graph.initializer[0]), dtype=np.float32
    )
    codes_w = np.clip(
        np.rint(original_weight / weight.scale) + weight.zero_point, -128, 127
    ).astype(np.int32)
    evaluator = OnnxEvaluator(result.model)
    for sample in matrix_samples():
        codes_x = np.clip(
            np.rint(sample["x"] / activation.scale) + activation.zero_point, 0, 255
        ).astype(np.int32)
        accumulated = (codes_x - activation.zero_point.astype(np.int32)) @ (
            codes_w - weight.zero_point.astype(np.int32)
        )
        expected = accumulated.astype(np.float32) * (activation.scale * weight.scale)
        actual = expect_ok(evaluator.run(sample, ("y",)))["y"]
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_no_output_statistics_or_fake_encoding(
    matrix_model: ModelProto, float_plan: QuantizationPlan
) -> None:
    """Omit output calibration and serialize the representation without a scale."""
    graph = expect_ok(describe(matrix_model))
    needs = expect_ok(
        FloatMatMulQuantizer(StaticW8A8()).requirements(graph.nodes[0], graph)
    )
    assert needs.ranges == (Requirement(tensor="x"),)
    assert not needs.histograms
    output = next(op for op in float_plan.operations if isinstance(op, FloatOutput))
    assert operation_record(output) == {
        "node": "projection",
        "index": 0,
        "target": "float_output",
    }
    with pytest.raises(ValidationError, match="requires quantized"):
        QuantizationPlan(operations=(output,))
    encoding = input_encoding(float_plan, 0)
    expect_error(
        float_plan.then(
            QuantizationPlan(
                operations=(
                    QuantizeOutput(node="projection", index=0, encoding=encoding),
                )
            )
        ),
        "duplicate output",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "signed_activation",
        "unsigned_weight",
        "wrong_axis",
        "nonzero_weight_zero",
        "wrong_output",
    ],
)
def test_reject_invalid_lowering_contract(
    matrix_model: ModelProto, float_plan: QuantizationPlan, mutation: str
) -> None:
    """Reject unsupported manual plans rather than accepting silent float fallback."""
    activation, weight = input_encoding(float_plan, 0), input_encoding(float_plan, 1)
    if mutation == "signed_activation":
        activation = Encoding(
            scale=activation.scale,
            zero_point=np.array(0, dtype=np.int8),
            granularity=PerTensor(),
        )
    elif mutation == "unsigned_weight":
        weight = Encoding(
            scale=weight.scale,
            zero_point=np.full(3, 128, dtype=np.uint8),
            granularity=PerChannel(axis=1),
        )
    elif mutation == "wrong_axis":
        weight = Encoding(
            scale=np.ones(2, dtype=np.float32),
            zero_point=np.zeros(2, dtype=np.int8),
            granularity=PerChannel(axis=0),
        )
    elif mutation == "nonzero_weight_zero":
        weight = Encoding(
            scale=weight.scale,
            zero_point=np.ones(3, dtype=np.int8),
            granularity=PerChannel(axis=1),
        )
    plan = QuantizationPlan(
        operations=(
            QuantizeInput(node="projection", index=0, encoding=activation),
            QuantizeInput(node="projection", index=1, encoding=weight),
            FloatOutput(
                node="projection", index=1 if mutation == "wrong_output" else 0
            ),
        )
    )
    expect_error(lower(matrix_model, plan), "float MatMul")


def test_source_audit_rejects_wrong_kernel_or_owner(
    matrix_model: ModelProto, matrix_samples: Samples, float_plan: QuantizationPlan
) -> None:
    """Require float-output integer execution with the actual source identity."""
    ordinary = expect_ok(
        Quantizer.from_config(QuantizationConfig())
        .quantize(matrix_model)
        .calibrate(Replay(matrix_samples))
    )
    expect_error(
        inspect_execution(matrix_model, compile_cpu(ordinary.model), (float_plan,)),
        "integer",
    )
    compiled = compile_cpu(expect_ok(lower(matrix_model, float_plan)))
    expect_ok(inspect_execution(matrix_model, compiled, (float_plan,)))
    altered = deepcopy(compiled)
    for node in altered.graph.node:
        if node.op_type == "MatMulIntegerToFloat":
            node.name = "unrelated_integer"
    expect_error(inspect_execution(matrix_model, altered, (float_plan,)), "integer")


@pytest.mark.parametrize("dtype", [IntegerType.INT8, IntegerType.UINT8])
def test_unsupported_format_has_explicit_exclusion(
    matrix_model: ModelProto, matrix_samples: Samples, dtype: IntegerType
) -> None:
    """Keep unsupported signedness floating with a recorded eligibility reason."""
    config = QuantizationConfig(
        activation_type=dtype,
        weight_type=IntegerType.UINT8
        if dtype == IntegerType.UINT8
        else IntegerType.INT8,
        matmul_outputs=FloatMatMulOutputs(),
    )
    result = expect_ok(
        Quantizer.from_config(config)
        .quantize(matrix_model)
        .calibrate(Replay(matrix_samples))
    )
    assert not isinstance(result.execution, Absent)
    assert not result.execution.integer_nodes
    assert any(
        "float-output MatMul" in item.reason
        for item in result.execution.unverified_nodes
    )


def test_shared_public_outputs_and_bias(
    matrix_model: ModelProto, matrix_samples: Samples
) -> None:
    """Preserve shared activation/output values and floating bias placement."""
    model = deepcopy(matrix_model)
    shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
    model.graph.node.extend(
        [
            helper.make_node("Add", ["y", "bias"], ["biased"], name="bias_add"),
            helper.make_node(
                "MatMul", ["x", "w"], ["sibling"], name="sibling_projection"
            ),
        ]
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([0.125, -0.5, 2], dtype=np.float32), "bias")
    )
    model.graph.output.extend(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name in ("biased", "sibling")
    )
    before = model.SerializeToString()
    result = expect_ok(
        Quantizer.from_config(QuantizationConfig(matmul_outputs=FloatMatMulOutputs()))
        .quantize(model)
        .calibrate(Replay(matrix_samples))
    )
    assert model.SerializeToString() == before
    assert [value.name for value in result.model.graph.output] == [
        "y",
        "biased",
        "sibling",
    ]
    assert (
        sum(node.op_type == "QuantizeLinear" for node in result.model.graph.node) == 1
    )
    assert not isinstance(result.execution, Absent)
    assert len(result.execution.integer_nodes) == 2
    evaluator = OnnxEvaluator(result.model)
    for sample in matrix_samples():
        values = expect_ok(evaluator.run(sample, ("y", "biased", "sibling")))
        np.testing.assert_array_equal(values["y"], values["sibling"])
        np.testing.assert_array_equal(
            values["biased"], values["y"] + np.array([0.125, -0.5, 2], dtype=np.float32)
        )


def test_dynamic_weights_stay_float_with_reason(matrix_model: ModelProto) -> None:
    """Do not claim dynamic-weight support from direct-kernel probe capability."""
    model = deepcopy(matrix_model)
    model.graph.input.append(
        helper.make_tensor_value_info("w", TensorProto.FLOAT, [2, 3])
    )
    del model.graph.initializer[:]
    result = expect_ok(
        Quantizer.from_config(QuantizationConfig(matmul_outputs=FloatMatMulOutputs()))
        .quantize(model)
        .calibrate(())
    )
    assert not isinstance(result.execution, Absent)
    assert not result.execution.integer_nodes
    assert any(
        "constant weights" in item.reason for item in result.execution.unverified_nodes
    )


def test_float_output_rejects_non_matmul(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Keep the new output contract specific to the verified MatMul lowering."""
    graph = expect_ok(describe(operator_case.model))
    plan = expect_ok(
        build_plan(
            graph,
            OnnxEvaluator(operator_case.model),
            samples,
            Rules(default=FloatMatMulQuantizer(StaticW8A8())),
        )
    )
    if graph.nodes[0].op == "MatMul":
        expect_ok(lower(operator_case.model, plan))
    else:
        expect_error(lower(operator_case.model, plan), "two-input MatMul")


def test_float_outputs_compose_with_candidate_selection(
    matrix_model: ModelProto, matrix_samples: Samples
) -> None:
    """Include floating contractions in existing observation and audit steps."""
    from quantsmith.config import JointSmoothQuantConfig, OutputSelectedSmoothingConfig
    from quantsmith.optimization.smoothing import ContractionOutputs

    config = QuantizationConfig(
        matmul_outputs=FloatMatMulOutputs(),
        smoothing=OutputSelectedSmoothingConfig(
            candidates=(JointSmoothQuantConfig(),), observations=ContractionOutputs()
        ),
    )
    result = expect_ok(
        Quantizer.from_config(config)
        .quantize(matrix_model)
        .calibrate(Replay(matrix_samples))
    )
    assert not isinstance(result.selection, Absent)
    assert result.selection.outputs == ("y",)
    assert result.selection.selected == "baseline"
    assert all(score.integer_nodes == 1 for score in result.selection.candidates)
