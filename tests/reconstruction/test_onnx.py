"""Verify reconstructed plans against real ONNX Runtime execution."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from onnx import ModelProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx import lower
from quantsmith.backends.onnx.reconstruction import reconstruct
from quantsmith.calibration import Samples
from quantsmith.config import (
    JointSmoothQuantConfig,
    QuantizationConfig,
    SmoothQuantConfig,
)
from quantsmith.domain import Encoding, FloatArray, PerChannel
from quantsmith.pipeline import RuntimeScaling
from quantsmith.plan import QuantizationPlan, QuantizeConstant, QuantizeInput
from quantsmith.reconstruction import Problem, Reconstructor, Replay, Solution
from quantsmith.reconstruction.affine import quantize, restore
from quantsmith.reconstruction.gptqv2 import GPTQv2
from quantsmith.reconstruction.qdrop import QDrop, array, forward
from quantsmith.result import Err, FailureKind, QuantSmithError, Result
from quantsmith.rules import ByName, Exclude, Rule, Rules
from quantsmith.runtime import OnnxEvaluator
from tests.conftest import OperatorCase
from tests.outcomes import expect_ok


@pytest.mark.parametrize("method", [GPTQv2(), QDrop(steps=3)])
def test_operator_export(
    operator_case: OperatorCase, samples: Samples, method: Reconstructor
) -> None:
    """Exercise all Gemm transposes and Conv through the actual public entry point."""
    original = operator_case.model.SerializeToString()
    rules: Rules[Reconstructor] = Rules(default=method)
    outcome = reconstruct(operator_case.model, samples, rules)
    if operator_case.model.graph.node[0].op_type == "Conv" and isinstance(
        method, GPTQv2
    ):
        assert isinstance(outcome, Err)
        assert outcome.error.kind == FailureKind.UNSUPPORTED
        return
    result = expect_ok(outcome)
    assert operator_case.model.SerializeToString() == original
    sample = next(iter(samples()))
    actual = expect_ok(OnnxEvaluator(result.model).run(sample, ("y",)))["y"]
    from quantsmith.backends.onnx import describe
    from quantsmith.backends.onnx.reconstruction import input_layout, operator

    graph = expect_ok(describe(operator_case.model))
    node = graph.nodes[0]
    match result.plans[0].operations:
        case (
            QuantizeInput(encoding=activation),
            QuantizeConstant(values=values, encoding=encoding),
        ):
            weights = restore(values, encoding)
        case _:
            pytest.fail("missing reconstructed codes")
    if operator_case.weight_axis == 1:
        weights = weights.T
    assert sample["x"].dtype == np.float32
    inputs = input_layout(node)(
        quantize(np.asarray(sample["x"], dtype=np.float32), activation)
    )
    expected = array(
        forward(
            operator(node), torch.tensor(inputs.copy()), torch.tensor(weights.copy())
        )
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.fixture
def shared_model() -> ModelProto:
    """Expose two consumers of one initializer to detect accidental global writes."""
    weight = np.array([[0.49, 0.17], [-0.33, 0.91]], dtype=np.float32)
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w"], ["hidden"], name="first"),
                helper.make_node("MatMul", ["hidden", "w"], ["y"], name="second"),
            ],
            "shared",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
            [numpy_helper.from_array(weight, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


@pytest.fixture
def shared_samples() -> Samples:
    """Use reproducible data with visible activation rounding error."""

    def replay() -> Iterable[Mapping[str, FloatArray]]:
        """Yield one fresh input batch for every pass."""
        yield {"x": np.array([[0.13, 0.91], [1.22, -0.45]], dtype=np.float32)}

    return replay


@dataclass
class RecordingMethod:
    """Observe reference/candidate divergence without retaining calibration data."""

    differences: list[float] = field(default_factory=list)

    def reconstruct(
        self, problem: Problem, replay: Replay
    ) -> Result[Solution, QuantSmithError]:
        """Inspect one paired batch, then delegate to the real numerical method."""
        batch = expect_ok(next(iter(replay())))
        self.differences.append(
            float(np.max(np.abs(batch.reference - batch.candidate)))
        )
        return GPTQv2().reconstruct(problem, replay)


def test_sequential_reconstruction_preserves_reference(
    shared_model: ModelProto, shared_samples: Samples
) -> None:
    """Later layers see actual upstream quantization while targets stay floating."""
    method = RecordingMethod()
    rules: Rules[Reconstructor] = Rules(default=method)
    result = expect_ok(reconstruct(shared_model, shared_samples, rules))
    assert len(result.plans) == 2
    assert method.differences[0] == 0
    assert method.differences[1] > 0


def test_exact_codes_are_local_to_consumer(
    shared_model: ModelProto, shared_samples: Samples
) -> None:
    """Replacing the first edge must leave the second consumer's weights intact."""
    encoding = Encoding(
        scale=np.ones(2, dtype=np.float32),
        zero_point=np.zeros(2, dtype=np.int8),
        granularity=PerChannel(axis=1),
    )
    operation = QuantizeConstant(
        node="first", index=1, values=np.eye(2, dtype=np.int8), encoding=encoding
    )
    candidate = expect_ok(
        lower(shared_model, QuantizationPlan(operations=(operation,)))
    )
    second = next(node for node in candidate.graph.node if node.name == "second")
    assert second.input[1] == "w"
    sample = next(iter(shared_samples()))
    actual = expect_ok(OnnxEvaluator(candidate).run(sample, ("y",)))["y"]
    expected = sample["x"] @ np.array([[0.49, 0.17], [-0.33, 0.91]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_explicit_exclusion(shared_model: ModelProto, shared_samples: Samples) -> None:
    """A skipped consumer stays FP32 and remains visible in the returned plans."""
    rules: Rules[Reconstructor] = Rules(
        default=GPTQv2(),
        overrides=(
            Rule(
                selector=ByName(pattern="second"),
                decision=Exclude(reason="keep output"),
            ),
        ),
    )
    result = expect_ok(reconstruct(shared_model, shared_samples, rules))
    assert result.plans[-1].excluded == ("second",)
    second = next(node for node in result.model.graph.node if node.name == "second")
    assert second.input[1] == "w"


@pytest.mark.parametrize(
    "smoothing", [SmoothQuantConfig(folding=RuntimeScaling()), JointSmoothQuantConfig()]
)
def test_smoothing_precedes_reconstruction(
    shared_model: ModelProto,
    shared_samples: Samples,
    smoothing: SmoothQuantConfig | JointSmoothQuantConfig,
) -> None:
    """Smooth once, then replay on the transformed reference and current graph."""
    rules: Rules[Reconstructor] = Rules(default=GPTQv2())
    result = expect_ok(
        reconstruct(
            shared_model,
            shared_samples,
            rules,
            config=QuantizationConfig(smoothing=smoothing),
        )
    )
    assert len(result.plans) > 2
    sample = next(iter(shared_samples()))
    actual = expect_ok(OnnxEvaluator(result.model).run(sample, ("y",)))["y"]
    assert np.isfinite(actual).all()


@pytest.fixture(params=[1, 2])
def convolution(request: pytest.FixtureRequest) -> ModelProto:
    """Exercise grouped geometry with asymmetric padding, stride, dilation and bias."""
    groups = int(request.param)
    weight = (
        np.random.default_rng(7).normal(size=(4, 2 // groups, 2, 3)).astype(np.float32)
    )
    bias = np.array([0.1, -0.3, 0.7, 0.2], dtype=np.float32)
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Conv",
                    ["x", "w", "bias"],
                    ["y"],
                    name="conv",
                    group=groups,
                    pads=[1, 2, 0, 1],
                    strides=[2, 1],
                    dilations=[2, 1],
                )
            ],
            "convolution",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 7, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 3, 9])],
            [
                numpy_helper.from_array(weight, "w"),
                numpy_helper.from_array(bias, "bias"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


@pytest.fixture
def convolution_samples() -> Samples:
    """Return a replayable convolution batch independent of the weight generator."""

    def replay() -> Iterable[Mapping[str, FloatArray]]:
        """Generate a fresh current batch on each invocation."""
        yield {
            "x": np.random.default_rng(11).normal(size=(2, 2, 7, 8)).astype(np.float32)
        }

    return replay


def test_grouped_convolution_export(
    convolution: ModelProto, convolution_samples: Samples
) -> None:
    """Verify reconstructed grouped Conv2d and its unchanged bias against ORT."""
    from quantsmith.backends.onnx import describe
    from quantsmith.backends.onnx.reconstruction import operator

    rules: Rules[Reconstructor] = Rules(default=QDrop(steps=3))
    result = expect_ok(reconstruct(convolution, convolution_samples, rules))
    sample = next(iter(convolution_samples()))
    graph = expect_ok(describe(convolution))
    match result.plans[0].operations:
        case (
            QuantizeInput(encoding=activation),
            QuantizeConstant(values=values, encoding=encoding),
        ):
            weights = restore(values, encoding)
        case _:
            pytest.fail("missing reconstructed codes")
    assert sample["x"].dtype == np.float32
    inputs = quantize(np.asarray(sample["x"], dtype=np.float32), activation)
    output = forward(
        operator(graph.nodes[0]), torch.tensor(inputs), torch.tensor(weights)
    )
    expected = array(output) + graph.weights["bias"].reshape(1, -1, 1, 1)
    actual = expect_ok(OnnxEvaluator(result.model).run(sample, ("y",)))["y"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)


def test_output_selected_static_objective_is_rejected(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Reject static candidate scoring at the weight-reconstruction boundary."""
    from quantsmith.config import OutputSelectedSmoothingConfig
    from tests.outcomes import expect_error

    config = QuantizationConfig(
        smoothing=OutputSelectedSmoothingConfig(candidates=(JointSmoothQuantConfig(),))
    )
    rules: Rules[Reconstructor] = Rules(default=GPTQv2())
    expect_error(
        reconstruct(operator_case.model, samples, rules, config=config),
        "separate candidate objective",
    )


def test_float_integer_outputs_require_static_pipeline(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Reject a static output representation at the reconstruction boundary."""
    from quantsmith.pipeline import FloatMatMulOutputs
    from tests.outcomes import expect_error

    config = QuantizationConfig(matmul_outputs=FloatMatMulOutputs())
    rules: Rules[Reconstructor] = Rules(default=GPTQv2())
    expect_error(
        reconstruct(operator_case.model, samples, rules, config=config),
        "static execution pipeline",
    )
