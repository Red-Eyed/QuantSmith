"""Numerical, ownership, and conservative-skip contracts for scale folding."""

from collections.abc import Iterator, Mapping
from copy import deepcopy

import numpy as np
import onnx
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx.optimization import optimize
from quantsmith.domain import FloatArray
from quantsmith.runtime import OnnxEvaluator
from tests.outcomes import expect_ok


@pytest.fixture(params=["Conv", "BatchNormalization"])
def scaled_model(request: pytest.FixtureRequest) -> ModelProto:
    """Build an affine producer with nonzero bias and nonuniform channel scales."""
    kind: str = request.param
    rng = np.random.default_rng(42)
    parameters = {"b": np.array([0.3, -0.5], dtype=np.float32)}
    if kind == "Conv":
        parameters["w"] = rng.normal(size=(2, 2, 1, 1)).astype(np.float32)
        inputs = ["x", "w", "b"]
    else:
        parameters.update(
            {
                "w": np.array([1.5, -0.2], dtype=np.float32),
                "mean": np.array([0.2, -0.8], dtype=np.float32),
                "variance": np.array([0.4, 1.2], dtype=np.float32),
            }
        )
        inputs = ["x", "w", "b", "mean", "variance"]
    parameters["s"] = np.array([2.0, 0.5], dtype=np.float32).reshape(1, 2, 1, 1)
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(kind, inputs, ["a"], name="producer"),
                helper.make_node("Div", ["a", "s"], ["y"], name="scale"),
            ],
            "scaling",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 3])],
            [
                numpy_helper.from_array(value, name)
                for name, value in parameters.items()
            ],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


def assert_equivalent(before: ModelProto, after: ModelProto) -> None:
    """Compare actual ORT outputs with nonuniform, signed activation values."""
    inputs = {
        "x": np.random.default_rng(5).normal(size=(1, 2, 3, 3)).astype(np.float32)
    }
    original = expect_ok(OnnxEvaluator(before).run(inputs, ("y",)))
    rewritten = expect_ok(OnnxEvaluator(after).run(inputs, ("y",)))
    np.testing.assert_allclose(rewritten["y"], original["y"], rtol=2e-6, atol=2e-6)


def test_fold_affine(scaled_model: ModelProto) -> None:
    """Fold weights and bias, retain output identity, and own all changes."""
    original = scaled_model.SerializeToString()
    optimized = expect_ok(optimize(scaled_model))
    assert scaled_model.SerializeToString() == original
    assert [node.op_type for node in optimized.graph.node] == [
        scaled_model.graph.node[0].op_type
    ]
    assert optimized.graph.node[0].output[0] == "y"
    assert "s" not in {tensor.name for tensor in optimized.graph.initializer}
    onnx.checker.check_model(optimized)
    assert_equivalent(scaled_model, optimized)
    assert (
        expect_ok(optimize(optimized)).SerializeToString()
        == optimized.SerializeToString()
    )


@pytest.mark.parametrize(
    "protection", ["consumer", "output", "broadcast", "negative", "custom"]
)
def test_skip_unsafe(scaled_model: ModelProto, protection: str) -> None:
    """Preserve shared values, exposed outputs, unsupported scales and domains."""
    model = deepcopy(scaled_model)
    if protection == "consumer":
        model.graph.node.append(
            helper.make_node("Identity", ["a"], ["sibling"], name="sibling")
        )
    elif protection == "output":
        model.graph.output.append(
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 3, 3])
        )
    elif protection in {"broadcast", "negative"}:
        value = (
            np.ones((1, 1, 3, 1), dtype=np.float32)
            if protection == "broadcast"
            else -np.ones((1, 2, 1, 1), dtype=np.float32)
        )
        model.graph.initializer[-1].CopyFrom(numpy_helper.from_array(value, "s"))
    else:
        model.graph.node[0].domain = "custom"
        model.opset_import.append(helper.make_opsetid("custom", 1))
    optimized = expect_ok(optimize(model))
    assert any(node.op_type == "Div" for node in optimized.graph.node)


def test_shared_weights(scaled_model: ModelProto) -> None:
    """Create private replacements when another producer shares its parameters."""
    sibling = deepcopy(scaled_model.graph.node[0])
    sibling.name = "sibling"
    sibling.output[0] = "other"
    scaled_model.graph.node.append(sibling)
    original = scaled_model.graph.initializer[0].SerializeToString()
    optimized = expect_ok(optimize(scaled_model))
    assert optimized.graph.node[-1].input[1] == "w"
    assert (
        next(
            value for value in optimized.graph.initializer if value.name == "b"
        ).SerializeToString()
        == original
    )
    assert_equivalent(scaled_model, optimized)


def test_pipeline_optimizes_before_recalibration(scaled_model: ModelProto) -> None:
    """Fold before fresh calibration while retaining usable quantized outputs."""
    from quantsmith.algorithms.smoothquant import SmoothQuant
    from quantsmith.algorithms.static import StaticW8A8
    from quantsmith.backends.onnx.pipeline import quantize
    from quantsmith.backends.onnx.stages import (
        QuantizationStage,
        TransformStage,
        VerifyStage,
    )
    from quantsmith.config import supported
    from quantsmith.pipeline import Pipeline
    from quantsmith.rules import ByName, Exclude, Rule, Rules

    model = deepcopy(scaled_model)
    model.graph.node[-1].CopyFrom(
        helper.make_node("Conv", ["a", "cw"], ["y"], name="consumer")
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.ones((2, 2, 1, 1), dtype=np.float32), "cw")
    )
    smoothing = Rules(
        default=Exclude(reason="only smooth the consumer"),
        overrides=(Rule(selector=ByName(pattern="consumer"), decision=SmoothQuant()),),
    )

    def samples() -> Iterator[Mapping[str, FloatArray]]:
        """Regenerate signed inputs for both calibration revisions."""
        yield {
            "x": np.random.default_rng(12).normal(size=(1, 2, 3, 3)).astype(np.float32)
        }

    result = expect_ok(
        quantize(
            model,
            samples,
            Pipeline(TransformStage(smoothing))
            .then(QuantizationStage(supported(StaticW8A8())))
            .then(VerifyStage()),
        )
    )
    assert not any(node.op_type == "Div" for node in result.model.graph.node)
    assert len(result.plans) == 2
    quantized = expect_ok(OnnxEvaluator(result.model).run(next(samples()), ("y",)))
    original = expect_ok(OnnxEvaluator(model).run(next(samples()), ("y",)))
    np.testing.assert_allclose(quantized["y"], original["y"], atol=0.1, rtol=0.02)


def test_chained_scaling(scaled_model: ModelProto) -> None:
    """Fold consecutive divisions while retaining an initializer graph output."""
    scaled_model.graph.node[-1].output[0] = "middle"
    scaled_model.graph.node.append(
        helper.make_node("Div", ["middle", "s"], ["y"], name="again")
    )
    scaled_model.graph.output.append(
        helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 2, 1, 1])
    )
    optimized = expect_ok(optimize(scaled_model))
    assert not any(node.op_type == "Div" for node in optimized.graph.node)
    assert "s" in {value.name for value in optimized.graph.initializer}
    assert_equivalent(scaled_model, optimized)


def test_invalid_model_is_result(scaled_model: ModelProto) -> None:
    """Malformed external graphs preserve the backend's expected failure contract."""
    from tests.outcomes import expect_error

    scaled_model.graph.node[-1].input[0] = "missing"
    expect_error(optimize(scaled_model), "missing")


def test_scope_preserves_existing_divisions(scaled_model: ModelProto) -> None:
    """A smoothing candidate cannot rewrite divisions outside its own output scope."""
    untouched = expect_ok(optimize(scaled_model, scope=frozenset({"new_smoothing"})))
    assert any(node.op_type == "Div" for node in untouched.graph.node)
    assert_equivalent(scaled_model, untouched)
