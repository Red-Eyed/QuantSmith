"""Verify transactional affine folding and unchanged fallback with real ORT."""

from collections.abc import Iterable, Mapping

import numpy as np
import onnx
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from quantsmith import Quantizer
from quantsmith.backends.onnx.folding import fold_only
from quantsmith.config import PercentileConfig, QuantizationConfig, SmoothQuantConfig
from quantsmith.domain import FloatArray
from quantsmith.pipeline import Folded, Unfoldable
from quantsmith.plan import QuantizationPlan, RescaleInput
from quantsmith.runtime import OnnxEvaluator, evaluate
from tests.outcomes import expect_ok


@pytest.fixture(params=["Mul", "MatMul", "Gemm", "Relu", "Sigmoid", "Shared", "Public"])
def affine_graph(request: pytest.FixtureRequest) -> ModelProto:
    """Build private affine paths plus nonlinear and branching barriers."""
    kind: str = request.param
    weight = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], dtype=np.float32
    )
    constants = [
        numpy_helper.from_array(weight, "w"),
        numpy_helper.from_array(np.array([2.0, 3.0, 4.0], dtype=np.float32), "gain"),
        numpy_helper.from_array(np.array([1.0, -2.0, 3.0], dtype=np.float32), "bias"),
    ]
    if kind == "Mul":
        nodes = [helper.make_node("Mul", ["x", "gain"], ["a"], name="affine")]
    elif kind == "Gemm":
        nodes = [helper.make_node("Gemm", ["x", "w", "bias"], ["a"], name="affine")]
    else:
        nodes = [helper.make_node("MatMul", ["x", "w"], ["a"], name="affine")]
    source = "a"
    if kind in {"Relu", "Sigmoid"}:
        nodes.append(helper.make_node(kind, ["a"], ["activated"], name="activation"))
        source = "activated"
    nodes.append(helper.make_node("MatMul", [source, "w"], ["y"], name="consumer"))
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])]
    if kind == "Shared":
        nodes.append(helper.make_node("Identity", ["a"], ["other"], name="sibling"))
        outputs.append(
            helper.make_tensor_value_info("other", TensorProto.FLOAT, [2, 3])
        )
    if kind == "Public":
        outputs.append(helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]))
    return onnx.shape_inference.infer_shapes(
        helper.make_model(
            helper.make_graph(
                nodes,
                kind,
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                outputs,
                constants,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=10,
        )
    )


def test_transactional_folding(affine_graph: ModelProto) -> None:
    """Fold private affine paths exactly and roll back every unsupported candidate."""
    original = affine_graph.SerializeToString()
    plan = QuantizationPlan(
        operations=(
            RescaleInput(
                node="consumer",
                scale=np.array([2.0, 3.0, 4.0], dtype=np.float32),
                activation_axis=1,
                weight_axis=0,
            ),
        )
    )
    folded, applied, outcomes = expect_ok(fold_only(affine_graph, plan))
    assert affine_graph.SerializeToString() == original
    assert not any(node.op_type == "Div" for node in folded.graph.node)
    if affine_graph.graph.name in {"Sigmoid", "Shared", "Public"}:
        assert isinstance(outcomes[0], Unfoldable)
        assert not applied.operations
        # Normalization may prune unused constants; compare effective parameters.
        before = {
            t.name: numpy_helper.to_array(t) for t in affine_graph.graph.initializer
        }
        for tensor in folded.graph.initializer:
            np.testing.assert_array_equal(
                numpy_helper.to_array(tensor), before[tensor.name]
            )
    else:
        assert outcomes == (Folded("consumer"),)
        assert applied.operations == plan.operations

    def samples() -> Iterable[Mapping[str, FloatArray]]:
        """Replay deterministic nontrivial positive and negative activations."""
        yield {"x": np.array([[-2.0, 1.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)}

    report = expect_ok(
        evaluate(
            OnnxEvaluator(affine_graph),
            OnnxEvaluator(folded),
            samples,
            tuple(v.name for v in folded.graph.output),
        )
    )
    assert report.maximum_absolute_error < 0.0001


@pytest.mark.parametrize("percentile", [False, True])
def test_unfoldable_matches_baseline(percentile: bool) -> None:
    """Graph-input smoothing falls through to byte-identical ordinary quantization."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")],
            "input",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
            [numpy_helper.from_array(np.eye(3, dtype=np.float32), "w")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    config = (
        QuantizationConfig(calibration=PercentileConfig())
        if percentile
        else QuantizationConfig()
    )
    data = ({"x": np.array([[1.0, 2.0, 3.0], [-2.0, -3.0, -4.0]], dtype=np.float32)},)
    baseline = expect_ok(Quantizer.from_config(config).quantize(model).calibrate(data))
    smoothed = expect_ok(
        Quantizer.from_config(
            QuantizationConfig(
                calibration=config.calibration, smoothing=SmoothQuantConfig()
            )
        )
        .quantize(model)
        .calibrate(data)
    )
    assert smoothed.model.SerializeToString() == baseline.model.SerializeToString()
    assert isinstance(smoothed.folding[0], Unfoldable)


def test_fold_only_pipeline(affine_graph: ModelProto) -> None:
    """Quantize the accepted revision without adding runtime smoothing operators."""
    data = ({"x": np.array([[1.0, 2.0, 3.0], [-2.0, -3.0, -4.0]], dtype=np.float32)},)
    result = expect_ok(
        Quantizer.from_config(QuantizationConfig(smoothing=SmoothQuantConfig()))
        .quantize(affine_graph)
        .calibrate(data)
    )
    assert len(result.plans) == 2
    assert len(result.folding) == (1 if affine_graph.graph.name == "Mul" else 2)
    assert not any(
        node.op_type in {"Div", "QLinearMul"} for node in result.model.graph.node
    )
    if affine_graph.graph.name != "Mul":
        assert isinstance(result.folding[0], Unfoldable)
    if affine_graph.graph.name in {"Mul", "MatMul", "Gemm", "Relu"}:
        assert isinstance(result.folding[-1], Folded)
    else:
        assert isinstance(result.folding[-1], Unfoldable)
    revision, _, _ = expect_ok(fold_only(affine_graph, result.plans[0]))
    recalibrated = expect_ok(
        Quantizer.from_config(QuantizationConfig()).quantize(revision).calibrate(data)
    )
    assert result.model.SerializeToString() == recalibrated.model.SerializeToString()
