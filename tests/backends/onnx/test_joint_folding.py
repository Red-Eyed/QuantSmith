"""Execute grouped smoothing with shared weights, barriers, and fresh calibration."""

from collections.abc import Iterable, Mapping

import numpy as np
import onnx
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from quantsmith import Quantizer
from quantsmith.backends.onnx.group_folding import fold_groups
from quantsmith.backends.onnx.quantizer_reuse import reuse_quantizers
from quantsmith.config import JointSmoothQuantConfig, QuantizationConfig
from quantsmith.domain import FloatArray
from quantsmith.pipeline import Folded, Unfoldable
from quantsmith.plan import QuantizationPlan, RescaleInput
from quantsmith.runtime import OnnxEvaluator, evaluate
from tests.outcomes import expect_ok


@pytest.fixture(params=["Mul", "MatMul", "Gemm", "Relu", "Sigmoid", "Shared", "Public"])
def grouped_graph(request: pytest.FixtureRequest) -> ModelProto:
    """Feed two projections from a shared affine value with optional barriers."""
    kind: str = request.param
    parameters = [
        numpy_helper.from_array(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "w"
        ),
        numpy_helper.from_array(np.array([2.0, -3.0], dtype=np.float32), "gain"),
        numpy_helper.from_array(np.array([0.25, -0.75], dtype=np.float32), "bias"),
    ]
    if kind == "Mul":
        nodes = [helper.make_node("Mul", ["x", "gain"], ["a"], name="producer")]
    elif kind == "Gemm":
        nodes = [helper.make_node("Gemm", ["x", "w", "bias"], ["a"], name="producer")]
    else:
        nodes = [helper.make_node("MatMul", ["x", "w"], ["a"], name="producer")]
    source = "a"
    if kind in {"Relu", "Sigmoid"}:
        nodes.append(helper.make_node(kind, ["a"], ["h"], name="activation"))
        source = "h"
    nodes.extend(
        helper.make_node("MatMul", [source, "w"], [name], name=name)
        for name in ("q", "k")
    )
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 2])
        for name in ("q", "k")
    ]
    if kind == "Shared":
        nodes.append(helper.make_node("Identity", [source], ["extra"], name="excluded"))
        outputs.append(
            helper.make_tensor_value_info("extra", TensorProto.FLOAT, [2, 2])
        )
    if kind == "Public":
        outputs.append(helper.make_tensor_value_info(source, TensorProto.FLOAT, [2, 2]))
    return onnx.shape_inference.infer_shapes(
        helper.make_model(
            helper.make_graph(
                nodes,
                kind,
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
                outputs,
                parameters,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=10,
        )
    )


@pytest.fixture
def group_samples() -> tuple[Mapping[str, FloatArray], ...]:
    """Retain only tiny test inputs; production calibration still streams batches."""
    return ({"x": np.array([[2.0, -1.0], [-3.0, 4.0]], dtype=np.float32)},)


def test_group_transaction(
    grouped_graph: ModelProto, group_samples: tuple[Mapping[str, FloatArray], ...]
) -> None:
    """Fold siblings once while preserving outputs and caller-owned parameters."""
    before = grouped_graph.SerializeToString()
    plan = QuantizationPlan(
        operations=tuple(
            RescaleInput(
                node=name,
                scale=np.array([2.0, 3.0], dtype=np.float32),
                activation_axis=-1,
                weight_axis=0,
            )
            for name in ("q", "k")
        )
    )
    folded, applied, outcomes = expect_ok(fold_groups(grouped_graph, plan))
    assert grouped_graph.SerializeToString() == before
    assert not any(node.op_type == "Div" for node in folded.graph.node)
    if grouped_graph.graph.name in {"Sigmoid", "Shared", "Public"}:
        assert not applied.operations
        assert all(isinstance(value, Unfoldable) for value in outcomes)
    else:
        assert len(applied.operations) == 2
        assert outcomes == (Folded("q"), Folded("k"))
        consumers = [node for node in folded.graph.node if node.name in {"q", "k"}]
        assert consumers[0].input[0] == consumers[1].input[0]

    def samples() -> Iterable[Mapping[str, FloatArray]]:
        """Replay the numerical fixture without caching runtime activations."""
        yield from group_samples

    report = expect_ok(
        evaluate(
            OnnxEvaluator(grouped_graph),
            OnnxEvaluator(folded),
            samples,
            tuple(value.name for value in folded.graph.output),
        )
    )
    assert report.maximum_absolute_error < 1e-4


@pytest.mark.parametrize("protection", ["exclude", "scale", "axis"])
def test_partial_group_rolls_back(grouped_graph: ModelProto, protection: str) -> None:
    """An excluded sibling or conflicting policy prevents changing either sibling."""
    first = RescaleInput(
        node="q",
        scale=np.array([2.0, 3.0], dtype=np.float32),
        activation_axis=-1,
        weight_axis=0,
    )
    second = RescaleInput(
        node="k",
        scale=np.array([2.0, 4.0 if protection == "scale" else 3.0], dtype=np.float32),
        activation_axis=1 if protection == "axis" else -1,
        weight_axis=0,
    )
    plan = QuantizationPlan(
        operations=(first,) if protection == "exclude" else (first, second)
    )
    folded, applied, outcomes = expect_ok(fold_groups(grouped_graph, plan))
    assert not applied.operations
    assert all(isinstance(value, Unfoldable) for value in outcomes)
    original = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in grouped_graph.graph.initializer
    }
    for tensor in folded.graph.initializer:
        np.testing.assert_array_equal(
            numpy_helper.to_array(tensor), original[tensor.name]
        )


def test_pipeline_recalibrates(
    grouped_graph: ModelProto, group_samples: tuple[Mapping[str, FloatArray], ...]
) -> None:
    """The compiled facade matches ordinary quantization on the folded revision."""
    result = expect_ok(
        Quantizer.from_config(QuantizationConfig(smoothing=JointSmoothQuantConfig()))
        .quantize(grouped_graph)
        .calibrate(group_samples)
    )
    folded, _, _ = expect_ok(fold_groups(grouped_graph, result.plans[0]))
    baseline = expect_ok(
        Quantizer.from_config(QuantizationConfig())
        .quantize(folded)
        .calibrate(group_samples)
    )
    assert (
        result.model.SerializeToString()
        == expect_ok(reuse_quantizers(baseline.model)).SerializeToString()
    )
    assert not any(
        node.op_type in {"Div", "QLinearMul"} for node in result.model.graph.node
    )


@pytest.mark.parametrize("observe_internal", [False, True])
def test_output_selection_protects_coordinates(
    grouped_graph: ModelProto,
    group_samples: tuple[Mapping[str, FloatArray], ...],
    observe_internal: bool,
) -> None:
    """Protect intermediate scoring cuts without changing the deployed interface."""
    from quantsmith.config import OutputSelectedSmoothingConfig
    from quantsmith.domain import Absent
    from quantsmith.optimization.smoothing import ModelOutputs, ObservedTensors

    observations = ObservedTensors(names=("a",)) if observe_internal else ModelOutputs()
    config = QuantizationConfig(
        smoothing=OutputSelectedSmoothingConfig(
            candidates=(JointSmoothQuantConfig(alpha=0.5),), observations=observations
        )
    )
    before = grouped_graph.SerializeToString()
    result = expect_ok(
        Quantizer.from_config(config).quantize(grouped_graph).calibrate(group_samples)
    )
    assert grouped_graph.SerializeToString() == before
    assert tuple(output.name for output in result.model.graph.output) == tuple(
        output.name for output in grouped_graph.graph.output
    )
    assert not isinstance(result.selection, Absent)
    report = result.selection
    assert report.outputs == (
        ("a",)
        if observe_internal
        else tuple(output.name for output in grouped_graph.graph.output)
    )
    assert report.candidates[0].candidate == "baseline"
    if observe_internal:
        assert not report.candidates[1].folded_nodes
    elif grouped_graph.graph.name in {"Mul", "MatMul", "Gemm", "Relu"}:
        assert set(report.candidates[1].folded_nodes) == {"q", "k"}
    assert min(score.mean_squared_error for score in report.candidates) == next(
        score.mean_squared_error
        for score in report.candidates
        if score.candidate == report.selected
    )
