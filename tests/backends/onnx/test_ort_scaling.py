"""Runtime numerical and ownership contracts for default ORT integer execution."""

from collections.abc import Iterator, Mapping
from copy import deepcopy

import numpy as np
import onnxruntime as ort
import pytest
from numpy.typing import NDArray
from onnx import ModelProto, TensorProto, helper, numpy_helper
from pydantic import ConfigDict, TypeAdapter

from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe
from quantsmith.backends.onnx.ort_scaling import lower_integer_scaling
from quantsmith.backends.onnx.ort_scaling.boundaries import with_boundaries
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.config import supported
from quantsmith.domain import FloatArray, IntegerType
from quantsmith.plan import QuantizeInput
from quantsmith.rules import ByName, Exclude, Rule, Rules
from tests.outcomes import expect_error, expect_ok

_OUTPUTS = TypeAdapter(
    list[NDArray[np.generic]],
    config=ConfigDict(strict=True, arbitrary_types_allowed=True),
)


def execute(
    model: ModelProto, inputs: Mapping[str, NDArray[np.generic]]
) -> NDArray[np.uint8]:
    """Validate native ORT outputs without changing the dtype being tested."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    outputs = _OUTPUTS.validate_python(session.run(None, dict(inputs)))
    assert len(outputs) == 1
    output = outputs[0]
    if output.dtype != np.uint8:
        raise ValueError("expected UINT8 codes")
    return np.asarray(output, dtype=np.uint8)


@pytest.fixture
def integer_boundary() -> ModelProto:
    """Cover every UINT8 code in two channels with nonzero activation zero points."""
    constants = {
        "source_scale": np.array(0.05, dtype=np.float32),
        "source_zero": np.array(128, dtype=np.uint8),
        "target_scale": np.array(0.03, dtype=np.float32),
        "target_zero": np.array(97, dtype=np.uint8),
        "divisor": np.array([1, 2], dtype=np.float32).reshape(1, 2, 1, 1),
    }
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "DequantizeLinear",
                    ["x", "source_scale", "source_zero"],
                    ["decoded"],
                    name="decode",
                ),
                helper.make_node(
                    "Div", ["decoded", "divisor"], ["scaled"], name="smooth"
                ),
                helper.make_node(
                    "QuantizeLinear",
                    ["scaled", "target_scale", "target_zero"],
                    ["y"],
                    name="encode",
                ),
            ],
            "integer_scaling",
            [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 2, 1, 256])],
            [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 2, 1, 256])],
            [numpy_helper.from_array(value, name) for name, value in constants.items()],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


def test_integer_scaling_codes_and_ownership(integer_boundary: ModelProto) -> None:
    """Exercise rounding, signed real values and saturation across all input codes."""
    before = integer_boundary.SerializeToString()
    result = expect_ok(lower_integer_scaling(integer_boundary, frozenset({"scaled"})))
    assert integer_boundary.SerializeToString() == before
    assert [node.op_type for node in result.graph.node] == ["QLinearMul"]
    inputs = {"x": np.tile(np.arange(256, dtype=np.uint8), (1, 2, 1, 1))}
    actual = execute(result, inputs)
    reference = execute(integer_boundary, inputs)
    assert actual.dtype == np.uint8
    assert np.max(np.abs(actual.astype(np.int16) - reference.astype(np.int16))) <= 1
    assert np.any(actual == 0) and np.any(actual == 255)
    assert (
        expect_ok(
            lower_integer_scaling(result, frozenset({"scaled"}))
        ).SerializeToString()
        == result.SerializeToString()
    )


@pytest.mark.parametrize(
    "reason",
    [
        "negative",
        "zero",
        "nonfinite",
        "tiny",
        "extreme_range",
        "signed_encoding",
        "exposed",
        "shared",
        "unselected",
        "custom",
    ],
)
def test_unsupported_scaling_stays_float(
    integer_boundary: ModelProto, reason: str
) -> None:
    """Decline unsupported coefficients, exposed divisions, and foreign operators."""
    model = deepcopy(integer_boundary)
    if reason in {"negative", "zero", "nonfinite", "tiny", "extreme_range"}:
        values = {
            "negative": [-1, 2],
            "zero": [0, 2],
            "nonfinite": [np.inf, 2],
            "tiny": [np.finfo(np.float32).tiny / 1024, 2],
            "extreme_range": [1, 10000],
        }
        model.graph.initializer[-1].CopyFrom(
            numpy_helper.from_array(
                np.array(values[reason], dtype=np.float32).reshape(1, 2, 1, 1),
                "divisor",
            )
        )
    elif reason == "signed_encoding":
        model.graph.initializer[3].CopyFrom(
            numpy_helper.from_array(np.array(0, dtype=np.int8), "target_zero")
        )
        model.graph.output[0].type.tensor_type.elem_type = TensorProto.INT8
    elif reason == "exposed":
        model.graph.output.append(
            helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [1, 2, 1, 256])
        )
    elif reason == "shared":
        model.graph.node.append(
            helper.make_node("Identity", ["scaled"], ["other"], name="sibling")
        )
    elif reason == "custom":
        model.graph.node[1].domain = "custom"
        model.opset_import.append(helper.make_opsetid("custom", 1))
    selected = frozenset() if reason == "unselected" else frozenset({"scaled"})
    result = expect_ok(lower_integer_scaling(model, selected))
    assert any(node.op_type == "Div" for node in result.graph.node)
    assert not any(node.domain == "com.microsoft" for node in result.graph.node)


def test_shared_decoded_value_is_preserved(integer_boundary: ModelProto) -> None:
    """Keep the original dequantizer when another output still needs it."""
    integer_boundary.graph.output.append(
        helper.make_tensor_value_info("decoded", TensorProto.FLOAT, [1, 2, 1, 256])
    )
    result = expect_ok(lower_integer_scaling(integer_boundary, frozenset({"scaled"})))
    assert [node.op_type for node in result.graph.node] == [
        "DequantizeLinear",
        "QLinearMul",
    ]


@pytest.fixture
def clipped_boundary() -> ModelProto:
    """Build a biased, signed convolution with quantized channel weights and ReLU6."""
    constants = {
        "w": np.array([2, 0, 0, -4], dtype=np.int8).reshape(2, 2, 1, 1),
        "ws": np.array([0.25, 0.5], dtype=np.float32),
        "wz": np.zeros(2, dtype=np.int8),
        "bias": np.array([1, -1], dtype=np.float32),
        "low": np.array(0, dtype=np.float32),
        "high": np.array(6, dtype=np.float32),
        "source_scale": np.array(6 / 255, dtype=np.float32),
        "source_zero": np.array(0, dtype=np.uint8),
        "target_scale": np.array(0.05, dtype=np.float32),
        "target_zero": np.array(0, dtype=np.uint8),
        "divisor": np.array([2, 0.5], dtype=np.float32).reshape(1, 2, 1, 1),
    }
    nodes = [
        helper.make_node(
            "DequantizeLinear", ["w", "ws", "wz"], ["weight"], name="weight_dq", axis=0
        ),
        helper.make_node("Conv", ["x", "weight", "bias"], ["projected"], name="conv"),
        helper.make_node(
            "Clip", ["projected", "low", "high"], ["clipped"], name="clip"
        ),
        helper.make_node(
            "QuantizeLinear",
            ["clipped", "source_scale", "source_zero"],
            ["codes"],
            name="source_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["codes", "source_scale", "source_zero"],
            ["decoded"],
            name="source_dq",
        ),
        helper.make_node("Div", ["decoded", "divisor"], ["scaled"], name="smooth"),
        helper.make_node(
            "QuantizeLinear",
            ["scaled", "target_scale", "target_zero"],
            ["y"],
            name="target_q",
        ),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "clipped",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 2, 2, 3])],
            [numpy_helper.from_array(value, name) for name, value in constants.items()],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


def test_integer_clip_matches_scaled_float_reference(
    clipped_boundary: ModelProto,
) -> None:
    """Clamp in the scaled domain, including negatives and values beyond ReLU6."""
    before = clipped_boundary.SerializeToString()
    result = expect_ok(lower_integer_scaling(clipped_boundary, frozenset({"scaled"})))
    assert clipped_boundary.SerializeToString() == before
    assert [node.op_type for node in result.graph.node] == [
        "DequantizeLinear",
        "Conv",
        "QuantizeLinear",
        "Min",
    ]
    values = np.array([-100, -5, 0, 2, 6, 100], dtype=np.float32)
    inputs = np.tile(values, 2).reshape(1, 2, 2, 3)
    weights = np.array([0.5, -2], dtype=np.float32).reshape(1, 2, 1, 1)
    bias = np.array([1, -1], dtype=np.float32).reshape(1, 2, 1, 1)
    divisor = np.array([2, 0.5], dtype=np.float32).reshape(1, 2, 1, 1)
    reference = np.clip(
        np.rint(np.clip(inputs * weights + bias, 0, 6) / divisor / 0.05), 0, 255
    ).astype(np.uint8)
    actual = execute(result, {"x": inputs})
    np.testing.assert_allclose(actual, reference, atol=1, rtol=0)


@pytest.mark.parametrize(
    "reason",
    [
        "shared_conv",
        "shared_weight",
        "graph_output",
        "nonzero_lower",
        "dynamic_bias",
        "wrong_axis",
    ],
)
def test_clip_fusion_preserves_unsupported_paths(
    clipped_boundary: ModelProto, reason: str
) -> None:
    """Fall back to integer multiplication without altering a protected producer."""
    model = deepcopy(clipped_boundary)
    if reason == "shared_conv":
        model.graph.node.append(
            helper.make_node("Identity", ["projected"], ["other"], name="sibling")
        )
    elif reason == "shared_weight":
        model.graph.node.append(
            helper.make_node("Identity", ["weight"], ["other"], name="sibling")
        )
    elif reason == "graph_output":
        model.graph.output.append(
            helper.make_tensor_value_info("clipped", TensorProto.FLOAT, [1, 2, 2, 3])
        )
    elif reason == "nonzero_lower":
        model.graph.initializer[4].CopyFrom(
            numpy_helper.from_array(np.array(-1, dtype=np.float32), "low")
        )
    elif reason == "dynamic_bias":
        del model.graph.initializer[3]
        model.graph.input.append(
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [2])
        )
    else:
        model.graph.node[0].attribute[0].i = 1
    result = expect_ok(lower_integer_scaling(model, frozenset({"scaled"})))
    assert any(node.op_type == "Clip" for node in result.graph.node)
    assert not any(node.op_type == "Min" for node in result.graph.node)


def test_explicit_exclusions_remain_authoritative(clipped_boundary: ModelProto) -> None:
    """An explicit exclusion prevents boundary quantization from claiming an edge."""
    graph = expect_ok(describe(clipped_boundary))
    rules = Rules(
        default=StaticW8A8(),
        overrides=(
            Rule(
                selector=ByName(pattern="smooth"), decision=Exclude(reason="keep float")
            ),
        ),
    )
    updated = with_boundaries(clipped_boundary, graph, rules, frozenset({"scaled"}))
    assert updated is rules


def test_malformed_graph_is_expected_failure(integer_boundary: ModelProto) -> None:
    """Preserve backend Result semantics for invalid external graph data."""
    integer_boundary.graph.node[1].input[0] = "missing"
    expect_error(
        lower_integer_scaling(integer_boundary, frozenset({"scaled"})), "missing"
    )


@pytest.fixture
def residual_model() -> ModelProto:
    """Give smoothing both a bounded activation and an input-sharing residual path."""
    constants = {
        "w": np.array([0.5, 0.1, -0.2, 0.75], dtype=np.float32).reshape(2, 2, 1, 1),
        "bias": np.array([0.3, 0.6], dtype=np.float32),
        "low": np.array(0, dtype=np.float32),
        "high": np.array(6, dtype=np.float32),
    }
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["x", "w", "bias"], ["a"], name="producer"),
                helper.make_node("Clip", ["a", "low", "high"], ["b"], name="clip"),
                helper.make_node("Conv", ["b", "w", "bias"], ["c"], name="middle"),
                helper.make_node("Add", ["c", "x"], ["d"], name="residual"),
                helper.make_node("Conv", ["d", "w", "bias"], ["y"], name="consumer"),
            ],
            "residual",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 3])],
            [numpy_helper.from_array(value, name) for name, value in constants.items()],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


def replay_inputs() -> Iterator[Mapping[str, FloatArray]]:
    """Replay three signed calibration batches with modest channel outliers."""
    rng = np.random.default_rng(31)
    for _ in range(3):
        yield {"x": rng.normal(size=(1, 2, 3, 3)).astype(np.float32)}


def test_pipeline_adds_calibrated_boundaries(residual_model: ModelProto) -> None:
    """Exercise smoothing, fresh statistics, clipping fusion and residual planning."""
    from quantsmith.runtime import OnnxEvaluator

    before = residual_model.SerializeToString()
    smoothing = Rules(
        default=Exclude(reason="only smooth consumers"),
        overrides=tuple(
            Rule(selector=ByName(pattern=name), decision=SmoothQuant())
            for name in ("middle", "consumer")
        ),
    )
    stages = (smoothing, supported(StaticW8A8()))
    deployed = expect_ok(quantize(residual_model, replay_inputs, stages))
    assert residual_model.SerializeToString() == before
    assert any(node.op_type == "QLinearMul" for node in deployed.model.graph.node)
    assert any(node.op_type == "Min" for node in deployed.model.graph.node)
    assert not any(node.op_type == "Div" for node in deployed.model.graph.node)
    from examples._shared.plans import coverage

    scope = coverage(expect_ok(describe(residual_model)), deployed.plans)
    assert set(scope.quantized_nodes) == {"producer", "middle", "consumer"}
    assert len(scope.quantized_nodes) == scope.eligible_nodes
    boundaries = {
        (operation.node, operation.index)
        for operation in deployed.plans[1].operations
        if isinstance(operation, QuantizeInput)
    }
    assert {("residual", 0), ("residual", 1)} <= boundaries
    for inputs in replay_inputs():
        expected = expect_ok(OnnxEvaluator(residual_model).run(inputs, ("y",)))["y"]
        actual = expect_ok(OnnxEvaluator(deployed.model).run(inputs, ("y",)))["y"]
        np.testing.assert_allclose(actual, expected, atol=0.08, rtol=0.05)


@pytest.mark.parametrize("dtype", [IntegerType.UINT8, IntegerType.INT8])
def test_static_only_has_no_scaling_boundaries(
    residual_model: ModelProto, dtype: IntegerType
) -> None:
    """Static-only stages do not acquire smoothing or residual boundary operations."""
    stages = (supported(StaticW8A8(activation_type=dtype)),)
    result = expect_ok(quantize(residual_model, replay_inputs, stages))
    assert not any(node.op_type == "QLinearMul" for node in result.model.graph.node)
    assert {operation.node for operation in result.plans[0].operations} == {
        "producer",
        "middle",
        "consumer",
    }


def test_residual_exclusion_survives_boundary_extension(
    residual_model: ModelProto,
) -> None:
    """Extend real smoothed edges while retaining an explicitly floating residual."""
    smoothing = Rules(
        default=Exclude(reason="only smooth last consumer"),
        overrides=(Rule(selector=ByName(pattern="consumer"), decision=SmoothQuant()),),
    )
    model = expect_ok(quantize(residual_model, replay_inputs, (smoothing,))).model
    graph = expect_ok(describe(model))
    original = supported(StaticW8A8())
    rules = Rules(
        default=original.default,
        overrides=original.overrides
        + (
            Rule(
                selector=ByName(pattern="residual"),
                decision=Exclude(reason="keep residual float"),
            ),
        ),
    )
    divisions = frozenset(node.outputs[0] for node in graph.nodes if node.op == "Div")
    updated = with_boundaries(model, graph, rules, divisions)
    assert updated is not rules
    residual = next(node for node in graph.nodes if node.name == "residual")
    assert isinstance(updated.resolve(residual, graph), Exclude)


def test_signed_smoothquant_retains_float_scaling(residual_model: ModelProto) -> None:
    """Signed activation stages retain their floating-point scaling paths."""
    stages = (
        supported(SmoothQuant()),
        supported(StaticW8A8(activation_type=IntegerType.INT8)),
    )
    result = expect_ok(quantize(residual_model, replay_inputs, stages))
    assert not any(node.op_type == "QLinearMul" for node in result.model.graph.node)
    assert any(node.op_type == "Div" for node in result.model.graph.node)
    assert {operation.node for operation in result.plans[1].operations} == {
        "producer",
        "middle",
        "consumer",
    }


def test_boundary_missing_statistics_is_expected_failure(
    residual_model: ModelProto,
) -> None:
    """Boundary planning reports missing collected ranges through Result."""
    from quantsmith.algorithms.contracts import Statistics
    from quantsmith.backends.onnx.ort_scaling.boundaries import BoundaryQuantizer

    graph = expect_ok(describe(residual_model))
    node = next(node for node in graph.nodes if node.name == "residual")
    algorithm = BoundaryQuantizer()
    needs = expect_ok(algorithm.requirements(node, graph))
    assert {request.tensor for request in needs.ranges} == {"c", "x"}
    expect_error(
        algorithm.plan(node, graph, Statistics(ranges={}, histograms={})),
        "statistics were not collected",
    )


@pytest.mark.parametrize("kind", ["integer", "clip"])
def test_rewrite_planning_is_pure(
    integer_boundary: ModelProto, clipped_boundary: ModelProto, kind: str
) -> None:
    """Repeated planning is deterministic and leaves every input protobuf intact."""
    from quantsmith.backends.onnx.ort_scaling import plan_boundary
    from quantsmith.backends.onnx.ort_scaling.graph import (
        RewritePatch,
        ScalingBoundary,
        apply_patches,
        index_graph,
        scaling_boundary,
    )
    from quantsmith.domain import Absent

    model = integer_boundary if kind == "integer" else clipped_boundary
    before = model.SerializeToString()
    index = index_graph(model)
    division = next(node for node in model.graph.node if node.op_type == "Div")
    match scaling_boundary(index, division):
        case Absent(reason=reason):
            pytest.fail(reason)
        case ScalingBoundary() as boundary:
            pass
    match plan_boundary(index, boundary):
        case Absent(reason=reason):
            pytest.fail(reason)
        case RewritePatch() as first:
            assert first == plan_boundary(index, boundary)
    assert model.SerializeToString() == before
    revised = index.reserve(first)
    assert revised is not index
    assert revised.producers is index.producers
    assert revised.names > index.names
    match plan_boundary(revised, boundary):
        case Absent(reason=reason):
            pytest.fail(reason)
        case RewritePatch() as second:
            assert {value.name for value in first.constants}.isdisjoint(
                value.name for value in second.constants
            )
    applied = apply_patches(deepcopy(model), (first,))
    expected = expect_ok(lower_integer_scaling(model, frozenset({"scaled"})))
    assert model.SerializeToString() == before
    assert applied.SerializeToString() == expected.SerializeToString()
