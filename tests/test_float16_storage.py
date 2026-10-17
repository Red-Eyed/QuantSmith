"""Exercise compact storage, caller ownership and actual ORT constant folding."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper
from pydantic import ValidationError

from quantsmith import Quantizer, Replay
from quantsmith.backends.onnx.float16_storage import store_float16
from quantsmith.config import Float16StorageConfig
from quantsmith.domain import Absent, InputArray
from quantsmith.runtime import OnnxEvaluator
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def storage_model() -> ModelProto:
    """Use a constant large enough to dominate serialized graph metadata."""
    values = np.linspace(-2, 2, 1024, dtype=np.float32).reshape(32, 32)
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")],
            "storage",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
            [numpy_helper.from_array(values, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )


def test_storage_owns_graph_and_needs_no_samples(storage_model: ModelProto) -> None:
    """Shrink storage without touching caller tensors or requesting calibration."""
    original = storage_model.SerializeToString()

    def unread() -> Iterable[Mapping[str, InputArray]]:
        """Fail if a storage-only pipeline asks for input data."""
        pytest.fail("storage transform read calibration inputs")
        yield {}

    result = expect_ok(
        Quantizer.from_config(Float16StorageConfig())
        .quantize(storage_model)
        .calibrate(Replay(unread))
    )
    assert storage_model.SerializeToString() == original
    assert len(result.model.SerializeToString()) < len(original) * 0.65
    assert result.model.graph.initializer[0].data_type == TensorProto.FLOAT16
    assert result.plans == ()
    assert isinstance(result.execution, Absent)
    onnx.checker.check_model(result.model)
    storage_model.graph.ClearField("initializer")
    assert len(result.model.graph.initializer) == 1


def test_ort_folds_storage_casts(storage_model: ModelProto, tmp_path: Path) -> None:
    """Execute rounded weights in FP32 with no additional optimized operators."""
    result = expect_ok(
        Quantizer.from_config(Float16StorageConfig())
        .quantize(storage_model)
        .calibrate([])
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    optimized = tmp_path / "optimized.onnx"
    options.optimized_model_filepath = str(optimized)
    ort.InferenceSession(
        result.model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    compiled = onnx.load(optimized)
    assert [node.op_type for node in compiled.graph.node] == ["MatMul"]
    assert all(
        value.data_type == TensorProto.FLOAT for value in compiled.graph.initializer
    )
    values = np.ones((1, 32), dtype=np.float32)
    actual = expect_ok(OnnxEvaluator(result.model).run({"x": values}, ("y",)))["y"]
    weights = numpy_helper.to_array(storage_model.graph.initializer[0])
    expected = values @ weights.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 65505.0])
def test_unsafe_constants_remain_unchanged(
    storage_model: ModelProto, value: float
) -> None:
    """Preserve nonfinite masks and constants that would overflow FP16."""
    storage_model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.full((32, 32), value, dtype=np.float32), "w")
    )
    result = expect_ok(
        Quantizer.from_config(Float16StorageConfig())
        .quantize(storage_model)
        .calibrate([])
    )
    assert result.model.graph.initializer[0] == storage_model.graph.initializer[0]
    assert len(result.model.graph.node) == 1


def test_threshold_preserves_small_constants(storage_model: ModelProto) -> None:
    """Honor the declared size threshold without inserting unused conversions."""
    result = expect_ok(
        Quantizer.from_config(Float16StorageConfig(minimum_elements=1025))
        .quantize(storage_model)
        .calibrate([])
    )
    assert result.model.graph.initializer[0] == storage_model.graph.initializer[0]
    assert len(result.model.graph.node) == 1


def test_overridable_weight_remains_fp32(storage_model: ModelProto) -> None:
    """Keep initializer defaults that are also runtime graph inputs overridable."""
    storage_model.graph.input.append(
        helper.make_tensor_value_info("w", TensorProto.FLOAT, [32, 32])
    )
    result = store_float16(storage_model, 256)
    assert result.graph.initializer[0] == storage_model.graph.initializer[0]
    assert len(result.graph.node) == 1
    expect_error(
        Quantizer.from_config(Float16StorageConfig())
        .quantize(storage_model)
        .calibrate([]),
        "overridable",
    )


def test_graph_output_and_name_collision(storage_model: ModelProto) -> None:
    """Restore exported initializer names and allocate unique storage identifiers."""
    storage_model.graph.output.append(
        helper.make_tensor_value_info("w", TensorProto.FLOAT, [32, 32])
    )
    storage_model.graph.initializer.append(
        numpy_helper.from_array(np.array(1, dtype=np.int64), "w_storage_fp16")
    )
    result = expect_ok(
        Quantizer.from_config(Float16StorageConfig())
        .quantize(storage_model)
        .calibrate([])
    )
    onnx.checker.check_model(result.model)
    assert result.model.graph.initializer[0].name == "w_storage_fp16_"
    actual = expect_ok(
        OnnxEvaluator(result.model).run({"x": np.ones((1, 32), np.float32)}, ("w",))
    )["w"]
    expected = numpy_helper.to_array(storage_model.graph.initializer[0])
    np.testing.assert_array_equal(
        actual, expected.astype(np.float16).astype(np.float32)
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"minimum_elements":0}',
        '{"minimum_elements":true}',
        '{"activation_type":"uint8"}',
    ],
)
def test_storage_config_rejects_invalid_options(payload: str) -> None:
    """Keep storage configuration strict and separate from integer quantization."""
    with pytest.raises(ValidationError):
        Float16StorageConfig.model_validate_json(payload)


def test_storage_config_roundtrip() -> None:
    """Preserve the explicit mode and threshold through JSON configuration."""
    config = Float16StorageConfig(minimum_elements=512)
    assert Float16StorageConfig.model_validate_json(config.model_dump_json()) == config


def test_storage_reports_invalid_graph(storage_model: ModelProto) -> None:
    """Preserve facade Result semantics for malformed models."""
    storage_model.graph.node[0].input[0] = "missing"
    expect_error(
        Quantizer.from_config(Float16StorageConfig())
        .quantize(storage_model)
        .calibrate([]),
        "missing",
    )
