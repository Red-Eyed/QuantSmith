"""Compact weight storage and numerical equivalence to runtime ONNX QDQ."""

from copy import deepcopy

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import AxisAttribute, lower
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.calibration import Samples
from quantsmith.config import supported
from quantsmith.domain import Encoding, Granularity, IntegerType, PerChannel, PerTensor
from quantsmith.plan import QuantizationPlan, QuantizeConstant, QuantizeInput
from quantsmith.runtime import OnnxEvaluator
from tests.conftest import OperatorCase
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def constant_model() -> ModelProto:
    """Expose ties and saturation without an operator obscuring rounding errors."""
    values = np.tile(
        np.array([-200, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 130, 200], np.float32),
        (2, 1),
    )
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["w"], ["y"], name="constant")],
            "rounding",
            [],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, values.shape)],
            [numpy_helper.from_array(values, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


@pytest.mark.parametrize("dtype", list(IntegerType))
@pytest.mark.parametrize(
    "granularity",
    [
        PerTensor(),
        PerChannel(axis=0),
        PerChannel(axis=-2),
        PerChannel(axis=1),
        PerChannel(axis=-1),
    ],
)
def test_stored_codes_match_runtime_quantization(
    constant_model: ModelProto, dtype: IntegerType, granularity: Granularity
) -> None:
    """Match ORT's FP32 division, ties-to-even rounding, zero points, and clipping."""
    shape: tuple[int, ...] = ()
    attributes: AxisAttribute = {}
    match granularity:
        case PerChannel(axis=axis):
            shape = (constant_model.graph.initializer[0].dims[axis],)
            attributes = {"axis": axis}
        case PerTensor():
            pass
    zero = 3 if dtype == IntegerType.INT8 else 129
    scale = np.full(shape, 0.5, dtype=np.float32)
    zero_point = np.full(shape, zero, dtype=dtype.value)
    if shape:
        scale *= np.exp2(np.arange(shape[0], dtype=np.float32) % 3)
        zero_point += np.arange(shape[0], dtype=zero_point.dtype)
    encoding = Encoding(
        scale=scale,
        zero_point=zero_point,
        granularity=granularity,
    )
    original = constant_model.SerializeToString()
    result = expect_ok(
        lower(
            constant_model,
            QuantizationPlan(
                operations=(QuantizeInput(node="constant", index=0, encoding=encoding),)
            ),
        )
    )
    reference = deepcopy(constant_model)
    reference.graph.initializer.extend(
        [
            numpy_helper.from_array(encoding.scale, "scale"),
            numpy_helper.from_array(encoding.zero_point, "zero"),
        ]
    )
    del reference.graph.node[:]
    reference.graph.node.extend(
        [
            helper.make_node(
                "QuantizeLinear", ["w", "scale", "zero"], ["q"], **attributes
            ),
            helper.make_node(
                "DequantizeLinear", ["q", "scale", "zero"], ["y"], **attributes
            ),
        ]
    )
    actual = expect_ok(OnnxEvaluator(result).run({}, ("y",)))["y"]
    expected = expect_ok(OnnxEvaluator(reference).run({}, ("y",)))["y"]
    np.testing.assert_array_equal(actual, expected)
    assert constant_model.SerializeToString() == original
    assert "w" not in {tensor.name for tensor in result.graph.initializer}
    assert not any(node.op_type == "QuantizeLinear" for node in result.graph.node)


@pytest.mark.parametrize("smoothing", [False, True])
def test_pipeline_prunes_replaced_weights(
    operator_case: OperatorCase, samples: Samples, smoothing: bool
) -> None:
    """Discard original and intermediate FP32 weights after the last consumer moves."""
    stages = (supported(StaticW8A8()),)
    if smoothing:
        stages = (supported(SmoothQuant()), *stages)
    result = expect_ok(quantize(operator_case.model, samples, stages))
    used = {name for node in result.model.graph.node for name in node.input}
    for tensor in result.model.graph.initializer:
        assert tensor.name in used
        assert tensor.name != "w"
        assert not tensor.name.endswith("_smooth_weight")
    assert any(t.data_type == TensorProto.INT8 for t in result.model.graph.initializer)


def test_initializer_graph_output_is_preserved(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """An initializer exported directly is live even after its node uses move."""
    model = deepcopy(operator_case.model)
    weight = model.graph.initializer[0]
    model.graph.output.append(
        helper.make_tensor_value_info("w", TensorProto.FLOAT, weight.dims)
    )
    result = expect_ok(quantize(model, samples, (supported(StaticW8A8()),)))
    retained = next(t for t in result.model.graph.initializer if t.name == "w")
    assert retained == weight


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_constant_is_rejected(
    constant_model: ModelProto, value: float
) -> None:
    """Direct plans cannot bypass finite-weight validation during constant folding."""
    constant_model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.full((2, 9), value, dtype=np.float32), "w")
    )
    encoding = Encoding(
        scale=np.array(1, dtype=np.float32),
        zero_point=np.array(0, dtype=np.int8),
        granularity=PerTensor(),
    )
    expect_error(
        lower(
            constant_model,
            QuantizationPlan(
                operations=(QuantizeInput(node="constant", index=0, encoding=encoding),)
            ),
        ),
        "finite",
    )


@pytest.fixture
def large_constant_model(constant_model: ModelProto) -> ModelProto:
    """Use enough weight data to distinguish compression from protobuf overhead."""
    model = deepcopy(constant_model)
    model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.ones((256, 256), dtype=np.float32), "w")
    )
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [256, 256])
    )
    return model


@pytest.mark.parametrize("reconstructed", [False, True])
def test_serialized_weight_compression(
    large_constant_model: ModelProto, reconstructed: bool
) -> None:
    """Ordinary and reconstructed constants both replace four-byte FP32 storage."""
    encoding = Encoding(
        scale=np.array(0.5, dtype=np.float32),
        zero_point=np.array(0, dtype=np.int8),
        granularity=PerTensor(),
    )
    operation: QuantizeInput | QuantizeConstant = QuantizeInput(
        node="constant", index=0, encoding=encoding
    )
    if reconstructed:
        operation = QuantizeConstant(
            node="constant",
            index=0,
            encoding=encoding,
            values=np.full((256, 256), 2, dtype=np.int8),
        )
    result = expect_ok(
        lower(large_constant_model, QuantizationPlan(operations=(operation,)))
    )
    assert (
        len(result.SerializeToString())
        < len(large_constant_model.SerializeToString()) * 0.3
    )
