"""Executable ORT CPU scale-axis probes independent of quantizer restrictions."""

from typing import Literal

import numpy as np
import onnxruntime as ort
import pytest
from numpy.typing import NDArray
from onnx import ModelProto, TensorProto, helper, numpy_helper
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    RuntimeException,
)
from pydantic import ConfigDict, TypeAdapter

_OUTPUTS = TypeAdapter(
    list[NDArray[np.generic]],
    config=ConfigDict(strict=True, arbitrary_types_allowed=True),
)
type Operator = Literal["QLinearConv", "QLinearMatMul"]


def kernel_model(
    op: Operator,
    activation_scale: NDArray[np.float32],
    weight_scale: NDArray[np.float32],
    output_scale: NDArray[np.float32],
    dynamic_weight: bool = False,
) -> ModelProto:
    """Build a direct integer operator with explicit scalar zero points."""
    weight = np.array([[1, 2, -1, 3], [2, -1, 1, 2], [1, 1, 2, -1]], dtype=np.int8)
    input_shape, output_shape = ([2, 3], [2, 4])
    if op == "QLinearConv":
        weight = np.einsum("co->oc", weight).reshape(4, 3, 1, 1)
        input_shape, output_shape = [1, 3, 1, 2], [1, 4, 1, 2]
    constants = [
        numpy_helper.from_array(value, name)
        for name, value in (
            ("as", activation_scale),
            ("az", np.array(0, dtype=np.uint8)),
            ("bs", weight_scale),
            ("bz", np.array(0, dtype=np.int8)),
            ("ys", output_scale),
            ("yz", np.array(0, dtype=np.uint8)),
        )
    ]
    inputs = [helper.make_tensor_value_info("x", TensorProto.UINT8, input_shape)]
    if dynamic_weight:
        inputs.append(
            helper.make_tensor_value_info("w", TensorProto.INT8, list(weight.shape))
        )
    else:
        constants.append(numpy_helper.from_array(weight, "w"))
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    op,
                    ["x", "as", "az", "w", "bs", "bz", "ys", "yz"],
                    ["y"],
                    name="integer",
                )
            ],
            "probe",
            inputs,
            [helper.make_tensor_value_info("y", TensorProto.UINT8, output_shape)],
            constants,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


def run(model: ModelProto, dynamic_weight: bool) -> NDArray[np.generic]:
    """Execute the direct kernel with optimization disabled and validate outputs."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    constants = {
        value.name: numpy_helper.to_array(value) for value in model.graph.initializer
    }
    shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    feed: dict[str, NDArray[np.generic]] = {
        "x": np.arange(1, 7, dtype=np.uint8).reshape(shape)
    }
    if dynamic_weight:
        weight = np.array([[1, 2, -1, 3], [2, -1, 1, 2], [1, 1, 2, -1]], dtype=np.int8)
        feed["w"] = (
            np.einsum("co->oc", weight).reshape(4, 3, 1, 1)
            if len(shape) == 4
            else weight
        )
    outputs = _OUTPUTS.validate_python(session.run(None, feed))
    assert len(outputs) == 1 and outputs[0].dtype == np.uint8
    assert "as" in constants
    return outputs[0]


@pytest.mark.parametrize("op", ["QLinearConv", "QLinearMatMul"])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("dynamic_weight", [False, True])
def test_scalar_activation_and_output_channel_weights(
    op: Operator, per_channel: bool, dynamic_weight: bool
) -> None:
    """Verify integer arithmetic for scalar activations and optional channel weights."""
    scale = np.array([0.5, 0.25, 0.125, 1.0] if per_channel else 0.5, dtype=np.float32)
    model = kernel_model(
        op,
        np.array(0.25, dtype=np.float32),
        scale,
        np.array(0.125, dtype=np.float32),
        dynamic_weight,
    )
    actual = run(model, dynamic_weight)
    weight = np.array([[1, 2, -1, 3], [2, -1, 1, 2], [1, 1, 2, -1]], dtype=np.float32)
    if op == "QLinearMatMul":
        product = np.arange(1, 7, dtype=np.float32).reshape(2, 3) @ weight
        expected = product * (0.25 * scale / 0.125)
    else:
        product = np.einsum(
            "nchw,co->nohw",
            np.arange(1, 7, dtype=np.float32).reshape(1, 3, 1, 2),
            weight,
        )
        expected = product * (0.25 * scale / 0.125).reshape(1, -1, 1, 1)
    np.testing.assert_array_equal(
        actual, np.clip(np.rint(expected), 0, 255).astype(np.uint8)
    )


@pytest.mark.parametrize("op", ["QLinearConv", "QLinearMatMul"])
@pytest.mark.parametrize("part", ["activation", "output"])
def test_vector_activation_scales_are_rejected(op: Operator, part: str) -> None:
    """Confirm multi-element activation scales fail in these direct CPU kernels."""
    activation = np.array(
        [0.25, 0.5, 1.0] if part == "activation" else 0.25, dtype=np.float32
    )
    output = np.array(
        [0.125, 0.25, 0.5, 1.0] if part == "output" else 0.125, dtype=np.float32
    )
    model = kernel_model(op, activation, np.full(4, 0.5, dtype=np.float32), output)
    with pytest.raises((Fail, InvalidArgument, RuntimeException), match="scale"):
        run(model, False)
