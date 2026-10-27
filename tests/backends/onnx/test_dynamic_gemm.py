"""Native Gemm dynamic quantization preserves layout, bias and excluded consumers."""

from itertools import product

import numpy as np
import onnxruntime as ort
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from examples._shared.plans import operation_record
from quantsmith import Quantizer
from quantsmith.algorithms.dynamic import DynamicW8A8
from quantsmith.backends.onnx.dynamic import DynamicQuantizationStage
from quantsmith.backends.onnx.stages import VerifyStage
from quantsmith.domain import Absent
from quantsmith.pipeline import Pipeline
from quantsmith.plan import DynamicQuantizeGemm
from quantsmith.rules import ByName, Exclude, Rule, Rules
from tests.outcomes import expect_ok


def build_gemm_model(trans_a: bool, trans_b: bool, bias_kind: str) -> ModelProto:
    """Share source weights with an excluded floating Gemm cache sibling."""
    weight = np.array(
        [[0.25, -2, 0.003, 0], [1.25, 0.5, -0.001, 0], [-0.75, 1, 0.007, 0]],
        dtype=np.float32,
    )
    initializers = [
        numpy_helper.from_array(weight.T.copy() if trans_b else weight, "w")
    ]
    shape: list[int | str] = [3, "rows"] if trans_a else ["rows", 3]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)]
    arguments = ["x", "w"]
    match bias_kind:
        case "none":
            pass
        case "runtime":
            arguments.append("bias")
            inputs.append(
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, ["rows", 4])
            )
        case _:
            arguments.append("bias")
            bias = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
            match bias_kind:
                case "row":
                    bias = bias.reshape(1, 4)
                case "matrix":
                    bias = np.tile(bias, (2, 1))
                case "scalar":
                    bias = np.asarray(0.5, dtype=np.float32)
            initializers.append(numpy_helper.from_array(bias, "bias"))
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Gemm",
                    arguments,
                    ["y"],
                    name="projection",
                    transA=int(trans_a),
                    transB=int(trans_b),
                ),
                helper.make_node(
                    "Gemm",
                    arguments,
                    ["cache"],
                    name="cache",
                    transA=int(trans_a),
                    transB=int(trans_b),
                ),
            ],
            "gemm",
            inputs,
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, ["rows", 4]),
                helper.make_tensor_value_info("cache", TensorProto.FLOAT, ["rows", 4]),
            ],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )


@pytest.mark.parametrize(
    ("trans_a", "trans_b", "bias_kind"),
    list(
        product(
            [False, True],
            [False, True],
            ["none", "vector", "row", "matrix", "scalar", "runtime"],
        )
    ),
)
def test_gemm_dynamic_layout_and_bias(
    trans_a: bool, trans_b: bool, bias_kind: str
) -> None:
    """Audit the full graph and compare multiple runtime ranges with FP32."""
    model = build_gemm_model(trans_a, trans_b, bias_kind)
    original = model.SerializeToString()
    rules = Rules(
        default=Exclude(reason="cache stays floating"),
        overrides=(
            Rule(selector=ByName(pattern="projection"), decision=DynamicW8A8()),
        ),
    )
    program = Pipeline(DynamicQuantizationStage(rules)).then(VerifyStage())
    result = expect_ok(Quantizer.from_pipeline(program).quantize(model).execute())
    assert model.SerializeToString() == original
    assert not isinstance(result.execution, Absent)
    (integer,) = result.execution.integer_nodes
    assert integer.node == "projection"
    assert integer.source_operator == "Gemm"
    (operation,) = result.plans[0].operations
    assert isinstance(operation, DynamicQuantizeGemm)
    assert operation_record(operation).get("target") == "dynamic_gemm"
    assert (
        result.model.graph.node[-1].SerializeToString()
        == model.graph.node[-1].SerializeToString()
    )
    retained = next(
        value for value in result.model.graph.initializer if value.name == "w"
    )
    assert (
        retained.SerializeToString() == model.graph.initializer[0].SerializeToString()
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(
            value.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        for value in (model, result.model)
    ]
    rng = np.random.default_rng(1)
    for rows in (2,) if bias_kind == "matrix" else (1, 5):
        for magnitude in (0.001, 1, 1000):
            activation = rng.normal(size=(rows, 3)).astype(np.float32) * magnitude
            feed = {"x": activation.T.copy() if trans_a else activation}
            if bias_kind == "runtime":
                feed["bias"] = rng.normal(size=(rows, 4)).astype(np.float32)
            expected, cache = sessions[0].run(None, feed)
            actual, integer_cache = sessions[1].run(None, feed)
            assert isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray)
            np.testing.assert_array_equal(cache, integer_cache)
            relative_mse = np.mean(np.square(expected - actual)) / max(
                float(np.mean(np.square(expected))), 1e-12
            )
            assert relative_mse < 0.001
