"""Exact quantizer reuse guards, ownership and numerical boundaries through ORT."""

from copy import deepcopy

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from quantsmith.backends.onnx.quantizer_reuse import plan_reuse, reuse_quantizers
from quantsmith.runtime import OnnxEvaluator
from tests.outcomes import expect_ok


@pytest.fixture
def reuse_model() -> ModelProto:
    """Build separate quantizers with equal encodings under different names."""
    nodes = []
    parameters = []
    for name in ("left", "right"):
        parameters.extend(
            (
                numpy_helper.from_array(
                    np.array(0.1, dtype=np.float32), name + "_scale"
                ),
                numpy_helper.from_array(np.array(0, dtype=np.uint8), name + "_zero"),
            )
        )
        nodes.extend(
            (
                helper.make_node(
                    "QuantizeLinear",
                    ["x", name + "_scale", name + "_zero"],
                    [name + "_q"],
                    name=name + "_Q",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [name + "_q", name + "_scale", name + "_zero"],
                    [name],
                    name=name + "_DQ",
                ),
            )
        )
    return helper.make_model(
        helper.make_graph(
            nodes,
            "reuse",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 3])
                for name in ("left", "right")
            ],
            parameters,
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


def test_reuse_is_exact_and_owned(reuse_model: ModelProto) -> None:
    """Preserve rounding and clipping exactly without mutating caller storage."""
    before = reuse_model.SerializeToString()
    plan = plan_reuse(reuse_model)
    assert len(plan.aliases) == 1
    rewritten = expect_ok(reuse_quantizers(reuse_model))
    assert reuse_model.SerializeToString() == before
    assert sum(node.op_type == "QuantizeLinear" for node in rewritten.graph.node) == 1
    sample = {
        "x": np.array([[-0.15, 0.05, 0.15], [25.55, 25.65, 0.0]], dtype=np.float32)
    }
    original = expect_ok(OnnxEvaluator(reuse_model).run(sample, ("left", "right")))
    actual = expect_ok(OnnxEvaluator(rewritten).run(sample, ("left", "right")))
    for name in ("left", "right"):
        np.testing.assert_array_equal(actual[name], original[name])
    assert not plan_reuse(rewritten).aliases
    assert (
        rewritten.SerializeToString()
        == expect_ok(reuse_quantizers(rewritten)).SerializeToString()
    )


@pytest.mark.parametrize(
    "difference",
    [
        "source",
        "scale",
        "zero",
        "dtype",
        "axis",
        "public",
        "omitted",
        "dynamic",
        "overridable",
        "domain",
    ],
)
def test_incompatible_quantizers_stay_separate(
    reuse_model: ModelProto, difference: str
) -> None:
    """Different numerical contracts or public identities cannot be merged."""
    model = deepcopy(reuse_model)
    quantizer = model.graph.node[2]
    match difference:
        case "source":
            quantizer.input[0] = "other"
            model.graph.input.append(
                helper.make_tensor_value_info("other", TensorProto.FLOAT, [2, 3])
            )
        case "scale":
            model.graph.initializer[2].CopyFrom(
                numpy_helper.from_array(np.array(0.2, dtype=np.float32), "right_scale")
            )
        case "zero":
            model.graph.initializer[3].CopyFrom(
                numpy_helper.from_array(np.array(1, dtype=np.uint8), "right_zero")
            )
        case "dtype":
            model.graph.initializer[3].CopyFrom(
                numpy_helper.from_array(np.array(0, dtype=np.int8), "right_zero")
            )
        case "axis":
            quantizer.attribute.append(helper.make_attribute("axis", 0))
        case "public":
            model.graph.output.append(
                helper.make_tensor_value_info("right_q", TensorProto.UINT8, [2, 3])
            )
        case "omitted":
            del quantizer.input[2:]
        case "dynamic":
            del model.graph.initializer[2]
            model.graph.input.append(
                helper.make_tensor_value_info("right_scale", TensorProto.FLOAT, [])
            )
        case "overridable":
            model.graph.input.append(
                helper.make_tensor_value_info("right_scale", TensorProto.FLOAT, [])
            )
        case "domain":
            quantizer.domain = "custom"
    before = model.SerializeToString()
    assert not plan_reuse(model).aliases
    assert model.SerializeToString() == before
