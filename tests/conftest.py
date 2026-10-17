"""Reusable ONNX operator fixtures with explicit layout variants."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.calibration import Samples
from quantsmith.domain import FloatArray


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contain incidental runtime files in each test's private working directory."""
    monkeypatch.chdir(tmp_path)


class OperatorCase(BaseModel):
    """Pair a graph with its calibration shape and expected output-channel axis."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    model: ModelProto = Field()
    input_shape: tuple[int, ...] = Field()
    weight_axis: int = Field()


@pytest.fixture(params=["MatMul", "Conv", "Gemm00", "Gemm01", "Gemm10", "Gemm11"])
def operator_case(request: pytest.FixtureRequest) -> OperatorCase:
    """Cover convolution and every Gemm transpose combination."""
    name: str = request.param
    rng = np.random.default_rng(31)
    input_shape, weight_shape, output_shape = ((4, 3), (3, 5), (4, 5))
    op, axis = ("MatMul", 1)
    attrs: dict[str, int] = {}
    if name == "Conv":
        op, axis = ("Conv", 0)
        input_shape, weight_shape, output_shape = (
            (2, 3, 6, 6),
            (5, 3, 3, 3),
            (2, 5, 4, 4),
        )
    elif name.startswith("Gemm"):
        op = "Gemm"
        trans_a, trans_b = (int(name[-2]), int(name[-1]))
        attrs = {"transA": trans_a, "transB": trans_b}
        if trans_a:
            input_shape = (3, 4)
        if trans_b:
            weight_shape, axis = ((5, 3), 0)
    weight = rng.normal(size=weight_shape).astype(np.float32)
    node = helper.make_node(op, ["x", "w"], ["y"], name="projection")
    node.attribute.extend(
        (helper.make_attribute(key, value) for key, value in attrs.items())
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            [numpy_helper.from_array(weight, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    return OperatorCase(model=model, input_shape=input_shape, weight_axis=axis)


@pytest.fixture
def samples(operator_case: OperatorCase) -> Samples:
    """Return deterministic replayable batches without storing a dataset."""

    def replay() -> Iterable[Mapping[str, FloatArray]]:
        """Recreate identical input data for every calibration pass."""
        rng = np.random.default_rng(47)
        for _ in range(4):
            yield {"x": rng.normal(size=operator_case.input_shape).astype(np.float32)}

    return replay
