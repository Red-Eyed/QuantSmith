"""Test decoding work accounting and rejection of malformed ORT responses."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import helper, numpy_helper

from examples.mobile_captioning.inference import (
    Predictor,
    generate,
    generate_context,
    run,
    session,
)
from quantsmith.domain import FloatArray


@pytest.mark.parametrize("stop_at_eos,expected_calls", [(True, 1), (False, 8)])
def test_eos_accounting(stop_at_eos: bool, expected_calls: int) -> None:
    """Fixed-work benchmarks must retain decoder calls after an early EOS."""
    calls: list[int] = []

    def encode(images: FloatArray) -> FloatArray:
        """Return a small synthetic visual context."""
        return np.zeros((1, 4, 8), dtype=np.float32)

    def decode(
        tokens: np.ndarray[tuple[int, ...], np.dtype[np.int64]], context: FloatArray
    ) -> FloatArray:
        """Prefer padding over EOS to verify suppression of invalid generation IDs."""
        calls.append(1)
        logits = np.zeros((1, 8, 5), dtype=np.float32)
        logits[..., 0] = 10
        logits[..., 2] = 5
        return logits

    result = generate_context(
        Predictor(encode, decode),
        encode(np.zeros(1, dtype=np.float32)),
        8,
        stop_at_eos=stop_at_eos,
    )
    assert len(calls) == expected_calls
    assert result == (2,) * expected_calls


def test_image_encoded_once() -> None:
    """Autoregressive generation must reuse visual context for every decoder step."""
    calls: list[int] = []

    def encode(images: FloatArray) -> FloatArray:
        """Count image executions independently from decoder executions."""
        calls.append(1)
        return np.zeros((1, 4, 8), dtype=np.float32)

    def decode(
        tokens: np.ndarray[tuple[int, ...], np.dtype[np.int64]], context: FloatArray
    ) -> FloatArray:
        """Generate a non-EOS token until the explicit length budget is reached."""
        logits = np.zeros((1, 4, 5), dtype=np.float32)
        logits[..., 4] = 1
        return logits

    assert generate(
        Predictor(encode, decode), np.zeros((1, 3, 32, 32), dtype=np.float32), 4
    ) == (4, 4, 4, 4)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "values",
    [
        (np.ones(1, dtype=np.float64),),
        (np.asarray([np.nan], dtype=np.float32),),
        (np.empty(0, dtype=np.float32),),
        (np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)),
    ],
)
def test_invalid_runtime_outputs(
    tmp_path: Path,
    values: tuple[NDArray[np.generic], ...],
) -> None:
    """Reject wrong output dtype, nonfinite/empty data, and output-count mismatches."""
    nodes: list[onnx.NodeProto] = []
    outputs: list[onnx.ValueInfoProto] = []
    for index, value in enumerate(values):
        name = f"out_{index}"
        constant = numpy_helper.from_array(value)
        nodes.append(helper.make_node("Constant", [], [name], value=constant))
        outputs.append(
            helper.make_tensor_value_info(name, constant.data_type, value.shape)
        )
    graph = helper.make_model(
        helper.make_graph(nodes, "boundary", [], outputs),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    path = tmp_path / "bad_output.onnx"
    onnx.save(graph, path)
    with pytest.raises(ValueError, match="captioning"):
        run(session(path, 1), {})
