"""Check synthetic Transformer equivalence, fixed grids and actual integer execution."""

from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import ModelProto, TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from pydantic import ValidationError

from examples.quantization_performance.kernels import run_arrays
from examples.quantization_performance.transformer_blocks import (
    FRAGMENTS,
    Fragment,
    Geometry,
    build_fragment,
    fixed_onnx,
    fragment_output,
    synthetic_inputs,
)
from examples.quantization_performance.transformer_kernels import (
    Config,
    errors,
    export_fragment,
    measure_case,
    verify_dynamic_sequence,
)
from quantsmith import Quantizer
from quantsmith.config import DynamicQuantizationConfig
from quantsmith.domain import Absent
from tests.outcomes import expect_ok


@pytest.fixture
def geometry() -> Geometry:
    """Keep fragment tests small while retaining multiple heads and batches."""
    return Geometry(batch=2, tokens=3, context=5, width=64)


@pytest.fixture
def matmul_graph() -> ModelProto:
    """Retain a learned projection followed by a dynamic-operand multiplication."""
    weight = np.random.default_rng(42).normal(0, 0.2, (4, 4)).astype(np.float32)
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["inputs", "weight"], ["projected"]),
                helper.make_node("MatMul", ["projected", "memory"], ["output"]),
            ],
            "synthetic",
            [
                helper.make_tensor_value_info("inputs", TensorProto.FLOAT, [1, 3, 4]),
                helper.make_tensor_value_info("memory", TensorProto.FLOAT, [1, 4, 2]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2])],
            [numpy_helper.from_array(weight, "weight")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )


@pytest.mark.parametrize("kind", FRAGMENTS)
def test_fragments_are_deterministic_finite(kind: Fragment, geometry: Geometry) -> None:
    """Exercise the implemented surrounding patterns without training or calibration."""
    inputs = synthetic_inputs(kind, geometry, 6)
    with torch.inference_mode():
        first = fragment_output(build_fragment(kind, geometry, 42), inputs)
        second = fragment_output(build_fragment(kind, geometry, 42), inputs)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_qkv_fused_and_separate_match(geometry: Geometry) -> None:
    """Compare packed and separate QKV with identical weights."""
    inputs = synthetic_inputs("qkv_fused", geometry, 3)
    with torch.inference_mode():
        separate = fragment_output(build_fragment("qkv_separate", geometry, 42), inputs)
        packed = fragment_output(build_fragment("qkv_fused", geometry, 42), inputs)
    torch.testing.assert_close(packed, separate)


def test_strided_and_probability_inputs(geometry: Geometry) -> None:
    """Preserve real noncontiguous storage and the AV probability input contract."""
    strided, _ = synthetic_inputs("strided", geometry, 3)
    probabilities, _ = synthetic_inputs("av", geometry, 3)
    assert not strided.is_contiguous()
    assert strided.stride()[-1] == 2
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0])
    )


@pytest.mark.parametrize("output", ["float", "quantized"])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("scope,expected_count", [("weights", 1), ("all", 2)])
def test_fixed_graph_is_owned_and_executes_integers(
    matmul_graph: ModelProto,
    output: Literal["float", "quantized"],
    signed: bool,
    scope: Literal["weights", "all"],
    expected_count: int,
) -> None:
    """Check both operand scopes and signedness against the same FP32 graph."""
    before = matmul_graph.SerializeToString()
    transformed = fixed_onnx(matmul_graph, 4, output, signed, scope)
    assert matmul_graph.SerializeToString() == before
    kernel = "MatMulIntegerToFloat" if output == "float" else "QLinearMatMul"
    assert (
        sum(node.op_type == kernel for node in transformed.graph.node) == expected_count
    )
    feed = {
        "inputs": np.linspace(-1, 1, 12, dtype=np.float32).reshape(1, 3, 4),
        "memory": np.linspace(-1, 1, 8, dtype=np.float32).reshape(1, 4, 2),
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reference = ort.InferenceSession(
        before, options, providers=["CPUExecutionProvider"]
    )
    runtime = ort.InferenceSession(
        transformed.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    np.testing.assert_allclose(
        run_arrays(runtime, feed)[0],
        run_arrays(reference, feed)[0],
        atol=0.06,
        rtol=0.06,
    )


def test_fixed_activation_grid_does_not_observe_inputs(
    matmul_graph: ModelProto,
) -> None:
    """Keep declared activation scales independent of generated activation values."""
    first = fixed_onnx(matmul_graph, 4, "float", False, "all")
    second = fixed_onnx(matmul_graph, 8, "float", False, "all")
    grids = {item.name: numpy_helper.to_array(item) for item in first.graph.initializer}
    changed = {
        item.name: numpy_helper.to_array(item) for item in second.graph.initializer
    }
    np.testing.assert_allclose(grids["synthetic_int8_0_s0"], 4 / 127)
    np.testing.assert_allclose(grids["synthetic_int8_1_s1"], 4 / 127)
    np.testing.assert_array_equal(
        grids["synthetic_int8_0_q1"], changed["synthetic_int8_0_q1"]
    )
    np.testing.assert_allclose(changed["synthetic_int8_0_s0"], 8 / 127)


def test_head_geometry_rejects_lossy_split() -> None:
    """Fail invalid shapes at the configuration boundary."""
    with pytest.raises(ValidationError, match="divisible"):
        Geometry(tokens=1, context=8, width=65)


def test_fixed_graph_rejects_nonfinite_weight(matmul_graph: ModelProto) -> None:
    """Validate source constants before constructing integer weight tensors."""
    invalid = numpy_helper.from_array(
        np.full((4, 4), np.nan, dtype=np.float32), "weight"
    )
    matmul_graph.graph.initializer[0].CopyFrom(invalid)
    with pytest.raises(ValueError, match="finite FP32"):
        fixed_onnx(matmul_graph, 4, "float", False, "weights")


def test_causal_block_cannot_observe_future_tokens(geometry: Geometry) -> None:
    """Retain the causal contract in the complete Stackformers encoder fragment."""
    module = build_fragment("transformer", geometry, 42)
    inputs, memory = synthetic_inputs("transformer", geometry, 3)
    altered = inputs.clone()
    altered[:, 1:] += 10
    with torch.inference_mode():
        first = fragment_output(module, (inputs, memory))
        changed = fragment_output(module, (altered, memory))
    torch.testing.assert_close(first[:, :1], changed[:, :1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "actual",
    [np.full((2,), np.nan, dtype=np.float32), np.zeros((3,), dtype=np.float32)],
)
def test_output_errors_reject_bad_runtime_values(
    actual: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
) -> None:
    """Keep nonfinite and wrong-shape outputs out of performance comparisons."""
    with pytest.raises(ValueError, match="finite FP32"):
        errors(actual, np.zeros((2,), dtype=np.float32))


def test_noop_dynamic_path_is_identifiable(tmp_path: Path) -> None:
    """A pure dynamic-operand QK fragment must not masquerade as an INT8 projection."""
    geometry = Geometry(tokens=2, context=4, width=64)
    config = Config(opsets=[18], runs=2, warmup=1, repetitions=2)
    with torch.inference_mode():
        case = measure_case("qk", geometry, tmp_path, config)
    dynamic = [row for row in case.measurements if row.scheme == "dynamic"]
    assert len(dynamic) == 1
    assert dynamic[0].integer_matmuls == 0
    assert all(len(row.median_us) == 2 for row in case.measurements)
    assert dynamic[0].relative_mse < 1e-10


@pytest.mark.parametrize("opset_mode,opset", [("convert", 18), ("native", 21)])
def test_projection_profiles_real_integer_kernels(
    tmp_path: Path, opset_mode: Literal["convert", "native"], opset: Literal[18, 21]
) -> None:
    """Check the runner reports actual ORT integer execution."""
    geometry = Geometry(tokens=2, context=2, width=64)
    config = Config(
        opsets=[opset], opset_mode=opset_mode, runs=2, warmup=1, repetitions=1
    )
    with torch.inference_mode():
        case = measure_case("linear", geometry, tmp_path, config)
    assert not case.unsupported
    for row in case.measurements:
        assert row.integer_matmuls == (0 if row.scheme == "fp32" else 1)
        assert len(row.median_us) == 1


@pytest.mark.parametrize("kind", FRAGMENTS)
@pytest.mark.parametrize("opset", [18, 21])
def test_one_export_runs_multiple_sequence_lengths(
    kind: Fragment, opset: int, tmp_path: Path
) -> None:
    """Reuse each FP32/integer graph and session across independent query/KV lengths."""
    geometry = Geometry(tokens=1, context=1, width=64)
    module = build_fragment(kind, geometry, 42)
    path = tmp_path / "dynamic.onnx"
    with torch.inference_mode():
        export_fragment(module, synthetic_inputs(kind, geometry, 3), path, opset)
    original = onnx.load(path)
    graphs = [original]
    dynamic_path = tmp_path / "dynamic_int8.onnx"
    quantize_dynamic(
        original,
        dynamic_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    graphs.append(onnx.load(dynamic_path))
    for output in ("float", "quantized"):
        for signed in (False, True):
            graphs.append(fixed_onnx(original, 4, output, signed, "all"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions: list[ort.InferenceSession] = []
    for graph in graphs:
        verify_dynamic_sequence(graph)
        sessions.append(
            ort.InferenceSession(
                graph.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
        )
    for tokens, context in ((1, 1), (2, 7), (7, 2), (31, 13)):
        varying = Geometry(tokens=tokens, context=context, width=64)
        inputs = synthetic_inputs(kind, varying, 7)
        with torch.inference_mode():
            expected = fragment_output(module, inputs).numpy()
        supplied = {"inputs": inputs[0].numpy(), "memory": inputs[1].numpy()}
        for index, runtime in enumerate(sessions):
            feed = {value.name: supplied[value.name] for value in runtime.get_inputs()}
            actual = run_arrays(runtime, feed)[0]
            assert actual.shape == expected.shape
            if index == 0:
                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_dynamic_validation_rejects_static_graph(matmul_graph: ModelProto) -> None:
    """Prevent a fixed graph from acquiring the report's dynamic-sequence claim."""
    with pytest.raises(ValueError, match="symbolic sequence"):
        verify_dynamic_sequence(matmul_graph)


@pytest.mark.parametrize("kind", ["attention", "transformer"])
def test_quantsmith_dynamic_stackformers_blocks(
    kind: Fragment, geometry: Geometry, tmp_path: Path
) -> None:
    """Quantize real Stackformers projections across query/KV lengths without data."""
    module = build_fragment(kind, geometry, 42)
    path = tmp_path / "source.onnx"
    with torch.inference_mode():
        export_fragment(module, synthetic_inputs(kind, geometry, 3), path)
    original = onnx.load(path)
    result = expect_ok(
        Quantizer.from_config(DynamicQuantizationConfig()).quantize(original).execute()
    )
    verify_dynamic_sequence(result.model)
    assert not isinstance(result.execution, Absent)
    assert len(result.execution.integer_nodes) >= 4
    assert all(
        node.source_operator == "MatMul" for node in result.execution.integer_nodes
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(
        result.model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    for tokens, context in ((3, 5), (1, 6), (1, 7), (7, 2)):
        varying = Geometry(
            batch=geometry.batch, tokens=tokens, context=context, width=geometry.width
        )
        inputs = synthetic_inputs(kind, varying, 7)
        with torch.inference_mode():
            expected = fragment_output(module, inputs).numpy()
        supplied = {"inputs": inputs[0].numpy(), "memory": inputs[1].numpy()}
        feed = {value.name: supplied[value.name] for value in runtime.get_inputs()}
        actual = run_arrays(runtime, feed)[0]
        assert actual.shape == expected.shape
        relative_mse = float(np.mean((actual - expected) ** 2) / np.mean(expected**2))
        assert relative_mse < 0.005
