"""Verify ORT preoptimization becomes the actual fresh quantization input."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TypeAlias

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper
from rich.console import Console

from examples.quantization_performance.kernels import run_arrays
from examples.quantization_performance.main import file_sha256
from examples.quantization_performance.preoptimization import (
    optimize_graph,
    provenance,
    quantize_revision,
    quantized_candidates,
    save_quantized,
)
from quantsmith.backends.onnx.integer_execution.audit import ExecutionReport
from quantsmith.domain import Absent, FloatArray
from quantsmith.result import Err, Ok

FloatSamples: TypeAlias = Callable[[], Iterable[Mapping[str, FloatArray]]]


@pytest.fixture
def unfused_model() -> ModelProto:
    """Keep Conv/BatchNorm/Relu explicit, followed by a separately quantizable Gemm."""
    rng = np.random.default_rng(7)
    constants = {
        "weight": rng.normal(size=(4, 3, 1, 1)).astype(np.float32),
        "gain": np.array([0.7, 1.1, 0.9, 1.2], dtype=np.float32),
        "offset": np.array([-0.2, 0.1, 0.3, -0.1], dtype=np.float32),
        "mean": np.array([0.1, -0.1, 0.2, 0.05], dtype=np.float32),
        "variance": np.array([0.8, 1.2, 0.9, 1.1], dtype=np.float32),
        "projection": rng.normal(size=(3, 64)).astype(np.float32),
        "bias": np.zeros(3, dtype=np.float32),
    }
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["images", "weight"], ["conv"], name="conv"),
                helper.make_node(
                    "BatchNormalization",
                    ["conv", "gain", "offset", "mean", "variance"],
                    ["normalized"],
                    name="batchnorm",
                ),
                helper.make_node("Relu", ["normalized"], ["relu"], name="relu"),
                helper.make_node("Flatten", ["relu"], ["flat"], name="flatten", axis=1),
                helper.make_node(
                    "Gemm",
                    ["flat", "projection", "bias"],
                    ["logits"],
                    name="projection",
                    transB=1,
                ),
                helper.make_node("Identity", ["images"], ["dead"], name="unmapped"),
            ],
            "unfused_fixture",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
            [numpy_helper.from_array(value, name) for name, value in constants.items()],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )


@pytest.fixture
def samples() -> FloatSamples:
    """Replay identical bounded FP32 inputs on every independent calibration pass."""

    def replay() -> Iterable[Mapping[str, FloatArray]]:
        """Generate each batch on demand without retaining activations."""
        for shift in (0.0, 0.3):
            yield {
                "images": np.linspace(
                    -1 + shift, 1 + shift, 48, dtype=np.float32
                ).reshape(1, 3, 4, 4)
            }

    return replay


@pytest.fixture
def saved_graphs(unfused_model: ModelProto, tmp_path: Path) -> tuple[Path, Path]:
    """Save original and ORT_ENABLE_ALL snapshots as distinct actual input files."""
    original, optimized = tmp_path / "original.onnx", tmp_path / "optimized.onnx"
    annotated = onnx.shape_inference.infer_shapes(unfused_model)
    onnx.save(annotated, original)
    optimize_graph(annotated, optimized, 1)
    return original, optimized


def test_preoptimization_preserves_owner_and_unfused_numerical_behavior(
    unfused_model: ModelProto, samples: FloatSamples, tmp_path: Path
) -> None:
    """Check fusion parity against optimization-disabled source execution."""
    before = unfused_model.SerializeToString()
    optimized = optimize_graph(unfused_model, tmp_path / "optimized.onnx", 1)
    assert unfused_model.SerializeToString() == before
    assert any(
        node.op_type == "BatchNormalization" for node in unfused_model.graph.node
    )
    assert all(node.op_type != "BatchNormalization" for node in optimized.graph.node)
    assert any(node.domain and "Conv" in node.op_type for node in optimized.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    source = ort.InferenceSession(before, options, providers=["CPUExecutionProvider"])
    target = ort.InferenceSession(
        optimized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    for feed in samples():
        np.testing.assert_allclose(
            run_arrays(target, feed)[0],
            run_arrays(source, feed)[0],
            rtol=1e-5,
            atol=1e-5,
        )


def test_saved_optimized_graph_is_actual_fresh_calibration_input(
    saved_graphs: tuple[Path, Path], samples: FloatSamples, tmp_path: Path
) -> None:
    """Freshly calibrate Gemm while preserving the optimized fused Conv in FP32."""
    original, optimized = saved_graphs
    passes: list[int] = []

    def tracked() -> Iterable[Mapping[str, FloatArray]]:
        """Count independent replay invocations without caching any samples."""
        passes.append(1)
        yield from samples()

    for revision, path in (("original", original), ("optimized", optimized)):
        previous = len(passes)
        match quantize_revision(path, tracked):
            case Err(error):
                pytest.fail(error.model_dump_json())
            case Ok(result):
                pass
        assert len(passes) > previous
        match result.execution:
            case ExecutionReport(integer_nodes=nodes):
                operators = {node.source_operator for node in nodes}
                assert "Gemm" in operators
                assert ("Conv" in operators) == (revision == "original")
            case Absent():
                pytest.fail("static quantization must audit its actual input revision")
        if revision == "optimized":
            record = save_quantized(
                "optimized", optimized, tmp_path / "quantized.onnx", result
            )
            assert record.input_graph.sha256 == file_sha256(optimized)
            assert record.input_graph.sha256 != file_sha256(original)
            assert any(
                "FusedConv" in name or ".nchwc::Conv" in name
                for name in record.output_graph.operators
            )


def test_erased_source_identity_is_explicitly_absent(
    unfused_model: ModelProto, saved_graphs: tuple[Path, Path]
) -> None:
    """Do not invent lineage for an original output removed by optimization."""
    links = provenance(unfused_model, onnx.load(saved_graphs[1]))
    missing = next(link for link in links if link.original_node == "unmapped")
    assert isinstance(missing.optimized_nodes, Absent)
    assert missing.optimized_nodes.reason
    projection = next(link for link in links if link.original_node == "projection")
    assert projection.optimized_nodes == ("projection",)


def test_expected_calibration_failures_do_not_trigger_restricted_fallback(
    saved_graphs: tuple[Path, Path], tmp_path: Path
) -> None:
    """Record documented failures without changing the optimization level."""

    def invalid_samples() -> Iterable[Mapping[str, FloatArray]]:
        """Return a finite-shape but invalid nonfinite calibration tensor."""
        yield {"images": np.full((1, 3, 4, 4), np.nan, dtype=np.float32)}

    choices, records, failures = quantized_candidates(
        saved_graphs[0], saved_graphs[1], invalid_samples, tmp_path, Console(quiet=True)
    )
    assert choices == records == ()
    assert {failure.revision for failure in failures} == {"original", "optimized"}
    optimized = next(value for value in failures if value.revision == "optimized")
    assert optimized.input_graph.sha256 == file_sha256(saved_graphs[1])


def test_custom_fusion_without_intermediate_metadata_reports_exact_failure(
    unfused_model: ModelProto, samples: FloatSamples, tmp_path: Path
) -> None:
    """Expose the missing fused-op type-inference contract without hiding it."""
    path = tmp_path / "unannotated_optimized.onnx"
    optimize_graph(unfused_model, path, 1)
    match quantize_revision(path, samples):
        case Err(error):
            assert error.operation == "ONNX execution"
            assert error.detail == "requested tensor lacks FP32 metadata: flat"
        case Ok():
            pytest.fail("unannotated custom fusion unexpectedly supplied flat metadata")
