"""Verify region boundaries, shape controls, and latency/quality selection semantics."""

from pathlib import Path

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from examples._shared.vision.profile_runtime import Report as ProfileReport
from examples.quantization_performance.kernels import (
    Case,
    Timing,
    converted,
    run_arrays,
    session,
    with_batch,
    with_broadcast,
)
from examples.quantization_performance.main import Model, Score, Variant
from examples.quantization_performance.regions import extract_region, profile_costs
from examples.quantization_performance.selection import (
    Candidate,
    Outcome,
    frontier,
    select_nodes,
)
from quantsmith.domain import FloatArray


@pytest.fixture
def model() -> ModelProto:
    """Provide a private Conv/Relu/residual chain with an external broadcast edge."""
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["images", "weight"], ["conv"], name="conv"),
                helper.make_node("Relu", ["conv"], ["relu"], name="relu"),
                helper.make_node("Add", ["relu", "residual"], ["output"], name="add"),
            ],
            "region",
            [
                helper.make_tensor_value_info(
                    "images", TensorProto.FLOAT, [1, 2, 4, 4]
                ),
                helper.make_tensor_value_info(
                    "residual", TensorProto.FLOAT, [1, 3, 1, 1]
                ),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
            [
                numpy_helper.from_array(
                    np.ones((3, 2, 1, 1), dtype=np.float32), "weight"
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )


@pytest.fixture
def feed() -> dict[str, FloatArray]:
    """Exercise both signs and a broadcast addition with finite FP32 tensors."""
    return {
        "images": np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 2, 4, 4),
        "residual": np.ones((1, 3, 1, 1), dtype=np.float32),
    }


def test_extraction_preserves_broadcast_boundary_and_ownership(
    model: ModelProto, feed: dict[str, FloatArray], tmp_path: Path
) -> None:
    """Preserve residual edges and constants without changing the source graph."""
    before = model.SerializeToString()
    extracted, region = extract_region(model, "conv", 2, 20)
    assert model.SerializeToString() == before
    assert region.nodes == ("conv", "relu", "add")
    assert {value.name: value.dimensions for value in region.inputs} == {
        "images": (1, 2, 4, 4),
        "residual": (1, 3, 1, 1),
    }
    expected = run_arrays(session(model, 1, tmp_path / "original.onnx"), feed)
    actual = run_arrays(session(extracted, 1, tmp_path / "extracted.onnx"), feed)
    np.testing.assert_array_equal(actual[0], expected[0])


def test_shared_outputs_stop_region_extension(model: ModelProto) -> None:
    """Do not silently absorb an activation when another consumer needs the producer."""
    model.graph.node.append(
        helper.make_node("Sigmoid", ["conv"], ["shared"], name="shared")
    )
    _, region = extract_region(model, "conv", 2, 20)
    assert region.nodes == ("conv",)


def test_symbolic_region_boundaries_are_rejected(model: ModelProto) -> None:
    """Do not guess concrete geometry for an unresolved symbolic batch dimension."""
    dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
    dimension.ClearField("dim_value")
    dimension.dim_param = "batch"
    with pytest.raises(ValueError, match="concrete dimensions"):
        extract_region(model, "conv", 0, 20)


@pytest.mark.parametrize(
    "dtype,value", [(TensorProto.INT64, 1), (TensorProto.FLOAT, float("nan"))]
)
def test_runtime_boundary_rejects_wrong_dtype_and_nonfinite_values(
    tmp_path: Path, dtype: int, value: int | float
) -> None:
    """Validate real runtime responses before admitting them as floating activations."""
    constant = helper.make_tensor("constant", dtype, [1], [value])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Constant", [], ["output"], value=constant)],
            "invalid_response",
            [],
            [helper.make_tensor_value_info("output", dtype, [1])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    with pytest.raises(ValueError, match="finite FP32"):
        run_arrays(session(model, 1, tmp_path / "invalid.onnx"), {})


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("opset", [18, 21])
def test_batch_and_opset_conversion_preserve_values(
    model: ModelProto,
    feed: dict[str, FloatArray],
    tmp_path: Path,
    batch: int,
    opset: int,
) -> None:
    """Adapt schemas and inferred batch shapes without changing the computed values."""
    expected = run_arrays(session(model, 1, tmp_path / "base.onnx"), feed)[0]
    changed = converted(with_batch(model, batch), opset)
    actual = run_arrays(
        session(changed, 1, tmp_path / "converted.onnx"),
        {name: np.repeat(value, batch, axis=0) for name, value in feed.items()},
    )[0]
    assert actual.shape == (batch, 3, 4, 4)
    np.testing.assert_array_equal(actual, np.repeat(expected, batch, axis=0))


@pytest.mark.parametrize("context", ["scalar", "channel", "full"])
def test_broadcast_controls_are_numerically_equivalent(
    model: ModelProto, feed: dict[str, FloatArray], tmp_path: Path, context: str
) -> None:
    """Change only broadcasting geometry while keeping coefficients identical."""
    expected = run_arrays(session(model, 1, tmp_path / "base.onnx"), feed)[0] * 1.125
    match context:
        case "scalar":
            changed = with_broadcast(model, "scalar")
        case "channel":
            changed = with_broadcast(model, "channel")
        case "full":
            changed = with_broadcast(model, "full")
        case _:
            raise AssertionError("unexpected test context")
    actual = run_arrays(session(changed, 1, tmp_path / "broadcast.onnx"), feed)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_profile_mapping_uses_exact_source_names(
    model: ModelProto, tmp_path: Path
) -> None:
    """Recognize ORT token suffixes without accepting arbitrary source prefixes."""
    trace = tmp_path / "trace.json"
    trace.write_text(
        '[{"cat":"Node","name":"conv_token_3_kernel_time","dur":10},'
        '{"cat":"Node","name":"conv_other_kernel_time","dur":100}]'
    )
    assert profile_costs(model, trace) == {"conv": 10}


@pytest.fixture
def core_case(model: ModelProto, tmp_path: Path) -> Case:
    """Make an attractive isolated quantized kernel with no deployment assumption."""
    _, region = extract_region(model, "conv", 0, 20)
    return Case(
        model=Model.RESNET18,
        source=region,
        context="core",
        batch=1,
        input_shapes={"images": (1, 2, 4, 4)},
        timings=tuple(
            Timing(
                scheme=scheme,
                opset=18,
                path=tmp_path / f"{scheme}.onnx",
                sha256="test",
                median_ms=(latency,),
                p95_ms=(latency,),
                relative_mse=0,
                optimized_operators={},
            )
            for scheme, latency in ((Variant.FP32, 1.0), (Variant.ORT_QDQ, 0.5))
        ),
    )


def test_region_cost_overrides_isolated_kernel_gain(core_case: Case) -> None:
    """Reject a locally fast kernel when its surrounding region is slower than FP32."""
    slow = core_case.timings[1].model_copy(update={"median_ms": (2.0,)})
    region = core_case.model_copy(
        update={"context": "region", "timings": (core_case.timings[0], slow)}
    )
    assert (
        select_nodes((core_case, region), Model.RESNET18, Variant.ORT_QDQ, 18, 0.05)
        == ()
    )
    assert select_nodes((core_case,), Model.RESNET18, Variant.ORT_QDQ, 18, 0.05) == (
        "conv",
    )


def test_artificial_broadcasts_cannot_select_real_model_nodes(core_case: Case) -> None:
    """Synthetic context changes do not justify rewriting a different original graph."""
    artificial = core_case.model_copy(update={"context": "channel"})
    assert select_nodes((artificial,), Model.RESNET18, Variant.ORT_QDQ, 18, 0.05) == ()


@pytest.fixture
def fp32_outcome(tmp_path: Path) -> Outcome:
    """Create a complete-model FP32 control with deterministic quality and latency."""
    profile = ProfileReport(
        model=tmp_path / "model.onnx",
        input=tmp_path / "input.npy",
        threads=1,
        warmup=1,
        runs=1,
        runtime_version="test",
        platform="test",
        model_bytes=1,
        median_ms=1,
        p95_ms=1,
        baseline_peak_mib=1,
        session_peak_mib=1,
        inference_peak_mib=1,
        optimized_operators={},
        profiled_operator_us={},
    )
    score = Score(samples=1, top1=1, agreement=1, logit_mse=0)
    return Outcome(
        candidate=Candidate(
            name="fp32",
            scope="fp32",
            scheme=Variant.FP32,
            opset=18,
            path=tmp_path / "model.onnx",
            selected=(),
        ),
        sha256="test",
        tuning=score,
        held_out=score,
        profiles=(profile,),
    )


def test_frontier_does_not_reward_more_quantized_nodes(fp32_outcome: Outcome) -> None:
    """Retain FP32 when added quantization increases latency at the same accuracy."""
    quantized = fp32_outcome.model_copy(
        update={
            "candidate": fp32_outcome.candidate.model_copy(
                update={"name": "quantized", "selected": ("a", "b", "c")}
            ),
            "profiles": (fp32_outcome.profiles[0].model_copy(update={"median_ms": 2}),),
        }
    )
    assert frontier((fp32_outcome, quantized)) == ("fp32",)


def test_frontier_ignores_final_held_out_labels(fp32_outcome: Outcome) -> None:
    """Final validation labels cannot override worse tuning quality and speed."""
    control = fp32_outcome.model_copy(
        update={"held_out": fp32_outcome.held_out.model_copy(update={"top1": 0})}
    )
    other = fp32_outcome.model_copy(
        update={
            "candidate": fp32_outcome.candidate.model_copy(update={"name": "other"}),
            "tuning": fp32_outcome.tuning.model_copy(update={"top1": 0}),
            "profiles": (fp32_outcome.profiles[0].model_copy(update={"median_ms": 2}),),
        }
    )
    assert frontier((control, other)) == ("fp32",)
