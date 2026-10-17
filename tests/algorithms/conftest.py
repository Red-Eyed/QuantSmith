"""NumPy-only fixtures; no ONNX models, runtime sessions, or downloaded data."""

from pathlib import Path

import pytest

from tests.algorithms.cases import OperatorLayout


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep isolated algorithm runs inside pytest's private working directories."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(
    params=[
        OperatorLayout(
            kind="MatMul", activation_axis=-1, input_weight_axis=0, output_weight_axis=1
        ),
        OperatorLayout(
            kind="BatchedMatMul",
            activation_axis=-1,
            input_weight_axis=1,
            output_weight_axis=2,
        ),
        OperatorLayout(
            kind="Conv", activation_axis=1, input_weight_axis=1, output_weight_axis=0
        ),
        OperatorLayout(
            kind="Gemm",
            attributes={"transA": 0, "transB": 0},
            activation_axis=1,
            input_weight_axis=0,
            output_weight_axis=1,
        ),
        OperatorLayout(
            kind="Gemm",
            attributes={"transA": 0, "transB": 1},
            activation_axis=1,
            input_weight_axis=1,
            output_weight_axis=0,
        ),
        OperatorLayout(
            kind="Gemm",
            attributes={"transA": 1, "transB": 0},
            activation_axis=0,
            input_weight_axis=0,
            output_weight_axis=1,
        ),
        OperatorLayout(
            kind="Gemm",
            attributes={"transA": 1, "transB": 1},
            activation_axis=0,
            input_weight_axis=1,
            output_weight_axis=0,
        ),
    ],
    ids=["matmul", "batched-matmul", "conv", "gemm00", "gemm01", "gemm10", "gemm11"],
)
def operator_layout(request: pytest.FixtureRequest) -> OperatorLayout:
    """Validate each parameter once before typed fixtures/tests consume it."""
    return OperatorLayout.model_validate(request.param)
