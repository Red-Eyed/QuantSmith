"""Actual ORT candidate scoring, calibration replay and typed pipeline behavior."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import pytest

from quantsmith import Quantizer, Replay
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.backends.onnx.smoothing_selection import (
    SelectSmoothingStage,
    SmoothingCandidate,
)
from quantsmith.backends.onnx.stages import QuantizationStage, VerifyStage
from quantsmith.calibration import Samples
from quantsmith.config import (
    JointSmoothQuantConfig,
    OutputSelectedSmoothingConfig,
    QuantizationConfig,
    supported,
)
from quantsmith.domain import Absent, InputArray
from quantsmith.optimization.smoothing import ObservedTensors
from quantsmith.pipeline import CalibrationContext, FloatGraph, Pipeline, QuantizedGraph
from quantsmith.result import Err, FailureKind, QuantSmithError, Result, failure
from quantsmith.runtime import OnnxEvaluator, evaluate
from tests.conftest import OperatorCase
from tests.outcomes import expect_error, expect_ok


@pytest.fixture
def selection_config() -> QuantizationConfig:
    """Declare two joint candidates and explicit operator-output scoring."""
    return QuantizationConfig(
        smoothing=OutputSelectedSmoothingConfig(
            candidates=(
                JointSmoothQuantConfig(alpha=0.25),
                JointSmoothQuantConfig(alpha=0.75),
            ),
            observations=ObservedTensors(names=("y",)),
        ),
        histogram_bins=32,
    )


def test_output_error_matches_real_execution(
    operator_case: OperatorCase, samples: Samples, selection_config: QuantizationConfig
) -> None:
    """Verify the retained score against independent optimized ORT execution."""
    before = operator_case.model.SerializeToString()
    result = expect_ok(
        Quantizer.from_config(selection_config)
        .quantize(operator_case.model)
        .calibrate(Replay(samples))
    )
    assert operator_case.model.SerializeToString() == before
    assert not isinstance(result.selection, Absent)
    report = result.selection
    assert len(report.candidates) == 3
    assert report.selected == "baseline"
    assert report.outputs == ("y",)
    measured = expect_ok(
        evaluate(
            OnnxEvaluator(
                operator_case.model,
                optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            ),
            OnnxEvaluator(
                result.model, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ),
            samples,
            ("y",),
        )
    )
    assert measured.mean_squared_error == report.candidates[0].mean_squared_error
    assert all(
        score.batches == 4 and score.integer_nodes == 1 for score in report.candidates
    )
    assert not isinstance(result.execution, Absent)
    assert len(result.execution.integer_nodes) == 1


def test_no_data_is_failure(
    operator_case: OperatorCase, selection_config: QuantizationConfig
) -> None:
    """Reject empty calibration rather than selecting an unmeasured baseline."""
    expect_error(
        Quantizer.from_config(selection_config)
        .quantize(operator_case.model)
        .calibrate(()),
        "no",
    )


def test_unknown_tensor_is_failure(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Preserve invalid observation failures at the runtime boundary."""
    config = QuantizationConfig(
        smoothing=OutputSelectedSmoothingConfig(
            candidates=(JointSmoothQuantConfig(),),
            observations=ObservedTensors(names=("missing",)),
        )
    )
    expect_error(
        Quantizer.from_config(config)
        .quantize(operator_case.model)
        .calibrate(Replay(samples)),
        "metadata",
    )


@dataclass(frozen=True)
class FailedCandidate:
    """Return a documented failure without asking the selector to hide it."""

    error: Err[QuantSmithError]

    def run(
        self, value: FloatGraph, context: CalibrationContext
    ) -> Result[QuantizedGraph, QuantSmithError]:
        """Propagate the exact expected diagnostic."""
        return self.error


def test_failed_candidate_is_not_silently_skipped(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Keep numerical/execution failures visible even when baseline succeeds."""
    expected = failure(FailureKind.EXECUTION, "candidate", "deliberate failure")
    stage = SelectSmoothingStage(
        (
            SmoothingCandidate("baseline", QuantizationStage(supported(StaticW8A8()))),
            SmoothingCandidate("failed", FailedCandidate(expected)),
        ),
        ObservedTensors(names=("y",)),
    )
    result = quantize(operator_case.model, samples, Pipeline(stage).then(VerifyStage()))
    assert result is expected


def test_streams_fresh_replay(
    operator_case: OperatorCase, selection_config: QuantizationConfig
) -> None:
    """Consume deterministic fresh batches without materializing a source iterable."""
    passes = 0

    def samples() -> Iterable[Mapping[str, InputArray]]:
        """Expose whether all candidates receive calibration and scoring replay."""
        nonlocal passes
        passes += 1
        for value in (0.5, 1.0):
            yield {"x": np.full(operator_case.input_shape, value, dtype=np.float32)}

    result = expect_ok(
        Quantizer.from_config(selection_config)
        .quantize(operator_case.model)
        .calibrate(Replay(samples))
    )
    assert not isinstance(result.selection, Absent)
    assert all(score.batches == 2 for score in result.selection.candidates)
    assert passes >= 6
