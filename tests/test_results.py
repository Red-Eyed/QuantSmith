"""Expected failures stay typed and do not expose partial graph mutations."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from onnx import numpy_helper

from quantsmith.algorithms import Algorithm, Needs, Statistics, build_plan
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe, load
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.backends.onnx.stages import (
    QuantizationStage,
    VerifyStage,
)
from quantsmith.calibration import Requirement, Samples, collect, extrema
from quantsmith.config import supported
from quantsmith.domain import FloatArray, Graph, InputArray, Node
from quantsmith.pipeline import Pipeline
from quantsmith.plan import QuantizationPlan
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)
from quantsmith.rules import Rules
from quantsmith.runtime import OnnxEvaluator, compare, evaluate
from tests.conftest import OperatorCase
from tests.outcomes import expect_error, expect_ok
from tests.test_calibration import IdentityEvaluator


@pytest.mark.parametrize("successful", [True, False])
def test_result_match(successful: bool) -> None:
    """Explicit matching preserves error identity and skips success-only work."""
    error = QuantSmithError(
        kind=FailureKind.INVALID_DATA, operation="composition", detail="bad input"
    )
    outcome: Result[int, QuantSmithError] = Ok(2) if successful else Err(error)
    visited: list[int] = []

    def transform(value: Result[int, QuantSmithError]) -> Result[int, QuantSmithError]:
        """Transform only a successful integer; propagate the original error variant."""
        match value:
            case Ok(number):
                visited.append(number)
                return Ok((number + 1) * 2)
            case Err() as rejected:
                return rejected

    result = transform(outcome)
    if successful:
        assert result == Ok(6)
        assert visited == [2]
    else:
        assert result is outcome
        assert expect_error(result, "bad input") is error
        assert visited == []


def test_result_preserves_native_payload() -> None:
    """Wrapping a numerical payload must not copy, coerce, or freeze its storage."""
    values = np.asarray([1, 2], dtype=np.float32)
    outcome = Ok(values)
    assert outcome.value is values
    assert values.flags.writeable


def test_failure_carries_validated_diagnostic() -> None:
    """The failure factory builds an error variant with the original diagnostic."""
    outcome = failure(FailureKind.INVALID_DATA, "admission", "bad input")
    assert outcome.error.kind is FailureKind.INVALID_DATA
    assert outcome.error.operation == "admission"
    assert outcome.error.detail == "bad input"


class RejectPlanning:
    """Exercise plugin rejection without throwing to stop the pipeline."""

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Reject admission before the collector consumes any data."""
        return failure(FailureKind.UNSUPPORTED, node.name, "plugin declined layout")

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Make accidental planning after admission rejection visible."""
        pytest.fail("a rejected plugin must never plan")


def test_plugin_rejection_stops_collection(operator_case: OperatorCase) -> None:
    """Propagate plugin errors and avoid calibration side effects after rejection."""

    def unused() -> Samples:
        """Create a source factory whose consumption is forbidden in this test."""

        def samples() -> tuple[()]:
            """Fail if admission mistakenly starts calibration."""
            pytest.fail("rejected layout must not consume samples")

        return samples

    rules: Rules[Algorithm] = Rules(default=RejectPlanning())
    outcome = build_plan(
        expect_ok(describe(operator_case.model)), IdentityEvaluator(), unused(), rules
    )
    error = expect_error(outcome, "plugin declined")
    assert error.kind is FailureKind.UNSUPPORTED


def test_missing_file(tmp_path: Path) -> None:
    """Expose I/O category and operation without forcing callers to catch OSError."""
    error = expect_error(load(tmp_path / "missing.onnx"), "missing.onnx")
    assert error.kind is FailureKind.IO


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_observation(value: float) -> None:
    """Reject nonfinite numerical observations as a typed data error."""
    error = expect_error(extrema(np.asarray([value], dtype=np.float32), ()), "finite")
    assert error.kind is FailureKind.INVALID_DATA


def test_foreign_checker_error(operator_case: OperatorCase, samples: Samples) -> None:
    """Translate ONNX's documented checker exception, which is not a ValueError."""
    model = deepcopy(operator_case.model)
    model.graph.node[0].input[0] = "nonexistent"
    before = model.SerializeToString()
    expect_error(
        quantize(
            model,
            samples,
            Pipeline(QuantizationStage(supported(StaticW8A8()))).then(VerifyStage()),
        ),
        "nonexistent",
    )
    assert model.SerializeToString() == before


def test_foreign_runtime_error(operator_case: OperatorCase) -> None:
    """Return an execution failure when ORT rejects an incompatible input dtype."""
    evaluator = OnnxEvaluator(operator_case.model)
    error = expect_error(
        evaluator.run(
            {"x": np.ones(operator_case.input_shape, dtype=np.int64)}, ("y",)
        ),
        "data type",
    )
    assert error.kind is FailureKind.EXECUTION


def test_mask_constants_and_invalid_weights(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Preserve masks with infinities while rejecting nonfinite selected weights."""
    model = deepcopy(operator_case.model)
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([0, -np.inf], dtype=np.float32), "mask")
    )
    graph = expect_ok(describe(model))
    assert np.isneginf(graph.weights["mask"][1])
    assert not graph.weights["mask"].flags.writeable
    weight = np.full(tuple(model.graph.initializer[0].dims), np.inf, dtype=np.float32)
    model.graph.initializer[0].CopyFrom(numpy_helper.from_array(weight, "w"))
    expect_error(
        quantize(
            model,
            samples,
            Pipeline(QuantizationStage(supported(StaticW8A8()))).then(VerifyStage()),
        ),
        "finite",
    )


def test_empty_evaluation() -> None:
    """Report empty held-out data through the evaluation return contract."""
    evaluator = IdentityEvaluator()
    error = expect_error(
        evaluate(evaluator, evaluator, lambda: iter(()), ("x",)), "no values"
    )
    assert error.kind is FailureKind.EMPTY


def test_missing_observed_output() -> None:
    """A missing plugin observation is diagnosed before array indexing."""

    class MissingEvaluator:
        """Return a valid result envelope with an incomplete observation mapping."""

        def run(
            self, sample: "Mapping[str, InputArray]", outputs: tuple[str, ...]
        ) -> "Result[Mapping[str, FloatArray], QuantSmithError]":
            """Demonstrate the runtime obligation beyond protocol signature checking."""
            return Ok({})

    expect_error(
        collect(MissingEvaluator(), lambda: iter(({},)), (Requirement(tensor="x"),)),
        "missing output x",
    )


def test_programming_error_is_not_swallowed() -> None:
    """Expected-outcome helpers must not hide arbitrary programming errors."""

    def broken() -> int:
        """Produce an unexpected error outside the documented admission contract."""
        raise TypeError("implementation bug")

    with pytest.raises(TypeError, match="implementation bug"):
        validate("programmer error", broken)


@pytest.mark.parametrize(
    "actual",
    [
        np.asarray([], dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
    ],
)
def test_invalid_evaluation_values(actual: FloatArray) -> None:
    """Invalid observations from a plugin cannot leak NaN metrics or reducer errors."""
    expect_error(compare(actual, np.zeros_like(actual)), "nonempty and finite")


def test_evaluation_shape_mismatch() -> None:
    """A plugin's changed output dimensions produce an explicit data error."""
    expect_error(
        compare(np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32)),
        "shapes differ",
    )
