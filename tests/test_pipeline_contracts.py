"""Prove open typed composition without registration or exception translation."""

from dataclasses import dataclass

import pytest

from quantsmith.calibration import Samples
from quantsmith.pipeline import CalibrationContext, Pipeline
from quantsmith.result import Err, FailureKind, Ok, QuantSmithError, Result, failure


@dataclass(frozen=True)
class Increment:
    """Demonstrate a third-party step with a native typed payload."""

    def run(
        self, value: int, context: CalibrationContext
    ) -> Result[int, QuantSmithError]:
        """Return a new value without registration or mutating the declaration."""
        return Ok(value + 1)


@dataclass(frozen=True)
class Reject:
    """Carry an expected failure whose identity must survive composition."""

    error: Err[QuantSmithError]

    def run(
        self, value: int, context: CalibrationContext
    ) -> Result[int, QuantSmithError]:
        """Return the original expected failure container."""
        return self.error


@dataclass(frozen=True)
class Broken:
    """Represent an unexpected plugin programming error."""

    def run(
        self, value: int, context: CalibrationContext
    ) -> Result[int, QuantSmithError]:
        """Raise instead of disguising a broken plugin as expected bad input."""
        raise RuntimeError("broken plugin")


def test_custom_composition(samples: Samples) -> None:
    """Compose native extension steps and keep earlier declarations reusable."""
    context = CalibrationContext(samples, 64)
    original = Pipeline(Increment())
    assert original.then(Increment()).run(1, context) == Ok(3)
    assert original.run(1, context) == Ok(2)


def test_failure_short_circuits(samples: Samples) -> None:
    """An expected failure stops execution before the deliberately broken step."""
    error = failure(FailureKind.INVALID_DATA, "test", "expected rejection")
    assert (
        Pipeline(Reject(error)).then(Broken()).run(1, CalibrationContext(samples, 64))
        is error
    )


def test_plugin_exception_propagates(samples: Samples) -> None:
    """Unexpected failures preserve their exception semantics."""
    with pytest.raises(RuntimeError, match="broken plugin"):
        Pipeline(Broken()).run(1, CalibrationContext(samples, 64))
