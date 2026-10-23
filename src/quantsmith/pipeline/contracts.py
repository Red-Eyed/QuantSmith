"""Typed composition for declarative ORT execution stages and extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol

from typing_extensions import TypeVar

from quantsmith.calibration import Samples
from quantsmith.calibration.contracts import DEFAULT_CHANNEL_HISTOGRAM_BYTES
from quantsmith.result import Err, Ok, QuantSmithError, Result

Before = TypeVar("Before", infer_variance=True)
Between = TypeVar("Between", infer_variance=True)
After = TypeVar("After", infer_variance=True)
Next = TypeVar("Next", infer_variance=True)


@dataclass(frozen=True)
class CalibrationContext:
    """Inject replay and bounded histogram collection into execution stages."""

    samples: Samples
    histogram_bins: int
    max_channel_histogram_bytes: int = DEFAULT_CHANNEL_HISTOGRAM_BYTES


class Step(Protocol[Before, After]):
    """Declare one typed transition without string registration or global dispatch."""

    def run(
        self, value: Before, context: CalibrationContext
    ) -> Result[After, QuantSmithError]:
        """Preserve expected failure semantics and caller ownership."""
        ...


@dataclass(frozen=True)
class Sequence(Generic[Before, Between, After]):
    """Connect stages only when the preceding output fits the following input."""

    first: Step[Before, Between]
    second: Step[Between, After]

    def run(
        self, value: Before, context: CalibrationContext
    ) -> Result[After, QuantSmithError]:
        """Stop at the first expected failure without interpreting plugin exceptions."""
        match self.first.run(value, context):
            case Err() as error:
                return error
            case Ok(intermediate):
                return self.second.run(intermediate, context)


@dataclass(frozen=True)
class Pipeline(Generic[Before, After]):
    """Store a declared executable program whose composition is checked statically."""

    step: Step[Before, After]

    def then(self, following: Step[After, Next]) -> Pipeline[Before, Next]:
        """Return a new typed program without mutating this declaration."""
        return Pipeline(Sequence(self.step, following))

    def run(
        self, value: Before, context: CalibrationContext
    ) -> Result[After, QuantSmithError]:
        """Interpret the already declared program through its typed step contract."""
        return self.step.run(value, context)
