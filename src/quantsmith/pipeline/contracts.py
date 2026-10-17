"""Typed composition for declarative ORT execution stages and extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from quantsmith.calibration import Samples
from quantsmith.result import Err, Ok, QuantSmithError, Result


@dataclass(frozen=True)
class CalibrationContext:
    """Inject replay and bounded histogram collection into execution stages."""

    samples: Samples
    histogram_bins: int


class Step[Before, After](Protocol):
    """Declare one typed transition without string registration or global dispatch."""

    def run(
        self, value: Before, context: CalibrationContext
    ) -> Result[After, QuantSmithError]:
        """Preserve expected failure semantics and caller ownership."""
        ...


@dataclass(frozen=True)
class Sequence[Before, Between, After]:
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
class Pipeline[Before, After]:
    """Store a declared executable program whose composition is checked statically."""

    step: Step[Before, After]

    def then[Next](self, following: Step[After, Next]) -> Pipeline[Before, Next]:
        """Return a new typed program without mutating this declaration."""
        return Pipeline(Sequence(self.step, following))

    def run(
        self, value: Before, context: CalibrationContext
    ) -> Result[After, QuantSmithError]:
        """Interpret the already declared program through its typed step contract."""
        return self.step.run(value, context)
