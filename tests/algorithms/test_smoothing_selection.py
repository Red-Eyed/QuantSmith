"""Pure candidate-selection invariants without graph or runtime dependencies."""

import pytest
from pydantic import ValidationError

from quantsmith.optimization.smoothing import (
    CandidateScore,
    ObservedTensors,
    SelectionReport,
    select,
)
from tests.outcomes import expect_error, expect_ok


def score(name: str, loss: float) -> CandidateScore:
    """Build a valid objective observation with an explicit candidate identity."""
    return CandidateScore(
        candidate=name,
        mean_squared_error=loss,
        maximum_absolute_error=loss,
        batches=1,
        integer_nodes=1,
    )


@pytest.mark.parametrize(
    ("losses", "winner"),
    [
        ((1.0, 0.5, 0.8), "joint-0"),
        ((0.5, 0.5, 0.8), "baseline"),
        ((0.2, 0.5, 0.8), "baseline"),
    ],
)
def test_selection(losses: tuple[float, float, float], winner: str) -> None:
    """Select lower error and preserve the ordered baseline on exact ties."""
    scores = tuple(
        score(name, loss)
        for name, loss in zip(("baseline", "joint-0", "joint-1"), losses, strict=True)
    )
    report = expect_ok(select(scores, ("y",)))
    assert report.selected == winner
    assert report.candidates == scores
    assert SelectionReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("loss", [-1.0, float("nan"), float("inf")])
def test_invalid_loss(loss: float) -> None:
    """Reject invalid objectives at the typed boundary."""
    with pytest.raises(ValidationError):
        score("baseline", loss)


def test_invalid_identities() -> None:
    """Reject absent objectives, ambiguous candidates and repeated outputs."""
    expect_error(select((), ("y",)), "required")
    expect_error(select((score("baseline", 0.0),), ()), "required")
    expect_error(
        select((score("baseline", 0.0), score("baseline", 1.0)), ("y",)), "unique"
    )
    expect_error(select((score("baseline", 0.0),), ("y", "y")), "unique")
    with pytest.raises(ValidationError):
        ObservedTensors(names=("",))
    with pytest.raises(ValidationError):
        SelectionReport(
            selected="missing", outputs=("y",), candidates=(score("baseline", 0.0),)
        )
