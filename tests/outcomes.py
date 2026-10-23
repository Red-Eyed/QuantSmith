"""Test assertions that preserve the failure diagnostic when success is expected."""

import pytest
from typing_extensions import TypeVar

from quantsmith.result import Err, Ok, QuantSmithError, Result

T = TypeVar("T", infer_variance=True)


def expect_ok(outcome: Result[T, QuantSmithError]) -> T:
    """Assert success and include the structured diagnostic on test failure."""
    match outcome:
        case Err(error):
            pytest.fail(f"{error.operation}: {error.kind}: {error.detail}")
        case Ok(value):
            return value


def expect_error(outcome: Result[T, QuantSmithError], detail: str) -> QuantSmithError:
    """Assert an expected rejection and retain its structured reason for checks."""
    match outcome:
        case Err(error):
            assert detail in error.detail
            return error
        case Ok():
            pytest.fail("expected an explicit failure")
