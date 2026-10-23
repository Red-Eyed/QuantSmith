"""Shared stderr execution and human/JSON presentation for example entry points."""

import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path

from pydantic import BaseModel
from typing_extensions import TypeVar

from quantsmith.result import Err, Ok, QuantSmithError, Result

ConfigT = TypeVar("ConfigT", infer_variance=True)
ReportT = TypeVar("ReportT", infer_variance=True, bound=BaseModel)


def run_and_report(
    config: ConfigT,
    run: Callable[[ConfigT], Result[ReportT, QuantSmithError]],
    *,
    output: Path,
    json_output: bool,
) -> None:
    """Keep logs off JSON stdout and fail on an expected QuantSmith error."""
    with redirect_stdout(sys.stderr):
        outcome = run(config)
    match outcome:
        case Err(error):
            print(error.model_dump_json(indent=2), file=sys.stderr)
            sys.exit(1)
        case Ok(report):
            pass
    print(
        report.model_dump_json(indent=2)
        if json_output
        else f"Report: {(output / 'index.html').resolve()}"
    )
