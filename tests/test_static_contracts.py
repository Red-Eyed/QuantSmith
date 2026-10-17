"""Verify rejection reasons separately from runtime numerical behavior."""

import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel, Field


class Diagnostic(BaseModel):
    """Validate the diagnostic fields used by the static contract assertions."""

    line: int = Field()
    name: str = Field()
    severity: str = Field()


class Report(BaseModel):
    """Parse Pyrefly output without leaking loose JSON into test logic."""

    errors: list[Diagnostic] = Field()


@pytest.mark.parametrize(
    ("source", "expected", "line"),
    [
        (
            "from quantsmith.domain import PerChannel\nvalue = PerChannel(axis=1)\n",
            "",
            0,
        ),
        (
            "from quantsmith.calibration import MinMaxStats\n"
            "def mutate(stats: MinMaxStats) -> None:\n"
            "    stats.minimum = stats.maximum\n",
            "read-only",
            3,
        ),
        (
            "from quantsmith.calibration import HistogramStats\n"
            "def mutate(stats: HistogramStats) -> None:\n"
            "    stats.counts = stats.counts.copy()\n",
            "read-only",
            3,
        ),
        (
            "from quantsmith.domain import Encoding\n"
            "def mutate(encoding: Encoding) -> None:\n"
            "    encoding.scale = encoding.scale.copy()\n",
            "read-only",
            3,
        ),
        (
            "from quantsmith.plan import RescaleInput\n"
            "def mutate(operation: RescaleInput) -> None:\n"
            "    operation.scale = operation.scale.copy()\n",
            "read-only",
            3,
        ),
        (
            "from PIL import Image\n"
            "from torch import nn\n"
            "from examples._shared.vision.models import LoadedModel\n"
            "def transform(image: Image.Image) -> str:\n"
            "    return 'invalid output'\n"
            "model = LoadedModel(nn.Identity(), transform, (), 'test')\n",
            "bad-argument-type",
            6,
        ),
        (
            "from quantsmith.result import Result\n"
            "from quantsmith.result import QuantSmithError\n"
            "def read(value: Result[int, QuantSmithError]) -> None:\n"
            "    integer: int = value\n",
            "bad-assignment",
            4,
        ),
        (
            "from quantsmith.result import Result\n"
            "from quantsmith.result import QuantSmithError\n"
            "def read(value: Result[int, QuantSmithError]) -> str:\n"
            "    return value.unwrap()\n",
            "missing-attribute",
            4,
        ),
        (
            "from quantsmith.result import Err, Ok, QuantSmithError, Result\n"
            "def describe(value: Result[int, QuantSmithError]) -> str:\n"
            "    match value:\n"
            "        case Ok(number):\n"
            "            return str(number)\n"
            "        case Err(error):\n"
            "            return error.detail\n",
            "",
            0,
        ),
        (
            "from quantsmith.result import Err, Ok, QuantSmithError, Result\n"
            "def convert(value: Result[int, QuantSmithError]) -> "
            "Result[str, QuantSmithError]:\n"
            "    match value:\n"
            "        case Ok(number):\n"
            "            return Ok(str(number))\n"
            "        case Err() as error:\n"
            "            return error\n",
            "",
            0,
        ),
        (
            "from typing import assert_never\n"
            "from quantsmith.result import Err, Ok, Result\n"
            "def read(value: Result[tuple[int, int], str]) -> int:\n"
            "    match value:\n"
            "        case Ok(axes):\n"
            "            first, second = axes\n"
            "            return first + second\n"
            "        case Err():\n"
            "            return 0\n"
            "        case _ as remaining:\n"
            "            assert_never(remaining)\n",
            "",
            0,
        ),
        (
            "from quantsmith.result import Err, Ok, Result\n"
            "def read(value: Result[int, str]) -> str:\n"
            "    match value:\n"
            "        case Ok(number):\n"
            "            return number\n"
            "        case Err(error):\n"
            "            return error\n",
            "bad-return",
            5,
        ),
        (
            "from typing import assert_never\n"
            "from quantsmith.result import Ok, Result\n"
            "def read(value: Result[int, str]) -> int:\n"
            "    match value:\n"
            "        case Ok(number):\n"
            "            return number\n"
            "        case _ as remaining:\n"
            "            assert_never(remaining)\n",
            "bad-argument-type",
            8,
        ),
        (
            "from quantsmith.result import Result\n"
            "def read(value: Result[int, str]) -> int:\n"
            "    return value.value\n",
            "missing-attribute",
            3,
        ),
        (
            "from quantsmith.result import Ok\nvalue = Ok(1)\nvalue.value = 2\n",
            "read-only",
            3,
        ),
        (
            "from quantsmith.result import Err\n"
            "value = Err('bad input')\n"
            "value.error = 'changed'\n",
            "read-only",
            3,
        ),
        (
            "from quantsmith.domain import PerChannel\n"
            'value = PerChannel(axis="bad")\n',
            "bad-argument-type",
            2,
        ),
        (
            "from quantsmith.domain import PerChannel\nvalue = PerChannel()\n",
            "missing-argument",
            2,
        ),
        (
            "from quantsmith.domain import PerTensor\nvalue = PerTensor(axis=1)\n",
            "unexpected-keyword",
            2,
        ),
        (
            "from quantsmith.config import supported\n"
            'value = supported("not a plugin")\n',
            "bad-argument-type",
            2,
        ),
        (
            "from quantsmith.domain import PerTensor, PerChannel\n"
            "from typing import assert_never\n"
            "def incomplete(value: PerTensor | PerChannel) -> None:\n"
            "    match value:\n"
            "        case PerTensor():\n"
            "            return\n"
            "        case _:\n"
            "            assert_never(value)\n",
            "bad-argument-type",
            8,
        ),
    ],
)
def test_contract(tmp_path: Path, source: str, expected: str, line: int) -> None:
    """Require exactly the intended error at the intended source location."""
    candidate = tmp_path / "contract.py"
    candidate.write_text(source)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(root),
            "--locked",
            "pyrefly",
            "check",
            "--config",
            str(root / "pyrefly.toml"),
            "--output-format",
            "json",
            str(candidate),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    report = Report.model_validate_json(result.stdout)
    errors = [
        (item.name, item.line) for item in report.errors if item.severity == "error"
    ]
    assert errors == ([(expected, line)] if expected else [])
    assert result.returncode == (1 if expected else 0)
