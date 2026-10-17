"""Validated command-line controls for this concrete model example."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import CliSuppress, SettingsConfigDict

from examples._shared.training.schema import Config as BaseConfig
from examples._shared.training.schema import TaskKind


class Config(BaseConfig):
    """Fix the model identity while exposing method and experiment controls."""

    model_config = SettingsConfigDict(env_prefix="QUANTSMITH_WINE_MLP_")
    tasks: CliSuppress[tuple[Literal[TaskKind.WINE]]] = Field(default=(TaskKind.WINE,))
    output: Path = Field(default=Path("work_dir/artifacts/wine_mlp"))

    rnn_steps: CliSuppress[int] = Field(default=600, ge=1)
    transformer_steps: CliSuppress[int] = Field(default=600, ge=1)
    sequence_length: CliSuppress[int] = Field(default=32, ge=1)
    continuation_length: CliSuppress[int] = Field(default=80, ge=1)
    calibration_samples: CliSuppress[int] = Field(default=64, ge=1)
    evaluation_samples: CliSuppress[int] = Field(default=128, ge=1)
