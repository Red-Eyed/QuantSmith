"""Validated command-line controls for this concrete model example."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import CliSuppress, SettingsConfigDict

from examples._shared.training.schema import Config as BaseConfig
from examples._shared.training.schema import TaskKind


class Config(BaseConfig):
    """Fix the model identity while exposing method and experiment controls."""

    model_config = SettingsConfigDict(env_prefix="QUANTSMITH_CHARACTER_RNN_")
    tasks: CliSuppress[tuple[Literal[TaskKind.TEXT]]] = Field(default=(TaskKind.TEXT,))
    output: Path = Field(default=Path("work_dir/artifacts/character_rnn"))

    mlp_steps: CliSuppress[int] = Field(default=300, ge=1)
    transformer_steps: CliSuppress[int] = Field(default=600, ge=1)
