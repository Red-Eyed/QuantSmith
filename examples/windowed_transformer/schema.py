"""Validated command-line controls for this concrete model example."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import CliSuppress, SettingsConfigDict

from examples._shared.training.schema import Config as BaseConfig
from examples._shared.training.schema import TaskKind


class Config(BaseConfig):
    """Fix the model identity while exposing method and experiment controls."""

    model_config = SettingsConfigDict(env_prefix="QUANTSMITH_WINDOWED_TRANSFORMER_")
    tasks: CliSuppress[tuple[Literal[TaskKind.WINDOWED]]] = Field(
        default=(TaskKind.WINDOWED,)
    )
    output: Path = Field(default=Path("work_dir/artifacts/windowed_transformer"))

    mlp_steps: CliSuppress[int] = Field(default=300, ge=1)
    rnn_steps: CliSuppress[int] = Field(default=600, ge=1)
