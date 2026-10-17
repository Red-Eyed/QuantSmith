"""Validated command-line controls for this concrete model example."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import CliSuppress, SettingsConfigDict

from examples._shared.vision.schema import DemoConfig as BaseConfig
from examples._shared.vision.schema import ModelName


class Config(BaseConfig):
    """Fix the model identity while exposing method and experiment controls."""

    model_config = SettingsConfigDict(env_prefix="QUANTSMITH_MOBILENET_V2_")
    models: CliSuppress[tuple[Literal[ModelName.MOBILENET_V2]]] = Field(
        default=(ModelName.MOBILENET_V2,)
    )
    output: Path = Field(default=Path("work_dir/artifacts/mobilenet_v2"))
