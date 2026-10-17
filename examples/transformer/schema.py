"""Validated command-line controls for this concrete model example."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import CliSuppress, SettingsConfigDict

from examples._shared.training.schema import Config as BaseConfig
from examples._shared.training.schema import TaskKind
from quantsmith.pipeline.policies import MatMulOutputs, QuantizedMatMulOutputs


class Config(BaseConfig):
    """Fix the model identity while exposing method and experiment controls."""

    model_config = SettingsConfigDict(env_prefix="QUANTSMITH_TRANSFORMER_")
    tasks: CliSuppress[tuple[Literal[TaskKind.TRANSFORMER]]] = Field(
        default=(TaskKind.TRANSFORMER,)
    )
    output: Path = Field(default=Path("work_dir/artifacts/transformer"))
    matmul_outputs: MatMulOutputs = Field(
        default_factory=QuantizedMatMulOutputs, discriminator="kind"
    )

    mlp_steps: CliSuppress[int] = Field(default=300, ge=1)
    rnn_steps: CliSuppress[int] = Field(default=600, ge=1)
