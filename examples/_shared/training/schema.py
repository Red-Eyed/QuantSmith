"""Typed experiment inputs, provenance, and bounded demonstration records."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict

from examples._shared.schema import (
    Coverage,
    Environment,
    Latency,
    Method,
    QuantizationMethod,
)
from quantsmith.domain import InputArray

type Tokens = np.ndarray[tuple[int, ...], np.dtype[np.int64]]


class TaskKind(StrEnum):
    """Tasks with different input types, architectures, and accuracy measures."""

    WINE = "wine_mlp"
    TEXT = "shakespeare_rnn"
    TRANSFORMER = "shakespeare_transformer"
    WINDOWED = "shakespeare_windowed_transformer"


class Config(BaseSettings, frozen=True):
    """Make training, sampling, calibration, and artifact budgets explicit."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True, env_prefix="QUANTSMITH_TRAINING_", populate_by_name=True
    )
    tasks: tuple[TaskKind, ...] = Field(
        default=(TaskKind.TRANSFORMER, TaskKind.WINDOWED, TaskKind.WINE), min_length=1
    )
    output: Path = Field(default=Path("work_dir/artifacts/training"))
    methods: list[QuantizationMethod] = Field(
        default=list(QuantizationMethod),
        min_length=1,
        description="Choose minmax, percentile, smoothquant; commas select several.",
    )
    cache: Path = Field(default=Path("work_dir/artifacts/cache"))
    seed: int = Field(default=42, ge=0)
    mlp_steps: int = Field(default=300, ge=1)
    rnn_steps: int = Field(default=600, ge=1)
    transformer_steps: int = Field(default=600, ge=1)
    batch_size: int = Field(default=32, ge=1)
    sequence_length: int = Field(default=32, ge=2)
    calibration_samples: int = Field(
        default=64,
        ge=1,
        description="Text calibration windows; Wine uses its full calibration split.",
    )
    evaluation_samples: int = Field(
        default=128,
        ge=1,
        description="Text evaluation windows; Wine uses its full held-out split.",
    )
    histogram_bins: int = Field(default=2048, ge=2)
    percentile: float = Field(default=99.99, gt=0, le=100)
    smoothquant_alpha: float = Field(default=0.5, ge=0, le=1)
    warmup: int = Field(default=5, ge=1)
    benchmark_runs: int = Field(default=30, ge=1)
    demonstrations: int = Field(default=8, ge=1)
    continuation_length: int = Field(default=80, ge=1)
    quiet: CliImplicitFlag[bool] = Field(default=False)
    json_output: CliImplicitFlag[bool] = Field(default=False, alias="json")


class Split(BaseModel):
    """Hold a deliberately small, bounded demo split and its source identities."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    inputs: InputArray = Field()
    targets: Tokens = Field()
    identities: tuple[int, ...] = Field()


class Dataset(BaseModel):
    """Keep training data separate from calibration and held-out evaluation."""

    kind: TaskKind = Field()
    train: Split = Field()
    calibration: Split = Field()
    evaluation: Split = Field()
    labels: tuple[str, ...] = Field()
    source: str = Field()
    sha256: str = Field()
    preprocessing: str = Field()
    train_end: int = Field()
    calibration_end: int = Field()


class Scores(BaseModel):
    """Aggregate classification/token metrics without retaining full model outputs."""

    observations: int = Field(ge=1)
    accuracy: float = Field(ge=0, le=1)
    cross_entropy: float = Field(ge=0)
    perplexity: float = Field(ge=1)
    agreement_with_onnx: float = Field(ge=0, le=1)
    agreement_with_torch: float = Field(ge=0, le=1)
    mse_vs_onnx: float = Field(ge=0)
    mse_vs_torch: float = Field(ge=0)
    max_abs_vs_onnx: float = Field(ge=0)


class Demonstration(BaseModel):
    """Show actual task inputs, reference labels, and model predictions."""

    identity: int = Field()
    method: Method = Field()
    context: str = Field()
    expected: str = Field()
    predicted: str = Field()
    continuation: str = Field(default="")


class Variant(BaseModel):
    """Pair behavior with its saved artifact and exact quantization coverage."""

    method: Method = Field()
    scores: Scores = Field()
    latency: Latency = Field()
    artifact: Path = Field()
    artifact_bytes: int = Field(ge=0)
    coverage: Coverage = Field()


class TaskReport(BaseModel):
    """Persist architecture, split provenance, training loss, and held-out behavior."""

    task: TaskKind = Field()
    architecture: str = Field()
    source: str = Field()
    sha256: str = Field()
    preprocessing: str = Field()
    train_size: int = Field()
    calibration_size: int = Field()
    evaluation_size: int = Field()
    initial_training_loss: float = Field()
    final_training_loss: float = Field()
    export_max_abs_error: float = Field(ge=0)
    variants: tuple[Variant, ...] = Field()
    demonstrations: tuple[Demonstration, ...] = Field()


class Report(BaseModel):
    """Capture configuration, runtime versions, and task-specific measurements."""

    created_at: datetime = Field()
    config: Config = Field()
    environment: Environment = Field()
    tasks: tuple[TaskReport, ...] = Field()
