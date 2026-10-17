"""Typed model, dataset, and experiment controls for image captioning."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict


class ModelConfig(BaseModel):
    """Fix image geometry and Transformer dimensions before training and export."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    image_size: int = Field(default=224, ge=32)
    dim: int = Field(default=256, ge=8)
    heads: int = Field(default=4, ge=1)
    encoder_layers: int = Field(default=2, ge=1)
    decoder_layers: int = Field(default=4, ge=1)
    ff_mult: float = Field(default=3, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(default=32, ge=4, le=128)

    @model_validator(mode="after")
    def check_geometry(self) -> Self:
        """Keep the spatial grid exact and the rotary head dimension even."""
        if self.image_size % 32:
            raise ValueError("image_size must be divisible by the backbone stride 32")
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("dim must divide into heads with even head dimensions")
        return self


class Config(BaseSettings):
    """Expose explicit bounded experiment budgets and separate execution phases."""

    model_config = SettingsConfigDict(
        env_prefix="QUANTSMITH_CAPTION_", cli_kebab_case=True, extra="forbid"
    )
    command: Literal["run", "train", "benchmark"] = "run"
    output: Path = Path("work_dir/artifacts/mobile_captioning")
    data: Path = Path("work_dir/data/coco2017")
    model: ModelConfig = Field(default_factory=ModelConfig)
    train_images: int = Field(default=512, ge=1)
    calibration_images: int = Field(default=16, ge=1)
    evaluation_images: int = Field(default=32, ge=2, le=5000)
    vocabulary_size: int = Field(default=8192, ge=8, le=32768)
    steps: int = Field(default=300, ge=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=0.0003, gt=0, allow_inf_nan=False)
    freeze_backbone: bool = True
    seed: int = Field(default=42, ge=0)
    threads: int = Field(default=1, ge=1)
    download_workers: int = Field(default=4, ge=1, le=32)
    warmup: int = Field(default=5, ge=0)
    runs: int = Field(default=30, ge=1)
    profile_runs: int = Field(default=5, ge=1)
    histogram_bins: int = Field(default=256, ge=2)
    quiet: CliImplicitFlag[bool] = False
    json_output: CliImplicitFlag[bool] = False


class Vocabulary(BaseModel):
    """Keep stable special IDs followed by training-only word types."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    words: tuple[str, ...]

    @model_validator(mode="after")
    def check_words(self) -> Self:
        """Reject ambiguous IDs or missing reserved symbols at admission."""
        if self.words[:4] != ("<pad>", "<bos>", "<eos>", "<unk>"):
            raise ValueError("vocabulary must start with pad, bos, eos, unk")
        if len(set(self.words)) != len(self.words):
            raise ValueError("vocabulary words must be unique")
        return self


class CheckpointInfo(BaseModel):
    """Bind weights to the exact architecture, vocabulary, and training controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    config: Config
    vocabulary: Vocabulary
    steps_completed: int = Field(ge=0)
    initial_loss: float
    final_loss: float
