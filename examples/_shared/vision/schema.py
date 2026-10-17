"""Validated configuration and serializable demonstration results."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict

from examples._shared.schema import (
    Coverage,
    Environment,
    Latency,
    Method,
    QuantizationMethod,
)


class ModelName(StrEnum):
    """Pretrained TorchVision architectures demonstrated by this suite."""

    RESNET18 = "resnet18"
    MOBILENET_V2 = "mobilenet_v2"
    EFFICIENTNET_B0 = "efficientnet_b0"


class DemoConfig(BaseSettings, frozen=True):
    """Control explicit sample budgets, reproducibility, and local artifacts."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_parse_args=False,
        extra="forbid",
        env_prefix="QUANTSMITH_DEMO_",
        populate_by_name=True,
    )
    models: tuple[ModelName, ...] = Field(default=tuple(ModelName), min_length=1)
    output: Path = Field(default=Path("work_dir/artifacts/vision"))
    methods: list[QuantizationMethod] = Field(
        default=list(QuantizationMethod),
        min_length=1,
        description="Choose minmax, percentile, smoothquant; commas select several.",
    )
    cache: Path = Field(default=Path("work_dir/artifacts/cache"))
    calibration_samples: int = Field(default=128, ge=1)
    evaluation_samples: int = Field(default=256, ge=1)
    seed: int = Field(default=42, ge=0)
    threads: int = Field(default=1, ge=1)
    histogram_bins: int = Field(default=2048, ge=2)
    percentile: float = Field(default=99.99, gt=0, le=100, allow_inf_nan=False)
    smoothquant_alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    warmup: int = Field(default=5, ge=1)
    benchmark_runs: int = Field(default=30, ge=1)
    gallery_images: int = Field(default=12, ge=1)
    quiet: CliImplicitFlag[bool] = Field(default=False)
    json_output: CliImplicitFlag[bool] = Field(default=False, alias="json")


class ImageRecord(BaseModel):
    """Identify an image and its original ImageNet-1K label."""

    model_config = ConfigDict(frozen=True, strict=True)
    path: Path = Field()
    label: int = Field(ge=0, lt=1000)
    class_name: str = Field(min_length=1)


class Metrics(BaseModel):
    """Aggregate task accuracy and output differences without retaining logits."""

    samples: int = Field(ge=1)
    top1_accuracy: float = Field(ge=0, le=1)
    top5_accuracy: float = Field(ge=0, le=1)
    agreement_with_torch: float = Field(ge=0, le=1)
    agreement_with_onnx: float = Field(ge=0, le=1)
    mse_vs_torch: float = Field(ge=0)
    mse_vs_onnx: float = Field(ge=0)
    max_abs_vs_torch: float = Field(ge=0)
    max_abs_vs_onnx: float = Field(ge=0)


class Prediction(BaseModel):
    """Store one bounded gallery prediction per inference variant."""

    method: Method = Field()
    label: int = Field(ge=0, lt=1000)
    name: str = Field()
    confidence: float = Field(ge=0, le=1)
    correct: bool = Field()


class GalleryItem(BaseModel):
    """Keep a portable image reference and before/after predictions."""

    image: Path = Field()
    truth: str = Field()
    predictions: tuple[Prediction, ...] = Field()


class VariantReport(BaseModel):
    """Pair measured behavior with saved model size and quantization coverage."""

    method: Method = Field()
    metrics: Metrics = Field()
    latency: Latency = Field()
    artifact: Path = Field()
    artifact_bytes: int = Field(ge=0)
    coverage: Coverage = Field()


class ModelReport(BaseModel):
    """Capture one pretrained model's full export and quantization experiment."""

    model: ModelName = Field()
    weights: str = Field()
    preprocessing: str = Field()
    export_max_abs_error: float = Field(ge=0)
    execution_target: Literal["onnxruntime"] = "onnxruntime"
    variants: tuple[VariantReport, ...] = Field()
    gallery: tuple[GalleryItem, ...] = Field()


class SuiteReport(BaseModel):
    """Persist reproducible configuration, provenance, and measured results."""

    created_at: datetime = Field()
    config: DemoConfig = Field()
    environment: Environment = Field()
    calibration_split: str = Field(default="Imagenette 160px / train")
    evaluation_split: str = Field(default="Imagenette 160px / val")
    models: tuple[ModelReport, ...] = Field()
