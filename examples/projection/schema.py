"""Validated controls and measurements for the deterministic projection example."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict

from examples._shared.schema import Coverage, Method, QuantizationMethod
from quantsmith.runtime import Evaluation


class Config(BaseSettings, frozen=True):
    """Select independent recipes and explicit streamed sample budgets."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True, env_prefix="QUANTSMITH_PROJECTION_", populate_by_name=True
    )
    output: Path = Field(default=Path("work_dir/artifacts/projection"))
    methods: list[QuantizationMethod] = Field(
        default=list(QuantizationMethod),
        min_length=1,
        description="Choose minmax, percentile, smoothquant; commas select several.",
    )
    calibration_samples: int = Field(default=8, ge=1)
    evaluation_samples: int = Field(default=4, ge=1)
    calibration_seed: int = Field(default=11, ge=0)
    evaluation_seed: int = Field(default=19, ge=0)
    histogram_bins: int = Field(default=2048, ge=2)
    percentile: float = Field(default=99.99, gt=0, le=100, allow_inf_nan=False)
    smoothquant_alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    quiet: CliImplicitFlag[bool] = Field(default=False)
    json_output: CliImplicitFlag[bool] = Field(default=False, alias="json")


class Variant(BaseModel, frozen=True):
    """Pair one independently calibrated artifact with held-out output error."""

    method: Method = Field()
    artifact: Path = Field()
    metrics: Evaluation = Field()
    coverage: Coverage = Field()


class Report(BaseModel, frozen=True):
    """Record reproducible controls, the FP32 baseline, and selected method results."""

    created_at: datetime = Field()
    quantsmith: str = Field()
    config: Config = Field()
    baseline: Path = Field()
    variants: tuple[Variant, ...] = Field()
