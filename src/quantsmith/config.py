"""Validated declarative construction of built-in ORT pipelines."""

from typing import Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms import Algorithm
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.smoothquant.joint import ContractionInputs, JointSmoothQuant
from quantsmith.algorithms.smoothquant.joint.ranges import (
    ChannelExtrema,
    ChannelSelection,
)
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx.stages import (
    QuantizationStage,
    ReuseQuantizersStage,
    TransformStage,
    VerifyStage,
)
from quantsmith.calibration import MinMax, Percentile
from quantsmith.calibration.contracts import DEFAULT_CHANNEL_HISTOGRAM_BYTES
from quantsmith.domain import IntegerType
from quantsmith.pipeline import FloatGraph, JointFoldOnly, Pipeline, QuantizationResult
from quantsmith.pipeline.policies import Folding, FoldOnly, Outputs, ProducerOutputs
from quantsmith.rules import ByOperator, Exclude, Rule, Rules


class MinMaxConfig(BaseModel):
    """Choose untrimmed observed extrema."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["minmax"] = Field(default="minmax")


class PercentileConfig(BaseModel):
    """Choose a validated central percentile interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["percentile"] = Field(default="percentile")
    percentile: float = Field(default=99.99, gt=0, le=100, allow_inf_nan=False)


class NoSmoothing(BaseModel):
    """Declare plain static quantization without unused smoothing parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["none"] = Field(default="none")


class SmoothQuantConfig(BaseModel):
    """Declare smoothing and its permitted folding behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["smoothquant"] = Field(default="smoothquant")
    alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    folding: Folding = Field(default_factory=FoldOnly, discriminator="kind")


class JointSmoothQuantConfig(BaseModel):
    """Coordinate sibling projections and permit only complete offline scale folding."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["joint_smoothquant"] = Field(default="joint_smoothquant")
    alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    channels: ChannelSelection = Field(
        default_factory=ChannelExtrema, discriminator="kind"
    )


class QuantizationConfig(BaseModel):
    """Validate built-in choices; extensions compose typed pipeline steps directly."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    activation_type: IntegerType = Field(default=IntegerType.UINT8)
    weight_type: IntegerType = Field(default=IntegerType.INT8)
    activation_symmetric: bool = Field(default=False, strict=True)
    weight_symmetric: bool = Field(default=True, strict=True)
    calibration: MinMaxConfig | PercentileConfig = Field(
        default_factory=MinMaxConfig, discriminator="kind"
    )
    smoothing: NoSmoothing | SmoothQuantConfig | JointSmoothQuantConfig = Field(
        default_factory=NoSmoothing, discriminator="kind"
    )
    outputs: Outputs = Field(default_factory=ProducerOutputs, discriminator="kind")
    histogram_bins: int = Field(default=2048, ge=2, strict=True)
    max_channel_histogram_bytes: int = Field(
        default=DEFAULT_CHANNEL_HISTOGRAM_BYTES, ge=1, strict=True
    )

    def algorithm(self) -> StaticW8A8:
        """Construct the static algorithm for execution or reconstruction adapters."""
        match self.calibration:
            case MinMaxConfig():
                calibration = MinMax()
            case PercentileConfig(percentile=percentile):
                calibration = Percentile(percentile=percentile)
            case _:
                assert_never(self.calibration)
        return StaticW8A8(
            activation_type=self.activation_type,
            weight_type=self.weight_type,
            activation_symmetric=self.activation_symmetric,
            weight_symmetric=self.weight_symmetric,
            calibration=calibration,
        )

    def pipeline(self) -> Pipeline[FloatGraph, QuantizationResult]:
        """Compile typed alternatives to explicit stages without a runtime registry."""
        quantization = QuantizationStage(supported(self.algorithm()), self.outputs)
        match self.smoothing:
            case NoSmoothing():
                return Pipeline(quantization).then(VerifyStage())
            case SmoothQuantConfig(alpha=alpha, folding=folding):
                transform = TransformStage(supported(SmoothQuant(alpha=alpha)), folding)
                return Pipeline(transform).then(quantization).then(VerifyStage())
            case JointSmoothQuantConfig(alpha=alpha, channels=channels):
                return (
                    Pipeline(joint_transform(alpha, channels))
                    .then(quantization)
                    .then(ReuseQuantizersStage())
                    .then(VerifyStage())
                )
            case _:
                assert_never(self.smoothing)


def supported(algorithm: Algorithm) -> Rules[Algorithm]:
    """Select built-in weight-bearing operators while leaving other nodes untouched."""
    return Rules(
        default=Exclude(reason="operator outside this stage's scope"),
        overrides=(
            Rule(
                selector=ByOperator(operators=frozenset({"Conv", "MatMul", "Gemm"})),
                decision=algorithm,
            ),
        ),
    )


_EXTREMA_CHANNELS = ChannelExtrema()


def joint_transform(
    alpha: float, channels: ChannelSelection = _EXTREMA_CHANNELS
) -> TransformStage:
    """Declare complete group folding while excluding unsupported channel layouts."""
    rules: Rules[Algorithm] = Rules(
        default=Exclude(reason="outside joint smoothing contraction layouts"),
        overrides=(
            Rule(
                selector=ContractionInputs(),
                decision=JointSmoothQuant(alpha=alpha, channels=channels),
            ),
        ),
    )
    return TransformStage(rules, JointFoldOnly())
