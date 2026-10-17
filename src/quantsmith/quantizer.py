"""ORT-only facade with typed configured/model-bound states and explicit replay."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass

from onnx import ModelProto

from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.calibration import Samples
from quantsmith.calibration.contracts import DEFAULT_CHANNEL_HISTOGRAM_BYTES
from quantsmith.config import Float16StorageConfig, QuantizationConfig
from quantsmith.domain import InputArray
from quantsmith.pipeline import FloatGraph, Pipeline, QuantizationResult
from quantsmith.result import FailureKind, QuantSmithError, Result, failure


@dataclass(frozen=True)
class Replay:
    """Supply a fresh deterministic iterable for every calibration pass."""

    samples: Samples


@dataclass(frozen=True)
class Configured:
    """Carry a validated pipeline declaration without a bound model."""

    pipeline: Pipeline[FloatGraph, QuantizationResult]
    histogram_bins: int
    max_channel_histogram_bytes: int


@dataclass(frozen=True)
class ModelBound:
    """Own a model snapshot and its declared execution program."""

    configuration: Configured
    model: ModelProto


@dataclass(frozen=True)
class Quantizer[State]:
    """Preserve lifecycle restrictions through generics instead of mutable flags."""

    state: State

    @classmethod
    def from_config(
        cls, config: QuantizationConfig | Float16StorageConfig
    ) -> Quantizer[Configured]:
        """Compile a validated configuration without running or capturing a model."""
        match config:
            case Float16StorageConfig():
                return cls.from_pipeline(config.pipeline())
            case QuantizationConfig():
                return Quantizer(
                    Configured(
                        config.pipeline(),
                        config.histogram_bins,
                        config.max_channel_histogram_bytes,
                    )
                )

    @classmethod
    def from_pipeline(
        cls,
        program: Pipeline[FloatGraph, QuantizationResult],
        *,
        histogram_bins: int = 2048,
        max_channel_histogram_bytes: int = DEFAULT_CHANNEL_HISTOGRAM_BYTES,
    ) -> Quantizer[Configured]:
        """Admit custom typed stages directly, without registration or string lookup."""
        if histogram_bins < 2:
            raise ValueError("histogram_bins must be at least two")
        if max_channel_histogram_bytes < 1:
            raise ValueError("max_channel_histogram_bytes must be positive")
        return Quantizer(
            Configured(program, histogram_bins, max_channel_histogram_bytes)
        )

    def quantize(
        self: Quantizer[Configured], model: ModelProto
    ) -> Quantizer[ModelBound]:
        """Bind an owned snapshot; expected model validation occurs at calibration."""
        return Quantizer(ModelBound(self.state, deepcopy(model)))

    def calibrate(
        self: Quantizer[ModelBound],
        samples: Iterable[Mapping[str, InputArray]] | Replay,
    ) -> Result[QuantizationResult, QuantSmithError]:
        """Run with bounded statistics and no dataset caching.

        Iterable sources must replay. One-shot iterators are rejected explicitly;
        wrap a factory in Replay for generators, streams and repeated loading.
        """
        match samples:
            case Replay(factory):
                source = factory
            case _:
                if isinstance(samples, Iterator):
                    return failure(
                        FailureKind.INVALID_DATA,
                        "calibration source",
                        "one-shot iterators cannot replay; "
                        "supply an iterable or Replay(factory)",
                    )

                def source() -> Iterable[Mapping[str, InputArray]]:
                    """Replay the dataset without retaining batches."""
                    return iter(samples)

        configuration = self.state.configuration
        return quantize(
            self.state.model,
            source,
            configuration.pipeline,
            histogram_bins=configuration.histogram_bins,
            max_channel_histogram_bytes=configuration.max_channel_histogram_bytes,
        )
