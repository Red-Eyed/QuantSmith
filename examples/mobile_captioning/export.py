"""Split ONNX export, parity checks, and declarative quantization candidates."""

from collections.abc import Iterable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Literal

import numpy as np
import onnx
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from examples._shared.plans import save_plans
from examples.mobile_captioning.data import Tokenizer, image, records
from examples.mobile_captioning.inference import predictor, run, session
from examples.mobile_captioning.model import Captioner
from examples.mobile_captioning.schema import Config
from quantsmith import Quantizer, Replay
from quantsmith.algorithms import Algorithm
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx.stages import QuantizationStage, VerifyStage
from quantsmith.calibration import Percentile
from quantsmith.config import PercentileConfig, QuantizationConfig
from quantsmith.domain import FloatArray, InputArray
from quantsmith.pipeline import (
    FloatGraph,
    FloatMatMulOutputs,
    FunctionOutputs,
    Pipeline,
    QuantizationResult,
)
from quantsmith.result import Err, Ok
from quantsmith.rules import ByOperator, Exclude, Rule, Rules


@dataclass(frozen=True)
class GraphPair:
    """Name the encoder and decoder that must be deployed together."""

    encoder: Path
    decoder: Path


class KeepFloat(BaseModel):
    """Declare an unchanged FP32 decoder as an intentional experiment choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["float"] = "float"


class ConvolutionOnly(BaseModel):
    """Quantize CNN convolutions and preserve visual attention in FP32."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["convolution_only"] = "convolution_only"
    algorithm: StaticW8A8
    histogram_bins: int = Field(ge=2)

    def pipeline(self) -> Pipeline[FloatGraph, QuantizationResult]:
        """Compose existing typed rules and stages without adding a library policy."""
        rules: Rules[Algorithm] = Rules(
            default=Exclude(reason="outside explicitly selected CNN convolutions"),
            overrides=(
                Rule(
                    selector=ByOperator(operators=frozenset({"Conv"})),
                    decision=self.algorithm,
                ),
            ),
        )
        return Pipeline(QuantizationStage(rules, FunctionOutputs())).then(VerifyStage())


@dataclass(frozen=True)
class Candidate:
    """Bind each graph's policy without implicit exclusions or fallback behavior."""

    name: str
    encoder: QuantizationConfig | ConvolutionOnly
    decoder: QuantizationConfig | KeepFloat


def candidates(bins: int) -> tuple[Candidate, ...]:
    """Keep controls and isolate activation-function and float-output changes."""
    minmax = QuantizationConfig(histogram_bins=bins)
    percentile = QuantizationConfig(calibration=PercentileConfig(), histogram_bins=bins)
    function = QuantizationConfig(outputs=FunctionOutputs(), histogram_bins=bins)
    float_minmax = QuantizationConfig(
        outputs=FunctionOutputs(),
        matmul_outputs=FloatMatMulOutputs(),
        histogram_bins=bins,
    )
    float_percentile = QuantizationConfig(
        calibration=PercentileConfig(),
        outputs=FunctionOutputs(),
        matmul_outputs=FloatMatMulOutputs(),
        histogram_bins=bins,
    )
    return (
        Candidate("minmax", minmax, minmax),
        Candidate("percentile", percentile, percentile),
        Candidate("function_minmax", function, function),
        Candidate("float_minmax", float_minmax, float_minmax),
        Candidate("float_percentile", float_percentile, float_percentile),
        Candidate("encoder_only_minmax", float_minmax, KeepFloat()),
        Candidate("encoder_only_percentile", float_percentile, KeepFloat()),
        Candidate(
            "cnn_only_minmax",
            ConvolutionOnly(algorithm=StaticW8A8(), histogram_bins=bins),
            KeepFloat(),
        ),
        Candidate(
            "cnn_only_percentile",
            ConvolutionOnly(
                algorithm=StaticW8A8(calibration=Percentile(percentile=99.99)),
                histogram_bins=bins,
            ),
            KeepFloat(),
        ),
    )


def native(tensor: Tensor) -> FloatArray:
    """Require native FP32 inference tensors before crossing into NumPy."""
    if tensor.dtype != torch.float32:
        raise ValueError("model must produce FP32 tensors")
    return tensor.detach().cpu().numpy()


def export(
    model: Captioner,
    pixels: FloatArray,
    tokens: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    config: Config,
) -> GraphPair:
    """Export fixed batch-one graphs and compare both stages to eager outputs."""
    directory = config.output / "fp32"
    directory.mkdir(parents=True, exist_ok=True)
    pair = GraphPair(directory / "encoder.onnx", directory / "decoder.onnx")
    with torch.inference_mode():
        context = model.encoder.forward(torch.from_numpy(pixels))
        expected = model.decoder.forward(torch.from_numpy(tokens), context)
    with (directory / "export.log").open("w") as log, redirect_stdout(log):
        torch.onnx.export(
            model.encoder,
            (torch.from_numpy(pixels),),
            str(pair.encoder),
            input_names=["images"],
            output_names=["context"],
            opset_version=18,
            dynamo=True,
            external_data=False,
            optimize=True,
        )
        torch.onnx.export(
            model.decoder,
            (torch.from_numpy(tokens), context),
            str(pair.decoder),
            input_names=["tokens", "context"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            external_data=False,
            optimize=True,
        )
    for path in (pair.encoder, pair.decoder):
        onnx.checker.check_model(onnx.load(path))
    runtime = predictor(
        session(pair.encoder, config.threads), session(pair.decoder, config.threads)
    )
    actual_context = runtime.encode(pixels)
    actual = runtime.decode(tokens, actual_context)
    np.testing.assert_allclose(actual_context, native(context), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(actual, native(expected), rtol=1e-3, atol=1e-4)
    return pair


def quantize(
    source: Path,
    destination: Path,
    policy: QuantizationConfig | ConvolutionOnly,
    samples: Replay,
) -> None:
    """Audit actual integer kernels and preserve plans and execution diagnostics."""
    match policy:
        case QuantizationConfig():
            quantizer = Quantizer.from_config(policy)
        case ConvolutionOnly():
            quantizer = Quantizer.from_pipeline(
                policy.pipeline(), histogram_bins=policy.histogram_bins
            )
    match quantizer.quantize(onnx.load(source)).calibrate(samples):
        case Err(error):
            raise RuntimeError(error.model_dump_json(indent=2))
        case Ok(result):
            onnx.save(result.model, destination)
            save_plans(destination.with_suffix(".plans.json"), result.plans)
            destination.with_suffix(".execution.json").write_text(
                result.execution.model_dump_json(indent=2) + "\n"
            )
    destination.with_suffix(".config.json").write_text(
        policy.model_dump_json(indent=2) + "\n"
    )


def encoder_samples(
    database: Path, config: Config
) -> Iterable[Mapping[str, InputArray]]:
    """Replay image inputs from a disjoint calibration partition."""
    for record in records(database, "calibration", config.calibration_images):
        yield {"images": image(record, config)}


def decoder_samples(
    encoder: Path,
    database: Path,
    config: Config,
    tokenizer: Tokenizer,
) -> Iterable[Mapping[str, InputArray]]:
    """Calibrate prefixes with this variant's actual encoded visual context."""
    runtime = session(encoder, config.threads)
    length = config.model.max_tokens
    prefix_lengths = tuple(sorted({1, min(4, length), min(8, length), length}))
    for record in records(database, "calibration", config.calibration_images):
        context = run(runtime, {"images": image(record, config)})
        encoded = tokenizer.encode(record.captions[0])[:-1]
        for prefix_length in prefix_lengths:
            prefix = np.zeros((1, length), dtype=np.int64)
            prefix[0, :prefix_length] = encoded[:prefix_length]
            yield {"tokens": prefix, "context": context}


def build_candidate(
    candidate: Candidate,
    fp32: GraphPair,
    database: Path,
    config: Config,
    tokenizer: Tokenizer,
) -> GraphPair:
    """Lower the encoder first, then calibrate the decoder against its context."""
    directory = config.output / candidate.name
    directory.mkdir(parents=True, exist_ok=True)
    pair = GraphPair(directory / "encoder.onnx", directory / "decoder.onnx")
    quantize(
        fp32.encoder,
        pair.encoder,
        candidate.encoder,
        Replay(lambda: encoder_samples(database, config)),
    )
    match candidate.decoder:
        case KeepFloat() as policy:
            copyfile(fp32.decoder, pair.decoder)
            pair.decoder.with_suffix(".config.json").write_text(
                policy.model_dump_json(indent=2) + "\n"
            )
        case QuantizationConfig() as policy:
            quantize(
                fp32.decoder,
                pair.decoder,
                policy,
                Replay(
                    lambda: decoder_samples(pair.encoder, database, config, tokenizer)
                ),
            )
    return pair
