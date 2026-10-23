"""Measure trained caption quality and ORT latency after native projection packing."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import onnx
import torch
from numpy.typing import NDArray
from onnxruntime.quantization import QuantType, quantize_dynamic
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict
from rich.console import Console
from stackformers import CrossAttention, SelfAttention
from stackformers.feedforward.swiglu import SwiGLU

from examples._shared.environment import environment
from examples.mobile_captioning.data import (
    Tokenizer,
    cache_images,
    image,
    prepare,
    records,
)
from examples.mobile_captioning.export import (
    GraphPair,
    build_candidate,
    candidates,
    native,
)
from examples.mobile_captioning.inference import predictor, session
from examples.mobile_captioning.main import (
    DatasetEvidence,
    Report,
    Variant,
    measure,
    print_table,
    save_summary,
)
from examples.mobile_captioning.model import Captioner
from examples.mobile_captioning.quality import evaluate
from examples.mobile_captioning.schema import CheckpointInfo, Config
from examples.mobile_captioning.training import load
from examples.quantization_performance.packed_stackformers import (
    PackedConversion,
    library_digest,
    save_library_snapshot,
    weight_digest,
)
from quantsmith.domain import FloatArray

Layout: TypeAlias = Literal["unpacked", "attention_packed", "packed"]


class Settings(BaseSettings):
    """Keep the trained checkpoint independent from new experiment artifacts."""

    model_config = SettingsConfigDict(cli_kebab_case=True)
    checkpoint: Path = Path("work_dir/artifacts/mobile_captioning")
    output: Path = Path("work_dir/artifacts/mobile_captioning_packed")
    evaluation_images: int = Field(default=32, ge=2, le=5000)
    calibration_images: int = Field(default=16, ge=1)
    runs: int = Field(default=100, ge=1)
    warmup: int = Field(default=20, ge=0)


class PackingEvidence(BaseModel):
    """Retain conversion ownership and eager parity before quantization."""

    layout: Layout
    converted_modules: tuple[str, ...]
    encoder_max_abs_error: float
    decoder_max_abs_error: float


class PackedReport(Report):
    """Bind caption results to the exact upgraded Stackformers source."""

    stackformers_source_sha256: str
    packing: tuple[PackingEvidence, ...]
    dynamic_decoder_lengths: tuple[int, ...]


def digest(path: Path) -> str:
    """Identify original weights and annotations without materializing files."""
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def convert(source: Captioner, layout: Layout) -> tuple[Captioner, tuple[str, ...]]:
    """Pack native child modules on an owned copy of the trained captioner."""
    target = deepcopy(source)
    converted: list[str] = []
    for name, child in tuple(target.named_modules()):
        selected = layout != "unpacked" and (
            isinstance(child, (SelfAttention, CrossAttention))
            or (layout == "packed" and isinstance(child, SwiGLU))
        )
        if not selected:
            continue
        packed = PackedConversion.model_validate(child).to_packed()
        target.set_submodule(name, packed)
        converted.append(name)
    return target.eval(), tuple(converted)


def export_pair(
    model: Captioner, pixels: FloatArray, tokens: NDArray[np.int64], directory: Path
) -> GraphPair:
    """Export opset 21 with a genuinely variable decoder prefix length."""
    directory.mkdir(parents=True, exist_ok=True)
    pair = GraphPair(directory / "encoder.onnx", directory / "decoder.onnx")
    images = torch.from_numpy(pixels)
    with torch.inference_mode():
        context = model.encoder.forward(images)
    with (directory / "export.log").open("w") as log, redirect_stdout(log):
        torch.onnx.export(
            model.encoder,
            (images,),
            str(pair.encoder),
            input_names=["images"],
            output_names=["context"],
            opset_version=21,
            dynamo=True,
            external_data=False,
        )
        torch.onnx.export(
            model.decoder,
            (torch.from_numpy(tokens[:, :3].copy()), context),
            str(pair.decoder),
            input_names=["tokens", "context"],
            output_names=["logits"],
            dynamic_shapes=({1: torch.export.Dim("prefix", min=1)}, {}),
            opset_version=21,
            dynamo=True,
            external_data=False,
        )
    for path in (pair.encoder, pair.decoder):
        onnx.checker.check_model(onnx.load(path))
    return pair


def verify_pair(
    pair: GraphPair,
    source: Captioner,
    pixels: FloatArray,
    tokens: NDArray[np.int64],
    *,
    quantized: bool,
) -> None:
    """Execute each deployed decoder at multiple lengths against the common source."""
    graph = onnx.load(pair.decoder)
    if not graph.graph.input[0].type.tensor_type.shape.dim[1].dim_param:
        raise ValueError("decoder prefix length was specialized")
    runtime = predictor(session(pair.encoder, 1), session(pair.decoder, 1))
    context = runtime.encode(pixels)
    with torch.inference_mode():
        reference_context = source.encoder.forward(torch.from_numpy(pixels))
        for length in (1, 3, tokens.shape[1]):
            prefix = tokens[:, :length].copy()
            actual = runtime.decode(prefix, context)
            expected = native(
                source.decoder.forward(torch.from_numpy(prefix), reference_context)
            )
            if actual.shape != expected.shape:
                raise ValueError("dynamic decoder returned the wrong shape")
            if not quantized:
                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def dynamic_pair(source: GraphPair, directory: Path) -> GraphPair:
    """Use native ORT dynamic INT8 only as a separately named runtime control."""
    directory.mkdir(parents=True, exist_ok=True)
    pair = GraphPair(directory / "encoder.onnx", directory / "decoder.onnx")
    for src, dst in ((source.encoder, pair.encoder), (source.decoder, pair.decoder)):
        quantize_dynamic(
            str(src),
            str(dst),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
    return pair


def measure_variants(
    pairs: list[tuple[str, GraphPair]],
    database: Path,
    config: Config,
    tokenizer: Tokenizer,
    console: Console,
) -> tuple[Variant, ...]:
    """Score matched images, then time graph pairs in forward and reverse order."""
    variants: dict[str, Variant] = {}
    for name, pair in pairs:
        console.print(f"{name}: scoring {config.evaluation_images} validation images")
        quality = evaluate(
            pair, database, config, tokenizer, pair.encoder.parent / "captions.jsonl"
        )
        (pair.encoder.parent / "quality.json").write_text(
            quality.model_dump_json(indent=2) + "\n"
        )
        variants[name] = Variant(name=name, quality=quality, repetitions=())
    for sweep, ordered in enumerate((pairs, list(reversed(pairs))), start=1):
        for name, pair in ordered:
            console.print(f"{name}: fresh-process timing sweep {sweep}/2")
            measurement = measure(pair, config, pair.encoder.parent / f"repeat_{sweep}")
            variants[name].repetitions += (measurement,)
    return tuple(variants.values())


def packing_parity(
    source: Captioner,
    target: Captioner,
    pixels: FloatArray,
    tokens: NDArray[np.int64],
    layout: Layout,
    converted: tuple[str, ...],
) -> PackingEvidence:
    """Require eager numerical parity before either graph undergoes quantization."""
    with torch.inference_mode():
        images = torch.from_numpy(pixels)
        prefix = torch.from_numpy(tokens)
        expected_context = source.encoder.forward(images)
        actual_context = target.encoder.forward(images)
        expected = source.decoder.forward(prefix, expected_context)
        actual = target.decoder.forward(prefix, actual_context)
    torch.testing.assert_close(actual_context, expected_context, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    return PackingEvidence(
        layout=layout,
        converted_modules=converted,
        encoder_max_abs_error=float((actual_context - expected_context).abs().max()),
        decoder_max_abs_error=float((actual - expected).abs().max()),
    )


def run(settings: Settings) -> None:
    """Reuse one trained state, disjoint calibration, scoring, and paired timing."""
    if settings.output.resolve() == settings.checkpoint.resolve():
        raise ValueError("new artifacts must not overwrite the original checkpoint")
    info = CheckpointInfo.model_validate_json(
        (settings.checkpoint / "checkpoint.json").read_text()
    )
    load_config = Config.model_validate(
        info.config.model_dump() | {"output": settings.checkpoint}
    )
    source, info = load(load_config)
    torch.set_num_threads(1)
    settings.output.mkdir(parents=True, exist_ok=True)
    save_library_snapshot(settings.output)
    config = Config.model_validate(
        load_config.model_dump()
        | {
            "output": settings.output,
            "command": "benchmark",
            "evaluation_images": settings.evaluation_images,
            "calibration_images": settings.calibration_images,
            "threads": 1,
            "runs": settings.runs,
            "warmup": settings.warmup,
        }
    )
    console = Console(stderr=True)
    database = prepare(config)
    cache_images(database, config, ("calibration", "evaluation"))
    tokenizer = Tokenizer(info.vocabulary, config.model.max_tokens)
    record = next(records(database, "calibration", 1))
    pixels = image(record, config)
    tokens = tokenizer.encode(record.captions[0])[None, :-1]
    np.save(settings.output / "benchmark_image.npy", pixels, allow_pickle=False)
    original_digest = weight_digest(source)
    pairs: list[tuple[str, GraphPair]] = []
    packing: list[PackingEvidence] = []
    layouts: tuple[Layout, ...] = ("unpacked", "attention_packed", "packed")
    for layout in layouts:
        console.print(f"{layout}: native conversion and dynamic ONNX export")
        target, converted = convert(source, layout)
        packing.append(
            packing_parity(source, target, pixels, tokens, layout, converted)
        )
        if weight_digest(source) != original_digest:
            raise ValueError("packing mutated the original trained weights")
        root = settings.output / layout
        fp32 = export_pair(target, pixels, tokens, root / "fp32")
        verify_pair(fp32, source, pixels, tokens, quantized=False)
        pairs.append((f"{layout}/fp32", fp32))
        variant_config = Config.model_validate(config.model_dump() | {"output": root})
        for candidate in candidates(config.histogram_bins):
            if candidate.name not in ("minmax", "float_percentile"):
                continue
            console.print(
                f"{layout}/{candidate.name}: QuantSmith calibration and audit"
            )
            pair = build_candidate(candidate, fp32, database, variant_config, tokenizer)
            verify_pair(pair, source, pixels, tokens, quantized=True)
            pairs.append((f"{layout}/{candidate.name}", pair))
        pair = dynamic_pair(fp32, root / "ort_dynamic")
        verify_pair(pair, source, pixels, tokens, quantized=True)
        pairs.append((f"{layout}/ort_dynamic", pair))
    variants = measure_variants(pairs, database, config, tokenizer, console)
    report = PackedReport(
        created_at=datetime.now(UTC),
        config=config,
        checkpoint=info,
        weights_sha256=digest(settings.checkpoint / "weights.pt"),
        environment=environment(),
        timm_version=version("timm"),
        caption_scorer_version=version("pycocoevalcap"),
        dataset=DatasetEvidence(
            train_annotations_sha256=digest(config.data / "captions_train2017.json"),
            validation_annotations_sha256=digest(config.data / "captions_val2017.json"),
        ),
        variants=tuple(variants),
        stackformers_source_sha256=library_digest(),
        packing=tuple(packing),
        dynamic_decoder_lengths=(1, 3, tokens.shape[1]),
    )
    (settings.output / "report.json").write_text(
        report.model_dump_json(indent=2) + "\n"
    )
    save_summary(report)
    print_table(report, console)


def main() -> None:
    """Parse typed experiment controls without modifying the project dependency pin."""
    run(CliApp.run(Settings))


if __name__ == "__main__":
    main()
