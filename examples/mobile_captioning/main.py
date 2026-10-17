"""Train image captioning, export split graphs, and compare measured ORT variants."""

import subprocess
import sys
from datetime import UTC, datetime
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from pydantic_settings import CliApp
from rich.console import Console
from rich.table import Table

from examples._shared.environment import environment
from examples._shared.schema import Environment
from examples.mobile_captioning.benchmark import Measurements
from examples.mobile_captioning.data import (
    Tokenizer,
    cache_images,
    image,
    prepare,
    records,
    vocabulary,
)
from examples.mobile_captioning.export import (
    GraphPair,
    build_candidate,
    candidates,
    export,
)
from examples.mobile_captioning.quality import Quality, evaluate
from examples.mobile_captioning.schema import CheckpointInfo, Config
from examples.mobile_captioning.training import load, train
from quantsmith.domain import Absent


class Variant(BaseModel):
    """Retain quality and every fresh-process repetition for a deployed graph pair."""

    name: str
    quality: Quality
    repetitions: tuple[Measurements, ...]


class DatasetEvidence(BaseModel):
    """Identify the exact public caption annotations used to construct splits."""

    train_annotations_sha256: str
    validation_annotations_sha256: str


class Report(BaseModel):
    """Bind results to their checkpoint, dataset, runtime, and explicit budgets."""

    created_at: datetime
    config: Config
    checkpoint: CheckpointInfo
    weights_sha256: str
    environment: Environment
    timm_version: str
    caption_scorer_version: str
    dataset: DatasetEvidence
    variants: tuple[Variant, ...]


def measure(pair: GraphPair, config: Config, destination: Path) -> Measurements:
    """Isolate RSS high-water marks in a fresh interpreter for each graph pair."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.mobile_captioning.benchmark",
            "--encoder",
            str(pair.encoder.resolve()),
            "--decoder",
            str(pair.decoder.resolve()),
            "--image",
            str((config.output / "benchmark_image.npy").resolve()),
            "--output",
            str(destination.resolve()),
            "--length",
            str(config.model.max_tokens),
            "--threads",
            str(config.threads),
            "--warmup",
            str(config.warmup),
            "--runs",
            str(config.runs),
            "--profile-runs",
            str(config.profile_runs),
        ],
        check=True,
    )
    return Measurements.model_validate_json(
        (destination / "measurements.json").read_text()
    )


def print_table(report: Report, console: Console) -> None:
    """Expose all variants without an invented universal quality/latency gate."""
    table = Table(title="MobileNetV4 + Stackformers captioning")
    for column in (
        "Variant",
        "CIDEr",
        "BLEU-4",
        "Caption ms",
        "Fixed steps ms",
        "ROM MiB",
        "Peak RSS MiB",
    ):
        table.add_column(column)
    for variant in report.variants:
        runs = variant.repetitions
        table.add_row(
            variant.name,
            f"{variant.quality.cider:.4f}",
            f"{variant.quality.bleu4:.4f}",
            "/".join(f"{run.generation.median_ms:.2f}" for run in runs),
            "/".join(f"{run.fixed_steps_generation.median_ms:.2f}" for run in runs),
            f"{runs[0].model_bytes / 2**20:.2f}",
            "/".join(f"{run.inference_peak_mib:.1f}" for run in runs),
        )
    console.print(table)


def save_summary(report: Report) -> None:
    """Keep a complete comparison table when terminal width truncates rich output."""
    lines = [
        "# Mobile captioning measurements",
        "",
        "Bounded training/evaluation study; no production caption-quality claim.",
        "",
        "| Variant | CIDEr | BLEU-4 | Caption ms (two sweeps) | "
        "Fixed steps ms | ROM MiB | Peak RSS MiB |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant in report.variants:
        runs = variant.repetitions
        caption = " / ".join(f"{run.generation.median_ms:.3f}" for run in runs)
        fixed = " / ".join(
            f"{run.fixed_steps_generation.median_ms:.3f}" for run in runs
        )
        rss = " / ".join(f"{run.inference_peak_mib:.1f}" for run in runs)
        lines.append(
            f"| {variant.name} | {variant.quality.cider:.5f} | "
            f"{variant.quality.bleu4:.5f} | {caption} | {fixed} | "
            f"{runs[0].model_bytes / 2**20:.3f} | {rss} |"
        )
    (report.config.output / "summary.md").write_text("\n".join(lines) + "\n")


def run(config: Config) -> Report | Absent:
    """Run the explicit data → training → export → optimization → reporting shell."""
    torch.set_num_threads(config.threads)
    config.output.mkdir(parents=True, exist_ok=True)
    console = Console(stderr=True, quiet=config.quiet)
    with console.status("Preparing disk-indexed COCO caption annotations"):
        database = prepare(config)
    match config.command:
        case "train":
            cache_images(database, config, ("train",))
            train(config, database, vocabulary(database, config))
            console.print(f"Saved checkpoint to {config.output}")
            if config.json_output:
                print((config.output / "checkpoint.json").read_text())
            return Absent(
                reason="training-only command does not produce a benchmark report"
            )
        case "run":
            if not (config.output / "checkpoint.json").exists():
                cache_images(database, config, ("train",))
                train(config, database, vocabulary(database, config))
        case "benchmark":
            pass
    model, info = load(config)
    cache_images(database, config, ("calibration", "evaluation"))
    tokenizer = Tokenizer(info.vocabulary, config.model.max_tokens)
    record = next(records(database, "calibration", 1))
    pixels = image(record, config)
    tokens = tokenizer.encode(record.captions[0])[None, :-1]
    np.save(config.output / "benchmark_image.npy", pixels, allow_pickle=False)
    with console.status("Exporting encoder/decoder and checking eager parity"):
        fp32 = export(model, pixels, tokens, config)
    del model
    pairs: list[tuple[str, GraphPair]] = [("fp32", fp32)]
    for candidate in candidates(config.histogram_bins):
        with console.status(f"Calibrating and auditing {candidate.name}"):
            pair = build_candidate(candidate, fp32, database, config, tokenizer)
        pairs.append((candidate.name, pair))
    qualities: dict[str, Quality] = {}
    for name, pair in pairs:
        console.print(f"{name}: evaluating generated captions")
        qualities[name] = evaluate(
            pair, database, config, tokenizer, pair.encoder.parent / "captions.jsonl"
        )
        (pair.encoder.parent / "quality.json").write_text(
            qualities[name].model_dump_json(indent=2) + "\n"
        )
    measurements: dict[str, list[Measurements]] = {name: [] for name, _ in pairs}
    for repetition, ordered in enumerate((pairs, list(reversed(pairs))), start=1):
        for name, pair in ordered:
            console.print(f"{name}: fresh-process timing repetition {repetition}/2")
            measurements[name].append(
                measure(pair, config, config.output / name / f"repeat_{repetition}")
            )
    with (config.output / "weights.pt").open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    with (config.data / "captions_train2017.json").open("rb") as source:
        train_digest = file_digest(source, "sha256").hexdigest()
    with (config.data / "captions_val2017.json").open("rb") as source:
        validation_digest = file_digest(source, "sha256").hexdigest()
    report = Report(
        created_at=datetime.now(UTC),
        config=config,
        checkpoint=info,
        weights_sha256=digest,
        environment=environment(),
        timm_version=version("timm"),
        caption_scorer_version=version("pycocoevalcap"),
        dataset=DatasetEvidence(
            train_annotations_sha256=train_digest,
            validation_annotations_sha256=validation_digest,
        ),
        variants=tuple(
            Variant(
                name=name,
                quality=qualities[name],
                repetitions=tuple(measurements[name]),
            )
            for name, _ in pairs
        ),
    )
    content = report.model_dump_json(indent=2)
    (config.output / "report.json").write_text(content + "\n")
    save_summary(report)
    print_table(report, console)
    if config.json_output:
        print(content)
    return report


def main() -> None:
    """Parse pydantic-settings CLI controls and execute the captioning experiment."""
    run(CliApp.run(Config))


if __name__ == "__main__":
    main()
