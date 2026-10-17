"""Portable HTML, Markdown, and JSON reports for heterogeneous task demonstrations."""

from collections.abc import Mapping
from html import escape
from pathlib import Path

from pydantic import BaseModel, Field

from examples._shared.schema import Coverage, Method
from examples._shared.training.measurement import Predictor, benchmark
from examples._shared.training.schema import (
    Config,
    Dataset,
    Report,
    Scores,
    TaskKind,
    TaskReport,
    Variant,
)
from quantsmith.domain import InputArray
from quantsmith.result import Err, Ok, QuantSmithError, Result


def task_html(task: TaskReport) -> str:
    """Render metrics and actual predictions with artifact links and method filters."""
    rows = []
    for variant in task.variants:
        score = variant.scores
        rows.append(
            f'<tr data-method="{variant.method.value}"><td>{variant.method.value}</td>'
            f"<td>{score.accuracy:.2%}</td><td>{score.cross_entropy:.3f}</td>"
            f"<td>{score.perplexity:.2f}</td><td>{score.agreement_with_onnx:.2%}</td>"
            f"<td>{score.mse_vs_onnx:.5g}</td><td>{variant.latency.median_ms:.3f}</td>"
            f'<td><a href="{escape(variant.artifact.as_posix())}">'
            f"{variant.artifact_bytes / 1024:.1f} KiB</a></td></tr>"
        )
    cards = []
    for demo in task.demonstrations:
        continuation = (
            f"<p>Greedy continuation</p><pre>{escape(demo.continuation)}</pre>"
            if task.task is not TaskKind.WINE
            else ""
        )
        cards.append(
            f'<details data-method="{demo.method.value}"><summary>'
            f"Case {demo.identity} · {demo.method.value}</summary>"
            f"<p>Input</p><pre>{escape(demo.context)}</pre>"
            f"<p>Truth</p><pre>{escape(demo.expected)}</pre>"
            f"<p>Prediction</p><pre>{escape(demo.predicted)}</pre>"
            f"{continuation}</details>"
        )
    return (
        f"<section><h2>{task.task.value}</h2><p>{escape(task.architecture)}</p>"
        f'<p><a href="{escape(task.source)}">Dataset source</a> · '
        f"{task.train_size} training items · "
        f"{task.calibration_size} calibration inputs · "
        f"{task.evaluation_size} held-out inputs</p>"
        f"<p>Fixed training-probe loss: {task.initial_training_loss:.3f} → "
        f"{task.final_training_loss:.3f}. Export max error: "
        f"{task.export_max_abs_error:.3g}.</p>"
        '<div class="scroll"><table><thead><tr><th>Method</th><th>Accuracy</th>'
        "<th>Cross entropy</th><th>Perplexity</th><th>ONNX agreement</th>"
        "<th>ONNX MSE</th><th>Median ms</th><th>Artifact</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f"<h3>Held-out demonstrations</h3>{''.join(cards)}</section>"
    )


def save(report: Report) -> None:
    """Persist each completed task so partial suites remain reviewable."""
    directory = report.config.output
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(report.model_dump_json(indent=2))
    options = '<option value="all">All methods</option>' + "".join(
        f'<option value="{variant.method.value}">{variant.method.value}</option>'
        for variant in report.tasks[0].variants
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>QuantSmith · Trained model experiments</title><style>"
        "body{max-width:1200px;margin:40px auto;padding:0 24px;background:#101720;"
        "color:#edf2f7;font:16px system-ui}a{color:#7ed5ff}section{margin:40px 0}"
        "table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;"
        "border-bottom:1px solid #344154}pre{white-space:pre-wrap;"
        "overflow-wrap:anywhere;"
        "background:#1d2938;padding:16px}details{margin:12px 0;padding:12px;"
        "border:1px solid #344154}select{padding:8px}.scroll{overflow:auto}</style>"
        "<h1>QuantSmith: trained model experiments</h1>"
        "<p>Real datasets, trained PyTorch models, FP32 ONNX export, and independently "
        "calibrated MinMax, Percentile, and SmoothQuant graphs.</p>"
        "<p>These small educational experiments are not production benchmarks. "
        "Language-model accuracy and perplexity use teacher forcing; "
        "continuations use greedy "
        "decoding with a fixed context. Wine accuracy is cultivar classification. "
        "The entire selected held-out split is measured; displayed cases are "
        "the first selected cases.</p>"
        "<p>CPU, one thread. Warmed timings exclude training, calibration, "
        "preprocessing, "
        "and session startup; the ORT evaluator disables graph optimization. "
        "Quantized constant weights use integer storage. Artifact size does not "
        "measure runtime RAM; speedup depends on execution-provider optimizations.</p>"
        f'<label>Show method <select id="method">{options}</select></label>'
        + "".join(task_html(task) for task in report.tasks)
        + '<p><a href="results.json">Full configuration and measured results</a></p>'
        '<script>document.querySelector("#method").addEventListener("change",event=>{'
        'document.querySelectorAll("[data-method]").forEach(item=>{'
        'item.hidden=event.target.value!=="all"&&item.dataset.method!==event.target.value'
        "})})</script></html>"
    )
    (directory / "index.html").write_text(page)
    summary = ["# Trained model experiment results\n"]
    for task in report.tasks:
        summary.extend(
            [
                f"## {task.task.value}\n",
                "| Method | Accuracy | Perplexity | ONNX MSE | Median ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for variant in task.variants:
            summary.append(
                f"| {variant.method.value} | {variant.scores.accuracy:.2%} | "
                f"{variant.scores.perplexity:.2f} | {variant.scores.mse_vs_onnx:.5g} | "
                f"{variant.latency.median_ms:.3f} |"
            )
    (directory / "summary.md").write_text("\n".join(summary) + "\n")


class Manifest(BaseModel):
    """Persist exact row/window selections and train-only vocabulary provenance."""

    labels: tuple[str, ...] = Field()
    training_ids: tuple[int, ...] = Field()
    calibration_ids: tuple[int, ...] = Field()
    evaluation_ids: tuple[int, ...] = Field()
    train_end: int = Field()
    calibration_end: int = Field()
    source_sha256: str = Field()


def artifact(directory: Path, method: Method) -> Path:
    """Resolve the original checkpoint or corresponding inference graph."""
    return directory / (
        "torch_fp32.pt" if method is Method.TORCH else f"{method.value}.onnx"
    )


def variant_reports(
    predictors: Mapping[Method, Predictor],
    scores: Mapping[Method, "Scores"],
    coverage: Mapping[Method, Coverage],
    example: InputArray,
    directory: Path,
    config: Config,
) -> Result[tuple[Variant, ...], QuantSmithError]:
    """Pair completed task metrics with measured inference and saved artifacts."""
    reports: list[Variant] = []
    for method, predict in predictors.items():
        match benchmark(predict, example, config):
            case Err() as error:
                return error
            case Ok(latency):
                pass
        path = artifact(directory, method)
        reports.append(
            Variant(
                method=method,
                scores=scores[method],
                latency=latency,
                artifact=path.relative_to(config.output),
                artifact_bytes=path.stat().st_size,
                coverage=coverage[method],
            )
        )
    return Ok(tuple(reports))


def save_manifest(data: Dataset, directory: Path) -> None:
    """Record split identities alongside checkpoints, keeping arrays out of JSON."""
    manifest = Manifest(
        labels=data.labels,
        training_ids=data.train.identities,
        calibration_ids=data.calibration.identities,
        evaluation_ids=data.evaluation.identities,
        train_end=data.train_end,
        calibration_end=data.calibration_end,
        source_sha256=data.sha256,
    )
    (directory / "selection.json").write_text(manifest.model_dump_json(indent=2))
