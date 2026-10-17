"""Portable HTML and Markdown reports built from measured, typed results."""

from collections.abc import Mapping
from html import escape
from pathlib import Path

from pydantic import BaseModel, Field

from examples._shared.schema import Coverage, Method
from examples._shared.vision.measurement import Runner, benchmark
from examples._shared.vision.schema import (
    DemoConfig,
    GalleryItem,
    ImageRecord,
    Metrics,
    ModelReport,
    SuiteReport,
    VariantReport,
)
from quantsmith.domain import FloatArray

STYLE = """
:root { color-scheme: light; font: 15px system-ui;
background: #f4f6f8; color: #18212c; }
body { max-width: 1450px; margin: auto; padding: 28px; }
h1 { font-size: 36px; margin-bottom: 8px; } h2 { margin-top: 42px; }
p { max-width: 1000px; line-height: 1.55; } a { color: #145ca8; }
section, .card { background: white; border: 1px solid #dfe4e9; border-radius: 12px; }
section { padding: 24px; margin: 20px 0; } .scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 11px 12px; border-bottom: 1px solid #e4e8ec; text-align: right; }
th:first-child, td:first-child { text-align: left; } th { color: #526171; }
.gallery { display: grid;
grid-template-columns: repeat(auto-fit, minmax(285px, 1fr)); gap: 18px; }
.card { overflow: hidden; padding: 16px; }
.card img { width: 100%; height: 180px;
object-fit: contain; background: #edf1f5; }
.card table { font-size: 12px; } .card td { padding: 7px 3px; }
.good { color: #13794d; } .bad { color: #af3535; }
label { display: inline-block; margin: 12px 20px 12px 0; } select { padding: 7px; }
.note { color: #566372; font-size: 13px; } code { font-size: 13px; }
.bar { display: inline-block; width: 80px; height: 6px;
background: #e4ebf1; margin-right: 8px; }
.bar span { display: block; height: 6px; background: #337cb8; }
"""


def variant_row(item: VariantReport) -> str:
    """Render actual metrics without assuming quantization improves them."""
    metrics, latency, coverage = item.metrics, item.latency, item.coverage
    count = f"{len(coverage.quantized_nodes)}/{coverage.eligible_nodes}"
    accuracy = metrics.top1_accuracy * 100
    return (
        f'<tr><td><a href="{escape(item.artifact.as_posix())}">'
        f"{item.method.value}</a></td>"
        f'<td><span class="bar"><span style="width:{accuracy:.2f}%">'
        f"</span></span>{accuracy:.2f}%</td>"
        f"<td>{metrics.top5_accuracy * 100:.2f}%</td>"
        f"<td>{metrics.agreement_with_onnx * 100:.2f}%</td>"
        f"<td>{metrics.mse_vs_onnx:.6g}</td><td>{metrics.max_abs_vs_onnx:.5g}</td>"
        f"<td>{latency.median_ms:.2f} / {latency.p95_ms:.2f}</td>"
        f"<td>{item.artifact_bytes / 2**20:.2f}</td><td>{count}</td></tr>"
    )


def gallery_card(item: GalleryItem) -> str:
    """Show a fixed sample with its label and every model variant's prediction."""
    predictions: list[str] = []
    changed = len({prediction.label for prediction in item.predictions}) > 1
    for prediction in item.predictions:
        color = "good" if prediction.correct else "bad"
        predictions.append(
            f'<tr><td>{prediction.method.value}</td><td class="{color}">'
            f"{escape(prediction.name)}</td>"
            f"<td>{prediction.confidence * 100:.1f}%</td></tr>"
        )
    return (
        f'<article class="card" data-changed="{str(changed).lower()}">'
        f'<img loading="lazy" src="{escape(item.image.as_posix())}" '
        f'alt="{escape(item.truth)}">'
        f"<p><strong>Label: {escape(item.truth)}</strong></p>"
        f"<table>{''.join(predictions)}</table></article>"
    )


def model_section(model: ModelReport) -> str:
    """Keep summary metrics and bounded visual examples together."""
    rows = "".join(variant_row(item) for item in model.variants)
    gallery = "".join(gallery_card(item) for item in model.gallery)
    scaling = (
        "<p>SmoothQuant uses opt-in ONNX Runtime integer scaling. "
        "It changes quantization rounding and requires ORT-specific operators.</p>"
        if model.ort_integer_scaling
        else ""
    )
    return (
        f'<section data-model="{model.model.value}"><h2>{model.model.value}</h2>'
        f"<p>{escape(model.weights)} · Export max absolute error: "
        f"{model.export_max_abs_error:.6g}</p>{scaling}"
        '<div class="scroll"><table><thead><tr><th>Artifact / variant</th>'
        "<th>Top-1</th><th>Top-5</th><th>FP32 agreement</th><th>Logit MSE</th>"
        "<th>Max error</th><th>Latency med / p95 ms</th>"
        "<th>MiB</th><th>Quantized nodes</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
        f"<details><summary>Preprocessing</summary><pre>{escape(model.preprocessing)}</pre></details>"
        "<h3>Same held-out images, before and after quantization</h3>"
        f'<div class="gallery">{gallery}</div></section>'
    )


def render_html(report: SuiteReport) -> str:
    """Return a self-contained interactive report with relative artifact links."""
    options = "".join(
        f'<option value="{item.model.value}">{item.model.value}</option>'
        for item in report.models
    )
    sections = "".join(model_section(item) for item in report.models)
    config = report.config
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantSmith — pretrained models</title><style>{STYLE}</style></head><body>
<h1>PyTorch → ONNX → QuantSmith</h1>
<p>Real pretrained models on real images. {config.calibration_samples} training-split
images for calibration; {config.evaluation_samples} disjoint validation images
for evaluation.
Top-1 and top-5 use the full 1,000-class ImageNet output space.</p>
<p class="note">Imagenette 160px, seed {config.seed}. This subset is not the ImageNet
validation benchmark. Every variant uses its model's exact pretrained preprocessing.
Gallery images are the first selected evaluation samples, not chosen by outcome.</p>
<p class="note">Latency: CPUExecutionProvider, batch 1, {config.threads} thread(s),
{config.warmup} warmups and {config.benchmark_runs} timed runs.
ORT optimizations enabled;
preprocessing, export, and session startup excluded. Adapter validation is included.
Artifact sizes are measured on disk, not runtime RAM. Quantized constant weights
use integer storage; unquantized tensors remain floating point. SmoothQuant is
applied only to eligible ungrouped operators.
Softmax scores in the gallery are not calibrated probabilities.</p>
<p><a href="results.json">Full JSON results</a> ·
<a href="summary.md">Markdown summary</a></p>
<label>Model <select id="model"><option value="all">All models</option>
{options}</select></label>
<label><input id="changes" type="checkbox">
Show only changed predictions in gallery</label>
{sections}<p class="note">Generated {report.created_at.isoformat()} ·
{escape(report.environment.platform)} · Torch {escape(report.environment.torch)} ·
ORT {escape(report.environment.onnxruntime)}</p>
<script>
function update() {{
 const model = document.getElementById('model').value;
 const changes = document.getElementById('changes').checked;
 document.querySelectorAll('section[data-model]').forEach(s => {{
   s.hidden = model !== 'all' && s.dataset.model !== model;
 }});
 document.querySelectorAll('.card').forEach(c => {{
   c.hidden = changes && c.dataset.changed !== 'true';
 }});
}}
document.getElementById('model').addEventListener('change', update);
document.getElementById('changes').addEventListener('change', update);
</script></body></html>"""


def render_markdown(report: SuiteReport) -> str:
    """Produce a compact shareable comparison with complete measurement context."""
    lines = [
        "# Measured pretrained-model demonstration",
        "",
        f"{report.config.calibration_samples} calibration images (train), "
        f"{report.config.evaluation_samples} held-out images (val), Imagenette 160px; "
        f"seed {report.config.seed}. Accuracy uses all 1,000 output classes.",
        "",
        "| Model | Variant | Top-1 | Top-5 | FP32 agreement | "
        "MSE vs ONNX | Median ms | MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in report.models:
        for item in model.variants:
            metric = item.metrics
            lines.append(
                f"| {model.model.value} | {item.method.value} | "
                f"{metric.top1_accuracy:.2%} | "
                f"{metric.top5_accuracy:.2%} | {metric.agreement_with_onnx:.2%} | "
                f"{metric.mse_vs_onnx:.6g} | {item.latency.median_ms:.2f} | "
                f"{item.artifact_bytes / 2**20:.2f} |"
            )
    for model in report.models:
        if model.ort_integer_scaling:
            lines.extend(
                [
                    "",
                    f"{model.model.value} SmoothQuant uses opt-in ORT integer scaling; "
                    "rounding changes and the graph requires ORT-specific operators.",
                ]
            )
    lines.extend(
        [
            "",
            f"CPU batch 1, {report.config.threads} thread(s), "
            f"{report.config.warmup} warmups, "
            f"{report.config.benchmark_runs} timed runs. "
            "ORT optimizations enabled; preprocessing/startup excluded. "
            "Timings include adapter validation.",
            "",
            "Quantized constant weights use integer storage. MiB measures file size; "
            "runtime RAM and speedup depend on execution-provider optimizations.",
            "",
            f"Environment: {report.environment.model_dump_json()}",
            "",
        ]
    )
    return "\n".join(lines)


def save_report(report: SuiteReport) -> None:
    """Write portable human-readable and machine-readable outputs together."""
    output = report.config.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(report.model_dump_json(indent=2))
    (output / "index.html").write_text(render_html(report))
    (output / "summary.md").write_text(render_markdown(report))


class SelectionManifest(BaseModel):
    """Persist relative file identities so every metric can be reproduced."""

    seed: int = Field()
    calibration: tuple[ImageRecord, ...] = Field()
    evaluation: tuple[ImageRecord, ...] = Field()


def save_selection(
    root: Path,
    directory: Path,
    calibration: tuple[ImageRecord, ...],
    evaluation: tuple[ImageRecord, ...],
    seed: int,
) -> None:
    """Store split-relative paths rather than machine-specific absolute locations."""

    def relative(record: ImageRecord) -> ImageRecord:
        """Preserve labels while removing the data-cache prefix."""
        return ImageRecord(
            path=record.path.relative_to(root),
            label=record.label,
            class_name=record.class_name,
        )

    manifest = SelectionManifest(
        seed=seed,
        calibration=tuple(relative(item) for item in calibration),
        evaluation=tuple(relative(item) for item in evaluation),
    )
    (directory / "selection.json").write_text(manifest.model_dump_json(indent=2))


def artifact_path(directory: Path, method: Method) -> Path:
    """Distinguish the original Torch checkpoint from exported inference graphs."""
    return directory / (
        "torch_fp32.pt" if method is Method.TORCH else f"{method.value}.onnx"
    )


def measured_variants(
    runners: Mapping[Method, Runner],
    metrics: Mapping[Method, Metrics],
    coverage: Mapping[Method, Coverage],
    directory: Path,
    example: FloatArray,
    config: DemoConfig,
) -> tuple[VariantReport, ...]:
    """Pair task metrics with warmed inference and exact saved artifacts."""
    return tuple(
        VariantReport(
            method=method,
            metrics=metrics[method],
            latency=benchmark(runner, example, config.warmup, config.benchmark_runs),
            artifact=artifact_path(directory, method).relative_to(config.output),
            artifact_bytes=artifact_path(directory, method).stat().st_size,
            coverage=coverage[method],
        )
        for method, runner in runners.items()
    )
