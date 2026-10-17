"""MobileNetV2 on real Imagenette images: preprocessing, export, and quantization.

Read run() first. Model loading, ONNX export, input replay, and quantization choices
are written here so you can replace them for your own image-classification task.
"""

from collections.abc import Iterable, Mapping
from contextlib import redirect_stdout
from datetime import UTC, datetime

import numpy as np
import onnx
import torch
from pydantic import BaseModel, Field
from pydantic_settings import CliApp
from rich.console import Console
from rich.progress import Progress
from torchvision import models

from examples._shared.cli import run_and_report
from examples._shared.environment import environment
from examples._shared.plans import coverage, save_plans
from examples._shared.schema import Coverage, Method, QuantizationMethod
from examples._shared.vision.data import download_dataset, read_input, select_records
from examples._shared.vision.measurement import Runner, measure
from examples._shared.vision.models import (
    LoadedModel,
    OrtRunner,
    TorchImageTransform,
    TorchRunner,
    checked_categories,
)
from examples._shared.vision.reporting import (
    measured_variants,
    save_report,
    save_selection,
)
from examples._shared.vision.schema import (
    ImageRecord,
    ModelName,
    ModelReport,
    SuiteReport,
)
from examples.mobilenet_v2.schema import Config
from quantsmith.algorithms import Algorithm
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.calibration import Percentile, Samples
from quantsmith.domain import FloatArray, Graph, Node
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.rules import Exclude, Rule, Rules


def run(config: Config) -> Result[SuiteReport, QuantSmithError]:
    """Execute MobileNetV2 → FP32 ONNX → selected INT8 recipes → held-out
    predictions.
    """
    config.output.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(config.cache / "torch"))
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    console = Console(stderr=True, quiet=config.quiet)
    with console.status("Loading MobileNetV2 ImageNet-1K pretrained weights"):
        loaded = load_model()
    with console.status("Preparing Imagenette train and val images"):
        dataset = download_dataset(config.cache)
    calibration = select_records(
        dataset / "train", loaded.categories, config.calibration_samples, config.seed
    )
    held_out = select_records(
        dataset / "val", loaded.categories, config.evaluation_samples, config.seed
    )
    save_selection(dataset, config.output, calibration, held_out, config.seed)
    example = read_input(calibration[0], loaded.preprocess)
    torch.save(loaded.model.state_dict(), config.output / "torch_fp32.pt")
    console.print("Exporting MobileNetV2 with its matching image preprocessing")
    with console.status("Exporting FP32 ONNX and checking output parity"):
        fp32, export_error = export_onnx(loaded, example, config)
    with Progress(console=console, disable=config.quiet) as progress:
        match quantized_variants(
            fp32, calibration_samples(calibration, loaded), config, progress
        ):
            case Err() as error:
                return error
            case Ok(configured):
                runners, scopes = configured
        runners[Method.TORCH] = TorchRunner(loaded.model)
        metrics, gallery = measure(
            runners,
            held_out,
            loaded.preprocess,
            loaded.categories,
            config.output,
            config,
            progress,
        )
        variants = measured_variants(
            runners, metrics, scopes, config.output, example, config
        )
    model_report = ModelReport(
        model=ModelName.MOBILENET_V2,
        weights=loaded.weights,
        preprocessing=repr(loaded.transform),
        export_max_abs_error=export_error,
        variants=variants,
        gallery=gallery,
    )
    report = SuiteReport(
        created_at=datetime.now(UTC),
        config=config,
        environment=environment(),
        models=(model_report,),
    )
    save_report(report)
    return Ok(report)


class ConstantWeights(BaseModel, frozen=True):
    """Select learned projections; input-dependent matrix products stay floating
    point.
    """

    smoothable: bool = Field(default=False)

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Require a constant weight input and respect SmoothQuant's layout support."""
        if node.op not in ("Conv", "MatMul", "Gemm") or len(node.inputs) < 2:
            return False
        if node.inputs[1] not in graph.weights:
            return False
        return not (
            self.smoothable
            and node.op == "Conv"
            and node.attributes.get("group", 1) != 1
        )


def recipes(config: Config) -> dict[QuantizationMethod, tuple[Rules[Algorithm], ...]]:
    """Define three QuantSmith configurations for independent adaptation."""
    minmax: Rules[Algorithm] = Rules(
        default=Exclude(reason="outside constant-weight operators"),
        overrides=(Rule(selector=ConstantWeights(), decision=StaticW8A8()),),
    )
    percentile: Rules[Algorithm] = Rules(
        default=Exclude(reason="outside constant-weight operators"),
        overrides=(
            Rule(
                selector=ConstantWeights(),
                decision=StaticW8A8(
                    calibration=Percentile(percentile=config.percentile)
                ),
            ),
        ),
    )
    smoothing: Rules[Algorithm] = Rules(
        default=Exclude(reason="outside SmoothQuant-supported operators"),
        overrides=(
            Rule(
                selector=ConstantWeights(smoothable=True),
                decision=SmoothQuant(alpha=config.smoothquant_alpha),
            ),
        ),
    )
    return {
        QuantizationMethod.MINMAX: (minmax,),
        QuantizationMethod.PERCENTILE: (percentile,),
        QuantizationMethod.SMOOTHQUANT: (smoothing, minmax),
    }


def load_model() -> LoadedModel:
    """Load MobileNetV2 ImageNet-1K pretrained weights with the checkpoint's matching
    RGB preprocessing.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    model = models.mobilenet_v2(weights=weights).eval()
    return LoadedModel(
        model,
        TorchImageTransform(weights.transforms()),
        checked_categories(weights),
        str(weights),
    )


def calibration_samples(
    records: tuple[ImageRecord, ...], loaded: LoadedModel
) -> Samples:
    """Re-read only train images; every pass uses identical weight preprocessing."""

    def samples() -> Iterable[Mapping[str, FloatArray]]:
        """Yield float32 RGB batches under the exported ONNX input name."""
        for record in records:
            yield {"images": read_input(record, loaded.preprocess)}

    return samples


def export_onnx(
    loaded: LoadedModel, example: FloatArray, config: Config
) -> tuple[onnx.ModelProto, float]:
    """Export fixed batch-one RGB inference and check parity before measuring INT8."""
    path = config.output / "onnx_fp32.onnx"
    with (config.output / "export.log").open("w") as log, redirect_stdout(log):
        torch.onnx.export(
            loaded.model,
            (torch.from_numpy(example),),
            str(path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            external_data=False,
            optimize=True,
        )
    fp32 = onnx.load(path)
    onnx.checker.check_model(fp32)
    expected = TorchRunner(loaded.model)(example)
    actual = OrtRunner(path, config.threads)(example)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    return fp32, float(np.max(np.abs(actual.astype(np.float64) - expected)))


def quantized_variants(
    fp32: onnx.ModelProto, source: Samples, config: Config, progress: Progress
) -> Result[tuple[dict[Method, Runner], dict[Method, Coverage]], QuantSmithError]:
    """Compare independently calibrated recipes without quantizing a previous INT8
    graph.
    """
    match describe(fp32):
        case Err() as error:
            return error
        case Ok(graph):
            pass
    runners: dict[Method, Runner] = {
        Method.ONNX: OrtRunner(config.output / "onnx_fp32.onnx", config.threads)
    }
    scopes = {
        Method.ONNX: Coverage(eligible_nodes=0),
        Method.TORCH: Coverage(eligible_nodes=0),
    }
    configurations = recipes(config)
    for recipe in config.methods:
        progress.console.print(
            f"{recipe.value}: calibrating depthwise convolutions and dense layers"
        )
        with progress.console.status(
            f"{recipe.value}: collecting statistics and lowering"
        ):
            outcome = quantize(
                fp32,
                source,
                configurations[recipe],
                histogram_bins=config.histogram_bins,
            )
        match outcome:
            case Err() as error:
                return error
            case Ok(result):
                pass
        method = recipe.variant()
        path = config.output / f"{method.value}.onnx"
        onnx.save(result.model, path)
        save_plans(config.output / f"{method.value}.plan.json", result.plans)
        runners[method] = OrtRunner(path, config.threads)
        scopes[method] = coverage(graph, result.plans)
    return Ok((runners, scopes))


def main() -> None:
    """Parse experiment controls and print a portable report path or JSON result."""
    config = CliApp.run(Config)
    run_and_report(config, run, output=config.output, json_output=config.json_output)


if __name__ == "__main__":
    main()
