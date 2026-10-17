"""Wine MLP: 13 normalized features → 32 → 16 → 3 cultivar logits: training, ONNX
export, calibration, and held-out predictions.

Read run() first, then export_onnx(), calibration_samples(), and
quantized_variants(). Replace the local model and data loader when applying this
workflow to your own task.
"""

from collections.abc import Iterable, Mapping
from contextlib import redirect_stdout
from datetime import UTC, datetime

import numpy as np
import onnx
import torch
from pydantic_settings import CliApp
from rich.console import Console
from rich.progress import Progress
from torch import nn

from examples._shared.cli import run_and_report
from examples._shared.environment import environment
from examples._shared.plans import coverage, save_plans
from examples._shared.recipes import Recipes
from examples._shared.schema import Coverage, Method
from examples._shared.training.data import wine
from examples._shared.training.measurement import Predictor, measure
from examples._shared.training.reporting import save, save_manifest, variant_reports
from examples._shared.training.schema import Dataset, Report, TaskKind, TaskReport
from examples._shared.training.training import eager, train
from examples.wine_mlp.model import WineMLP
from examples.wine_mlp.schema import Config
from quantsmith import Quantizer, Replay
from quantsmith.algorithms import Algorithm
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx import describe
from quantsmith.backends.onnx.stages import (
    QuantizationStage,
    TransformStage,
    VerifyStage,
)
from quantsmith.calibration import Percentile, Samples
from quantsmith.domain import FloatArray, Graph, InputArray, Node
from quantsmith.pipeline import FloatGraph, Pipeline, QuantizationResult, RuntimeScaling
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)
from quantsmith.rules import Exclude, Rule, Rules
from quantsmith.runtime import OnnxEvaluator


def run(config: Config) -> Result[Report, QuantSmithError]:
    """Execute the concrete model → ONNX → quantization → prediction workflow."""
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    config.output.mkdir(parents=True, exist_ok=True)
    console = Console(stderr=True, quiet=config.quiet)
    with console.status("Preparing Wine training, calibration, and held-out rows"):
        data = wine(config)
    config = config.model_copy(
        update={
            "calibration_samples": len(data.calibration.inputs),
            "evaluation_samples": len(data.evaluation.inputs),
        }
    )
    save_manifest(data, config.output)
    model = WineMLP()
    with Progress(console=console, disable=config.quiet) as progress:
        initial_loss, final_loss = train(model, data, config, progress)
        model.eval()
        torch.save(model.state_dict(), config.output / "torch_fp32.pt")
        example = data.calibration.inputs[:1]
        progress.console.print("Exporting 13-feature classification to FP32 ONNX")
        with console.status("Exporting FP32 ONNX and checking output parity"):
            exported = export_onnx(model, example, config)
        match exported:
            case Err() as error:
                return error
            case Ok(exported_model):
                fp32, export_error = exported_model
        match quantized_variants(fp32, data, config, progress):
            case Err() as error:
                return error
            case Ok(configured):
                predictors, scopes = configured

        def torch_predict(inputs: InputArray) -> Result[FloatArray, QuantSmithError]:
            """Use eager Torch as an independent export/quantization reference."""
            return validate("Torch prediction", lambda: eager(model, inputs))

        predictors[Method.TORCH] = torch_predict
        match measure(predictors, data, config, progress):
            case Err() as error:
                return error
            case Ok(measurements):
                scores, demonstrations = measurements
        match variant_reports(
            predictors, scores, scopes, example, config.output, config
        ):
            case Err() as error:
                return error
            case Ok(variants):
                pass
    task = TaskReport(
        task=TaskKind.WINE,
        architecture="Wine MLP: 13 normalized features → 32 → 16 → 3 cultivar logits",
        source=data.source,
        sha256=data.sha256,
        preprocessing=data.preprocessing,
        train_size=len(data.train.inputs),
        calibration_size=len(data.calibration.inputs),
        evaluation_size=len(data.evaluation.inputs),
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        export_max_abs_error=export_error,
        variants=variants,
        demonstrations=demonstrations,
    )
    report = Report(
        created_at=datetime.now(UTC),
        config=config,
        environment=environment(),
        tasks=(task,),
    )
    save(report)
    return Ok(report)


class ConstantWeights:
    """Select learned projections; input-dependent matrix products stay floating
    point.
    """

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Require a constant weight input and respect SmoothQuant's layout support."""
        if node.op not in ("MatMul", "Gemm") or len(node.inputs) < 2:
            return False
        if node.inputs[1] not in graph.weights:
            return False
        return True


def recipes(config: Config) -> Recipes[Pipeline[FloatGraph, QuantizationResult]]:
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
                selector=ConstantWeights(),
                decision=SmoothQuant(alpha=config.smoothquant_alpha),
            ),
        ),
    )
    return Recipes(
        minmax=Pipeline(QuantizationStage(minmax, config.outputs)).then(VerifyStage()),
        percentile=Pipeline(QuantizationStage(percentile, config.outputs)).then(
            VerifyStage()
        ),
        smoothquant=Pipeline(TransformStage(smoothing, RuntimeScaling()))
        .then(QuantizationStage(minmax, config.outputs))
        .then(VerifyStage()),
    )


def calibration_samples(data: Dataset) -> Samples:
    """Replay normalized float32 feature rows from the calibration split, preserving
    their dtype.
    """

    def samples() -> Iterable[Mapping[str, InputArray]]:
        """Yield batch-one inputs using the exact ONNX input name chosen below."""
        for index in range(len(data.calibration.inputs)):
            yield {"features": data.calibration.inputs[index : index + 1]}

    return samples


def onnx_predictor(model: onnx.ModelProto) -> Predictor:
    """Bind one CPU session for logits; propagate expected execution failures."""
    evaluator = OnnxEvaluator(model)

    def predict(inputs: InputArray) -> Result[FloatArray, QuantSmithError]:
        """Keep task predictions in native float32 arrays."""
        match evaluator.run({"features": inputs}, ("logits",)):
            case Err() as error:
                return error
            case Ok(outputs):
                return Ok(outputs["logits"])

    return predict


def export_onnx(
    model: nn.Module, example: InputArray, config: Config
) -> Result[tuple[onnx.ModelProto, float], QuantSmithError]:
    """Export the FP32 model and reject output changes before quantization starts."""
    path = config.output / "onnx_fp32.onnx"
    with (config.output / "export.log").open("w") as log, redirect_stdout(log):
        torch.onnx.export(
            model,
            (torch.from_numpy(example),),
            str(path),
            input_names=["features"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            external_data=False,
            optimize=True,
        )
    fp32 = onnx.load(path)
    onnx.checker.check_model(fp32)
    match onnx_predictor(fp32)(example):
        case Err() as error:
            return error
        case Ok(actual):
            pass
    expected = eager(model, example)
    if not np.allclose(actual, expected, rtol=1e-3, atol=1e-4):
        return failure(
            FailureKind.INVALID_DATA,
            "export parity",
            "FP32 ONNX export changed model outputs",
        )
    return Ok((fp32, float(np.max(np.abs(actual - expected)))))


def quantized_variants(
    fp32: onnx.ModelProto, data: Dataset, config: Config, progress: Progress
) -> Result[tuple[dict[Method, Predictor], dict[Method, Coverage]], QuantSmithError]:
    """Start every selected method from FP32; SmoothQuant recollects after rescaling."""
    match describe(fp32):
        case Err() as error:
            return error
        case Ok(graph):
            pass
    predictors = {Method.ONNX: onnx_predictor(fp32)}
    scopes = {
        Method.ONNX: Coverage(eligible_nodes=0),
        Method.TORCH: Coverage(eligible_nodes=0),
    }
    configurations = recipes(config)
    for recipe in config.methods:
        progress.console.print(f"{recipe.value}: calibrating dense feature projections")
        with progress.console.status(
            f"{recipe.value}: collecting statistics and lowering"
        ):
            outcome = (
                Quantizer.from_pipeline(
                    configurations.select(recipe), histogram_bins=config.histogram_bins
                )
                .quantize(fp32)
                .calibrate(Replay(calibration_samples(data)))
            )
        match outcome:
            case Err() as error:
                return error
            case Ok(result):
                pass
        method = recipe.variant()
        onnx.save(result.model, config.output / f"{method.value}.onnx")
        save_plans(config.output / f"{method.value}.plan.json", result.plans)
        predictors[method] = onnx_predictor(result.model)
        scopes[method] = coverage(graph, result.plans)
    return Ok((predictors, scopes))


def main() -> None:
    """Parse typed CLI controls and print the report path or its complete JSON."""
    config = CliApp.run(Config)
    run_and_report(config, run, output=config.output, json_output=config.json_output)


if __name__ == "__main__":
    main()
