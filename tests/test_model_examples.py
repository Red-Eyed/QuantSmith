"""Offline regressions for real Torch export, Transformer causality, and metrics."""

from collections.abc import Callable, Iterator, Mapping
from typing import Literal

import numpy as np
import onnx
import pytest
import torch
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliApp
from torch import nn
from torchvision import models

from examples._shared.schema import QuantizationMethod
from examples._shared.training.data import text_windows, wine
from examples._shared.training.measurement import Totals
from examples._shared.training.schema import Config, TaskKind
from examples._shared.training.training import eager
from examples._shared.vision.schema import DemoConfig, ModelName
from examples.character_rnn import main as rnn_example
from examples.character_rnn.model import CharacterRNN
from examples.efficientnet_b0 import main as efficientnet_example
from examples.mobilenet_v2 import main as mobilenet_example
from examples.projection import main as projection_example
from examples.projection.model import example_model
from examples.resnet18 import main as resnet_example
from examples.transformer import main as transformer_example
from examples.transformer.model import CharacterTransformer
from examples.windowed_transformer import main as windowed_example
from examples.windowed_transformer.model import WindowedCharacterTransformer
from examples.wine_mlp import main as wine_example
from examples.wine_mlp.model import WineMLP
from quantsmith import Quantizer
from quantsmith.algorithms.smoothquant.joint.ranges import (
    ChannelExtrema,
    ChannelMSE,
    ChannelPercentile,
    ChannelSelection,
)
from quantsmith.backends.onnx import describe
from quantsmith.backends.onnx.integer_execution.activation_function_ranges import (
    activation_function_outputs,
)
from quantsmith.backends.onnx.integer_execution.audit import (
    ExecutionReport,
    compile_cpu,
)
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.backends.onnx.stages import FinishTransform
from quantsmith.config import (
    JointSmoothQuantConfig,
    QuantizationConfig,
    joint_transform,
)
from quantsmith.domain import Absent, FloatArray, InputArray
from quantsmith.pipeline import (
    FloatGraph,
    Folded,
    FunctionOutputs,
    Pipeline,
    ProducerOutputs,
    QuantizationResult,
)
from quantsmith.pipeline.policies import Outputs
from quantsmith.runtime import OnnxEvaluator
from tests.outcomes import expect_error, expect_ok

type ModelKind = TaskKind | ModelName | Literal["projection"]


class ExportCase(BaseModel):
    """Pair a real Torch architecture with its serialized graph and native input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: nn.Module = Field()
    graph: onnx.ModelProto = Field()
    inputs: InputArray = Field()
    kind: ModelKind = Field()
    input_name: str = Field()
    output_name: str = Field()


@pytest.fixture(scope="module", autouse=True)
def cpu_threads() -> None:
    """Keep CPU tests reproducible and avoid native thread oversubscription."""
    torch.set_num_threads(1)


@pytest.fixture(scope="module", params=[*TaskKind, *ModelName, "projection"])
def export_case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ExportCase:
    """Export seven Torch architectures offline and the built-in ONNX projection."""
    kind: ModelKind = request.param
    torch.manual_seed(7)
    builders: dict[TaskKind, Callable[[int], nn.Module]] = {
        TaskKind.TEXT: CharacterRNN,
        TaskKind.TRANSFORMER: CharacterTransformer,
        TaskKind.WINDOWED: WindowedCharacterTransformer,
    }
    match kind:
        case "projection":
            graph = example_model()
            model = nn.Linear(8, 4, bias=False).eval()
            weight = np.asarray(
                onnx.numpy_helper.to_array(graph.graph.initializer[0]), dtype=np.float32
            )
            with torch.no_grad():
                model.weight.copy_(torch.from_numpy(weight.T.copy()))
            return ExportCase(
                model=model,
                graph=graph,
                inputs=np.ones((1, 8), dtype=np.float32),
                kind=kind,
                input_name="x",
                output_name="y",
            )
        case TaskKind.WINE:
            model = WineMLP().eval()
            inputs: InputArray = np.ones((1, 13), dtype=np.float32)
        case TaskKind() as task:
            model = builders[task](8).eval()
            inputs = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
            if task in (TaskKind.TRANSFORMER, TaskKind.WINDOWED):
                inputs = np.tile(inputs, (1, 2))
        case ModelName.RESNET18:
            model = models.resnet18(weights=None).eval()
            inputs = np.ones((1, 3, 32, 32), dtype=np.float32)
        case ModelName.MOBILENET_V2:
            model = models.mobilenet_v2(weights=None).eval()
            inputs = np.ones((1, 3, 32, 32), dtype=np.float32)
        case ModelName.EFFICIENTNET_B0:
            model = models.efficientnet_b0(weights=None).eval()
            inputs = np.ones((1, 3, 32, 32), dtype=np.float32)
    destination = tmp_path_factory.mktemp(str(kind)) / "fp32.onnx"
    torch.onnx.export(
        model,
        (torch.from_numpy(inputs),),
        str(destination),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    graph = onnx.load(destination)
    return ExportCase(
        model=model,
        graph=graph,
        inputs=inputs,
        kind=kind,
        input_name="input",
        output_name="logits",
    )


def example_stages(
    kind: ModelKind, method: QuantizationMethod, outputs: Outputs
) -> Pipeline[FloatGraph, QuantizationResult]:
    """Exercise explicitly declared example programs, including output policy types."""
    match kind:
        case TaskKind.TRANSFORMER:
            return transformer_example.recipes(
                transformer_example.Config(histogram_bins=64, outputs=outputs)
            ).select(method)
        case TaskKind.WINDOWED:
            return windowed_example.recipes(
                windowed_example.Config(histogram_bins=64, outputs=outputs)
            ).select(method)
        case TaskKind.TEXT:
            return rnn_example.recipes(
                rnn_example.Config(histogram_bins=64, outputs=outputs)
            ).select(method)
        case TaskKind.WINE:
            return wine_example.recipes(
                wine_example.Config(histogram_bins=64, outputs=outputs)
            ).select(method)
        case ModelName.RESNET18:
            return resnet_example.recipes(
                resnet_example.Config(histogram_bins=64, outputs=outputs)
            ).select(method)
        case ModelName.MOBILENET_V2:
            return mobilenet_example.recipes(
                mobilenet_example.Config(histogram_bins=64, outputs=outputs)
            ).select(method)
        case ModelName.EFFICIENTNET_B0:
            return efficientnet_example.recipes(
                efficientnet_example.Config(histogram_bins=64, outputs=outputs)
            ).select(method)
        case "projection":
            return (
                projection_example.recipes(
                    projection_example.Config(histogram_bins=64, outputs=outputs)
                )
                .select(method)
                .pipeline()
            )


@pytest.mark.parametrize("method", list(QuantizationMethod))
@pytest.mark.parametrize("activation_function_aware", [False, True])
def test_torch_export_quantization_roundtrip(
    export_case: ExportCase,
    method: QuantizationMethod,
    activation_function_aware: bool,
) -> None:
    """Quantize independent recipes while preserving shape and source ownership."""
    before = export_case.graph.SerializeToString()

    def source() -> Iterator[Mapping[str, InputArray]]:
        """Replay native feature/token inputs without conversion to another dtype."""
        yield {export_case.input_name: export_case.inputs}

    result = expect_ok(
        quantize(
            export_case.graph,
            source,
            example_stages(
                export_case.kind,
                method,
                FunctionOutputs() if activation_function_aware else ProducerOutputs(),
            ),
            histogram_bins=64,
        )
    )
    lowered = result.model
    onnx.checker.check_model(lowered)
    actual = expect_ok(
        OnnxEvaluator(lowered).run(
            {export_case.input_name: export_case.inputs}, (export_case.output_name,)
        )
    )[export_case.output_name]
    expected = eager(export_case.model, export_case.inputs)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert float(np.mean((actual - expected) ** 2)) < 0.1
    assert export_case.graph.SerializeToString() == before
    if export_case.kind not in (TaskKind.TRANSFORMER, TaskKind.WINDOWED):
        return
    assert_transformer_execution(export_case, result.execution, lowered)
    if activation_function_aware:
        baseline = expect_ok(
            quantize(
                export_case.graph,
                source,
                example_stages(export_case.kind, method, ProducerOutputs()),
                histogram_bins=64,
            )
        )
        assert lowered.SerializeToString() == baseline.model.SerializeToString()
    changed = export_case.inputs.copy()
    changed[:, 4:] = 0
    causal = expect_ok(
        OnnxEvaluator(lowered).run(
            {export_case.input_name: changed}, (export_case.output_name,)
        )
    )[export_case.output_name]
    np.testing.assert_allclose(actual[:, :4], causal[:, :4], atol=1e-6, rtol=0)


def assert_transformer_execution(
    case: ExportCase, execution: ExecutionReport | Absent, lowered: onnx.ModelProto
) -> None:
    """Verify integer learned projections and float dynamic attention/function paths."""
    assert isinstance(execution, ExecutionReport)
    graph = expect_ok(describe(case.graph))
    learned = {
        node.name
        for node in graph.nodes
        if transformer_example.ConstantWeights().matches(node, graph)
    }
    dynamic = {
        node.name
        for node in graph.nodes
        if node.op == "MatMul" and node.inputs[1] not in graph.weights
    }
    assert len(learned) == 15
    assert len(dynamic) == 4
    assert {item.node for item in execution.integer_nodes} == learned
    assert all(
        item.executed_operator == "QLinearMatMul" for item in execution.integer_nodes
    )
    assert dynamic <= {item.node for item in execution.unverified_nodes}
    assert not activation_function_outputs(case.graph)
    compiled = compile_cpu(lowered)
    operators = [node.op_type for node in compiled.graph.node]
    assert operators.count("MatMul") + operators.count("FusedMatMul") == len(dynamic)
    assert operators.count("Softmax") == 2
    assert operators.count("Sigmoid") + operators.count("QuickGelu") == 2


@pytest.mark.parametrize(
    "builder", [CharacterTransformer, WindowedCharacterTransformer]
)
def test_transformer_causality(builder: Callable[[int], nn.Module]) -> None:
    """Future token changes must not change earlier next-token predictions."""
    model = builder(8).eval()
    first = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    second = first.copy()
    second[:, 4:] = 0
    np.testing.assert_allclose(
        eager(model, first)[:, :4], eager(model, second)[:, :4], atol=1e-6
    )


def test_dynamic_attention_edges_remain_float(export_case: ExportCase) -> None:
    """Constant-weight recipes cannot pretend to quantize dynamic QK/AV products."""
    graph = expect_ok(describe(export_case.graph))
    dynamic = [
        node
        for node in graph.nodes
        if node.op == "MatMul" and node.inputs[1] not in graph.weights
    ]
    if export_case.kind in (TaskKind.TRANSFORMER, TaskKind.WINDOWED):
        assert len(dynamic) >= 4
    for node in dynamic:
        assert not transformer_example.ConstantWeights().matches(node, graph)


def test_wine_split_and_normalization() -> None:
    """Prove training-only normalization and disjoint real dataset identities."""
    data = wine(Config())
    train, calibration, evaluation = (
        set(split.identities)
        for split in (data.train, data.calibration, data.evaluation)
    )
    assert not (train & calibration or train & evaluation or calibration & evaluation)
    assert len(train | calibration | evaluation) == 178
    np.testing.assert_allclose(data.train.inputs.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(data.train.inputs.std(axis=0), 1, atol=1e-5)
    assert data.calibration.identities == wine(Config()).calibration.identities


def test_text_window_boundaries() -> None:
    """Input and shifted targets must stay inside their assigned source span."""
    tokens = np.arange(200, dtype=np.int64)
    split = expect_ok(text_windows(tokens, 100, 200, 8, 8))
    assert int(split.inputs.min()) >= 100
    assert int(split.targets.max()) < 200
    np.testing.assert_array_equal(split.targets, split.inputs + 1)
    expect_error(text_windows(tokens, 100, 200, 100, 8), "available")


def test_token_metrics() -> None:
    """Count each token separately and verify exact entropy on uniform logits."""
    logits: FloatArray = np.zeros((1, 2, 2), dtype=np.float32)
    totals = Totals()
    totals.update(logits, logits, logits, np.asarray([[0, 1]], dtype=np.int64))
    scores = totals.finish()
    assert scores.observations == 2
    assert scores.accuracy == 0.5
    assert scores.cross_entropy == pytest.approx(np.log(2))
    assert scores.perplexity == pytest.approx(2)
    assert scores.mse_vs_onnx == 0


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("percentile", [QuantizationMethod.PERCENTILE]),
        (
            "minmax,percentile",
            [QuantizationMethod.MINMAX, QuantizationMethod.PERCENTILE],
        ),
    ],
)
def test_typed_task_cli(selection: str, expected: list[QuantizationMethod]) -> None:
    """Select methods while each example retains its explicit model identity."""
    config = CliApp.run(
        transformer_example.Config, cli_args=["--methods", selection, "--quiet"]
    )
    assert config.tasks == (TaskKind.TRANSFORMER,)
    assert config.methods == expected
    assert config.quiet


@pytest.mark.parametrize("config_type", [Config, DemoConfig])
def test_cli_configuration_roundtrip(
    config_type: type[Config] | type[DemoConfig],
) -> None:
    """Saved reports must accept both the CLI alias and serialized Python field name."""
    config = CliApp.run(config_type, cli_args=["--json", "--quiet"])
    restored = config_type.model_validate_json(config.model_dump_json())
    assert restored == config
    assert restored.json_output


@pytest.mark.parametrize(
    "export_case", [TaskKind.TRANSFORMER, TaskKind.WINDOWED, *ModelName], indirect=True
)
@pytest.mark.parametrize(
    "channels",
    [ChannelExtrema(), ChannelPercentile(percentile=99.9), ChannelMSE(candidates=4)],
)
def test_joint_model_regions(
    export_case: ExportCase, channels: ChannelSelection
) -> None:
    """Preserve float parity and integer projection coverage after group folding."""
    data = ({export_case.input_name: export_case.inputs},)
    program = Pipeline(joint_transform(0.5, channels)).then(FinishTransform())
    transformed = expect_ok(
        Quantizer.from_pipeline(program, histogram_bins=64)
        .quantize(export_case.graph)
        .calibrate(data)
    )
    if export_case.kind in {TaskKind.TRANSFORMER, TaskKind.WINDOWED}:
        assert sum(isinstance(value, Folded) for value in transformed.folding) == 11
    original = expect_ok(
        OnnxEvaluator(export_case.graph).run(data[0], (export_case.output_name,))
    )
    folded = expect_ok(
        OnnxEvaluator(transformed.model).run(data[0], (export_case.output_name,))
    )
    np.testing.assert_allclose(
        folded[export_case.output_name],
        original[export_case.output_name],
        rtol=1e-5,
        atol=1e-5,
    )
    result = expect_ok(
        Quantizer.from_config(
            QuantizationConfig(
                smoothing=JointSmoothQuantConfig(channels=channels), histogram_bins=64
            )
        )
        .quantize(export_case.graph)
        .calibrate(data)
    )
    if export_case.kind in {TaskKind.TRANSFORMER, TaskKind.WINDOWED}:
        assert_transformer_execution(export_case, result.execution, result.model)
    else:
        assert isinstance(result.execution, ExecutionReport)
        assert result.execution.integer_nodes
    assert not any(node.op_type == "QLinearMul" for node in result.model.graph.node)


def test_output_selected_smoothing_real_exports(export_case: ExportCase) -> None:
    """Score real exported CNNs and Stackformers with unchanged public outputs."""
    from quantsmith.config import OutputSelectedSmoothingConfig

    config = QuantizationConfig(
        smoothing=OutputSelectedSmoothingConfig(
            candidates=(JointSmoothQuantConfig(alpha=0.5),)
        ),
        histogram_bins=64,
    )
    inputs = {export_case.input_name: export_case.inputs}
    before = export_case.graph.SerializeToString()
    result = expect_ok(
        Quantizer.from_config(config).quantize(export_case.graph).calibrate((inputs,))
    )
    assert export_case.graph.SerializeToString() == before
    assert not isinstance(result.selection, Absent)
    report = result.selection
    assert len(report.candidates) == 2
    assert report.outputs == (export_case.output_name,)
    assert min(row.mean_squared_error for row in report.candidates) == next(
        row.mean_squared_error
        for row in report.candidates
        if row.candidate == report.selected
    )
    assert not isinstance(result.execution, Absent)
    assert all(
        row.integer_nodes == len(result.execution.integer_nodes)
        for row in report.candidates
    )


@pytest.mark.parametrize("export_case", [TaskKind.TRANSFORMER], indirect=True)
def test_float_matmul_global_transformer_export(export_case: ExportCase) -> None:
    """Verify the opt-in global recipe keeps integer projections with float outputs."""
    from examples.transformer.schema import Config as TransformerConfig
    from quantsmith.pipeline import FloatMatMulOutputs
    from quantsmith.plan import FloatOutput

    config = TransformerConfig(matmul_outputs=FloatMatMulOutputs())
    program = transformer_example.recipes(config).minmax
    before = export_case.graph.SerializeToString()
    result = expect_ok(
        Quantizer.from_pipeline(program, histogram_bins=64)
        .quantize(export_case.graph)
        .calibrate(({export_case.input_name: export_case.inputs},))
    )
    assert export_case.graph.SerializeToString() == before
    assert isinstance(result.execution, ExecutionReport)
    assert len(result.execution.integer_nodes) == 15
    assert all(
        node.executed_operator == "MatMulIntegerToFloat"
        for node in result.execution.integer_nodes
    )
    assert (
        sum(
            isinstance(op, FloatOutput)
            for plan in result.plans
            for op in plan.operations
        )
        == 15
    )
    compiled = compile_cpu(result.model)
    assert (
        sum(node.op_type == "MatMulIntegerToFloat" for node in compiled.graph.node)
        == 15
    )
    assert not any(node.op_type == "DequantizeLinear" for node in compiled.graph.node)
