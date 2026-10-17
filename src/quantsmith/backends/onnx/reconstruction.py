"""Sequential ONNX reconstruction with an immutable floating-point reference."""

from collections.abc import Callable
from glob import escape

from onnx import ModelProto
from pydantic import TypeAdapter

from quantsmith.algorithms import Algorithm
from quantsmith.algorithms.shell import build_plan
from quantsmith.algorithms.smoothquant import SmoothQuant
from quantsmith.backends.onnx import admit, describe, lower, normalize
from quantsmith.backends.onnx.pipeline import quantize
from quantsmith.backends.onnx.stages import FinishTransform, TransformStage
from quantsmith.calibration import Evaluator, Samples
from quantsmith.config import (
    JointSmoothQuantConfig,
    NoSmoothing,
    QuantizationConfig,
    SmoothQuantConfig,
    joint_transform,
    supported,
)
from quantsmith.domain import (
    Absent,
    Encoding,
    FloatArray,
    Graph,
    IntArray,
    Node,
    PerChannel,
)
from quantsmith.pipeline import Pipeline, QuantizationResult
from quantsmith.plan import QuantizationPlan, QuantizeConstant, QuantizeInput
from quantsmith.reconstruction import Conv2d, Linear, Problem, Reconstructor, Solution
from quantsmith.reconstruction.replay import paired_replay
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)
from quantsmith.rules import ByName, Exclude, Rule, Rules
from quantsmith.runtime import OnnxEvaluator

_DEFAULT_CONFIG = QuantizationConfig()


def canonical_weight(
    node: Node, weight: FloatArray
) -> tuple[FloatArray, Callable[[IntArray], IntArray]]:
    """Return output-first weights and an inverse bound to the original layout."""
    transpose = node.op == "MatMul" or (
        node.op == "Gemm" and not node.attributes.get("transB", 0)
    )

    def restore(codes: IntArray) -> IntArray:
        """Restore the exact layout used by this consumer's original initializer."""
        return codes.T if transpose else codes

    return (weight.T if transpose else weight), restore


def input_layout(node: Node) -> Callable[[FloatArray], FloatArray]:
    """Bind Gemm's optional input transpose once at the operator boundary."""
    transpose = node.op == "Gemm" and bool(node.attributes.get("transA", 0))

    def transform(value: FloatArray) -> FloatArray:
        """Apply the bound contraction layout without changing data values."""
        return value.T if transpose else value

    return transform


def operator(node: Node) -> Linear | Conv2d:
    """Admit explicit supported operator geometry; reject unmodeled semantics."""
    match node.op:
        case "MatMul":
            return Linear()
        case "Gemm":
            if node.attributes.get("alpha", 1) != 1:
                raise ValueError("reconstruction currently requires Gemm alpha=1")
            return Linear()
        case "Conv":
            values = {
                key: node.attributes[key]
                for key in ("strides", "dilations", "pads")
                if key in node.attributes
            }
            values["groups"] = node.attributes.get("group", 1)
            return TypeAdapter(Conv2d).validate_python(values)
        case _:
            raise ValueError(f"unsupported reconstruction operator: {node.op}")


def problem_for(
    node: Node, graph: Graph, plan: QuantizationPlan
) -> tuple[Problem, Callable[[Solution], QuantizationPlan]]:
    """Bind canonicalization and export so the inverse cannot drift from the input."""
    match plan.operations:
        case (
            QuantizeInput(index=0, encoding=activation),
            QuantizeInput(index=1, encoding=weight_encoding),
        ):
            pass
        case _:
            raise ValueError(
                "reconstruction initialization requires activation and weight encodings"
            )
    weight, restore = canonical_weight(node, graph.weights[node.inputs[1]])
    problem = Problem(
        weight=weight,
        weight_encoding=Encoding(
            scale=weight_encoding.scale,
            zero_point=weight_encoding.zero_point,
            granularity=PerChannel(axis=0),
        ),
        activation_encoding=activation,
        operator=operator(node),
    )

    def export(solution: Solution) -> QuantizationPlan:
        """Validate method output and restore this consumer's original layout."""
        if solution.codes.shape != weight.shape:
            raise ValueError("reconstructor returned the wrong weight shape")
        return QuantizationPlan(
            operations=(
                QuantizeInput(
                    node=node.name, index=0, encoding=solution.activation_encoding
                ),
                QuantizeConstant(
                    node=node.name,
                    index=1,
                    values=restore(solution.codes),
                    encoding=weight_encoding,
                ),
            )
        )

    return problem, export


def initial_plan(
    node: Node,
    graph: Graph,
    evaluator: Evaluator,
    samples: Samples,
    config: QuantizationConfig,
) -> Result[QuantizationPlan, QuantSmithError]:
    """Calibrate only the selected consumer on its current graph revision."""
    algorithm = config.algorithm()
    rules: Rules[Algorithm] = Rules(
        default=Exclude(reason="outside reconstruction unit"),
        overrides=(
            Rule(selector=ByName(pattern=escape(node.name)), decision=algorithm),
        ),
    )
    return build_plan(
        graph, evaluator, samples, rules, histogram_bins=config.histogram_bins
    )


def reconstruct_node(
    node: Node,
    graph: Graph,
    reference: Evaluator,
    candidate: Evaluator,
    samples: Samples,
    method: Reconstructor,
    config: QuantizationConfig,
) -> Result[QuantizationPlan, QuantSmithError]:
    """Initialize grids, reconstruct from paired inputs, and encode exact decisions."""
    match initial_plan(node, graph, candidate, samples, config):
        case Err() as error:
            return error
        case Ok(initial):
            pass
    match validate(node.name, lambda: problem_for(node, graph, initial)):
        case Err() as error:
            return error
        case Ok(payload):
            problem, export = payload
    replay = paired_replay(
        reference, candidate, samples, node.inputs[0], input_layout(node)
    )
    match method.reconstruct(problem, replay):
        case Err() as error:
            return error
        case Ok(solution):
            return validate(node.name, lambda: export(solution))


def reconstruct_unit(
    model: ModelProto,
    node: Node,
    reference: Evaluator,
    samples: Samples,
    method: Reconstructor,
    config: QuantizationConfig,
) -> Result[QuantizationResult, QuantSmithError]:
    """Lower one completed unit before any subsequent calibration takes place."""
    proto = next(value for value in model.graph.node if value.name == node.name)
    # String attributes are outside the domain graph's numeric attribute schema.
    if any(
        attr.name == "auto_pad" and attr.s not in (b"", b"NOTSET")
        for attr in proto.attribute
    ):
        return failure(
            FailureKind.UNSUPPORTED,
            node.name,
            "reconstruction requires explicit Conv padding",
        )
    match describe(model):
        case Err() as error:
            return error
        case Ok(graph):
            pass
    match reconstruct_node(
        node, graph, reference, OnnxEvaluator(model), samples, method, config
    ):
        case Err() as error:
            return error
        case Ok(plan):
            pass
    match lower(model, plan):
        case Err() as error:
            return error
        case Ok(candidate):
            return Ok(
                QuantizationResult(
                    model=candidate,
                    plans=(plan,),
                    execution=Absent(
                        reason="reconstruction did not audit integer execution"
                    ),
                )
            )


def _reconstruct(
    model: ModelProto,
    samples: Samples,
    rules: Rules[Reconstructor],
    *,
    config: QuantizationConfig = _DEFAULT_CONFIG,
) -> Result[QuantizationResult, QuantSmithError]:
    """Reconstruct selected operators in graph order, lowering after each unit.

    Supply a floating-point graph (optionally already smoothed). Unsupported
    selected layouts fail explicitly. Exclusions remain FP32. The caller owns
    calibration limits and replay ordering. No dataset or activation cache is kept.
    """
    match admit("reconstruction admission", lambda: normalize(model)):
        case Err() as error:
            return error
        case Ok(current):
            pass
    match describe(current):
        case Err() as error:
            return error
        case Ok(original):
            pass
    reference = OnnxEvaluator(current)
    plans: list[QuantizationPlan] = []
    excluded: list[str] = []
    for node in original.nodes:
        match rules.resolve(node, original):
            case Exclude():
                excluded.append(node.name)
                continue
            case method:
                pass
        match reconstruct_unit(current, node, reference, samples, method, config):
            case Err() as error:
                return error
            case Ok(result):
                current = result.model
                plans.extend(result.plans)
    if excluded:
        plans.append(QuantizationPlan(excluded=tuple(excluded)))
    return Ok(
        QuantizationResult(
            model=current,
            plans=tuple(plans),
            execution=Absent(reason="reconstruction did not audit integer execution"),
        )
    )


def reconstruct(
    model: ModelProto,
    samples: Samples,
    rules: Rules[Reconstructor],
    *,
    config: QuantizationConfig = _DEFAULT_CONFIG,
) -> Result[QuantizationResult, QuantSmithError]:
    """Optionally smooth, then reconstruct with fresh per-revision calibration.

    The input graph is not mutated. Selected unsupported layouts fail explicitly;
    rules control exclusions. Replay must be deterministic and caller-bounded.
    Each returned plan applies to the preceding plan's lowered graph revision.
    """
    match config.smoothing:
        case NoSmoothing():
            return _reconstruct(model, samples, rules, config=config)
        case SmoothQuantConfig(alpha=alpha, folding=folding):
            transform = TransformStage(supported(SmoothQuant(alpha=alpha)), folding)
        case JointSmoothQuantConfig(alpha=alpha, channels=channels):
            transform = joint_transform(alpha, channels)
    match describe(model):
        case Err() as error:
            return error
        case Ok(original):
            pass
    # Resolve user selectors before smoothing adds implementation-detail nodes.
    resolved: Rules[Reconstructor] = Rules(
        default=Exclude(reason="inserted smoothing operation"),
        overrides=tuple(
            Rule(
                selector=ByName(pattern=escape(node.name)),
                decision=rules.resolve(node, original),
            )
            for node in original.nodes
        ),
    )
    match quantize(
        model,
        samples,
        Pipeline(transform).then(FinishTransform()),
        histogram_bins=config.histogram_bins,
        max_channel_histogram_bytes=config.max_channel_histogram_bytes,
    ):
        case Err() as error:
            return error
        case Ok(smoothed):
            pass
    match _reconstruct(smoothed.model, samples, resolved, config=config):
        case Err() as error:
            return error
        case Ok(result):
            return Ok(
                QuantizationResult(
                    model=result.model,
                    plans=smoothed.plans + result.plans,
                    execution=result.execution,
                    folding=smoothed.folding + result.folding,
                )
            )
