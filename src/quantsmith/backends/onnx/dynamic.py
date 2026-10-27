"""Runtime activation quantization and calibration-free ONNX projection stages."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

from quantsmith.algorithms.contracts import Algorithm, Needs, Statistics
from quantsmith.algorithms.dynamic import DynamicW8A8
from quantsmith.backends.onnx import describe, lower, validate_encoding
from quantsmith.backends.onnx.integer_execution.audit import UnverifiedNode
from quantsmith.domain import Encoding, Graph, Node
from quantsmith.pipeline.contracts import CalibrationContext
from quantsmith.pipeline.state import FloatGraph, QuantizedGraph
from quantsmith.plan import DynamicQuantizeGemm, DynamicQuantizeMatMul, QuantizationPlan
from quantsmith.reconstruction.affine import codes
from quantsmith.result import Err, FailureKind, Ok, QuantSmithError, Result, failure
from quantsmith.rules import Exclude, Rules


def _weight_inputs(
    model: ModelProto,
    node: NodeProto,
    encoding: Encoding,
    names: Callable[[str], str],
) -> tuple[str, str, str]:
    """Store consumer-local integer weights without rewriting shared constants."""
    constants = {tensor.name: tensor for tensor in model.graph.initializer}
    if node.input[1] not in constants:
        raise ValueError("dynamic MatMul requires constant FP32 weights")
    tensor = constants[node.input[1]]
    if tensor.data_type != TensorProto.FLOAT or len(tensor.dims) != 2:
        raise ValueError("dynamic MatMul requires rank-two FP32 weights")
    validate_encoding(model, node.input[1], encoding)
    weight = np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
    if not weight.size or not np.all(np.isfinite(weight)):
        raise ValueError("dynamic MatMul weights must be nonempty and finite")
    integer_weight, weight_scale, weight_zero = (
        names(node.name + suffix)
        for suffix in ("_weight_q", "_weight_scale", "_weight_zero")
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(codes(weight, encoding), integer_weight),
            numpy_helper.from_array(encoding.scale, weight_scale),
            numpy_helper.from_array(np.array(0, dtype=np.int8), weight_zero),
        ]
    )
    return integer_weight, weight_scale, weight_zero


def _require_float_activation(model: ModelProto, source: str) -> None:
    """Reject missing or non-FP32 activation metadata at the ONNX boundary."""
    for tensor in model.graph.initializer:
        if tensor.name == source:
            if tensor.data_type != TensorProto.FLOAT:
                raise ValueError("dynamic MatMul requires FP32 activations")
            return
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if value.name == source:
            if value.type.tensor_type.elem_type != TensorProto.FLOAT:
                raise ValueError("dynamic MatMul requires FP32 activations")
            return
    raise ValueError("dynamic MatMul requires FP32 activation metadata")


def lower_dynamic_matmul(
    model: ModelProto,
    node: NodeProto,
    operation: DynamicQuantizeMatMul,
    names: Callable[[str], str],
) -> list[NodeProto]:
    """Emit standard ONNX arithmetic with a source-named integer contraction.

    ORT may fuse the quantization, cast and scale multiplication. The serialized
    graph retains per-tensor UINT8 activation semantics without custom operators.
    """
    if (
        node.domain
        or node.op_type != "MatMul"
        or len(node.input) != 2
        or len(node.output) != 1
        or node.attribute
    ):
        raise ValueError("dynamic quantization requires a standard two-input MatMul")
    _require_float_activation(model, node.input[0])
    integer_weight, weight_scale, weight_zero = _weight_inputs(
        model, node, operation.weight_encoding, names
    )
    activation, scale, zero, integer, floating, combined = (
        names(node.name + suffix)
        for suffix in (
            "_input_q",
            "_input_scale",
            "_input_zero",
            "_int32",
            "_float",
            "_scale",
        )
    )
    output = node.output[0]
    quantizer = helper.make_node(
        "DynamicQuantizeLinear",
        [node.input[0]],
        [activation, scale, zero],
        name=names(node.name + "_dynamic_input"),
    )
    node.CopyFrom(
        helper.make_node(
            "MatMulInteger",
            [activation, integer_weight, zero, weight_zero],
            [integer],
            name=node.name,
        )
    )
    return [
        quantizer,
        node,
        helper.make_node(
            "Cast",
            [integer],
            [floating],
            name=names(node.name + "_cast"),
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Mul",
            [scale, weight_scale],
            [combined],
            name=names(node.name + "_scales"),
        ),
        helper.make_node(
            "Mul",
            [floating, combined],
            [output],
            name=names(node.name + "_restore"),
        ),
    ]


def _decision(
    method: Algorithm, node: Node, graph: Graph
) -> Result[QuantizationPlan | Exclude, QuantSmithError]:
    """Adapt built-in exclusions and enforce the calibration-free stage contract."""
    match method.requirements(node, graph):
        case Err(error) if (
            isinstance(method, DynamicW8A8) and error.kind == FailureKind.UNSUPPORTED
        ):
            return Ok(Exclude(reason=error.detail))
        case Err() as error:
            return error
        case Ok(needs):
            if needs != Needs():
                return failure(
                    FailureKind.CONFLICT,
                    node.name,
                    "dynamic stage requires an algorithm without calibration needs",
                )
    match method.plan(node, graph, Statistics(ranges={}, histograms={})):
        case Err() as error:
            return error
        case Ok(plan):
            if any(
                not isinstance(item, DynamicQuantizeMatMul | DynamicQuantizeGemm)
                for item in plan.operations
            ):
                return failure(
                    FailureKind.CONFLICT,
                    node.name,
                    "dynamic stage requires dynamic matrix operations",
                )
            return Ok(plan)


@dataclass(frozen=True)
class DynamicQuantizationStage:
    """Plan and lower dynamic projections without evaluating or reading samples.

    Rules select calibration-free algorithms or explicitly exclude nodes.
    Unsupported built-in layouts remain floating. Plugin failures propagate
    according to Algorithm's established Result/exception contract.
    """

    rules: Rules[Algorithm] = field(
        default_factory=lambda: Rules(default=DynamicW8A8())
    )

    def run(
        self, value: FloatGraph, context: CalibrationContext
    ) -> Result[QuantizedGraph, QuantSmithError]:
        """Preserve graph revision ownership and reject unresolved runtime smoothing."""
        if value.scaling:
            return failure(
                FailureKind.CONFLICT,
                "dynamic quantization",
                "fold smoothing scales before dynamic quantization",
            )
        match describe(value.revision.model):
            case Err() as error:
                return error
            case Ok(graph):
                pass
        plan = QuantizationPlan()
        exclusions: list[UnverifiedNode] = []
        for node in graph.nodes:
            decision = self.rules.resolve(node, graph)
            match decision:
                case Exclude():
                    resolved: QuantizationPlan | Exclude = decision
                case algorithm:
                    match _decision(algorithm, node, graph):
                        case Err() as error:
                            return error
                        case Ok(resolved):
                            pass
            match resolved:
                case Exclude(reason=reason):
                    exclusions.append(
                        UnverifiedNode(node=node.name, operator=node.op, reason=reason)
                    )
                    patch = QuantizationPlan(excluded=(node.name,))
                case QuantizationPlan() as patch:
                    pass
            match plan.then(patch):
                case Err() as error:
                    return error
                case Ok(plan):
                    pass
        match lower(value.revision.model, plan):
            case Err() as error:
                return error
            case Ok(model):
                revision = replace(
                    value.revision, model=model, plans=value.revision.plans + (plan,)
                )
                return Ok(QuantizedGraph(revision, tuple(exclusions)))
