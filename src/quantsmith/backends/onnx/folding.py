"""Transactional fold-only smoothing with explicit accepted/skipped outcomes."""

from collections import Counter
from typing import assert_never

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

from quantsmith.backends.onnx import (
    Names,
    admit,
    lower,
    normalize,
    remove_unused_initializers,
)
from quantsmith.backends.onnx.optimization import channel_parameters, optimize
from quantsmith.domain import Absent, FloatArray
from quantsmith.pipeline.state import Folded, FoldingOutcome, Unfoldable
from quantsmith.plan import QuantizationPlan, RescaleInput
from quantsmith.result import Err, Ok, QuantSmithError, Result


def _constant(model: ModelProto, name: str) -> FloatArray | Absent:
    """Borrow finite FP32 parameters only from the owned candidate revision."""
    for tensor in model.graph.initializer:
        if tensor.name == name and tensor.data_type == TensorProto.FLOAT:
            value = np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
            if np.all(np.isfinite(value)):
                return value
    return Absent(reason="not a finite FP32 initializer")


def _parameters(
    model: ModelProto, node: NodeProto, scale: FloatArray
) -> tuple[tuple[int, FloatArray], ...]:
    """Absorb last-channel scaling into affine gains or matrix output channels."""
    existing = channel_parameters(model, node, scale)
    if existing:
        return tuple(existing)
    if (
        scale.ndim < 1
        or any(size != 1 for size in scale.shape[:-1])
        or np.any(scale <= 0)
    ):
        return ()
    channels = scale.reshape(-1)
    match node.op_type:
        case "Mul":
            for index, name in enumerate(node.input):
                gain = _constant(model, name)
                if isinstance(gain, np.ndarray) and gain.shape == channels.shape:
                    return ((index, gain / channels),)
            return ()
        case "MatMul" | "Gemm":
            weight = _constant(model, node.input[1])
            if isinstance(weight, Absent) or weight.ndim != 2:
                return ()
            transposed = node.op_type == "Gemm" and any(
                attr.name == "transB" and attr.i != 0 for attr in node.attribute
            )
            axis = 0 if transposed else 1
            if weight.shape[axis] != channels.size:
                return ()
            divisor = channels[:, None] if transposed else channels[None, :]
            parameters = [(1, weight / divisor)]
            if len(node.input) > 2 and node.input[2]:
                bias = _constant(model, node.input[2])
                if isinstance(bias, Absent) or bias.shape != channels.shape:
                    return ()
                parameters.append((2, bias / channels))
            return tuple(parameters)
        case _:
            return ()


def _fold_target(
    model: ModelProto, source: str
) -> tuple[NodeProto, NodeProto] | Absent:
    """Find a private affine producer through positively homogeneous Relu nodes."""
    producers = {name: node for node in model.graph.node for name in node.output}
    uses = Counter(name for node in model.graph.node for name in node.input)
    public = {value.name for value in model.graph.output}
    outer = producers.get(source)
    if outer is None:
        return Absent(reason="graph input has no affine producer")
    current = source
    while current in producers:
        node = producers[current]
        if (
            node.domain
            or len(node.output) != 1
            or uses[current] != 1
            or current in public
        ):
            return Absent(
                reason="producer is public, shared or outside supported domains"
            )
        if node.op_type == "Relu":
            current = node.input[0]
            continue
        if node.op_type in {"Conv", "BatchNormalization", "Mul", "MatMul", "Gemm"}:
            return node, outer
        return Absent(reason="intervening operation cannot absorb channel scaling")
    return Absent(reason="activation function reaches a graph input")


def fold_affine(model: ModelProto, introduced: str) -> ModelProto:
    """Fold private last-channel Div nodes while cloning shared parameter storage."""
    result = normalize(model)
    ranks = {
        value.name: len(value.type.tensor_type.shape.dim)
        for value in (
            *result.graph.input,
            *result.graph.value_info,
            *result.graph.output,
        )
    }
    names = Names(result)
    removed: set[str] = set()
    obsolete: set[str] = set()
    for division in result.graph.node:
        if (
            division.domain
            or division.op_type != "Div"
            or division.output[0] != introduced
        ):
            continue
        source = division.input[0]
        target = _fold_target(result, source)
        if isinstance(target, Absent):
            continue
        producer, outer = target
        scale = _constant(result, division.input[1])
        if isinstance(scale, Absent) or ranks.get(source) != scale.ndim:
            continue
        parameters = _parameters(result, producer, scale)
        if not parameters or any(
            not np.all(np.isfinite(value)) for _, value in parameters
        ):
            continue
        for index, value in parameters:
            name = names.new(producer.name + "_folded_scale")
            result.graph.initializer.append(numpy_helper.from_array(value, name))
            producer.input[index] = name
        outer.output[0] = division.output[0]
        removed.add(division.name)
        obsolete.add(source)
    kept = [node for node in result.graph.node if node.name not in removed]
    metadata = [
        value for value in result.graph.value_info if value.name not in obsolete
    ]
    del result.graph.node[:]
    result.graph.node.extend(kept)
    del result.graph.value_info[:]
    result.graph.value_info.extend(metadata)
    remove_unused_initializers(result)
    onnx.checker.check_model(result)
    return onnx.shape_inference.infer_shapes(result, strict_mode=True)


def _attempt(
    model: ModelProto, operation: RescaleInput
) -> Result[ModelProto | Unfoldable, QuantSmithError]:
    """Discard a candidate completely if its introduced activation division remains."""
    consumer = next(node for node in model.graph.node if node.name == operation.node)
    eligible = _fold_target(model, consumer.input[0])
    if isinstance(eligible, Absent):
        return Ok(Unfoldable(node=operation.node, reason=eligible.reason))
    match lower(model, QuantizationPlan(operations=(operation,))):
        case Err() as error:
            return error
        case Ok(candidate):
            introduced = next(
                node.input[0]
                for node in candidate.graph.node
                if node.name == operation.node
            )
    match optimize(candidate, scope=frozenset({introduced})):
        case Err() as error:
            return error
        case Ok(candidate):
            pass
    match admit("fold affine scales", lambda: fold_affine(candidate, introduced)):
        case Err() as error:
            return error
        case Ok(candidate):
            pass
    if any(
        node.op_type == "Div" and node.output[0] == introduced
        for node in candidate.graph.node
    ):
        return Ok(
            Unfoldable(
                node=operation.node,
                reason="activation scale has no supported private affine producer",
            )
        )
    return Ok(candidate)


def fold_only(
    model: ModelProto, plan: QuantizationPlan
) -> Result[
    tuple[ModelProto, QuantizationPlan, tuple[FoldingOutcome, ...]], QuantSmithError
]:
    """Apply only foldable rescalings; skipped consumers retain original weights."""
    current = normalize(model)
    accepted: list[RescaleInput] = []
    skipped: list[str] = []
    outcomes: list[FoldingOutcome] = []
    for operation in plan.operations:
        if not isinstance(operation, RescaleInput):
            raise ValueError("fold-only smoothing requires a rescaling-only plan")
        match _attempt(current, operation):
            case Err() as error:
                return error
            case Ok(outcome):
                match outcome:
                    case Unfoldable():
                        outcomes.append(outcome)
                        skipped.append(operation.node)
                    case ModelProto() as candidate:
                        current = candidate
                        accepted.append(operation)
                        outcomes.append(Folded(operation.node))
                    case _ as remaining:
                        assert_never(remaining)
    applied = QuantizationPlan(
        operations=tuple(accepted), excluded=plan.excluded + tuple(skipped)
    )
    return Ok((current, applied, tuple(outcomes)))
