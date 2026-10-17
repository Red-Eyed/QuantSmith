"""Transactional common-scale folding for complete groups of activation consumers."""

from collections import defaultdict
from dataclasses import dataclass
from typing import assert_never

import numpy as np
from onnx import ModelProto

from quantsmith.backends.onnx import admit, lower, normalize
from quantsmith.backends.onnx.folding import fold_affine
from quantsmith.domain import Absent
from quantsmith.pipeline.state import Folded, FoldingOutcome, Unfoldable
from quantsmith.plan import QuantizationPlan, RescaleInput
from quantsmith.result import Err, Ok, QuantSmithError, Result


@dataclass(frozen=True)
class ConsumerGroup:
    """Bind all proposed rescalings of one original activation tensor."""

    source: str
    operations: tuple[RescaleInput, ...]


def consumer_groups(
    model: ModelProto, plan: QuantizationPlan
) -> tuple[ConsumerGroup, ...]:
    """Group by graph tensor identity while preserving plan declaration order."""
    nodes = {node.name: node for node in model.graph.node}
    grouped: defaultdict[str, list[RescaleInput]] = defaultdict(list)
    for operation in plan.operations:
        if not isinstance(operation, RescaleInput):
            raise ValueError("joint folding requires a rescaling-only plan")
        grouped[nodes[operation.node].input[0]].append(operation)
    return tuple(
        ConsumerGroup(source, tuple(operations))
        for source, operations in grouped.items()
    )


def eligibility(model: ModelProto, group: ConsumerGroup) -> ConsumerGroup | Absent:
    """Require equal scaling and ownership of every use before changing a producer."""
    first = group.operations[0]
    if any(
        operation.activation_axis != first.activation_axis
        or not np.array_equal(operation.scale, first.scale)
        for operation in group.operations
    ):
        return Absent(reason="consumer group has incompatible channel scales or axes")
    if any(value.name == group.source for value in model.graph.output):
        return Absent(reason="shared activation is a public graph output")
    selected = {(operation.node, 0) for operation in group.operations}
    users = {
        (node.name, index)
        for node in model.graph.node
        for index, name in enumerate(node.input)
        if name == group.source
    }
    if users != selected:
        return Absent(
            reason="shared activation has an excluded or incompatible consumer"
        )
    return group


def coalesce(model: ModelProto, group: ConsumerGroup) -> tuple[ModelProto, str]:
    """Merge candidate divisions before folding their common producer."""
    result = normalize(model)
    nodes = {node.name: node for node in result.graph.node}
    inputs = tuple(nodes[operation.node].input[0] for operation in group.operations)
    introduced = inputs[0]
    obsolete = frozenset(inputs[1:])
    for operation in group.operations:
        nodes[operation.node].input[0] = introduced
    kept = [
        node
        for node in result.graph.node
        if not (node.op_type == "Div" and node.output[0] in obsolete)
    ]
    metadata = [
        value for value in result.graph.value_info if value.name not in obsolete
    ]
    del result.graph.node[:]
    result.graph.node.extend(kept)
    del result.graph.value_info[:]
    result.graph.value_info.extend(metadata)
    return result, introduced


def attempt_group(
    model: ModelProto, group: ConsumerGroup
) -> Result[ModelProto | Absent, QuantSmithError]:
    """Accept a candidate only when its common runtime scale disappears."""
    eligible = eligibility(model, group)
    if isinstance(eligible, Absent):
        return Ok(eligible)
    match lower(model, QuantizationPlan(operations=group.operations)):
        case Err() as error:
            return error
        case Ok(candidate):
            candidate, introduced = coalesce(candidate, group)
    match admit("fold joint affine scales", lambda: fold_affine(candidate, introduced)):
        case Err() as error:
            return error
        case Ok(candidate):
            pass
    if any(
        node.op_type == "Div" and node.output[0] == introduced
        for node in candidate.graph.node
    ):
        return Ok(
            Absent(reason="shared scale has no supported private affine producer")
        )
    return Ok(candidate)


def fold_groups(
    model: ModelProto, plan: QuantizationPlan
) -> Result[
    tuple[ModelProto, QuantizationPlan, tuple[FoldingOutcome, ...]], QuantSmithError
]:
    """Apply complete groups or leave every member's original parameters unchanged."""
    current = normalize(model)
    accepted: list[RescaleInput] = []
    skipped: list[str] = []
    outcomes: list[FoldingOutcome] = []
    for group in consumer_groups(current, plan):
        match attempt_group(current, group):
            case Err() as error:
                return error
            case Ok(outcome):
                match outcome:
                    case Absent(reason=reason):
                        skipped.extend(operation.node for operation in group.operations)
                        outcomes.extend(
                            Unfoldable(operation.node, reason)
                            for operation in group.operations
                        )
                    case ModelProto() as candidate:
                        current = candidate
                        accepted.extend(group.operations)
                        outcomes.extend(
                            Folded(operation.node) for operation in group.operations
                        )
                    case _ as remaining:
                        assert_never(remaining)
    applied = QuantizationPlan(
        operations=tuple(accepted), excluded=plan.excluded + tuple(skipped)
    )
    return Ok((current, applied, tuple(outcomes)))
