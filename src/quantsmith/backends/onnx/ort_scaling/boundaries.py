"""Declare optional calibrated UINT8 boundaries without changing algorithm protocols."""

from dataclasses import dataclass

from onnx import ModelProto, TensorProto
from pydantic import BaseModel

from quantsmith.algorithms.contracts import Algorithm, Needs, Statistics
from quantsmith.algorithms.encoding import encode
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.calibration import Requirement
from quantsmith.domain import Graph, IntegerType, Node, PerTensor
from quantsmith.plan import QuantizationPlan, QuantizeInput
from quantsmith.result import FailureKind, Ok, QuantSmithError, Result, failure
from quantsmith.rules import Exclude, Rule, Rules


class BoundaryQuantizer(BaseModel, frozen=True):
    """Quantize Div input or both residual Add operands using fresh MinMax ranges."""

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Request bounded per-tensor statistics on this graph revision."""
        return Ok(
            Needs(ranges=tuple(Requirement(tensor=name) for name in _inputs(node)))
        )

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Expose every additional rounding boundary in the returned stage plan."""
        if any(Requirement(tensor=name) not in stats.ranges for name in _inputs(node)):
            return failure(
                FailureKind.INVALID_DATA,
                node.name,
                "boundary statistics were not collected",
            )
        operations = tuple(
            QuantizeInput(
                node=node.name,
                index=index,
                encoding=encode(
                    stats.ranges[Requirement(tensor=name)],
                    IntegerType.UINT8,
                    False,
                    PerTensor(),
                ),
            )
            for index, name in enumerate(_inputs(node))
        )
        return Ok(QuantizationPlan(operations=operations))


@dataclass(frozen=True)
class BoundaryNames:
    """Match exact node identities without interpreting ONNX names as glob patterns."""

    names: frozenset[str]

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Limit the optional algorithm to the selected boundary nodes."""
        return node.name in self.names


def _inputs(node: Node) -> tuple[str, ...]:
    """Keep division's coefficient constant outside activation quantization."""
    return node.inputs[:2] if node.op == "Add" else node.inputs[:1]


def _explicitly_excluded(node: Node, graph: Graph, rules: Rules[Algorithm]) -> bool:
    """Respect the last matching override, as required by the ordered rules contract."""
    excluded = False
    for rule in rules.overrides:
        if rule.selector.matches(node, graph):
            excluded = isinstance(rule.decision, Exclude)
    return excluded


def with_boundaries(
    model: ModelProto,
    graph: Graph,
    rules: Rules[Algorithm],
    divisions: frozenset[str],
) -> Rules[Algorithm]:
    """Extend a UINT8 static stage only within the smoothed quantized region.

    Explicit exclusion overrides remain authoritative. Boundary ranges always use
    MinMax even if the selected weight-bearing algorithm uses Percentile clipping.
    """
    producers = {output: node for node in graph.nodes for output in node.outputs}
    pending: list[Node] = []
    for node in graph.nodes:
        decision = rules.resolve(node, graph)
        if (
            isinstance(decision, StaticW8A8)
            and decision.activation_type == IntegerType.UINT8
        ):
            pending.append(node)
    visited: set[str] = set()
    eligible: list[Node] = []
    float_names = {
        value.name
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
        if value.type.tensor_type.elem_type == TensorProto.FLOAT
    }
    traversable = {
        "Conv",
        "Gemm",
        "MatMul",
        "Clip",
        "Relu",
        "Add",
        "Div",
        "Reshape",
        "ReduceMean",
        "BatchNormalization",
    }
    while pending:
        node = pending.pop()
        if node.name in visited or node.op not in traversable:
            continue
        visited.add(node.name)
        excluded = _explicitly_excluded(node, graph, rules)
        if excluded:
            continue
        if (
            (node.op == "Add" or (node.op == "Div" and node.outputs[0] in divisions))
            and isinstance(rules.resolve(node, graph), Exclude)
            and all(name in float_names for name in _inputs(node))
        ):
            eligible.append(node)
        pending.extend(producers[name] for name in node.inputs if name in producers)
    if not any(node.op == "Div" for node in eligible):
        return rules
    return Rules(
        default=rules.default,
        overrides=rules.overrides
        + (
            Rule(
                selector=BoundaryNames(frozenset(node.name for node in eligible)),
                decision=BoundaryQuantizer(),
            ),
        ),
    )
