"""Declare optional calibrated UINT8 boundaries without changing algorithm protocols."""

from collections.abc import Iterator
from dataclasses import dataclass

from onnx import ModelProto, TensorProto
from pydantic import BaseModel

from quantsmith.algorithms.contracts import Algorithm, Needs, Statistics
from quantsmith.algorithms.encoding import encode
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.calibration import Requirement
from quantsmith.domain import Graph, IntegerType, Node, PerTensor
from quantsmith.plan import QuantizationPlan, QuantizeInput, QuantizeOutput
from quantsmith.result import FailureKind, Ok, QuantSmithError, Result, failure
from quantsmith.rules import Exclude, Rule, Rules


class BoundaryQuantizer(BaseModel, frozen=True):
    """Quantize Div input or both residual Add operands using fresh MinMax ranges."""

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Request bounded per-tensor statistics on this graph revision."""
        tensors = _inputs(node) + (node.outputs if node.op == "Add" else ())
        return Ok(Needs(ranges=tuple(Requirement(tensor=name) for name in tensors)))

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
        if node.op != "Add":
            return Ok(QuantizationPlan(operations=operations))
        request = Requirement(tensor=node.outputs[0])
        if request not in stats.ranges:
            return failure(
                FailureKind.INVALID_DATA,
                node.name,
                "output statistics were not collected",
            )
        output = QuantizeOutput(
            node=node.name,
            index=0,
            encoding=encode(
                stats.ranges[request], IntegerType.UINT8, False, PerTensor()
            ),
        )
        return Ok(QuantizationPlan(operations=operations + (output,)))


@dataclass(frozen=True)
class BoundaryNames:
    """Match exact node identities without interpreting ONNX names as glob patterns."""

    names: frozenset[str]

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Limit boundary quantization to the selected node identities."""
        return node.name in self.names


def _inputs(node: Node) -> tuple[str, ...]:
    """Keep division's coefficient constant outside activation quantization."""
    return node.inputs[:2] if node.op == "Add" else node.inputs[:1]


def _explicitly_excluded(node: Node, graph: Graph, rules: Rules[Algorithm]) -> bool:
    """Respect the last matching override in the ordered rules contract."""
    decisions = tuple(
        rule.decision for rule in rules.overrides if rule.selector.matches(node, graph)
    )
    match decisions:
        case (*_, Exclude()):
            return True
        case _:
            return False


def _static_uint8(node: Node, graph: Graph, rules: Rules[Algorithm]) -> bool:
    """Recognize the built-in deployment contract without claiming custom methods."""
    match rules.resolve(node, graph):
        case StaticW8A8(activation_type=IntegerType.UINT8):
            return True
        case _:
            return False


def _ancestors(
    graph: Graph, roots: tuple[Node, ...], rules: Rules[Algorithm]
) -> Iterator[Node]:
    """Walk the supported region once, stopping at explicit exclusion boundaries."""
    producers = {output: node for node in graph.nodes for output in node.outputs}
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
    pending = list(roots)
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if (
            node.name in visited
            or node.op not in traversable
            or _explicitly_excluded(node, graph, rules)
        ):
            continue
        visited.add(node.name)
        yield node
        pending.extend(producers[name] for name in node.inputs if name in producers)


def _is_boundary(node: Node, divisions: frozenset[str]) -> bool:
    """Classify residual operands and only the divisions introduced by smoothing."""
    match node:
        case Node(op="Add"):
            return True
        case Node(op="Div", outputs=(output,)):
            return output in divisions
        case _:
            return False


def _unclaimed(node: Node, graph: Graph, rules: Rules[Algorithm]) -> bool:
    """Leave explicitly configured custom algorithms in control of their nodes."""
    match rules.resolve(node, graph):
        case Exclude():
            return True
        case _:
            return False


def with_boundaries(
    model: ModelProto, graph: Graph, rules: Rules[Algorithm], divisions: frozenset[str]
) -> Rules[Algorithm]:
    """Extend UINT8 residual regions and optional remaining smoothing divisions.

    Explicit exclusions and custom algorithms remain authoritative. Boundary
    ranges use MinMax even when weight-bearing operators use Percentile clipping.
    """
    roots = tuple(node for node in graph.nodes if _static_uint8(node, graph, rules))
    float_names = frozenset(
        value.name
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
        if value.type.tensor_type.elem_type == TensorProto.FLOAT
    )
    eligible = tuple(
        node
        for node in _ancestors(graph, roots, rules)
        if _is_boundary(node, divisions)
        and _unclaimed(node, graph, rules)
        and all(name in float_names for name in _inputs(node))
    )
    if not eligible:
        return rules
    extension: Rule[Algorithm] = Rule(
        selector=BoundaryNames(frozenset(node.name for node in eligible)),
        decision=BoundaryQuantizer(),
    )
    return Rules(default=rules.default, overrides=rules.overrides + (extension,))
