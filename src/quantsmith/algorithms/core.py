"""Pure rule selection and plan assembly from supplied graph statistics."""

from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms.contracts import Algorithm, Needs, Statistics
from quantsmith.domain import Graph, Node
from quantsmith.plan import QuantizationPlan
from quantsmith.result import Err, Ok, QuantSmithError, Result
from quantsmith.rules import Exclude, Rules


class SelectedNode(BaseModel):
    """Bind one selected graph node to its configured algorithm."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    node: Node = Field()
    algorithm: Algorithm = Field()


class Selection(BaseModel):
    """Carry decisions and calibration needs bound to the original graph snapshot.

    Ranges include the first-pass bounds needed by histogram requests. Algorithms
    and selectors must honor their pure contracts; arbitrary plugins are not frozen.
    """

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    graph: Graph = Field()
    selected: tuple[SelectedNode, ...] = Field()
    excluded: tuple[str, ...] = Field()
    needs: Needs = Field()


def select_algorithms(
    graph: Graph, rules: Rules[Algorithm]
) -> Result[Selection, QuantSmithError]:
    """Resolve rules and deduplicate needs without executing or changing the graph.

    Requirement failures preserve the plugin's diagnostic and stop selection.
    """
    selected: list[SelectedNode] = []
    excluded: list[str] = []
    needs: list[Needs] = []
    for node in graph.nodes:
        match rules.resolve(node, graph):
            case Exclude():
                excluded.append(node.name)
            case algorithm:
                match algorithm.requirements(node, graph):
                    case Err() as error:
                        return error
                    case Ok(requirement_needs):
                        needs.append(requirement_needs)
                selected.append(SelectedNode(node=node, algorithm=algorithm))
    ranges = tuple(
        dict.fromkeys(
            item
            for need in needs
            for item in need.ranges
            + need.histograms
            + tuple(request.range for request in need.channel_histograms)
        )
    )
    histograms = tuple(
        dict.fromkeys(item for need in needs for item in need.histograms)
    )
    channel_histograms = tuple(
        dict.fromkeys(item for need in needs for item in need.channel_histograms)
    )
    return Ok(
        Selection(
            graph=graph,
            selected=tuple(selected),
            excluded=tuple(excluded),
            needs=Needs(
                ranges=ranges,
                histograms=histograms,
                channel_histograms=channel_histograms,
            ),
        )
    )


def assemble_plan(
    selection: Selection, stats: Statistics
) -> Result[QuantizationPlan, QuantSmithError]:
    """Combine algorithm decisions using supplied statistics; perform no calibration.

    The caller supplies statistics for the selection's graph revision. Missing
    statistics, algorithm failures, and plan conflicts return the original failure.
    """
    result = QuantizationPlan(excluded=selection.excluded)
    for selected in selection.selected:
        match selected.algorithm.plan(selected.node, selection.graph, stats):
            case Err() as error:
                return error
            case Ok(patch):
                pass
        match result.then(patch):
            case Err() as error:
                return error
            case Ok(result):
                pass
    return Ok(result)
