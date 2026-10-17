"""Observe private activation functions when choosing integer producer ranges."""

from collections import defaultdict
from dataclasses import dataclass

from onnx import ModelProto, NodeProto

from quantsmith.algorithms.contracts import Algorithm, Needs, Statistics
from quantsmith.calibration import Requirement
from quantsmith.domain import Graph, Node
from quantsmith.plan import QuantizationPlan
from quantsmith.result import Err, FailureKind, Ok, QuantSmithError, Result, failure


def activation_function_outputs(model: ModelProto) -> dict[str, str]:
    """Map standard Conv nodes to private Relu/static-bound Clip observations.

    A public or shared pre-function value cannot adopt the following function's
    clipping interval: its other consumers still require the unmodified range.
    The pipeline supplies a normalized flat graph with all uses visible.
    """
    consumers: dict[str, list[NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for tensor in node.input:
            consumers[tensor].append(node)
    public = {value.name for value in model.graph.output}
    inputs = {value.name for value in model.graph.input}
    constants = {value.name for value in model.graph.initializer} - inputs
    outputs: dict[str, str] = {}
    for node in model.graph.node:
        if node.domain or node.op_type != "Conv" or len(node.output) != 1:
            continue
        tensor = node.output[0]
        following = consumers[tensor]
        if tensor in public or len(following) != 1:
            continue
        activation = following[0]
        if activation.op_type == "Clip" and any(
            bound and bound not in constants for bound in activation.input[1:]
        ):
            continue
        if (
            not activation.domain
            and activation.op_type in {"Relu", "Clip"}
            and len(activation.output) == 1
            and activation.input[0] == tensor
        ):
            outputs[node.name] = activation.output[0]
    return outputs


@dataclass(frozen=True)
class ActivationFunctionRanges:
    """Redirect built-in per-tensor output observations to the following function.

    Producer QDQ stays before the activation function. Its interval comes from
    the function's output, changing clipping and rounding intentionally.
    """

    base: Algorithm
    tensor: str

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Keep axes and input requests intact for extrema and histogram replay."""
        match self.base.requirements(node, graph):
            case Err() as error:
                return error
            case Ok(needs):
                pass

        def substitute(request: Requirement) -> Requirement:
            """Redirect only the producer output on the current graph revision."""
            if request.tensor == node.outputs[0]:
                return Requirement(tensor=self.tensor, axes=request.axes)
            return request

        return Ok(
            Needs(
                ranges=tuple(substitute(request) for request in needs.ranges),
                histograms=tuple(substitute(request) for request in needs.histograms),
            )
        )

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Alias observed statistics without modifying caller-owned mappings."""
        source = Requirement(tensor=self.tensor)
        target = Requirement(tensor=node.outputs[0])
        if source not in stats.ranges:
            return failure(
                FailureKind.INVALID_DATA,
                node.name,
                "activation-function output statistics were not collected",
            )
        ranges = {**stats.ranges, target: stats.ranges[source]}
        histograms = dict(stats.histograms)
        if source in histograms:
            histograms[target] = histograms[source]
        return self.base.plan(
            node, graph, Statistics(ranges=ranges, histograms=histograms)
        )
