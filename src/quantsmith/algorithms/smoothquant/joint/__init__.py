"""Joint channel balancing for projections that consume the same activation."""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quantsmith.algorithms.contracts import Needs, Statistics
from quantsmith.algorithms.smoothquant.joint.ranges import (
    ChannelExtrema,
    ChannelMSE,
    ChannelPercentile,
    ChannelSelection,
    select_ranges,
)
from quantsmith.algorithms.smoothquant.scaling import rescaling
from quantsmith.calibration import ChannelRequirement, Requirement, extrema
from quantsmith.domain import Absent, FloatArray, Graph, Node
from quantsmith.domain.layout import channel_axes
from quantsmith.plan import QuantizationPlan
from quantsmith.result import Err, FailureKind, Ok, QuantSmithError, Result, failure


def joint_magnitude(
    node: Node, graph: Graph, activation_axis: int
) -> Result[FloatArray, QuantSmithError]:
    """Aggregate channel maxima across compatible constant-weight siblings."""
    magnitudes: list[FloatArray] = []
    for sibling in graph.nodes:
        if not sibling.inputs or sibling.inputs[0] != node.inputs[0]:
            continue
        match channel_axes(sibling, graph):
            case Err():
                continue
            case Ok(axes):
                sibling_activation, weight_axis = axes
        if sibling_activation != activation_axis:
            continue
        match extrema(graph.weights[sibling.inputs[1]], (weight_axis,)):
            case Err() as error:
                return error
            case Ok(bounds):
                magnitudes.append(
                    np.maximum(np.abs(bounds.minimum), np.abs(bounds.maximum))
                )
    if any(value.shape != magnitudes[0].shape for value in magnitudes):
        return failure(
            FailureKind.INVALID_DATA,
            node.name,
            "shared weight channel dimensions differ",
        )
    # The selected node is itself a compatible sibling after channel_axes succeeds.
    return Ok(np.maximum.reduce(magnitudes))


@dataclass(frozen=True)
class ContractionInputs:
    """Select constant-weight layouts whose input channels support smoothing."""

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Keep dynamic attention and grouped convolutions outside smoothing."""
        match channel_axes(node, graph):
            case Ok():
                return True
            case Err():
                return False


class JointSmoothQuant(BaseModel):
    """Balance sibling projections jointly for complete group folding."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    channels: ChannelSelection = Field(
        default_factory=ChannelExtrema, discriminator="kind"
    )

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Request the same bounded per-channel statistics as independent smoothing."""
        match channel_axes(node, graph):
            case Err() as error:
                return error
            case Ok(axes):
                activation_axis, _ = axes
        match self.channels:
            case ChannelExtrema():
                return Ok(
                    Needs(
                        ranges=(
                            Requirement(tensor=node.inputs[0], axes=(activation_axis,)),
                        )
                    )
                )
            case ChannelPercentile() | ChannelMSE():
                return Ok(
                    Needs(
                        channel_histograms=(
                            ChannelRequirement(
                                tensor=node.inputs[0], axis=activation_axis
                            ),
                        )
                    )
                )

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Plan one consumer of the common transform without mutation."""
        match channel_axes(node, graph):
            case Err() as error:
                return error
            case Ok(axes):
                activation_axis, weight_axis = axes
        request = Requirement(tensor=node.inputs[0], axes=(activation_axis,))
        if request not in stats.ranges:
            return failure(
                FailureKind.INVALID_DATA,
                node.name,
                "channel statistics were not collected",
            )
        channel_request = ChannelRequirement(
            tensor=node.inputs[0], axis=activation_axis
        )
        histogram = stats.channel_histograms.get(
            channel_request, Absent(reason="channel histogram was not collected")
        )
        match select_ranges(self.channels, stats.ranges[request], histogram):
            case Err() as error:
                return error
            case Ok(observed):
                pass
        match joint_magnitude(node, graph, activation_axis):
            case Err() as error:
                return error
            case Ok(weight):
                pass
        match rescaling(
            node.name,
            activation_axis,
            weight_axis,
            observed,
            weight,
            self.alpha,
        ):
            case Err() as error:
                return error
            case Ok(operation):
                return Ok(QuantizationPlan(operations=(operation,)))
