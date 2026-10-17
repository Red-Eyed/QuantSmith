"""Pure channel balancing shared by independent and joint SmoothQuant planners."""

import numpy as np

from quantsmith.calibration import MinMaxStats
from quantsmith.domain import FloatArray
from quantsmith.plan import RescaleInput
from quantsmith.result import FailureKind, QuantSmithError, Result, failure, validate


def rescaling(
    node: str,
    activation_axis: int,
    weight_axis: int,
    observed: MinMaxStats,
    weight: FloatArray,
    alpha: float,
) -> Result[RescaleInput, QuantSmithError]:
    """Balance live channels and keep identity scaling for dead channels."""
    activation = np.maximum(np.abs(observed.minimum), np.abs(observed.maximum)).astype(
        np.float64
    )
    magnitude = weight.astype(np.float64)
    if activation.shape != magnitude.shape:
        return failure(
            FailureKind.INVALID_DATA,
            node,
            "activation and weight channel dimensions differ",
        )
    active = (activation > 0) & (magnitude > 0)
    scale = np.ones_like(activation)
    scale[active] = activation[active] ** alpha / magnitude[active] ** (1 - alpha)
    return validate(
        node,
        lambda: RescaleInput(
            node=node,
            scale=scale.astype(np.float32),
            activation_axis=activation_axis,
            weight_axis=weight_axis,
        ),
    )
