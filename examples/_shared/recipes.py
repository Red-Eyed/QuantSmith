"""Named typed example recipes instead of runtime pipeline lookup dictionaries."""

from dataclasses import dataclass
from typing import assert_never

from examples._shared.schema import QuantizationMethod


@dataclass(frozen=True)
class Recipes[Recipe]:
    """Require one declared configuration for each demonstrated quantization method."""

    minmax: Recipe
    percentile: Recipe
    smoothquant: Recipe

    def select(self, method: QuantizationMethod) -> Recipe:
        """Dispatch the CLI's closed method alternative without string keys."""
        match method:
            case QuantizationMethod.MINMAX:
                return self.minmax
            case QuantizationMethod.PERCENTILE:
                return self.percentile
            case QuantizationMethod.SMOOTHQUANT:
                return self.smoothquant
            case _:
                assert_never(method)
