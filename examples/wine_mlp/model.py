"""Concrete Torch architecture used by this quantization demonstration."""

import torch
from torch import Tensor, nn
from typing_extensions import override


class WineMLP(nn.Module):
    """Classify normalized chemical measurements with three dense projections."""

    def __init__(self) -> None:
        """Use a compact architecture that can be trained reproducibly on CPU."""
        super().__init__()
        self.first = nn.Linear(13, 32)
        self.second = nn.Linear(32, 16)
        self.head = nn.Linear(16, 3)

    @override
    def forward(self, features: Tensor) -> Tensor:
        """Return three cultivar logits for each feature row."""
        hidden = torch.relu(self.first.forward(features))
        hidden = torch.relu(self.second.forward(hidden))
        return self.head.forward(hidden)
