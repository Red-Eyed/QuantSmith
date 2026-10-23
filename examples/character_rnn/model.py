"""Concrete Torch architecture used by this quantization demonstration."""

import torch
from torch import Tensor, nn
from typing_extensions import override


class CharacterRNN(nn.Module):
    """Predict next characters through embeddings and a shared recurrent cell."""

    def __init__(self, vocabulary: int) -> None:
        """Expose recurrent dense operations during fixed-context ONNX export."""
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 32)
        self.cell = nn.RNNCell(32, 64)
        self.head = nn.Linear(64, vocabulary)

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        """Reset state for each context and return one next-token logit per position."""
        embedded = self.embedding.forward(tokens)
        hidden = embedded.new_zeros((tokens.shape[0], 64))
        outputs: list[Tensor] = []
        for index in range(tokens.shape[1]):
            hidden = self.cell.forward(embedded[:, index], hidden)
            outputs.append(self.head.forward(hidden))
        return torch.stack(outputs, dim=1)
