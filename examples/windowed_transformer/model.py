"""Causal window-eight Transformer with RoPE, RMSNorm, and SwiGLU on character token
IDs, using Stackformers attention.
"""

import torch
from stackformers import TransformerEncoder, make_padded_input, windowed_encoder_config
from torch import Tensor, nn
from typing_extensions import override


class WindowedCharacterTransformer(nn.Module):
    """Embed characters, apply causal attention, and predict the next character."""

    def __init__(self, vocabulary: int) -> None:
        """Use two CPU-sized layers with explicit attention and feedforward settings."""
        super().__init__()
        config = windowed_encoder_config(
            dim=64, heads=4, num_layers=2, causal=True, ff_mult=2, window_size=8
        )
        self.embedding = nn.Embedding(vocabulary, 64)
        self.transformer = TransformerEncoder(config)
        self.output = nn.Linear(64, vocabulary)

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        """Return one next-character logit vector at every fixed-context position."""
        embeddings = self.embedding.forward(tokens)
        sequence = make_padded_input(
            embeddings, torch.ones_like(tokens, dtype=torch.bool)
        )
        hidden = self.transformer.forward(sequence)
        return self.output.forward(hidden)
