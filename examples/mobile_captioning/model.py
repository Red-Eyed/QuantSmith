"""Pretrained mobile spatial features and Stackformers encoder–decoder layers."""

import timm
import torch
from einops import rearrange
from pydantic import ConfigDict, TypeAdapter
from stackformers import (
    TransformerDecoder,
    TransformerEncoder,
    make_padded_input,
    plain_decoder_config,
    plain_encoder_config,
)
from stackformers.positional.config import NoPosEncodingConfig
from stackformers.presets.encoder import TransformerEncoderConfig
from timm.models.mobilenetv3 import MobileNetV3
from torch import Tensor, nn
from typing_extensions import override

from examples.mobile_captioning.schema import ModelConfig

_BACKBONE = TypeAdapter(
    MobileNetV3, config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)


class ImageEncoder(nn.Module):
    """Encode an image once into a fixed spatial context shared by decoder steps."""

    def __init__(self, config: ModelConfig, *, pretrained: bool) -> None:
        """Give the visual grid learned row and column positions."""
        super().__init__()
        self.backbone = _BACKBONE.validate_python(
            timm.create_model(
                "mobilenetv4_conv_small.e2400_r224_in1k",
                pretrained=pretrained,
                num_classes=0,
            )
        )
        channels = TypeAdapter(int).validate_python(self.backbone.num_features)
        self.projection = nn.Linear(channels, config.dim)
        grid = config.image_size // 32
        self.rows = nn.Parameter(torch.randn(grid, config.dim) * 0.02)
        self.columns = nn.Parameter(torch.randn(grid, config.dim) * 0.02)
        base = plain_encoder_config(
            config.dim,
            config.heads,
            config.encoder_layers,
            ff_mult=config.ff_mult,
        )
        encoder_config = TransformerEncoderConfig(
            attn=base.attn,
            ff=base.ff,
            norm=base.norm,
            pos_encoding=NoPosEncodingConfig(),
            num_layers=base.num_layers,
        )
        self.transformer = TransformerEncoder(encoder_config)

    @override
    def forward(self, images: Tensor) -> Tensor:
        """Return batch × spatial positions × hidden dimensions in row-major order."""
        features = self.backbone.forward_features(images)
        tokens = self.projection.forward(rearrange(features, "b c h w -> b (h w) c"))
        positions = rearrange(self.rows, "h d -> h 1 d") + rearrange(
            self.columns, "w d -> 1 w d"
        )
        tokens = tokens + rearrange(positions, "h w d -> 1 (h w) d")
        valid = torch.ones_like(tokens[..., 0], dtype=torch.bool)
        return self.transformer.forward(make_padded_input(tokens, valid))


class CaptionDecoder(nn.Module):
    """Predict tokens using causal self-attention and visual cross-attention."""

    def __init__(self, config: ModelConfig, vocabulary: int) -> None:
        """Use Stackformers for both attention paths and the feedforward layers."""
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, config.dim, padding_idx=0)
        self.transformer = TransformerDecoder(
            plain_decoder_config(
                config.dim,
                config.heads,
                config.decoder_layers,
                ff_mult=config.ff_mult,
            )
        )
        self.output = nn.Linear(config.dim, vocabulary)

    @override
    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        """Return all prefix logits; padding is masked and later tokens are causal."""
        query = make_padded_input(self.embedding.forward(tokens), tokens != 0)
        visual = make_padded_input(
            context, torch.ones_like(context[..., 0], dtype=torch.bool)
        )
        return self.output.forward(self.transformer.forward(query, visual))


class Captioner(nn.Module):
    """Own the two exportable graphs without putting decoding policy in the model."""

    def __init__(
        self, config: ModelConfig, vocabulary: int, *, pretrained: bool
    ) -> None:
        """Initialize the CNN from ImageNet only when starting fresh training."""
        super().__init__()
        self.encoder = ImageEncoder(config, pretrained=pretrained)
        self.decoder = CaptionDecoder(config, vocabulary)

    @override
    def forward(self, images: Tensor, tokens: Tensor) -> Tensor:
        """Apply teacher forcing during training using the same inference graphs."""
        return self.decoder.forward(tokens, self.encoder.forward(images))
