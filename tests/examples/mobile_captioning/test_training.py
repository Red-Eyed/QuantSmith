"""Verify bounded training and native model checkpoint loading."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from examples.mobile_captioning.model import Captioner
from examples.mobile_captioning.schema import Config, ModelConfig, Vocabulary
from examples.mobile_captioning.training import load, train


def test_training_checkpoint(
    database: Path,
    config: Config,
    vocabulary: Vocabulary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip trained weights while keeping the frozen backbone in eval mode."""
    config.output.mkdir(parents=True)
    directory = config.data / "train2017"
    directory.mkdir(parents=True)
    for identity in (1, 2, 3):
        Image.new("RGB", (64, 64), color=(identity * 30, 20, 90)).save(
            directory / f"{identity:012d}.jpg"
        )

    def local_model(
        settings: ModelConfig, vocabulary: int, *, pretrained: bool
    ) -> Captioner:
        """Initialize actual layers locally so tests do not fetch ImageNet weights."""
        return Captioner(settings, vocabulary, pretrained=False)

    monkeypatch.setattr("examples.mobile_captioning.training.Captioner", local_model)
    torch.set_num_threads(1)
    model = train(config, database, vocabulary)
    restored, info = load(config)
    assert info.steps_completed == 2
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], expected, rtol=0, atol=0
        )
    assert not model.encoder.backbone.training
