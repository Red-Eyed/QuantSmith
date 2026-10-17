"""Small real MobileNetV4/Stackformers fixtures without network access."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.mobile_captioning.data import index_annotations
from examples.mobile_captioning.export import GraphPair, export
from examples.mobile_captioning.model import Captioner
from examples.mobile_captioning.schema import Config, ModelConfig, Vocabulary


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Keep a real CNN with one encoder/decoder layer and a small spatial grid."""
    return Config(
        output=tmp_path / "artifacts",
        data=tmp_path / "data",
        model=ModelConfig(
            image_size=64,
            dim=64,
            heads=1,
            encoder_layers=1,
            decoder_layers=1,
            max_tokens=8,
        ),
        train_images=3,
        calibration_images=2,
        evaluation_images=2,
        vocabulary_size=16,
        steps=2,
        batch_size=2,
        quiet=True,
    )


@pytest.fixture
def vocabulary() -> Vocabulary:
    """Give tests stable special IDs and caption words."""
    return Vocabulary(
        words=("<pad>", "<bos>", "<eos>", "<unk>", "a", "cat", "sits", ".")
    )


@pytest.fixture
def model(config: Config, vocabulary: Vocabulary) -> Captioner:
    """Construct the requested architecture with deterministic untrained weights."""
    torch.set_num_threads(1)
    torch.manual_seed(9)
    return Captioner(config.model, len(vocabulary.words), pretrained=False).eval()


@pytest.fixture
def graphs(model: Captioner, config: Config) -> GraphPair:
    """Export real CNN, global self-attention, and decoder cross-attention graphs."""
    return export(
        model,
        np.zeros((1, 3, 64, 64), dtype=np.float32),
        np.asarray([[1, 4, 5, 0, 0, 0, 0, 0]], dtype=np.int64),
        config,
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """Index small COCO-shaped metadata with overlapping partition-local image IDs."""
    path = tmp_path / "captions.sqlite3"
    for partition in ("train2017", "val2017"):
        source = tmp_path / f"{partition}.json"
        source.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "id": identity * 10 + caption,
                            "image_id": identity,
                            "caption": "a cat sits .",
                        }
                        for identity in (1, 2, 3, 20, 40)
                        for caption in (1, 2)
                    ]
                }
            )
        )
        index_annotations(source, path, partition)
    return path
