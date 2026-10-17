"""Verify split isolation, bounded loading, and caption token contracts."""

import json
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from examples.mobile_captioning.data import (
    Tokenizer,
    batches,
    index_annotations,
    records,
)
from examples.mobile_captioning.schema import Config, Vocabulary


def test_disjoint_splits(database: Path) -> None:
    """Train and calibration use separate image IDs; validation uses separate files."""
    train = {(item.partition, item.image_id) for item in records(database, "train", 10)}
    calibration = {
        (item.partition, item.image_id) for item in records(database, "calibration", 10)
    }
    evaluation = {
        (item.partition, item.image_id) for item in records(database, "evaluation", 10)
    }
    assert len(train) == 3
    assert len(calibration) == 2
    assert train.isdisjoint(calibration)
    assert (train | calibration).isdisjoint(evaluation)


def test_bounded_batch(
    database: Path,
    config: Config,
    vocabulary: Vocabulary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-batch admission must not download or open the following image."""
    directory = config.data / "train2017"
    directory.mkdir(parents=True)
    for identity in (1, 2):
        Image.new("RGB", (96, 48), color="red").save(directory / f"{identity:012d}.jpg")

    def forbidden(url: str, destination: Path) -> None:
        """Fail if a network download is attempted for an unadmitted image."""
        assert destination.exists()

    monkeypatch.setattr("examples.mobile_captioning.data.download", forbidden)
    stream = batches(database, config, Tokenizer(vocabulary, 8), epoch=0)
    images, tokens = next(stream)
    assert images.shape == (2, 3, 64, 64)
    assert tokens.shape == (2, 9)


def test_token_truncation(vocabulary: Vocabulary) -> None:
    """A truncated caption retains EOS and shifted targets fit the decoder length."""
    tokenizer = Tokenizer(vocabulary, 4)
    encoded = tokenizer.encode("a cat sits a cat sits")
    np.testing.assert_array_equal(encoded, [1, 4, 5, 6, 2])
    assert tokenizer.decode(tuple(int(token) for token in encoded)) == "a cat sits"


def test_limited_metadata(database: Path) -> None:
    """The iterator obeys its image budget without building a global ID collection."""
    selected = tuple(islice(records(database, "train", 1), 2))
    assert len(selected) == 1


def test_unknown_word(vocabulary: Vocabulary) -> None:
    """Unknown words remain observable in task-quality reports."""
    tokenizer = Tokenizer(vocabulary, 8)
    assert tokenizer.encode("unseen")[1] == 3


def test_vocabulary_excludes_calibration(
    database: Path,
    config: Config,
) -> None:
    """Calibration-only words must not leak into training vocabulary construction."""
    import sqlite3

    from examples.mobile_captioning.data import vocabulary

    config.output.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE captions SET caption='leakedword' WHERE image=20")
    assert "leakedword" not in vocabulary(database, config).words


def test_malformed_annotations(tmp_path: Path) -> None:
    """Reject a malformed external caption before publishing an indexed partition."""
    source = tmp_path / "bad.json"
    source.write_text(
        json.dumps(
            {
                "annotations": [
                    {"id": 1, "image_id": 2, "caption": 17},
                ]
            }
        )
    )
    with pytest.raises(ValueError):
        index_annotations(source, tmp_path / "bad.sqlite3", "train2017")
