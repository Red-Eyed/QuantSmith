"""Disk-indexed COCO captions with bounded image and minibatch admission."""

import re
import shutil
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from urllib.request import urlopen

import ijson
import numpy as np
import torch
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from remotezip import RemoteZip
from rich.console import Console
from rich.progress import Progress
from torch import Tensor

from examples.mobile_captioning.schema import Config, Vocabulary
from quantsmith.domain import FloatArray

type Split = Literal["train", "calibration", "evaluation"]
type Partition = Literal["train2017", "val2017"]
_ROOT = "https://s3.amazonaws.com/images.cocodataset.org"
_IDENTITY_ROW = TypeAdapter(tuple[int])
_CAPTION_ROW = TypeAdapter(tuple[str])


class Annotation(BaseModel):
    """Validate one caption record without retaining the surrounding JSON tree."""

    model_config = ConfigDict(extra="ignore", strict=True)
    id: int = Field(ge=0)
    image_id: int = Field(ge=0)
    caption: str = Field(min_length=1, max_length=4096)


class CaptionRecord(BaseModel):
    """Own at most one image's bounded set of captions."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    image_id: int
    partition: Partition
    captions: tuple[str, ...] = Field(min_length=1, max_length=10)


def words(text: str) -> tuple[str, ...]:
    """Use one explicit lowercase punctuation tokenizer for training and metrics."""
    return tuple(re.findall(r"\w+|[^\w\s]", text.lower()))


def download(url: str, path: Path) -> None:
    """Stream a file to an atomic cache entry; failed transfers stay incomplete."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        urlopen(url, timeout=90) as source,
        NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as target,
    ):
        temporary = Path(target.name)
        shutil.copyfileobj(source, target, length=1024 * 1024)
    temporary.replace(path)


def annotations(root: Path, partition: Partition) -> Path:
    """Fetch only caption members using ZIP range reads, avoiding image archives."""
    path = root / f"captions_{partition}.json"
    if path.exists():
        return path
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with RemoteZip(f"{_ROOT}/annotations/annotations_trainval2017.zip") as archive:
        with archive.open(f"annotations/{path.name}") as source:
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    temporary.replace(path)
    return path


def index_annotations(path: Path, database: Path, partition: Partition) -> None:
    """Commit one annotation stream atomically with bounded SQLite page caching."""
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA cache_size = -4096")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS captions "
            "(partition TEXT, id INTEGER, image INTEGER, caption TEXT, "
            "PRIMARY KEY(partition, id))"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS image_idx ON captions(partition, image)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS indexed (partition TEXT PRIMARY KEY)"
        )
        if connection.execute(
            "SELECT 1 FROM indexed WHERE partition=?", (partition,)
        ).fetchone():
            return
        with path.open("rb") as source:
            for raw in ijson.items(source, "annotations.item"):
                item = Annotation.model_validate(raw)
                connection.execute(
                    "INSERT INTO captions VALUES (?, ?, ?, ?)",
                    (partition, item.id, item.image_id, item.caption),
                )
        connection.execute("INSERT INTO indexed VALUES (?)", (partition,))


def prepare(config: Config) -> Path:
    """Automatically fetch and index annotation metadata, keeping images lazy."""
    database = config.data / "captions.sqlite3"
    for partition in ("train2017", "val2017"):
        index_annotations(annotations(config.data, partition), database, partition)
    return database


def cache_images(database: Path, config: Config, splits: tuple[Split, ...]) -> None:
    """Fetch only admitted images with a bounded window of independent requests."""
    console = Console(stderr=True, quiet=config.quiet)
    with ThreadPoolExecutor(max_workers=config.download_workers) as pool:
        with Progress(console=console, disable=config.quiet) as progress:
            for split in splits:
                match split:
                    case "train":
                        limit = config.train_images
                    case "calibration":
                        limit = config.calibration_images
                    case "evaluation":
                        limit = config.evaluation_images
                stream = records(database, split, limit)
                task = progress.add_task(f"Caching {split} images", total=limit)

                def cache(record: CaptionRecord) -> None:
                    """Give each independent transfer its own temporary file."""
                    name = f"{record.image_id:012d}.jpg"
                    download(
                        f"{_ROOT}/{record.partition}/{name}",
                        config.data / record.partition / name,
                    )

                while window := tuple(islice(stream, config.download_workers)):
                    for _ in pool.map(cache, window):
                        progress.advance(task)
                progress.remove_task(task)


def records(database: Path, split: Split, limit: int) -> Iterator[CaptionRecord]:
    """Read disjoint deterministic image splits with one image's references live."""
    partition = "val2017" if split == "evaluation" else "train2017"
    match split:
        case "train":
            predicate = "image % 20 != 0"
        case "calibration":
            predicate = "image % 20 = 0"
        case "evaluation":
            predicate = "1"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA cache_size = -4096")
        query = (
            "SELECT DISTINCT image FROM captions WHERE partition=? AND "
            + predicate
            + " ORDER BY image LIMIT ?"
        )
        for raw in connection.execute(query, (partition, limit)):
            (identity,) = _IDENTITY_ROW.validate_python(raw)
            captions = tuple(
                _CAPTION_ROW.validate_python(row)[0]
                for row in connection.execute(
                    "SELECT caption FROM captions "
                    "WHERE partition=? AND image=? ORDER BY id",
                    (partition, identity),
                )
            )
            yield CaptionRecord(
                image_id=identity, partition=partition, captions=captions
            )


def vocabulary(database: Path, config: Config) -> Vocabulary:
    """Count only admitted training captions using a disk-backed word table."""
    with sqlite3.connect(config.output / "vocabulary.sqlite3") as connection:
        connection.execute("DROP TABLE IF EXISTS words")
        connection.execute("CREATE TABLE words (word TEXT PRIMARY KEY, count INTEGER)")
        for record in records(database, "train", config.train_images):
            for caption in record.captions:
                for word in words(caption):
                    connection.execute(
                        "INSERT INTO words VALUES (?, 1) ON CONFLICT(word) "
                        "DO UPDATE SET count=count+1",
                        (word,),
                    )
        chosen = tuple(
            _CAPTION_ROW.validate_python(row)[0]
            for row in connection.execute(
                "SELECT word FROM words ORDER BY count DESC, word LIMIT ?",
                (config.vocabulary_size - 4,),
            )
        )
    return Vocabulary(words=("<pad>", "<bos>", "<eos>", "<unk>", *chosen))


class Tokenizer:
    """Bind bounded vocabulary lookups and preserve BOS/EOS under truncation."""

    def __init__(self, vocabulary: Vocabulary, length: int) -> None:
        """Own the vocabulary mapping once rather than rebuilding it per caption."""
        self.vocabulary = vocabulary
        self.length = length
        self.indices = {word: index for index, word in enumerate(vocabulary.words)}

    def encode(self, caption: str) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
        """Return max_tokens+1 IDs for shifted teacher forcing with padding."""
        encoded = np.zeros(self.length + 1, dtype=np.int64)
        content = words(caption)[: self.length - 1]
        encoded[0] = 1
        for index, word in enumerate(content, start=1):
            encoded[index] = self.indices.get(word, 3)
        encoded[len(content) + 1] = 2
        return encoded

    def decode(self, tokens: tuple[int, ...]) -> str:
        """Render generated IDs, stopping at EOS and omitting padding and BOS."""
        result: list[str] = []
        for token in tokens:
            if token == 2:
                break
            if token not in (0, 1):
                result.append(self.vocabulary.words[token])
        return " ".join(result)


def image(record: CaptionRecord, config: Config) -> FloatArray:
    """Load one cached RGB image with explicit square resize and ImageNet scaling."""
    name = f"{record.image_id:012d}.jpg"
    path = config.data / record.partition / name
    download(f"{_ROOT}/{record.partition}/{name}", path)
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        resized = rgb.resize(
            (config.model.image_size, config.model.image_size), Image.Resampling.BICUBIC
        )
        pixels = np.asarray(resized, dtype=np.float32) / np.float32(255)
    pixels = (
        pixels - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(np.moveaxis(pixels, -1, 0)[None])


def batches(
    database: Path,
    config: Config,
    tokenizer: Tokenizer,
    epoch: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Cycle reference captions across epochs while retaining only one minibatch."""
    images: list[FloatArray] = []
    tokens: list[np.ndarray[tuple[int, ...], np.dtype[np.int64]]] = []
    for record in records(database, "train", config.train_images):
        images.append(image(record, config))
        tokens.append(tokenizer.encode(record.captions[epoch % len(record.captions)]))
        if len(images) == config.batch_size:
            yield (
                torch.from_numpy(np.concatenate(images)),
                torch.from_numpy(np.stack(tokens)),
            )
            images.clear()
            tokens.clear()
    if images:
        yield (
            torch.from_numpy(np.concatenate(images)),
            torch.from_numpy(np.stack(tokens)),
        )
