"""Bounded caption-level COCO scorers and streaming teacher-forced metrics."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pydantic import BaseModel, Field, TypeAdapter
from rich.console import Console

from examples.mobile_captioning.data import Tokenizer, image, records, words
from examples.mobile_captioning.export import GraphPair
from examples.mobile_captioning.inference import generate_context, predictor, session
from examples.mobile_captioning.schema import Config


class CaptionExample(BaseModel):
    """Preserve generated captions and references without selecting attractive cases."""

    image_id: int
    caption: str
    references: tuple[str, ...]
    generated_tokens: int


class Quality(BaseModel):
    """Report generated-caption quality separately from teacher-forced prediction."""

    images: int
    cider: float = Field(allow_inf_nan=False)
    bleu4: float = Field(allow_inf_nan=False)
    token_accuracy: float
    cross_entropy: float
    unknown_target_fraction: float
    average_generated_tokens: float
    tokenization: str = (
        "lowercase regex words and punctuation; not official PTB tokenization"
    )


def evaluate(
    pair: GraphPair,
    database: Path,
    config: Config,
    tokenizer: Tokenizer,
    destination: Path,
) -> Quality:
    """Bound scorer memory by the explicit evaluation budget, never by train size."""
    predict = predictor(
        session(pair.encoder, config.threads), session(pair.decoder, config.threads)
    )
    references: dict[int, list[str]] = {}
    hypotheses: dict[int, list[str]] = {}
    correct, count, unknown, generated_count = 0, 0, 0, 0
    entropy = 0.0
    console = Console(stderr=True, quiet=config.quiet)
    with destination.open("w") as output:
        for row, record in enumerate(
            records(database, "evaluation", config.evaluation_images)
        ):
            context = predict.encode(image(record, config))
            generated = generate_context(predict, context, config.model.max_tokens)
            caption = tokenizer.decode(generated)
            example = CaptionExample(
                image_id=record.image_id,
                caption=caption,
                references=record.captions,
                generated_tokens=len(generated),
            )
            output.write(example.model_dump_json() + "\n")
            references[record.image_id] = [
                " ".join(words(text)) for text in record.captions
            ]
            hypotheses[record.image_id] = [caption]
            encoded = tokenizer.encode(record.captions[0])
            logits = predict.decode(encoded[None, :-1], context)[0].astype(np.float64)
            targets = encoded[1:]
            valid = targets != 0
            correct += int(np.sum((logits.argmax(axis=-1) == targets) & valid))
            count += int(valid.sum())
            unknown += int(np.sum(targets == 3))
            shifted = logits - logits.max(axis=-1, keepdims=True)
            logs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            entropy -= float(logs[np.arange(len(targets)), targets][valid].sum())
            generated_count += len(generated)
            if (row + 1) % 10 == 0:
                console.print(
                    f"Evaluated {row + 1}/{config.evaluation_images} captions"
                )
    if len(references) < 2 or count == 0:
        raise ValueError(
            "caption evaluation requires at least two images and valid targets"
        )
    cider_result = TypeAdapter(
        tuple[float, NDArray[np.float64]], config={"arbitrary_types_allowed": True}
    ).validate_python(Cider().compute_score(references, hypotheses))
    bleu_result = TypeAdapter(tuple[list[float], list[list[float]]]).validate_python(
        Bleu(4).compute_score(references, hypotheses, verbose=0)
    )
    return Quality(
        images=len(references),
        cider=cider_result[0],
        bleu4=bleu_result[0][3],
        token_accuracy=correct / count,
        cross_entropy=entropy / count,
        unknown_target_fraction=unknown / count,
        average_generated_tokens=generated_count / len(references),
    )
