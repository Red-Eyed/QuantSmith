"""Bounded teacher-forced training and strictly bound model checkpoints."""

from pathlib import Path

import torch
from pydantic import ConfigDict, TypeAdapter
from rich.console import Console
from rich.progress import Progress
from torch import Tensor, nn

from examples.mobile_captioning.data import Tokenizer, batches
from examples.mobile_captioning.model import Captioner
from examples.mobile_captioning.schema import CheckpointInfo, Config, Vocabulary

_STATE = TypeAdapter(
    dict[str, Tensor], config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)


def train(config: Config, database: Path, vocabulary: Vocabulary) -> Captioner:
    """Start from pretrained visual weights and train on bounded streaming batches."""
    torch.manual_seed(config.seed)
    model = Captioner(config.model, len(vocabulary.words), pretrained=True)
    if config.freeze_backbone:
        model.encoder.backbone.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    tokenizer = Tokenizer(vocabulary, config.model.max_tokens)
    loss = nn.CrossEntropyLoss(ignore_index=0)
    steps, epoch = 0, 0
    initial, final = 0.0, 0.0
    console = Console(stderr=True, quiet=config.quiet)
    with Progress(console=console, disable=config.quiet) as progress:
        task = progress.add_task("Training caption decoder", total=config.steps)
        while steps < config.steps:
            seen = False
            model.train()
            if config.freeze_backbone:
                model.encoder.backbone.eval()
            for images, tokens in batches(database, config, tokenizer, epoch):
                seen = True
                optimizer.zero_grad(set_to_none=True)
                logits = model.forward(images, tokens[:, :-1])
                objective = loss.forward(
                    logits.reshape(-1, len(vocabulary.words)), tokens[:, 1:].reshape(-1)
                )
                if not torch.isfinite(objective):
                    raise ValueError("training produced a nonfinite loss")
                final = float(objective.detach())
                if steps == 0:
                    initial = final
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps += 1
                if steps % max(1, config.steps // 10) == 0:
                    console.print(f"Training {steps}/{config.steps}: loss {final:.3f}")
                progress.update(
                    task, advance=1, description=f"Training: loss {final:.3f}"
                )
                if steps == config.steps:
                    break
            if not seen:
                raise ValueError("training partition is empty")
            epoch += 1
    model.eval()
    torch.save(model.state_dict(), config.output / "weights.pt")
    info = CheckpointInfo(
        config=config,
        vocabulary=vocabulary,
        steps_completed=steps,
        initial_loss=initial,
        final_loss=final,
    )
    (config.output / "checkpoint.json").write_text(
        info.model_dump_json(indent=2) + "\n"
    )
    return model


def load(config: Config) -> tuple[Captioner, CheckpointInfo]:
    """Reject mismatched model geometry instead of evaluating different weights."""
    info = CheckpointInfo.model_validate_json(
        (config.output / "checkpoint.json").read_text()
    )
    if info.config.model != config.model:
        raise ValueError("model configuration differs from checkpoint.json")
    model = Captioner(info.config.model, len(info.vocabulary.words), pretrained=False)
    state = _STATE.validate_python(
        torch.load(config.output / "weights.pt", map_location="cpu", weights_only=True)
    )
    expected = model.state_dict()
    for name, value in state.items():
        if name not in expected:
            raise ValueError(f"unexpected checkpoint tensor: {name}")
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"checkpoint tensor shape/dtype mismatch: {name}")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"nonfinite checkpoint tensor: {name}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, info
