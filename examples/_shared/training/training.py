"""Bounded minibatch training and validated eager Torch predictions."""

from collections.abc import Callable
from typing import TypeAlias, assert_never

import numpy as np
import torch
from pydantic import ConfigDict, TypeAdapter
from rich.progress import Progress
from torch import Tensor, nn

from examples._shared.training.schema import Config, Dataset, TaskKind
from quantsmith.domain import FloatArray, InputArray

_TENSOR_OUTPUT = TypeAdapter(
    Tensor, config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)


def checked_tensor(model: nn.Module, inputs: Tensor) -> Tensor:
    """Validate the dynamically typed Torch module boundary once."""
    return _TENSOR_OUTPUT.validate_python(model(inputs))


def eager(model: nn.Module, inputs: InputArray) -> FloatArray:
    """Execute eager CPU inference and detach native float32 logits."""
    with torch.inference_mode():
        output = checked_tensor(model, torch.from_numpy(inputs))
    return np.asarray(output.detach().cpu().numpy(), dtype=np.float32)


def minibatch(
    data: Dataset, config: Config, rng: np.random.Generator
) -> tuple[Tensor, Tensor]:
    """Allocate one minibatch without storing training windows in the dataset."""
    match data.kind:
        case TaskKind.WINE:
            indices = rng.integers(0, len(data.train.inputs), config.batch_size)
            inputs, targets = data.train.inputs[indices], data.train.targets[indices]
        case TaskKind.TEXT | TaskKind.TRANSFORMER | TaskKind.WINDOWED:
            offsets = rng.integers(
                0, len(data.train.inputs) - config.sequence_length, config.batch_size
            )
            inputs = np.stack(
                [data.train.inputs[i : i + config.sequence_length] for i in offsets]
            )
            targets = np.stack(
                [data.train.targets[i : i + config.sequence_length] for i in offsets]
            )
        case _:
            assert_never(data.kind)
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def train(
    model: nn.Module, data: Dataset, config: Config, progress: Progress
) -> tuple[float, float]:
    """Train only on the training split; report losses on one fixed training probe."""
    model.train()
    match data.kind:
        case TaskKind.WINE:
            steps = config.mlp_steps
        case TaskKind.TEXT:
            steps = config.rnn_steps
        case TaskKind.TRANSFORMER | TaskKind.WINDOWED:
            steps = config.transformer_steps
        case _:
            assert_never(data.kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed)
    probe, targets = minibatch(data, config, np.random.default_rng(config.seed + 1))

    def probe_loss() -> float:
        """Compare the same training minibatch before and after optimization."""
        with torch.no_grad():
            logits = checked_tensor(model, probe).reshape(-1, len(data.labels))
            return float(loss.forward(logits, targets.reshape(-1)))

    initial = probe_loss()
    task = progress.add_task(f"{data.kind.value}: training", total=steps)
    for _ in range(steps):
        inputs, expected = minibatch(data, config, rng)
        optimizer.zero_grad(set_to_none=True)
        logits = checked_tensor(model, inputs).reshape(-1, len(data.labels))
        loss.forward(logits, expected.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        progress.advance(task)
    progress.remove_task(task)
    model.eval()
    return initial, probe_loss()


Predict: TypeAlias = Callable[[InputArray], FloatArray]
