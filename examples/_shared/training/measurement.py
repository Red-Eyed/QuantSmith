"""Task metrics, bounded prediction demonstrations, and warmed inference timings."""

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from rich.progress import Progress

from examples._shared.schema import Latency, Method
from examples._shared.training.schema import (
    Config,
    Dataset,
    Demonstration,
    Scores,
    TaskKind,
    Tokens,
)
from quantsmith.domain import FloatArray, InputArray
from quantsmith.result import Err, Ok, QuantSmithError, Result

Predictor: TypeAlias = Callable[[InputArray], Result[FloatArray, QuantSmithError]]


class Totals(BaseModel):
    """Keep scalar accumulators rather than storing evaluation logits."""

    count: int = Field(default=0)
    correct: int = Field(default=0)
    entropy: float = Field(default=0)
    agreement: int = Field(default=0)
    torch_agreement: int = Field(default=0)
    squared: float = Field(default=0)
    torch_squared: float = Field(default=0)
    maximum: float = Field(default=0)
    elements: int = Field(default=0)

    def update(
        self,
        actual: FloatArray,
        reference: FloatArray,
        eager: FloatArray,
        targets: Tokens,
    ) -> None:
        """Use stable log-softmax and float64 reductions for token/class metrics."""
        logits = actual.reshape(-1, actual.shape[-1]).astype(np.float64)
        truth = targets.reshape(-1)
        chosen = logits.argmax(axis=-1)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probability = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        self.count += truth.size
        self.correct += int(np.sum(chosen == truth))
        self.entropy -= float(log_probability[np.arange(truth.size), truth].sum())
        self.agreement += int(
            np.sum(chosen == reference.reshape(logits.shape).argmax(axis=-1))
        )
        self.torch_agreement += int(
            np.sum(chosen == eager.reshape(logits.shape).argmax(axis=-1))
        )
        difference = actual.astype(np.float64) - reference
        self.squared += float(np.sum(difference**2))
        self.torch_squared += float(np.sum((actual.astype(np.float64) - eager) ** 2))
        self.maximum = max(self.maximum, float(np.max(np.abs(difference))))
        self.elements += actual.size

    def finish(self) -> Scores:
        """Build a report after the nonempty held-out split has been evaluated."""
        entropy = self.entropy / self.count
        return Scores(
            observations=self.count,
            accuracy=self.correct / self.count,
            cross_entropy=entropy,
            perplexity=float(np.exp(entropy)),
            agreement_with_onnx=self.agreement / self.count,
            agreement_with_torch=self.torch_agreement / self.count,
            mse_vs_onnx=self.squared / self.elements,
            mse_vs_torch=self.torch_squared / self.elements,
            max_abs_vs_onnx=self.maximum,
        )


def continuation(
    predict: Predictor, context: Tokens, labels: tuple[str, ...], length: int
) -> Result[str, QuantSmithError]:
    """Greedily generate text with exactly the exported fixed-size rolling context."""
    current = context.copy()
    generated: list[str] = []
    for _ in range(length):
        match predict(current):
            case Err() as error:
                return error
            case Ok(logits):
                index = int(np.argmax(logits[0, -1]))
        generated.append(labels[index])
        current = np.concatenate(
            (current[:, 1:], np.asarray([[index]], dtype=np.int64)), axis=1
        )
    return Ok("".join(generated))


def demonstrate(
    data: Dataset,
    row: int,
    method: Method,
    logits: FloatArray,
    predict: Predictor,
    config: Config,
) -> Result[Demonstration, QuantSmithError]:
    """Display early held-out cases in selection order, without cherry-picking."""
    expected = data.evaluation.targets[row].reshape(-1)
    indices = logits.reshape(-1, len(data.labels)).argmax(axis=-1)
    generated = ""
    match data.kind:
        case TaskKind.WINE:
            context = "normalized features: " + np.array2string(
                data.evaluation.inputs[row], precision=2
            )
            truth, prediction = (
                data.labels[int(expected[0])],
                data.labels[int(indices[0])],
            )
        case TaskKind.TEXT | TaskKind.TRANSFORMER | TaskKind.WINDOWED:
            tokens = np.asarray(data.evaluation.inputs[row : row + 1], dtype=np.int64)
            context = "".join(data.labels[int(i)] for i in tokens[0])
            truth = "".join(data.labels[int(i)] for i in expected)
            prediction = "".join(data.labels[int(i)] for i in indices)
            match continuation(
                predict, tokens, data.labels, config.continuation_length
            ):
                case Err() as error:
                    return error
                case Ok(generated):
                    pass
    return Ok(
        Demonstration(
            identity=data.evaluation.identities[row],
            method=method,
            context=context,
            expected=truth,
            predicted=prediction,
            continuation=generated,
        )
    )


def measure(
    predictors: Mapping[Method, Predictor],
    data: Dataset,
    config: Config,
    progress: Progress,
) -> Result[tuple[dict[Method, Scores], tuple[Demonstration, ...]], QuantSmithError]:
    """Evaluate identical held-out inputs across variants, one row/window at a time."""
    totals = {method: Totals() for method in predictors}
    demonstrations: list[Demonstration] = []
    task = progress.add_task(
        f"{data.kind.value}: held-out evaluation", total=len(data.evaluation.inputs)
    )
    for row in range(len(data.evaluation.inputs)):
        inputs = data.evaluation.inputs[row : row + 1]
        outputs: dict[Method, FloatArray] = {}
        for method, predict in predictors.items():
            match predict(inputs):
                case Err() as error:
                    return error
                case Ok(output):
                    outputs[method] = output
        for method, logits in outputs.items():
            totals[method].update(
                logits,
                outputs[Method.ONNX],
                outputs[Method.TORCH],
                data.evaluation.targets[row : row + 1],
            )
            if row < config.demonstrations:
                match demonstrate(
                    data, row, method, logits, predictors[method], config
                ):
                    case Err() as error:
                        return error
                    case Ok(demo):
                        demonstrations.append(demo)
        progress.advance(task)
    progress.remove_task(task)
    return Ok(
        (
            {method: total.finish() for method, total in totals.items()},
            tuple(demonstrations),
        )
    )


def benchmark(
    predict: Predictor, inputs: InputArray, config: Config
) -> Result[Latency, QuantSmithError]:
    """Measure warmed batch-one inference with preprocessing and startup excluded."""
    timings: list[float] = []
    for index in range(config.warmup + config.benchmark_runs):
        start = perf_counter()
        match predict(inputs):
            case Err() as error:
                return error
            case Ok():
                pass
        if index >= config.warmup:
            timings.append((perf_counter() - start) * 1000)
    return Ok(
        Latency(
            median_ms=float(np.median(timings)),
            p95_ms=float(np.percentile(timings, 95)),
            runs=len(timings),
        )
    )
