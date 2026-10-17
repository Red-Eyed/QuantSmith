"""Validated ONNX Runtime execution and streaming numerical evaluation."""

from collections.abc import Mapping
from copy import deepcopy
from time import perf_counter

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from onnx import ModelProto, TensorProto
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    InvalidGraph,
    NotImplemented,
    RuntimeException,
)
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from quantsmith.backends.onnx import normalize
from quantsmith.calibration import Evaluator, Samples
from quantsmith.domain import FloatArray, InputArray
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)

_OUTPUT_ARRAYS = TypeAdapter(
    list[NDArray[np.generic]],
    config=ConfigDict(strict=True, arbitrary_types_allowed=True),
)
_NO_OPTIMIZATION = ort.GraphOptimizationLevel.ORT_DISABLE_ALL


class OnnxEvaluator:
    """Cache one instrumented CPU session, exposing only requested tensors."""

    def __init__(
        self,
        model: ModelProto,
        *,
        optimization: ort.GraphOptimizationLevel = _NO_OPTIMIZATION,
    ) -> None:
        """Own a validated graph; session construction is deferred to execution."""
        self.model = normalize(model)
        self.optimization = optimization
        self.sessions: dict[tuple[str, ...], ort.InferenceSession] = {}

    def _session(self, outputs: tuple[str, ...]) -> ort.InferenceSession:
        """Expose requested tensors and apply the declared optimization level."""
        if outputs in self.sessions:
            return self.sessions[outputs]
        model = deepcopy(self.model)
        metadata = {
            value.name: value
            for value in (
                *model.graph.input,
                *model.graph.output,
                *model.graph.value_info,
            )
        }
        del model.graph.output[:]
        for name in outputs:
            if (
                name not in metadata
                or metadata[name].type.tensor_type.elem_type != TensorProto.FLOAT
            ):
                raise ValueError(f"requested tensor lacks FP32 metadata: {name}")
            model.graph.output.append(metadata[name])
        options = ort.SessionOptions()
        options.graph_optimization_level = self.optimization
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        self.sessions.clear()
        self.sessions[outputs] = session
        return session

    def _run(
        self, sample: Mapping[str, InputArray], outputs: tuple[str, ...]
    ) -> Mapping[str, FloatArray]:
        """Validate ORT's dynamic outputs before exposing them to algorithms."""
        session = self._session(outputs)
        arrays = _OUTPUT_ARRAYS.validate_python(
            session.run(list(outputs), dict(sample))
        )
        if len(arrays) != len(outputs):
            raise ValueError("runtime returned an invalid output sequence")
        result: dict[str, FloatArray] = {}
        for name, value in zip(outputs, arrays, strict=True):
            if value.dtype != np.dtype(np.float32):
                raise ValueError("runtime returned a non-FP32 tensor")
            array = np.asarray(value, dtype=np.float32)
            if not array.size or not np.all(np.isfinite(array)):
                raise ValueError("runtime returned empty or nonfinite data")
            result[name] = array
        return result

    def run(
        self, sample: Mapping[str, InputArray], outputs: tuple[str, ...]
    ) -> Result[Mapping[str, FloatArray], QuantSmithError]:
        """Translate documented ORT execution failures and tensor validation errors."""
        try:
            return validate("ONNX execution", lambda: self._run(sample, outputs))
        except (
            Fail,
            InvalidArgument,
            InvalidGraph,
            NotImplemented,
            RuntimeException,
        ) as error:
            return failure(FailureKind.EXECUTION, "ONNX execution", str(error))


class Evaluation(BaseModel):
    """Summarize elementwise error and measured wall time across streamed batches."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    mean_squared_error: float = Field()
    maximum_absolute_error: float = Field()
    batches: int = Field()
    candidate_seconds: float = Field()


class OutputError(BaseModel):
    """Carry one checked tensor's aggregate error without retaining its arrays."""

    squared: float = Field()
    maximum: float = Field()
    elements: int = Field()


def compare(
    actual: FloatArray, expected: FloatArray
) -> Result[OutputError, QuantSmithError]:
    """Reject malformed plugin observations before performing numerical reduction."""
    if actual.shape != expected.shape:
        return failure(
            FailureKind.INVALID_DATA, "evaluation", "evaluation output shapes differ"
        )
    if (
        not actual.size
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        return failure(
            FailureKind.INVALID_DATA,
            "evaluation",
            "evaluation outputs must be nonempty and finite",
        )
    difference = actual.astype(np.float64) - expected
    return Ok(
        OutputError(
            squared=float(np.sum(difference * difference)),
            maximum=float(np.max(np.abs(difference))),
            elements=difference.size,
        )
    )


def evaluate(
    reference: Evaluator,
    candidate: Evaluator,
    samples: Samples,
    outputs: tuple[str, ...],
) -> Result[Evaluation, QuantSmithError]:
    """Compare streamed outputs; return data failures, including empty sources.

    Timings include session startup. Unexpected dependency errors propagate.
    """
    squared, maximum, seconds = (0.0, 0.0, 0.0)
    count, batches = (0, 0)
    if not outputs:
        return failure(FailureKind.EMPTY, "evaluation", "evaluation requires outputs")
    for sample in samples():
        match reference.run(sample, outputs):
            case Err() as error:
                return error
            case Ok(expected):
                pass
        start = perf_counter()
        match candidate.run(sample, outputs):
            case Err() as error:
                return error
            case Ok(actual):
                pass
        seconds += perf_counter() - start
        for name in outputs:
            if name not in actual or name not in expected:
                return failure(
                    FailureKind.INVALID_DATA, "evaluation", f"missing output {name}"
                )
            match compare(actual[name], expected[name]):
                case Err() as error:
                    return error
                case Ok(error):
                    squared += error.squared
                    maximum = max(maximum, error.maximum)
                    count += error.elements
        batches += 1
    if not count:
        return failure(
            FailureKind.EMPTY, "evaluation", "evaluation source yielded no values"
        )
    return Ok(
        Evaluation(
            mean_squared_error=squared / count,
            maximum_absolute_error=maximum,
            batches=batches,
            candidate_seconds=seconds,
        )
    )
