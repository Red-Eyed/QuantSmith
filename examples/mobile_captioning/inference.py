"""Validated ORT sessions and image-once autoregressive caption generation."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from pydantic import ConfigDict, TypeAdapter

from quantsmith.domain import FloatArray, InputArray

_OUTPUTS = TypeAdapter(
    list[NDArray[np.generic]],
    config=ConfigDict(arbitrary_types_allowed=True, strict=True),
)
type Encode = Callable[[FloatArray], FloatArray]
type Decode = Callable[[NDArray[np.int64], FloatArray], FloatArray]


def session(path: Path, threads: int, *, profile: bool = False) -> ort.InferenceSession:
    """Use identical final ORT optimization for baseline and quantized graphs."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = profile
    options.profile_file_prefix = str(path.parent / f"{path.stem}_ort")
    if profile:
        options.optimized_model_filepath = str(path.with_suffix(".optimized.onnx"))
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run(runtime: ort.InferenceSession, feed: Mapping[str, InputArray]) -> FloatArray:
    """Reject wrong output count, dtype, empty tensors, or nonfinite logits."""
    outputs = _OUTPUTS.validate_python(runtime.run(None, dict(feed)))
    if len(outputs) != 1:
        raise ValueError("captioning graph must return exactly one tensor")
    output = outputs[0]
    if output.dtype != np.float32 or not output.size or not np.all(np.isfinite(output)):
        raise ValueError("captioning output must be nonempty finite FP32")
    return np.asarray(output, dtype=np.float32)


@dataclass(frozen=True)
class Predictor:
    """Separate graph execution from the pure greedy decoding policy."""

    encode: Encode
    decode: Decode


def predictor(
    encoder: ort.InferenceSession, decoder: ort.InferenceSession
) -> Predictor:
    """Bind native inputs to the explicit exported graph input names."""

    def encode(images: FloatArray) -> FloatArray:
        """Run visual encoding once for a complete caption."""
        return run(encoder, {"images": images})

    def decode(tokens: NDArray[np.int64], context: FloatArray) -> FloatArray:
        """Recompute a fixed padded prefix; Stackformers has no cache interface."""
        return run(decoder, {"tokens": tokens, "context": context})

    return Predictor(encode, decode)


def generate_context(
    predict: Predictor, context: FloatArray, length: int, *, stop_at_eos: bool = True
) -> tuple[int, ...]:
    """Greedily decode with a fixed buffer; never sample padding or another BOS."""
    tokens = np.zeros((1, length), dtype=np.int64)
    tokens[0, 0] = 1
    result: list[int] = []
    for position in range(length):
        logits = predict.decode(tokens, context)
        if logits.ndim != 3 or logits.shape[:2] != tokens.shape or logits.shape[2] < 4:
            raise ValueError("decoder logits do not match the caption buffer")
        scores = logits[0, position].copy()
        scores[:2] = -np.inf
        token = int(np.argmax(scores))
        result.append(token)
        if token == 2 and stop_at_eos:
            break
        if position + 1 < length:
            tokens[0, position + 1] = token
    return tuple(result)


def generate(predict: Predictor, images: FloatArray, length: int) -> tuple[int, ...]:
    """Include one image encoding and all decoder calls in end-to-end inference."""
    return generate_context(predict, predict.encode(images), length)
