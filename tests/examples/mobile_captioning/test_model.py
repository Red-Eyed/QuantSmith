"""Check conditioning, causal masks, and real exported integer execution."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from examples.mobile_captioning.data import CaptionRecord, Tokenizer
from examples.mobile_captioning.export import (
    Candidate,
    ConvolutionOnly,
    GraphPair,
    KeepFloat,
    build_candidate,
)
from examples.mobile_captioning.inference import predictor, session
from examples.mobile_captioning.model import Captioner
from examples.mobile_captioning.schema import Config, ModelConfig, Vocabulary
from quantsmith import Quantizer, Replay
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx.integer_execution.audit import ExecutionReport
from quantsmith.config import QuantizationConfig
from quantsmith.domain import InputArray
from quantsmith.pipeline import FloatMatMulOutputs, QuantizedMatMulOutputs
from quantsmith.result import Err, Ok


def test_causal_decoder(model: Captioner) -> None:
    """Changing a future token must not change earlier predictions."""
    context = torch.randn(1, 4, 64)
    tokens = torch.tensor([[1, 4, 5, 6, 0, 0, 0, 0]])
    changed = tokens.clone()
    changed[0, 2] = 7
    with torch.inference_mode():
        original = model.decoder.forward(tokens, context)
        modified = model.decoder.forward(changed, context)
    torch.testing.assert_close(original[:, :2], modified[:, :2])
    assert not torch.allclose(original[:, 2], modified[:, 2])


def test_visual_conditioning(model: Captioner) -> None:
    """Cross-attention must change logits when the image context changes."""
    tokens = torch.ones((1, 8), dtype=torch.int64)
    with torch.inference_mode():
        first = model.decoder.forward(tokens, torch.randn(1, 4, 64))
        second = model.decoder.forward(tokens, torch.randn(1, 4, 64))
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("policy", [QuantizedMatMulOutputs(), FloatMatMulOutputs()])
def test_real_integer_decoder(
    graphs: GraphPair,
    config: Config,
    policy: QuantizedMatMulOutputs | FloatMatMulOutputs,
) -> None:
    """Quantize actual Stackformers cross-attention projections and audit execution."""

    def samples() -> Iterable[Mapping[str, InputArray]]:
        """Replay deterministic visual contexts and two masked caption prefixes."""
        rng = np.random.default_rng(7)
        for valid in (1, 5):
            tokens = np.zeros((1, 8), dtype=np.int64)
            tokens[0, :valid] = 1
            yield {
                "tokens": tokens,
                "context": rng.normal(size=(1, 4, 64)).astype(np.float32),
            }

    outcome = (
        Quantizer.from_config(QuantizationConfig(matmul_outputs=policy))
        .quantize(onnx.load(graphs.decoder))
        .calibrate(Replay(samples))
    )
    match outcome:
        case Err(error):
            pytest.fail(error.model_dump_json())
        case Ok(result):
            assert isinstance(result.execution, ExecutionReport)
            expected = (
                "MatMulIntegerToFloat"
                if isinstance(policy, FloatMatMulOutputs)
                else "QLinearMatMul"
            )
            assert any(
                node.executed_operator == expected
                for node in result.execution.integer_nodes
            )


def test_encoder_parity(graphs: GraphPair, config: Config) -> None:
    """Keep the CNN/encoder and decoder deployment boundary fully executable."""
    runtime = predictor(session(graphs.encoder, 1), session(graphs.decoder, 1))
    context = runtime.encode(np.zeros((1, 3, 64, 64), dtype=np.float32))
    assert context.shape == (1, 4, 64)
    logits = runtime.decode(np.ones((1, 8), dtype=np.int64), context)
    assert logits.shape == (1, 8, 8)


def test_real_integer_encoder(graphs: GraphPair) -> None:
    """Verify MobileNetV4 convolutions compile to actual integer kernels."""

    def samples() -> Iterable[Mapping[str, InputArray]]:
        """Provide deterministic finite RGB calibration without network access."""
        rng = np.random.default_rng(15)
        for _ in range(2):
            yield {"images": rng.normal(size=(1, 3, 64, 64)).astype(np.float32)}

    match (
        Quantizer.from_config(QuantizationConfig())
        .quantize(onnx.load(graphs.encoder))
        .calibrate(Replay(samples))
    ):
        case Err(error):
            pytest.fail(error.model_dump_json())
        case Ok(result):
            assert isinstance(result.execution, ExecutionReport)
            assert any(
                node.executed_operator == "QLinearConv"
                for node in result.execution.integer_nodes
            )


@pytest.mark.parametrize(
    "encoder_policy",
    [
        QuantizationConfig(),
        ConvolutionOnly(algorithm=StaticW8A8(), histogram_bins=32),
    ],
)
def test_encoder_only_preserves_decoder(
    graphs: GraphPair,
    database: Path,
    config: Config,
    vocabulary: Vocabulary,
    monkeypatch: pytest.MonkeyPatch,
    encoder_policy: QuantizationConfig | ConvolutionOnly,
) -> None:
    """An encoder-only candidate must deploy the original decoder bytes unchanged."""

    def pixels(
        record: CaptionRecord, settings: Config
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        """Use deterministic image admission without external downloads."""
        return np.full((1, 3, 64, 64), record.image_id / 100, dtype=np.float32)

    monkeypatch.setattr("examples.mobile_captioning.export.image", pixels)
    candidate = Candidate("encoder_only", encoder_policy, KeepFloat())
    pair = build_candidate(
        candidate, graphs, database, config, Tokenizer(vocabulary, 8)
    )
    assert pair.decoder.read_bytes() == graphs.decoder.read_bytes()
    assert not pair.decoder.with_suffix(".plans.json").exists()
    if isinstance(encoder_policy, ConvolutionOnly):
        audit = ExecutionReport.model_validate_json(
            pair.encoder.with_suffix(".execution.json").read_text()
        )
        assert audit.integer_nodes
        assert {node.source_operator for node in audit.integer_nodes} <= {"Conv", "Add"}


@pytest.mark.parametrize(
    "config",
    [
        {"image_size": 63},
        {"dim": 65, "heads": 4},
        {"dim": 12, "heads": 4},
    ],
)
def test_invalid_geometry(config: dict[str, int]) -> None:
    """Fail malformed spatial and rotary dimensions before model construction."""
    with pytest.raises(ValueError):
        ModelConfig.model_validate(config)
