"""Exercise the fluent facade with real ORT execution and replayable inputs."""

from collections.abc import Iterable, Mapping

import pytest

from quantsmith import Quantizer, Replay
from quantsmith.calibration import Samples
from quantsmith.config import PercentileConfig, QuantizationConfig, SmoothQuantConfig
from quantsmith.domain import InputArray
from tests.conftest import OperatorCase
from tests.outcomes import expect_error, expect_ok


def test_facade_owns_model(operator_case: OperatorCase, samples: Samples) -> None:
    """Binding snapshots the graph before callers change their protobuf."""
    job = Quantizer.from_config(QuantizationConfig()).quantize(operator_case.model)
    operator_case.model.graph.ClearField("node")
    result = expect_ok(job.calibrate(Replay(samples)))
    assert result.plans[0].operations
    assert result.execution


def test_iterable_and_factory_agree(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Reiterable batches and an explicit replay factory produce identical graphs."""
    job = Quantizer.from_config(
        QuantizationConfig(calibration=PercentileConfig(), histogram_bins=64)
    ).quantize(operator_case.model)
    iterable = expect_ok(job.calibrate(tuple(samples())))
    replay = expect_ok(job.calibrate(Replay(samples)))
    assert iterable.model.SerializeToString() == replay.model.SerializeToString()
    expect_error(job.calibrate(iter(samples())), "one-shot")


def test_factory_replays_without_caching(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Smoothing and histogram collection request fresh source passes."""
    passes = 0

    def replay() -> Iterable[Mapping[str, InputArray]]:
        """Track complete source passes without retaining any activations."""
        nonlocal passes
        passes += 1
        yield from samples()

    job = Quantizer.from_config(
        QuantizationConfig(
            smoothing=SmoothQuantConfig(),
            calibration=PercentileConfig(),
            histogram_bins=64,
        )
    ).quantize(operator_case.model)
    expect_ok(job.calibrate(Replay(replay)))
    assert passes >= 3


def test_invalid_model_does_not_read_samples(
    operator_case: OperatorCase, samples: Samples
) -> None:
    """Admission fails before touching a potentially expensive data source."""
    operator_case.model.graph.node[0].input[0] = "missing"

    def unread() -> Iterable[Mapping[str, InputArray]]:
        """Fail if a malformed model starts calibration."""
        pytest.fail("invalid graph consumed calibration samples")
        yield from samples()

    outcome = Quantizer.from_config(QuantizationConfig()).quantize(operator_case.model)
    expect_error(outcome.calibrate(Replay(unread)), "missing")
