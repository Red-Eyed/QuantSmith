"""Admit an ONNX graph and execute an explicitly declared typed ORT pipeline."""

from onnx import ModelProto

from quantsmith.backends.onnx import admit, normalize
from quantsmith.calibration import Samples
from quantsmith.pipeline import (
    CalibrationContext,
    FloatGraph,
    Pipeline,
    QuantizationResult,
    Revision,
)
from quantsmith.result import Err, Ok, QuantSmithError, Result


def quantize(
    model: ModelProto,
    samples: Samples,
    program: Pipeline[FloatGraph, QuantizationResult],
    *,
    histogram_bins: int = 2048,
) -> Result[QuantizationResult, QuantSmithError]:
    """Admit an owned graph and interpret its explicit stage declaration."""
    match admit("ONNX validation", lambda: normalize(model)):
        case Err() as error:
            return error
        case Ok(current):
            source = normalize(current)
            value = FloatGraph(Revision(source, current))
            return program.run(value, CalibrationContext(samples, histogram_bins))
