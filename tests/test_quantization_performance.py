"""Verify streamed ORT calibration preserves every sample's extrema."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from PIL import Image
from pydantic_settings import CliApp
from torch import Tensor, nn

from examples._shared.vision.models import LoadedModel
from examples._shared.vision.schema import ImageRecord
from examples.quantization_performance import main as performance
from quantsmith.domain import FloatArray


def test_cli_parses_model_and_variant_lists() -> None:
    """Accept documented lowercase comma-separated selections."""
    config = CliApp.run(
        performance.Config,
        cli_args=[
            "--models",
            "mobilenet_v2",
            "--variants",
            "fp32,ort_qdq_tensor_u8s8",
        ],
    )
    assert config.models == [performance.Model.MOBILENET_V2]
    assert config.variants == [performance.Variant.FP32, performance.Variant.ORT_TENSOR]


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> performance.Reader:
    """Expose black and bright batches from metadata without image-file I/O."""

    def transform(image: Image.Image) -> Tensor:
        """Supply an unused transform because input loading is replaced below."""
        return torch.zeros((3, 2, 2))

    def read_input(
        record: ImageRecord, preprocess: Callable[[Image.Image], FloatArray]
    ) -> FloatArray:
        """Use labels as deterministic synthetic intensity values."""
        return np.full((1, 3, 2, 2), record.label, dtype=np.float32)

    monkeypatch.setattr(performance, "read_input", read_input)
    records = tuple(
        ImageRecord(path=Path(f"{label}.png"), label=label, class_name="synthetic")
        for label in (0, 255)
    )
    return performance.Reader(
        records, LoadedModel(nn.Identity(), transform, (), "test")
    )


def test_reader_range_replay(reader: performance.Reader) -> None:
    """Range selection can replay any batch and stops without reading beyond it."""
    assert len(reader) == 2
    reader.set_range(1, 2)
    np.testing.assert_array_equal(reader.get_next()["images"], 255)
    assert reader.get_next() == {}
    reader.set_range(0, 1)
    np.testing.assert_array_equal(reader.get_next()["images"], 0)
    assert reader.get_next() == {}
    with pytest.raises(StopIteration):
        next(reader)


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 0), (0, 3)])
def test_reader_rejects_invalid_range(
    reader: performance.Reader, start: int, end: int
) -> None:
    """Reject invalid slices before decoding or altering the selected range."""
    with pytest.raises(ValueError, match="invalid calibration range"):
        reader.set_range(start, end)


def test_strided_minmax_preserves_all_sample_extrema(
    reader: performance.Reader, tmp_path: Path
) -> None:
    """Actual ORT calibration merges extrema across one-image strides."""
    source, destination = tmp_path / "fp32.onnx", tmp_path / "quantized.onnx"
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Conv", ["images", "weight"], ["output"])],
            "streamed_calibration",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
            [
                numpy_helper.from_array(
                    np.ones((1, 3, 1, 1), dtype=np.float32), "weight"
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.save(model, source)
    quantize_static(
        source,
        destination,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        extra_options={"CalibStridedMinMax": 1},
    )
    parameters = {
        value.name: numpy_helper.to_array(value)
        for value in onnx.load(destination).graph.initializer
    }
    np.testing.assert_allclose(parameters["images_scale"], 1.0)
    np.testing.assert_allclose(parameters["output_scale"], 3.0)
