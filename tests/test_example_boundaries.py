"""Exercise typed example serialization and real Torch preprocessing boundaries."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights
from torchvision.transforms._presets import ImageClassification
from typing_extensions import override

from examples._shared.plans import PlanRecord, save_plans
from examples._shared.training.data import wine_arrays
from examples._shared.training.training import checked_tensor
from examples._shared.vision.models import (
    LoadedModel,
    TorchImageTransform,
    TorchRunner,
    checked_categories,
)
from quantsmith.domain import Encoding, PerChannel, PerTensor
from quantsmith.plan import QuantizationPlan, QuantizeInput, RescaleInput


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_plan_json_compatibility(
    tmp_path: Path, per_channel: bool, dtype: type[np.int8] | type[np.uint8]
) -> None:
    """Keep every existing JSON field, including scalar shape and integer dtype."""
    shape = (2,) if per_channel else ()
    encoding = Encoding(
        scale=np.full(shape, 0.5, dtype=np.float32),
        zero_point=np.zeros(shape, dtype=dtype),
        granularity=PerChannel(axis=1) if per_channel else PerTensor(),
    )
    plans = (
        QuantizationPlan(
            operations=(
                RescaleInput(
                    node="n",
                    scale=np.asarray([1, 2], dtype=np.float32),
                    activation_axis=1,
                    weight_axis=0,
                ),
            ),
        ),
        QuantizationPlan(
            operations=(QuantizeInput(node="n", index=1, encoding=encoding),),
            excluded=("skip",),
        ),
    )
    destination = tmp_path / "plan.json"
    save_plans(destination, plans)
    expected: list[PlanRecord] = [
        {
            "operations": [
                {
                    "node": "n",
                    "scale": {"dtype": "float32", "shape": (2,), "values": [1.0, 2.0]},
                    "activation_axis": 1,
                    "weight_axis": 0,
                }
            ],
            "excluded": (),
        },
        {
            "operations": [
                {
                    "node": "n",
                    "index": 1,
                    "encoding": {
                        "scale": {
                            "dtype": "float32",
                            "shape": shape,
                            "values": [0.5] * (2 if per_channel else 1),
                        },
                        "zero_point": {
                            "dtype": np.dtype(dtype).name,
                            "shape": shape,
                            "values": [0.0] * (2 if per_channel else 1),
                        },
                        "granularity": {"axis": 1} if per_channel else {},
                    },
                }
            ],
            "excluded": ("skip",),
        },
    ]
    assert destination.read_text() == json.dumps(expected, indent=2)


@pytest.fixture
def image() -> Image.Image:
    """Provide an RGB input without downloading a dataset or model weights."""
    return Image.new("RGB", (280, 280), color=(64, 128, 192))


def test_torchvision_preprocessing(image: Image.Image) -> None:
    """Validate a real TorchVision recipe and preserve its report representation."""
    recipe = ImageClassification(crop_size=224)
    transform = TorchImageTransform(recipe)
    loaded = LoadedModel(nn.Identity(), transform, (), "test")
    actual = loaded.preprocess(image)
    expected = recipe(image).numpy()[None, ...]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual.shape == (1, 3, 224, 224)
    assert actual.flags.c_contiguous
    assert repr(transform) == repr(recipe)


def test_transform_rejects_non_tensor(image: Image.Image) -> None:
    """A Torch module can return arbitrary values despite its callable surface."""
    transform = TorchImageTransform(nn.Identity())
    with pytest.raises(ValueError, match="Tensor"):
        transform(image)


@pytest.mark.parametrize("shape", [(3, 4), (4, 4, 4)])
def test_preprocessing_rejects_invalid_shape(
    image: Image.Image, shape: tuple[int, ...]
) -> None:
    """A typed Tensor still requires CHW and RGB validation."""

    def transform(image: Image.Image) -> Tensor:
        """Return a tensor with a deliberately invalid image layout."""
        return torch.zeros(shape)

    loaded = LoadedModel(nn.Identity(), transform, (), "test")
    with pytest.raises(ValueError, match="CHW|RGB"):
        loaded.preprocess(image)


class TupleOutput(nn.Module):
    """Represent a Torch module that violates the single-tensor output contract."""

    @override
    def forward(self, inputs: Tensor) -> tuple[Tensor]:
        """Return a container even though its element is a valid tensor."""
        return (inputs,)


def test_torch_output_validation() -> None:
    """Tensor validation preserves autograd identity and rejects tuple outputs."""
    inputs = torch.ones((1, 1000), requires_grad=True)
    assert checked_tensor(nn.Identity(), inputs) is inputs
    with pytest.raises(ValueError, match="Tensor"):
        checked_tensor(TupleOutput(), inputs)
    with pytest.raises(ValueError, match="Tensor"):
        TorchRunner(TupleOutput())(np.ones((1, 1000), dtype=np.float32))


@pytest.mark.parametrize(
    "categories",
    [
        ["label"] * 999,
        ["label"] * 1001,
        [""] * 1000,
        [3] * 1000,
        ("label",) * 1000,
    ],
)
def test_category_metadata_rejection(
    categories: list[str | int] | tuple[str, ...],
) -> None:
    """Metadata validation rejects coercions, empty labels, and wrong cardinality."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    with patch.dict(weights.meta, {"categories": categories}):
        with pytest.raises(ValueError):
            checked_categories(weights)


def test_category_metadata() -> None:
    """Read actual checkpoint metadata without fetching checkpoint weights."""
    labels = checked_categories(ResNet18_Weights.IMAGENET1K_V1)
    assert len(labels) == 1000
    assert labels[0] == "tench"


@pytest.mark.parametrize(
    "payload",
    [
        ([1, 2], [0, 1]),
        [np.zeros((2, 3)), np.zeros(2)],
        (np.zeros((2, 3)),),
        (np.zeros((2, 3)), np.zeros(2), np.zeros(2)),
    ],
)
def test_wine_loader_rejection(
    payload: tuple[NDArray[np.generic] | list[int], ...] | list[NDArray[np.generic]],
) -> None:
    """Require exactly two native arrays without coercing wrong response layouts."""
    with patch("examples._shared.training.data.load_wine", return_value=payload):
        with pytest.raises(ValueError):
            wine_arrays()
