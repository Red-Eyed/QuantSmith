# ResNet-18 × quantization methods

Read [`main.py`](main.py) for the workflow. To run this example, use the
commands below from the repository root.

Pretrained residual image classification on Imagenette. This example measures ImageNet top-1/top-5 accuracy and before/after image predictions.

Start at [`run()` in main.py](main.py), then follow `export_onnx()` (for Torch
models), `calibration_samples()`, and the quantization configuration in the same
file. The example writes the workflow directly instead of delegating it to a suite.

## What the code demonstrates

`load_model()` explicitly loads TorchVision `ResNet18_Weights.IMAGENET1K_V1`
and its matching resize/crop/normalization transform. `export_onnx()` shows the
actual `torch.onnx.export` call with input name `images` and output name `logits`.

`calibration_samples()` reopens selected train images and applies exactly the
checkpoint's inference preprocessing. `quantized_variants()` calls QuantSmith directly
for residual convolutions and the classifier. The gallery compares the same val
images across Torch FP32, ONNX FP32, and the selected INT8 methods.

The ONNX input contract is `images`: float32 `[1, 3, 224, 224]`. Configuration is in [`schema.py`](schema.py).
All chosen recipes start from the same FP32 graph and use the same calibration
selection. The input factory is called again when fresh statistics are required.

## MinMax

```sh
uv run --locked --group examples -m examples.resnet18.main --methods minmax
```

Use the complete observed interval for static W8A8. The code configures
`StaticW8A8()` (or `QuantizationConfig()` in the projection).

## Percentile

```sh
uv run --locked --group examples -m examples.resnet18.main --methods percentile
```

Clip activation outliers before static W8A8. The code configures
`StaticW8A8(calibration=Percentile(percentile=config.percentile))`.
`--percentile 99.99` and `--histogram-bins 2048` expose the clipping controls.

## SmoothQuant

```sh
uv run --locked --group examples -m examples.resnet18.main --methods smoothquant
```

The stages are `(smoothing, minmax)`: `SmoothQuant(alpha=config.smoothquant_alpha)`
first balances channels, then a fresh calibration pass applies `StaticW8A8()`.
`--smoothquant-alpha` controls the balancing exponent. Smoothing alone is not the
INT8 result.

## Compare columns

```sh
uv run --locked --group examples -m examples.resnet18.main
uv run --locked --group examples -m examples.resnet18.main --methods minmax,percentile
uv run --locked --group examples -m examples.resnet18.main --help
```

Omitting `--methods` runs all three columns. Reports go to
`work_dir/artifacts/resnet18/index.html`, `results.json`, and `summary.md`. The directory
also contains FP32/INT8 ONNX graphs and full per-stage `.plan.json` decisions.
`--output` chooses another directory; `--quiet` disables progress and `--json`
prints the report as JSON. Downloads and output setup are automatic.

## Apply it to your task

Replace the builder/checkpoint in `load_model()` and provide its exact inference preprocessing. Replace the image loader with your own train/calibration and held-out sources. Update category names and output-shape validation if your classifier does not have 1,000 logits.

The QuantSmith boundary in the example is deliberately small:

```python
outcome = quantize(
    fp32,
    calibration_samples(calibration, loaded),
    recipes(config)[QuantizationMethod.PERCENTILE],
    histogram_bins=config.histogram_bins,
)
```

Model training, datasets, and HTML reports are demonstration scaffolding; QuantSmith
needs the FP32 ONNX graph, a replayable input factory, and the selected stage rules.
The result contains `model` and inspectable `plans`, or a typed `Err` diagnostic.

[Back to the model × method matrix](../README.md).

Imagenette train images calibrate; val images evaluate. The default is 128/256
images. Labels stay in the checkpoint's full 1,000-class ImageNet space. Timings
use warmed optimized CPU ORT; QDQ does not itself promise faster inference.
