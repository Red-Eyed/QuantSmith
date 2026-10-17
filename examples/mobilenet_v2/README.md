# MobileNetV2 × quantization methods

Read [`main.py`](main.py) for the workflow. To run this example, use the
commands below from the repository root.

Pretrained image classification with depthwise convolutions. This example measures ImageNet accuracy and changes in predicted classes.

Start at [`run()` in main.py](main.py), then follow `export_onnx()` (for Torch
models), `calibration_samples()`, and the quantization configuration in the same
file. The example writes the workflow directly instead of delegating it to a suite.

## What the code demonstrates

`load_model()` explicitly loads `MobileNet_V2_Weights.IMAGENET1K_V2` and
the checkpoint's transforms. This architecture combines pointwise convolutions,
grouped depthwise convolutions, and a classifier; it is not a residual-CNN alias.

Read the grouped-convolution condition in `ConstantWeights.matches()`. Ordinary
static quantization includes depthwise layers. SmoothQuant excludes grouped Conv
because QuantSmith's rescaling supports ungrouped Conv; the following MinMax stage
still quantizes supported depthwise weights. The real val-image gallery shows
whether class predictions change.

The ONNX input contract is `images`: float32 `[1, 3, 224, 224]`. Configuration is in [`schema.py`](schema.py).
All chosen recipes start from the same FP32 graph and use the same calibration
selection. The input factory is called again when fresh statistics are required.

## MinMax

```sh
uv run --locked --group examples -m examples.mobilenet_v2.main --methods minmax
```

Use the complete observed interval for static W8A8. The code configures
`StaticW8A8()` (or `QuantizationConfig()` in the projection).

## Percentile

```sh
uv run --locked --group examples -m examples.mobilenet_v2.main --methods percentile
```

Clip activation outliers before static W8A8. The code configures
`StaticW8A8(calibration=Percentile(percentile=config.percentile))`.
`--percentile 99.99` and `--histogram-bins 2048` expose the clipping controls.

## SmoothQuant

```sh
uv run --locked --group examples -m examples.mobilenet_v2.main --methods smoothquant
```

The stages are `(smoothing, minmax)`: `SmoothQuant(alpha=config.smoothquant_alpha)`
first balances channels, then a fresh calibration pass applies `StaticW8A8()`.
`--smoothquant-alpha` controls the balancing exponent. Smoothing alone is not the
INT8 result.

## Activation-function-aware calibration

```sh
uv run --locked --group examples -m examples.mobilenet_v2.main --activation-function-aware true
```

This opt-in changes the output encoding of eligible private `Conv → Relu/Clip`
edges to use statistics observed after the activation function. ORT can then
eliminate conversion boundaries between integer regions. It applies to all
selected static recipes, including the MinMax stage after SmoothQuant; smoothing
still precedes fresh calibration. Shared/public Conv outputs, dynamic Clip bounds,
and other activation functions retain their existing policy.

Clipping and rounding change, so compare held-out accuracy as well as latency.
The full example configuration, including this flag, is saved in
`configuration.json`. Default behavior is unchanged. See
[the design and measured tradeoffs](../../docs/integer-execution.md#activation-function-aware-calibration).

## Compare columns

```sh
uv run --locked --group examples -m examples.mobilenet_v2.main
uv run --locked --group examples -m examples.mobilenet_v2.main --methods minmax,percentile
uv run --locked --group examples -m examples.mobilenet_v2.main --help
```

Omitting `--methods` runs all three columns. Reports go to
`work_dir/artifacts/mobilenet_v2/index.html`, `results.json`, and `summary.md`. The directory
also contains FP32/INT8 ONNX graphs and full per-stage `.plan.json` decisions.
`--output` chooses another directory; `--quiet` disables progress and `--json`
prints the report as JSON. Downloads and output setup are automatic.

## Apply it to your task

Replace the pretrained builder and matching preprocessing for your mobile model. Keep the grouped-convolution check if you use depthwise layers. Replace image sources, categories, and classifier-output validation for your task; compare measured quality before choosing a method.

The QuantSmith boundary in the example is deliberately small:

```python
outcome = quantize(
    fp32,
    calibration_samples(calibration, loaded),
    recipes(config)[QuantizationMethod.PERCENTILE],
    histogram_bins=config.histogram_bins,
    activation_function_aware=config.activation_function_aware,
)
```

Model training, datasets, and HTML reports are demonstration scaffolding; QuantSmith
needs the FP32 ONNX graph, a replayable input factory, and the selected stage rules.
The result contains `model` and inspectable `plans`, or a typed `Err` diagnostic.

[Back to the model × method matrix](../README.md).

Imagenette train images calibrate; val images evaluate. The default is 128/256
images. Labels stay in the checkpoint's full 1,000-class ImageNet space. Timings
use warmed optimized CPU ORT; QDQ does not itself promise faster inference.

## Inspect runtime overhead

The profiling command accepts a saved finite FP32 `.npy` input matching `images`.
For example, save the existing example's preprocessed calibration input:

```sh
uv run --locked --group examples - <<'PY'
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.models import MobileNet_V2_Weights

root = Path("work_dir/artifacts/cache/imagenette/imagenette2-160/train")
image_path = next(root.rglob("*.JPEG"))
with Image.open(image_path) as image:
    tensor = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()(image.convert("RGB"))
np.save("work_dir/profile_input.npy", tensor.numpy()[None, ...])
PY
uv run --locked --group examples -m examples._shared.vision.profile_runtime \
  --model work_dir/artifacts/mobilenet_v2/int8_smoothquant.onnx \
  --input work_dir/profile_input.npy \
  --output work_dir/artifacts/mobilenet_v2/profile --runs 300
```

The example run above provisions the model and dataset automatically. Repeat the
profiling invocation sequentially for each model, choosing a separate output
directory; each invocation starts a fresh process. `--quiet` suppresses progress.
The command always prints JSON and saves `report.json`, `optimized.onnx`, and an
ORT trace. Latency is measured with profiling disabled; operator totals come from
a separate profiling session and are summed microseconds across all `--runs`.
RSS values are whole-process cumulative high-water marks at three checkpoints,
not current RSS, isolated tensor allocations, or proof of steady-state RAM savings.
The optimized graph can contain hardware-specific operators and is diagnostic,
not a portable deployment artifact.

## ORT integer SmoothQuant execution

```sh
uv run --locked --group examples -m examples.mobilenet_v2.main \
  --methods smoothquant \
  --output work_dir/artifacts/mobilenet_v2_ort
```

SmoothQuant targets ONNX Runtime CPU execution by default. It adds calibrated UINT8
boundaries around smoothing and residual additions, folds eligible channel scales
into convolution output encodings, applies integer clipping, and replaces remaining
supported divisions with ORT integer multiplication. It keeps smoothing on all
originally selected layers, but changes rounding and quantizes reciprocal
coefficients. No additional flag is required. JSON reports record
`execution_target: "onnxruntime"`; HTML and Markdown explain the execution target.

On the recorded arm64 Mac experiment with 128 calibration and 256 held-out images,
three alternating 500-run measurements gave about 3.40 ms for ORT SmoothQuant versus
3.36 ms for MinMax and 6.42 ms for original SmoothQuant. Top-1 was 166/256 correct,
versus 162/256 for MinMax and 167/256 for original SmoothQuant. These results do not
establish accuracy equivalence or universal speedups. Process peak RSS remained
slightly above MinMax. See the [experiment log](../_shared/vision/DEVLOG.md).
