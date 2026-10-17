# QuantSmith

**Post-training quantization experiments with decisions you can inspect.**

QuantSmith turns an FP32 ONNX model and representative inputs into a W8A8 QDQ graph.
It also returns the plans behind that graph: which consumer edges were quantized,
which producer outputs were quantized, which layers were excluded, and which scales, zero points, or channel transforms
were chosen. Use those decisions to investigate an accuracy drop, change one
choice, and measure again on held-out data.

The aim is a research workflow you can reproduce and modify:

1. **Establish an FP32 baseline** with your exported ONNX model.
2. **Replay representative calibration inputs** for MinMax, Percentile, or
   SmoothQuant followed by fresh W8A8 calibration.
3. **Inspect the stage plans** and use rules to select methods or exclude
   sensitive layers.
4. **Compare held-out results** across the baseline and quantized graphs.

QuantSmith also includes [GPTQv2 and operator-level QDrop-based reconstruction](src/quantsmith/reconstruction/README.md)
for experiments that spend additional calibration computation on weight choices.
The methods are documented with their assumptions, numerical examples, and
implementation limits.

## Package name

Version 0.8.0 changes the distribution and Python import name from `qraft` to
`quantsmith`. Replace `qraft` imports with `quantsmith` and `QraftError` with
`QuantSmithError`. The `qraft` package on PyPI belongs to an unrelated project.
Example report metadata now uses the `quantsmith` field, and example environment
variables use the `QUANTSMITH_` prefix.

## Start with an experiment

Each example is a readable source walkthrough with model export, calibration,
method selection, and held-out evaluation in its own `main.py`. Its README has
the commands to run it.

| Research question | Example |
| --- | --- |
| How do methods behave on global or local causal attention? | [Transformer](examples/transformer/README.md) · [Windowed Transformer](examples/windowed_transformer/README.md) |
| How do integer CNN and Transformer projections affect complete image captioning? | [MobileNetV4 + Stackformers captioner](examples/mobile_captioning/README.md) |
| How do activation outliers affect image classification? | [ResNet-18](examples/resnet18/README.md) · [MobileNetV2](examples/mobilenet_v2/README.md) · [EfficientNet-B0](examples/efficientnet_b0/README.md) |
| What is the smallest complete API example? | [ONNX projection](examples/projection/README.md) |
| How do the methods apply beyond those models? | [Wine MLP](examples/wine_mlp/README.md) · [Character RNN](examples/character_rnn/README.md) |
| What does reconstruction change? | [Held-out reconstruction comparison](examples/reconstruction/README.md) |

The [models × quantization methods matrix](examples/README.md) links every
MinMax, Percentile, and SmoothQuant walkthrough. These are experiments, including
cases where quantization loses substantial accuracy. For example, a historical
256-image [EfficientNet-B0 demonstration](examples/efficientnet_b0/README.md)
measured 75.00% FP32 top-1, 44.92% with MinMax, and 69.53% with Percentile.
Those subset results illustrate why calibration choice matters; they are not
accuracy guarantees for other data.

## Quantize your ONNX model

For a model with a float32 input named `features` and representative rows in
`calibration.npy`:

```python
from collections.abc import Iterator, Mapping

import numpy as np
import onnx

from quantsmith import Quantizer, Replay
from quantsmith.config import QuantizationConfig, SmoothQuantConfig
from quantsmith.domain import InputArray
from quantsmith.result import Err, Ok

features = np.load("calibration.npy", mmap_mode="r")


def samples() -> Iterator[Mapping[str, InputArray]]:
    """Replay representative inputs for every calibration pass."""
    for index in range(len(features)):
        yield {"features": np.asarray(features[index : index + 1], dtype=np.float32)}


config = QuantizationConfig(smoothing=SmoothQuantConfig())
outcome = (
    Quantizer.from_config(config)
    .quantize(onnx.load("model.onnx"))
    .calibrate(Replay(samples))
)
match outcome:
    case Err(error):
        print(error.kind, error.operation, error.detail)
    case Ok(result):
        onnx.save(result.model, "model_w8a8.onnx")
        # result.plans records the decisions made at each stage.
```

Use the exact input names, shapes, and dtypes of your exported graph. The sample
factory must replay the same inputs on each calibration pass; reserve different
data for evaluation. `QuantizationConfig()` selects MinMax by default.
`smoothing=SmoothQuantConfig()` balances channels before collecting fresh quantization
statistics. For histogram clipping, use `PercentileConfig`. To choose behavior
per layer, compose [rules](src/quantsmith/rules/README.md) with `StaticW8A8`,
`SmoothQuant`, and `Exclude`.

For smaller files with FP32 execution, use
[`Float16StorageConfig` or `BFloat16StorageConfig`](docs/float16-storage.md)
and `.calibrate([])`. They store eligible constants in two bytes and rely on ORT
graph optimization to
restore them once at initialization. Runtime RAM can increase; evaluate accuracy
after rounding. BF16 supports a wider range but has coarser precision than FP16.

## What you can inspect and change

| Experiment control | Where to look |
| --- | --- |
| Calibration ranges and clipping | [MinMax and Percentile](src/quantsmith/algorithms/README.md) |
| Channel balancing and per-layer decisions | [Algorithm guides](src/quantsmith/algorithms/README.md) · [Rules](src/quantsmith/rules/README.md) |
| Weight reconstruction | [GPTQv2 and QDrop](src/quantsmith/reconstruction/README.md) |
| Replaying inputs and collecting bounded statistics | [Calibration](src/quantsmith/calibration/README.md) |
| ONNX graph validation and QDQ lowering | [ONNX backend](src/quantsmith/backends/onnx/README.md) |
| Component boundaries, execution contract, restrictions and open work | [Architecture](docs/architecture.md) · [Integer execution](docs/integer-execution.md) |

Plans expose the chosen encodings and exclusions; examples save graphs, plans,
reports, and held-out predictions or output errors under `work_dir/artifacts/`.
QuantSmith streams calibration samples and retains bounded statistics rather than caching
the full dataset. Its planning algorithms can also be tested from supplied
statistics without executing ONNX.

## Scope and interpretation

QuantSmith currently targets FP32 ONNX graphs with constant-weight Conv, MatMul, and
Gemm operators. It supports INT8 or UINT8 encodings, per-tensor activations,
per-output-channel weights, and CPU ONNX Runtime calibration. SmoothQuant covers
ungrouped Conv and matrix projections. Reconstruction has its own
[operator-specific limits](src/quantsmith/reconstruction/README.md).

The static ORT pipeline completes input/output quantization boundaries and INT32
biases, then verifies selected supported operators compile to CPU integer kernels.
Unsupported layouts remain FP32 with reasons in `result.execution`; unexpected
floating fallbacks fail explicitly. Constants use compact integer storage and
unused originals are pruned. Public outputs remain FP32 after dequantization.
The low-level lowering and reconstruction APIs have separate, narrower contracts.

Integer execution does not establish lower latency or RAM. Output quantization
also changes rounding and can reduce accuracy. Measure the final graph on your
workload; see [observations and restrictions](docs/integer-execution.md).
GPU calibration, packed low-bit export, weight-only
quantization, and automatic reconstruction blocks are outside the current scope.

Want to extend a method or reproduce a result? See [CONTRIBUTION.md](CONTRIBUTION.md).

See the [typed pipeline design](docs/declarative-pipeline.md) for lifecycle types,
extension points, policy migration, and fold-only smoothing restrictions.

[Joint SmoothQuant and exact quantizer reuse](docs/joint-smoothing.md) add coordinated
offline smoothing and an optional step that reuses identical activation quantizers.
The report separates lossless reuse from changes in quantized task accuracy.

[Per-channel histogram calibration](docs/channel-histograms.md) provides typed
percentile and histogram-MSE policies for joint smoothing, streamed collection,
and explicit memory bounds, with the full five-model comparison.
