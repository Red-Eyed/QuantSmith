# Models × quantization methods

Choose a row and a method. Each model folder has a README and a concrete
[`main.py`](transformer/main.py) that shows model setup, ONNX export, replayable
calibration inputs, QuantSmith configuration, and held-out comparison. The code is
intentionally explicit: you can read one example without following a shared runner.

| Model / task | MinMax | Percentile | SmoothQuant → MinMax |
| --- | --- | --- | --- |
| [Global causal Transformer](transformer/README.md) | [Read](transformer/README.md#minmax) | [Read](transformer/README.md#percentile) | [Read](transformer/README.md#smoothquant) |
| [Windowed causal Transformer](windowed_transformer/README.md) | [Read](windowed_transformer/README.md#minmax) | [Read](windowed_transformer/README.md#percentile) | [Read](windowed_transformer/README.md#smoothquant) |
| [MobileNetV4 + Stackformers captioner](mobile_captioning/README.md) | [Read](mobile_captioning/README.md) | [Read](mobile_captioning/README.md) | Outside this study |
| [Wine MLP](wine_mlp/README.md) | [Read](wine_mlp/README.md#minmax) | [Read](wine_mlp/README.md#percentile) | [Read](wine_mlp/README.md#smoothquant) |
| [ResNet-18](resnet18/README.md) | [Read](resnet18/README.md#minmax) | [Read](resnet18/README.md#percentile) | [Read](resnet18/README.md#smoothquant) |
| [MobileNetV2](mobilenet_v2/README.md) | [Read](mobilenet_v2/README.md#minmax) | [Read](mobilenet_v2/README.md#percentile) | [Read](mobilenet_v2/README.md#smoothquant) |
| [EfficientNet-B0](efficientnet_b0/README.md) | [Read](efficientnet_b0/README.md#minmax) | [Read](efficientnet_b0/README.md#percentile) | [Read](efficientnet_b0/README.md#smoothquant) |
| [Character RNN](character_rnn/README.md) | [Read](character_rnn/README.md#minmax) | [Read](character_rnn/README.md#percentile) | [Read](character_rnn/README.md#smoothquant) |
| [Small projection](projection/README.md) | [Read](projection/README.md#minmax) | [Read](projection/README.md#percentile) | [Read](projection/README.md#smoothquant) |

The Transformer examples are the primary reading path. The
[image captioner](mobile_captioning/README.md) adds visual cross-attention,
separate encoder/decoder graphs, and complete autoregressive generation benchmarks.
Start with
[the projection](projection/README.md) if you want the smallest QuantSmith API example.
The RNN is a secondary demonstration.

## Read one example

Open its README, then read `run()` near the top of `main.py`. The same file contains
the exporter, calibration factory, and quantization configurations. A local
`model.py` holds the trained model's architecture; `schema.py` holds CLI controls.
Downloads, minibatch training, metric accumulation, and HTML rendering are support
helpers in [`_shared`](_shared/README.md).

Running an example is optional. Commands and runtime options live in that
example’s own README.

## What the method columns mean

- **MinMax:** static W8A8 using the full observed activation interval.
- **Percentile:** static W8A8 after histogram-based clipping of activation outliers.
- **SmoothQuant → MinMax:** balance activation/weight channels, lower that floating
  transform, then collect fresh statistics and apply static W8A8.

Every selected method starts from the same FP32 model. The eight Torch examples
check FP32 ONNX parity first and compare against both Torch and ONNX baselines.
The projection is constructed directly in ONNX and compares against its FP32 graph.
Calibration and held-out data remain separate.

## Input contracts

| Example | ONNX input after preprocessing |
| --- | --- |
| [Global causal Transformer](transformer/README.md) | `tokens`: int64 `[1, 32]` |
| [Windowed causal Transformer](windowed_transformer/README.md) | `tokens`: int64 `[1, 32]` |
| [Mobile captioner](mobile_captioning/README.md) | Encoder `images`: float32 `[1, 3, 224, 224]`; decoder `tokens`: int64 `[1, 32]`, `context`: float32 `[1, 49, 256]` |
| [Wine MLP](wine_mlp/README.md) | `features`: float32 `[1, 13]` |
| [ResNet-18](resnet18/README.md) | `images`: float32 `[1, 3, 224, 224]` |
| [MobileNetV2](mobilenet_v2/README.md) | `images`: float32 `[1, 3, 224, 224]` |
| [EfficientNet-B0](efficientnet_b0/README.md) | `images`: float32 `[1, 3, 224, 224]` |
| [Character RNN](character_rnn/README.md) | `tokens`: int64 `[1, 32]` |
| [Small projection](projection/README.md) | `x`: float32 `[16, 8]` |

Text context length is configurable before fixed-shape export. Classification image
transforms come from the pretrained checkpoint; captioning uses an explicit square
resize and ImageNet normalization shared by training and evaluation.
QuantSmith takes native NumPy input arrays;
the factory must yield the exact ONNX input names and replay the same inputs on
every calibration pass.

## Outputs and interpretation

Classification and language examples write `work_dir/artifacts/<example>/index.html`, `results.json`, and
`summary.md`, plus its FP32/quantized ONNX graphs and stage plans. Reports show
actual held-out predictions or output errors, not just a successful export.
The trained language models are small educational models, not pretrained LLMs.
Quantized constant weights use integer storage and unused initializers are removed.
Artifact size does not establish runtime RAM usage, kernel fusion, or speedup;
compare measured task quality and latency in the report.

The captioner instead saves a typed `report.json`, per-image `captions.jsonl`, split
graphs, integer audits, and raw ORT traces. Its bounded default run is a smoke
experiment; it does not establish production captioning quality.
