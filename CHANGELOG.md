# Changelog

## 0.8.3

### New Features

- Add opt-in `pipeline.quantize(..., ort_integer_scaling=True)` and MobileNetV2
  `--ort-integer-scaling` execution. Calibrated UINT8 boundaries, integer clipping,
  and ORT scaling recover efficient convolution and residual-add kernels. This
  mode requires ORT-specific operators and changes quantization rounding; evaluate
  accuracy on the final graph. Portable ONNX remains the default.

### Improvements

- Fold eligible positive channel divisions into single-consumer ONNX Conv and
  inference BatchNormalization producers after SmoothQuant lowering and before
  fresh calibration. Preserve shared values and skip unsupported patterns.
- Add a vision runtime profiling command that saves optimized graphs, operator
  timing totals, JSON reports, and process RSS high-water marks measured before
  session creation, after creation, and after inference. General SmoothQuant
  latency and RAM improvements remain dependent on model and runtime behavior.
- Identify the ORT mode in saved JSON, HTML, and Markdown reports, and count
  constant-weight operator coverage separately from additional activation boundaries.
