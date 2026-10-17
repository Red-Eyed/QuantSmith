# Changelog

## 0.8.4

### Backwards Incompatible Changes

- Target ONNX Runtime by default. SmoothQuant followed by built-in UINT8 static
  quantization automatically uses calibrated integer scaling where supported.
  Remove `ort_integer_scaling` from `pipeline.quantize` calls and
  `--ort-integer-scaling` from MobileNetV2 commands; the former opt-in behavior is
  now the default. Graphs may contain ORT contrib operators and changed rounding.
- Replace the vision report model field `ort_integer_scaling` with
  `execution_target: "onnxruntime"`. Report consumers should read the target rather
  than treating integer execution as a selectable mode.

### Improvements

- Separate graph recognition, typed numerical validation, pure rewrite planning,
  and owned patch application. Rewrite planners leave indexed nodes unchanged and
  return deterministic patches or `Absent` with an unsupported-pattern reason.
- Express pipeline progression as named stage transformations with explicit
  smoothing/quantization dispatch. Preserve fresh calibration, numerical guards,
  unsupported-pattern fallbacks, and expected-failure contracts.

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
