# Changelog

## 0.9.0

### Tracked Regressions

- Additional output rounding can reduce task accuracy, and complete integer
  coverage does not guarantee lower latency. In the local 256-image experiment,
  MobileNetV2 MinMax top-1 drops from 63.28% to 58.59%; EfficientNetB0 MinMax drops
  from 44.92% to 28.52%. Review held-out results before accepting an artifact.
  Activation-aware boundaries and conversion elimination remain
  [open work](docs/integer-execution.md#restrictions-and-tbd).
- Complete producer boundaries can prevent the specialized SmoothQuant integer
  clipping rewrite. Clip can remain FP32, and MobileNetV2 SmoothQuant is slower
  than the previous implementation on the measured CPU target.

### Backwards Incompatible Changes

- Change the built-in static ORT pipeline from consumer-only QDQ to calibrated
  input/output boundaries and INT32 biases. Plans gain `QuantizeOutput`; custom
  exhaustive plan consumers must handle it. Example JSON marks these records
  with `target: "output"`. Recalibrate and reevaluate existing models because
  output rounding and accumulator-safe weight scale adjustment change numerics.
- Leave unsupported static layouts in FP32 with explicit reasons. Selected
  admitted operations that fail to compile to integer CPU kernels now return
  an execution error instead of silently retaining a floating fallback.

### New Features

- Add `PipelineResult.execution` with compiled integer operator evidence and
  reasons for source nodes outside the execution contract. Reconstruction and
  other unaudited results expose `Absent` rather than claiming integer coverage.
  Pretrained vision examples save `<variant>.execution.json`.

### Performance

- Obtain integer kernels for all 20 ResNet18, 52 MobileNetV2 and 81 EfficientNetB0
  convolutions across MinMax, Percentile and SmoothQuant in the local ORT CPU
  experiment. ResNet18 MinMax improves from 28.59 to 9.52 ms but remains slower
  than FP32. See the design document for full accuracy, memory and latency
  observations; these are not universal acceptance guarantees.

### Documentation

- Add Mermaid architecture and integer-execution design documents covering
  component boundaries, bias arithmetic, fallback policy, evidence, restrictions,
  alternatives and TBD work. Correct obsolete storage claims in the README.

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
