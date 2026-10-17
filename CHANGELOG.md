# Changelog

## 0.10.0

### Highlights

- Add the typed `Quantizer.from_config(config).quantize(model).calibrate(samples)`
  interface and explicit, composable ORT pipelines.
- Add `Float16StorageConfig` for smaller serialized weights with FP32 execution
  and no calibration. ORT optimization folds restoration casts at initialization.

### Tracked Regressions

- Static W8A8 Transformer projections can remain slower than FP32 even when ORT
  executes integer kernels. Removing explicit dequantization does not guarantee
  lower latency. Retain a measured FP32 or mixed-precision control.
- FP16 storage rounds weights and expands them to FP32 at initialization. It does
  not guarantee lower RAM or unchanged task accuracy; startup cost is unmeasured.

### Backwards Incompatible Changes

- Replace `QuantizationConfig.stages()` with `pipeline()` and `PipelineResult`
  with `QuantizationResult`. Prefer the new fluent facade; custom stages implement
  typed graph transitions. See [migration](docs/declarative-pipeline.md#typed-policies-and-migration).
- Replace `smoothquant=True` and `smoothquant_alpha=x` with
  `smoothing=SmoothQuantConfig(alpha=x)`, and `activation_function_aware=True` with
  `outputs=FunctionOutputs()`. SmoothQuant defaults to `FoldOnly()`; explicitly
  select `RuntimeScaling()` to retain runtime scale operations.
- Require replayable calibration iterables. Wrap generator factories in
  `Replay(factory)`; one-shot iterators return a structured error.
- Extend plan operations with `FloatOutput`. Custom exhaustive consumers must
  handle it. Floating-output integer MatMul changes output rounding; recalibrate
  and reevaluate task accuracy when enabling it.

### New Features

- Add joint fold-only SmoothQuant, histogram-informed channel selection with a
  bounded workspace, and optional smoothing selection by executed quantized
  output error. Selection error is not a task-accuracy guarantee.
- Add explicit boundary-calibration policies and `FloatMatMulOutputs()` for
  supported static integer MatMul/Gemm paths with floating outputs and ORT audit.
- Add a MobileNetV4 + Stackformers image-captioning walkthrough with disjoint
  calibration/evaluation, caption metrics, fresh-process latency/RSS measurements,
  and raw ORT profiling traces. The bounded training run is experimental.

### Performance

- Reuse identical activation quantizers without changing their tested outputs.
- Reduce captioning ONNX storage from 31.041 to 16.287 MiB with FP16 constants
  restored to FP32. All 32 evaluated captions match the baseline and no warmed
  latency regression is observed; peak process RSS rises by about 5.7 MiB.
  See [measurement limits](docs/float16-storage.md).

### Documentation

- Document typed extension contracts, folding, histogram budgets, smoothing
  selection, integer execution restrictions, and compact storage with Mermaid
  architecture diagrams and measured limitations.

## 0.9.1

### New Features

- Add opt-in `quantize(..., activation_function_aware=True)` for private
  `Conv → Relu/Clip` edges. Calibrate the Conv output encoding from the activation
  function output, preserving the chosen MinMax/Percentile policy. Defaults and
  standalone algorithms remain unchanged. This changes clipping and rounding;
  reevaluate task accuracy. Shared/public outputs and dynamic Clip bounds are
  excluded. MobileNetV2 exposes the matching CLI option and saves its configuration.

### Performance

- Reduce MobileNetV2 SmoothQuant latency from 4.22 to 3.20 ms and MinMax from
  3.76 to 2.78 ms in the local one-thread ORT comparison, keeping all 52 integer
  convolutions. MinMax Q/DQ counts fall from 36/36 to 2/2 with unchanged ONNX size.
  On 256 held-out images, top-1 changes from 61.33% to 64.84% for SmoothQuant
  and 58.59% to 63.67% for MinMax. Percentile loses accuracy despite lower latency;
  these gains are not universal. See [design and evidence](docs/integer-execution.md#activation-function-aware-calibration).

### Developers

- Add regression tests for eligibility, histogram replay, clipping and degenerate
  ranges, preserved statistics ownership, disabled behavior, and ORT integer
  coverage with conversion elimination.

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
