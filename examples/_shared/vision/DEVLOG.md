# Pretrained model experiment log

## 2026-10-17 — Functional rewrite planning and default ORT execution

The 0.8.3 implementation mixed recognition, numerical validation, name allocation,
and graph mutation in conditional-heavy functions. Refactor the ORT path into a
read-only graph index, named pattern/parameter records, pure rewrite planners,
and a separate patch-application boundary. Rule selection returns the first
applicable concrete patch or an `Absent` reason; it does not mutate matched nodes.
Name reservations return a new index sharing existing connectivity. Stage
progression returns new state records and explicitly dispatches between smoothing
and quantization. No public optimizer protocol or Result chaining is introduced.

ONNX Runtime is now the execution target. Remove the opt-in pipeline keyword and
MobileNetV2 CLI flag, automatically extending supported UINT8 SmoothQuant stages.
Reports identify `execution_target: "onnxruntime"`. Fresh calibration, explicit
exclusions, numerical guards, and unsupported/signed fallbacks remain intact.

The default-path MobileNetV2 rerun uses the same seed 42, 128 calibration images,
and 256 held-out images. MinMax is 3.364 ms and 162/256 correct; SmoothQuant is
3.398 ms and 166/256 correct. Artifact sizes remain 3,941,012 and 3,934,455 bytes.
Direct deployment-session comparison with the previously verified 0.8.3 ORT graph
(commit `9c74ffa`) gives exactly zero maximum absolute logit difference across all
256 held-out images. Outputs are under `work_dir/artifacts/mobilenet_v2_ort_functional/`.
These measurements preserve the prior performance/accuracy result and do not
establish new accuracy or RAM improvements.

Regression coverage additionally checks that repeated rewrite planning is
identical, does not change any input protobuf bytes, reserves names without
mutating connectivity, and applies to the same graph as the lowering entry point.
`just check` passes Ruff, strict Pyrefly, and all 483 tests.

## 2026-10-17 — SmoothQuant runtime fusion experiments

The direct-producer folding draft removes two MobileNetV2 divisions, recovering
19 QLinearConv nodes rather than 17. A complete rerun with seed 42, 128 calibration
images, and 256 held-out images gives 65.63% top-1 and 6.18 ms median latency.
This does not meet the MinMax latency target.

Fresh sequential CPU profiles using ORT 1.30.0, one thread, ten warmup runs, and
300 timed runs reproduce 4.44 ms FP32, 3.41 ms MinMax, and 6.42 ms original
SmoothQuant. A separate profiled session attributes about 0.32 ms per inference
to Div, so division cost alone cannot explain the slowdown. Ordinary MinMax has
36 QLinearConv nodes; original SmoothQuant has 17 and more floating convolutions.

Experimental artifacts are under `work_dir/smoothquant_optimization/`; these are
not integrated into the supported pipeline. All variants use the same 128 real
calibration images. Observed ablations:

| Variant | Median latency | Top-1 on 256 held-out images |
| --- | ---: | ---: |
| Extra QDQ boundaries before scaling | 4.19 ms | 65.63% |
| Quantized ORT multiplication at scaling boundaries | 3.75 ms | 65.23% |
| Foldable-only selective smoothing | 3.34 ms | 62.89% |
| Integer clipping plus quantized scaling | 3.55 ms | 65.23% |
| Integer clipping/scaling plus residual-add boundaries | 3.39 ms | 64.84% |

The final experimental graph preserves smoothing on all originally selected
layers. It absorbs channel scaling into eligible Conv weight encodings and biases,
replaces ReLU6 bounds with broadcast integer Min, and handles remaining scales with
`com.microsoft::QLinearMul`. Additional calibrated QDQ boundaries on residual Add
inputs recover 46 QLinearConv and five QLinearAdd nodes. This is an ORT-specific
quantization mode with changed rounding, not a numerically equivalent rewrite of
the original INT8 graph. Reciprocal coefficients also undergo UINT8 quantization.
No equivalent transformation is claimed for arbitrary operators or shared paths.

Three alternating, sequential fresh-process comparisons, each with 500 timed
runs on the same seeded synthetic FP32 input `[1, 3, 224, 224]`, give MinMax medians
3.356/3.353/3.362 ms and experimental medians 3.391/3.428/3.403 ms. Median-of-medians
is 3.356 versus 3.403 ms, approximately 1.4% slower for the experiment. Whole-process
inference RSS high-water marks are 117.3–117.6 MiB versus 118.6–118.7 MiB; this does
not demonstrate RAM savings. Serialized experimental size is 3,931,077 bytes,
compared with 3,941,012 bytes for MinMax and 3,988,864 for original SmoothQuant.

A final evaluation through deployment-style ORT sessions with optimizations enabled
confirms 162/256 correct for MinMax, 167/256 for original SmoothQuant, and 166/256
for the experiment. The experiment loses one correct prediction relative to
original SmoothQuant and remains four above MinMax. This small subset does not
establish population-level accuracy equivalence. The approved mode is now integrated behind
`pipeline.quantize(..., ort_integer_scaling=True)` and MobileNetV2's
`--ort-integer-scaling`. Portable ONNX remains the default. The integrated full
128/256-image rerun reproduces 64.84% top-1 and 3.40 ms median latency. Its serialized
size is 3,934,455 bytes; longer deterministic generated names account for the small
increase over the prototype. Runtime guards preserve shared values and exposed
outputs, skip signed activation encodings, and retain unsupported coefficients or
layouts. Plans include the additional calibrated rounding boundaries; graph
initializers contain the final rewritten encodings. Reports explicitly record
the mode, and coverage counts only eligible constant-weight operators.

Dedicated tests cover all UINT8 codes, negative real values, saturation, integer
clipping against the scaled floating-point reference, malformed graphs, shared
storage, explicit exclusions, pipeline calibration, and disabled/signed fallbacks.
`just check` passes Ruff, strict Pyrefly, and all 482 tests. The final combined
MinMax/SmoothQuant invocation saved its explicitly labeled report under
`work_dir/artifacts/mobilenet_v2_ort_verified/`: 3.360 ms MinMax versus 3.405 ms ORT
SmoothQuant, 162/256 versus 166/256 correct, and 53/53 eligible operators in both
coverage summaries. A separate fresh-process profile of the final graph confirms
46 QLinearConv nodes, five QLinearAdd nodes, and 118.77 MiB process inference RSS
high-water mark. No general accuracy or RAM guarantee is made.

## 2026-10-17 — Compact integer weight export in QuantSmith 0.8.1

MobileNetV2 SmoothQuant exports grew to 26.73 MiB because lowering retained both
original and rescaled FP32 weights. Ordinary QDQ exports also retained FP32 weight
storage. ONNX lowering now stores quantized constants as INT8/UINT8 followed by
dequantization and removes unused initializers after every stage, preserving shared
uses and constants exported directly as graph outputs.

The full MobileNetV2 rerun used the same seed 42, 128 calibration and 256 held-out
images. FP32 is 13.56 MiB; MinMax and Percentile are 3.76 MiB; SmoothQuant is
3.80 MiB. Top-1 remains 64.06%, 63.28%, 62.50%, and 65.23% respectively. Logit MSE
is unchanged at the report's displayed precision. Median one-thread optimized CPU
ORT latency is 4.36 ms FP32, 3.39 ms MinMax, 3.37 ms Percentile, and 6.32 ms
SmoothQuant. SmoothQuant improves accuracy on this subset but remains slower.

File compression does not establish RAM savings. Separate fresh processes with
one CPU ORT session, a zero FP32 input of shape `[1, 3, 224, 224]`, and 30 runs
showed approximately 99 MiB resident memory for FP32 and MinMax, versus 111 MiB
for SmoothQuant on this arm64 Mac. These process measurements include Python,
NumPy, runtime setup, and inference, and are not a cross-platform RAM benchmark.
Inspection of the optimized graphs found 36 `QLinearConv` nodes for MinMax but
only 17 for SmoothQuant, alongside 36 retained scaling `Div` nodes. Provider
fusion and activation-buffer improvements require separate work; this change
fixes serialized storage without claiming lower runtime RAM.

Regression tests compare stored-code dequantization with actual ONNX Runtime QDQ
for rounding ties, saturation, signed/unsigned zero points, and per-channel axes.
They also cover source ownership, shared consumers, live initializer outputs,
reconstructed integer storage, and removal of obsolete SmoothQuant constants.

## 2026-09-26 — Real CNN exports and held-out measurements

Added explicit TorchVision ImageNet checkpoints for ResNet-18, MobileNetV2, and
EfficientNet-B0. Every model uses its matching preprocessing and all 1,000 classes.
Imagenette train images calibrate; val images evaluate. Seeded selection stores
only bounded file metadata, and inference streams one decoded image at a time.

The full Qraft 0.1.1 run used 128 calibration and 256 held-out images. Torch/FP32
ONNX top-1 accuracy was 66.02%, 64.06%, and 75.00% respectively; export MSE was below
1.4e-11. MinMax top-1 was 67.58%, 63.28%, and 44.92%. Percentile gave 67.19%, 62.50%,
and 69.53%. SmoothQuant gave 66.02%, 65.23%, and 50.00%. The EfficientNet regressions
remain visible in reports; no subset or gallery was chosen by prediction outcome.

One-thread optimized CPU ORT measurements showed different kernel behavior across
architectures: MobileNetV2 improved from about 4.41 ms FP32 to 3.37 ms for ordinary
QDQ; ResNet-18 and EfficientNet-B0 slowed down. SmoothQuant does not universally
improve accuracy or speed. QDQ retains original float weights; the example shell
prunes only dead initializer copies after private rescaling.

After the Qraft 0.2.0 Result migration, all three entry points were checked again
with four calibration and eight held-out images. Full 0.1.1 measurements retain
their original provenance; smoke output lives in a separate directory. Numerical
regressions additionally cover real offline Torch exports in the test suite.
