# Pretrained model experiment log

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
