# Global causal Transformer development log

## 2026-10-17 — Integrated integer float-output mode

Expose `matmul_outputs` on the global-transformer example and preserve quantized
outputs as the default. The opt-in mode uses direct integer MatMul with FP32
outputs plus exact quantizer reuse. All 15 learned projections are source-audited;
compiled activation quantizers remain at nine, dequantizers fall from 25 to zero,
and two QuickGelu fusions return. MinMax's facade/example graphs are byte-identical.

Public MinMax and Percentile results match the prior experimental candidates bit
for bit over 1,024 contexts each. Integrated MinMax timings are 85.33 / 84.50 µs
versus INT8/reuse 91.83 / 90.88 µs, with ROM 305.5 versus 344.4 KiB. Top-1 is
40.695%, perplexity 7.93785 on 32,768 targets; FP32 is 40.750%, 7.92877. FP32
latency is 84.08 / 83.33 µs, so this approaches rather than beats FP32 execution.
All 893 tests pass. See [design and restrictions](../../docs/float-matmul-outputs.md).

## 2026-10-17 — Executed-output smoothing selection

Global-transformer output MSE prefers the unsmoothed baseline for MinMax and
Percentile against joint-extrema alpha-0.5 smoothing. Held-out results therefore
remain baseline: MinMax top-1 40.94%, perplexity 7.9762; Percentile top-1 40.60%,
perplexity 7.9658. Both alternatives retain 15 audited integer projections.
This does not establish faster-than-FP32 inference. Accepted recipes remain
unchanged; see [the complete study](../../docs/output-based-smoothing.md).

## 2026-10-17 — Per-channel histogram range selection

Add dedicated typed channel requirements/distributions, streamed replay, owned
read-only statistics, and an explicit storage budget propagated through the facade.
Joint smoothing can select extrema, channel percentile, or histogram-estimated
MSE ranges. These guide offline scales; final activation grids remain per-tensor,
and fresh static calibration still follows every transformed revision.

All 756 tests pass with Ruff and strict Pyrefly. Real exports cover all three
policies on the three CNNs and both Stackformers models. All 12 public transformer
configurations reproduce the independent study graphs byte for byte. The full
50-case study retains ROM, runtime process peak RSS, latency, quality and raw ORT
traces, plus forward/reverse process repetitions. Histogram storage and float
parity are reported separately from inference memory.

Observed results are mixed: ResNet18 channel Percentile + static Percentile
improves top-1 from 66.80% for joint extrema to 69.53%; MobileNetV2 channel MSE +
MinMax improves 58.20% to 60.16%. EfficientNet-B0 channel Percentile + static
Percentile declines 63.28% to 62.50%. MSE's channel-grid proxy does not optimize
final network loss. Existing recipes/defaults remain unchanged pending review;
no universal quality tolerance or automatic winner is introduced. See
[the complete experiment](../../docs/channel-histograms.md).


## 2026-10-17 — Joint smoothing and exact activation-quantizer reuse

Add a pure common-scale SmoothQuant planner, transactional complete-group affine
folding, and an explicit typed reuse step before the final ORT integer audit.
The Joint SmoothQuant config declares the complete pipeline. Ordinary configs and
accepted model recipes keep their previous declarations.

Both Stackformers checkpoints fold 11 of 15 edges versus 1 with independent
fold-only smoothing. Exact reuse reduces executed activation quantizers from
15 to 9 and retains all 15 integer projections. Plain MinMax/Percentile pipelines
with reuse preserve optimized held-out logits bit for bit (128 contexts each).
Repeated MinMax timings show small reductions, roughly 1–3%; FP32 remains faster.
Joint smoothing changes encodings and task scores. CNN smoothing candidates show
accuracy declines and are not promoted; smaller files or fewer nodes are not
sufficient acceptance criteria. No automatic quality tolerance is introduced.

All 703 tests pass, including numerical/rollback, exact reuse guards, real CNN
and Stackformers exports, compiled coverage, and static phase rejection. Ruff and
strict Pyrefly pass. Full ROM, process RSS, latency, quality, repetitions and raw
ORT artifact locations are in [the experiment report](../../docs/joint-smoothing.md).


## 2026-09-26 — Readable standalone demonstration

The local model and entry point replace the previous shared suite wrapper. Model
construction, named ONNX inputs, calibration replay, and all three method recipes
are visible together so readers can adapt them. Architecture dimensions and
training/data split policies are retained; parameter paths now belong to the local
model definition. Existing checkpoints from the older suite may use different keys.

Offline checks cover ONNX/INT8 numerical round trips and, for Transformers, causal
behavior and float dynamic-attention edges. Small smoke runs verify the complete
CLI/report path; they do not establish production accuracy or deployment speed.
