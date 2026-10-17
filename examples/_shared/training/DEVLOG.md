# Model experiment log

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
[the complete experiment](../../../docs/channel-histograms.md).


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
ORT artifact locations are in [the experiment report](../../../docs/joint-smoothing.md).


## 2026-10-17 — Typed ORT-only pipeline refactor

Verification: 642 tests pass; Ruff and strict Pyrefly pass. All 15 accepted
CNN/transformer × method exports reproduce their saved graph bytes exactly.

Replace inferred stage kinds and runtime recipe lookup with typed `Pipeline`
composition, explicit transform/quantization/audit transitions, and the fluent
`Quantizer.from_config(config).quantize(model).calibrate(iterable)` facade.
Existing model recipes explicitly retain runtime smoothing and producer-output
calibration. The broader activation candidate remains on
`experiment/broad-activation-regressions` at `b08407d`; it is not part of this branch.

`FoldOnly` smoothing accepts private affine folds transactionally and leaves
unsupported regions unchanged before ordinary MinMax/Percentile calibration.
Tests verify FP32 equivalence, rollback, parameter ownership, deterministic replay,
static rejection of illegal phase composition, and byte-identical plain-method
fallback when all scales are unfoldable. Foldability alone does not establish
whole-model task accuracy or faster inference. See
[the design](../../../docs/declarative-pipeline.md).


## 2026-09-26 — Prioritize Transformer coverage

The CNN-only demonstration did not exercise integer token inputs, embeddings,
attention masks, or sequence accuracy. Added global and windowed causal language
models using published Stackformers 4.6.0, alongside a tabular MLP. Transformers
run first by default; the recurrent model is optional secondary coverage.

Both Transformer presets use two layers, dimension 64, four heads, RoPE, RMSNorm,
and SwiGLU. Training uses 600 CPU steps, batch 32, context 32. Tiny Shakespeare is
pinned by revision and checksum, with contiguous 80/10/10 train/calibration/test
spans. Calibration uses 64 windows; evaluation uses 128 windows / 4,096 target
characters. Vocabulary is fitted on training text. No held-out tuning was performed.

The first real export revealed that graph admission rejected causal masks with
`-inf`. Graph constants now preserve source values, while algorithms/statistics
enforce finite selected weights and activations. Dynamic attention products remain
floating point; the recipes quantize 15 constant projection/feedforward operators
in each Transformer. SmoothQuant lowers independently and recollects statistics.

Verified on Qraft 0.2.0, Python 3.12.13, Torch 2.14.0, ONNX 1.23.0, ORT 1.30.0,
macOS arm64, one CPU thread:

| Model | FP32 accuracy | FP32 perplexity | MinMax perplexity | Percentile perplexity | SmoothQuant perplexity |
|---|---:|---:|---:|---:|---:|
| Global causal Transformer | 41.06% | 7.96 | 7.97 | 7.97 | 7.97 |
| Windowed causal Transformer | 40.92% | 7.93 | 7.96 | 7.95 | 7.95 |
| Optional character RNN | 38.75% | 8.21 | 8.22 | 8.22 | 8.22 |

Transformer eager/export maximum errors on the parity probe were below 4e-6.
Held-out eager/export logit MSE was below 6e-13. Fixed training-probe CE fell from
4.29 to 1.87 (global) and 4.29 to 1.84 (windowed). The models learn character
structure, but greedy continuations remain limited; these are small trained models,
not pretrained LLMs or quality text-generation benchmarks.

Wine uses 105 training, 36 calibration, and 37 evaluation rows, with stratified
splits and training-only normalization. Its three-layer MLP reached 100% accuracy
on the 37 held-out rows in FP32 and all three quantized variants. That small split
does not establish generalization to other datasets.

The current QDQ graphs retain float weights and were slower than FP32 here.
Transformer warm median latency was about 0.09 ms in FP32 versus 1.3–1.4 ms with
QDQ. Timings use the observability evaluator with optimization disabled and are
not comparable to the CNN suite's optimized ORT measurements. No compression or
speedup claim is made. Artifact provenance and bounded prediction/continuation
examples are saved in HTML/JSON; only aggregate statistics are retained during evaluation.
