# Mobile captioning experiment log

## 2026-10-18 — Packed attention with trained-caption accuracy

The width-768 synthetic Transformer benefited from native projection packing,
especially when it enabled ORT `DynamicQuantizeMatMul` fusion. Test whether that
gain transfers to the saved, trained width-256 caption model and QuantSmith's
calibrated static quantization. Reuse the original checkpoint without retraining;
strict loading into Stackformers 5.0.0rc13 succeeds. Convert owned copies through
native `to_packed()`: ten self/cross-attention modules for attention-only packing,
or those plus six SwiGLUs for full packing. The CNN, vocabulary projection, model
dimensions, and trained weights are otherwise shared.

### Method and evidence

`packed_benchmark.py` compares three layouts under FP32, QuantSmith MinMax,
QuantSmith Float Percentile, and a separately named native ORT dynamic INT8 control.
The two QuantSmith policies include both encoder and decoder quantization; the
native control quantizes constant-weight MatMul projections and keeps the CNN
floating. Consequently, compare packing within each policy, rather than interpreting
static-versus-dynamic timing as an isolated activation-quantization experiment.

All ONNX exports use native opset 21. Each exported and quantized decoder is reused
at prefix lengths 1, 3, and 32; symbolic metadata and actual output shapes are checked.
FP32 ORT outputs at all three lengths match the common unpacked eager source within
the declared tolerance. On the representative image/prefix, both packed eager
layouts are bit-identical to unpacked encoder context and decoder logits; source
parameter names and bytes remain unchanged. All FP32 layouts generate the same
captions on all 32 validation images.

Calibration uses the same 16 disjoint COCO training images and teacher-forced prefix
replay as the original study. Quality uses the same 32 validation images and
references; all candidates score every image. Timing uses one representative image,
one CPU thread, 20 warmups and 100 calls per metric in each of two fresh-process
forward/reverse sweeps. Profiling runs afterward in separate sessions. All candidates
generate 12 tokens on the timing image. Equal-work timing performs 32 decoder steps,
including one image encoding. Generation retains the existing padded 32-token
buffer; dynamic graph support does not mean incremental KV caching or growing-prefix
generation was added.

### Results

| Layout / policy | CIDEr | BLEU-4 | Caption ms, sweeps 1 / 2 | Fixed 32-step ms | ONNX MiB |
| --- | --- | --- | --- | --- | --- |
| Unpacked FP32 | 0.24330 | 0.11071 | 11.702 / 11.800 | 26.470 / 26.597 | 31.215 |
| Attention-packed FP32 | 0.24330 | 0.11071 | 12.493 / 12.419 | 28.258 / 27.908 | 31.183 |
| Fully packed FP32 | 0.24330 | 0.11071 | 11.911 / 11.987 | 27.122 / 27.336 | 31.170 |
| Unpacked QuantSmith MinMax | 0.23288 | 0.09475 | 16.150 / 16.246 | 39.172 / 40.080 | 11.346 |
| Attention-packed QuantSmith MinMax | 0.23500 | 0.09489 | 16.092 / 16.189 | 39.130 / 39.264 | 11.296 |
| Fully packed QuantSmith MinMax | 0.23234 | 0.09375 | 16.070 / 16.118 | 39.064 / 39.398 | 11.277 |
| Unpacked QuantSmith Float Percentile | 0.24358 | 0.11093 | 15.162 / 15.289 | 37.254 / 37.567 | 11.151 |
| Attention-packed QuantSmith Float Percentile | 0.24358 | 0.11093 | 15.066 / 15.105 | 36.931 / 37.130 | 11.155 |
| Fully packed QuantSmith Float Percentile | 0.24358 | 0.11093 | 15.115 / 15.161 | 37.127 / 37.090 | 11.153 |
| Unpacked ORT dynamic control | 0.24339 | 0.11071 | 14.975 / 14.947 | 35.146 / 34.943 | 14.399 |
| Attention-packed ORT dynamic control | 0.24140 | 0.11131 | 14.357 / 14.410 | 33.466 / 33.705 | 14.400 |
| Fully packed ORT dynamic control | 0.24339 | 0.11071 | 13.453 / 13.395 | 31.191 / 31.175 | 14.397 |

Float Percentile preserves its aggregate quality scores across layouts, but its
captions match FP32 on only 23/32 images. Teacher-forced token accuracy is 37.50%,
versus FP32's 37.76%; cross-entropy is 3.90450 versus 3.90415. Thus stable aggregate
scores do not mean identical outputs. Fully packed native dynamic INT8 matches
31/32 FP32 captions, with token accuracy 37.76% and cross-entropy 3.90451. MinMax
reduces both caption metrics in every layout. The tiny score differences between
Float Percentile, dynamic INT8, and FP32 do not establish a quality improvement.

### Interpretation

Packing does not establish a useful latency gain for either measured QuantSmith
policy. Fully packed Float Percentile remains about 29% slower than unpacked FP32,
although ONNX storage falls about 64%. Attention-only packing yields a small timing
change that does not alter that conclusion. Packing itself also regresses FP32
timing here, especially for the attention-only layout.

The optimized decoder provides direct evidence against using integer-operation
counts as a performance objective:

| Decoder policy | Unpacked | Attention packed | Fully packed |
| --- | --- | --- | --- |
| QuantSmith MinMax: `QLinearMatMul` | 45 | 33 | 29 |
| QuantSmith Float Percentile: `MatMulIntegerToFloat` | 45 | 33 | 29 |
| Native dynamic: `MatMulIntegerToFloat` | 28 | 12 | 4 |
| Native dynamic: `DynamicQuantizeMatMul` | 17 | 21 | 25 |

All three Float Percentile layouts retain 26 `QuantizeLinear` nodes in the decoder
and no fused `DynamicQuantizeMatMul`. Static calibration specifies fixed activation
grids; substituting a dynamically quantized fused kernel would change the numerical
policy. Native dynamic quantization benefits from full packing by about 10% in both
sweeps, consistent with its changed operator mix, but remains about 14% slower than
unpacked FP32. The large width-768 synthetic speedup does not transfer to this
width-256 caption workload. These observations do not isolate the CPU microkernel
cause or establish which policy wins for other trained models, shapes, or devices.
Encoder-only policies were not rerun in this study; their earlier results remain
separate from these whole-model comparisons.

Raw evidence: `work_dir/artifacts/mobile_captioning_packed/` contains the typed
`report.json`, `summary.md`, all generated captions, quality JSON, two fresh-process
measurements per candidate, optimized graphs, ORT traces, QuantSmith plans and
integer audits, and an archive of the exact Stackformers Python source. Environment:
Python 3.12.13, Torch 2.14.0, ORT 1.30.0, ONNX 1.23.0, macOS arm64. Checkpoint SHA-256:
`a381f59fde2682c5e42b299cdd7863ce33ad55b2313f170ab585623ba7445b83`.
Stackformers source SHA-256:
`6a3bac8920cf5206eeff12bde4091d79c10de31a6379175f37c3b3666ac22800`.

The exact run used the frozen rc13 uv environment retained from the preceding
synthetic study, injected through `PYTHONPATH`; the project's pinned example
dependencies remain unchanged. To run against the current local upgraded library:

```sh
uv run --locked --group examples \
  --with /Users/vadstup/Projects/personal/stackformers \
  --refresh-package stackformers \
  -m examples.mobile_captioning.packed_benchmark \
  --output work_dir/artifacts/mobile_captioning_packed_new
```

The runner loads the saved checkpoint, prepares missing data automatically, and
rejects using the original checkpoint directory as the output. Evaluation and
calibration budgets are explicit CLI options. The 300-update, 512-training-image
checkpoint and 32-image evaluation remain a smoke study, not evidence of useful
production caption accuracy. Checks: Ruff and strict Pyrefly for the runner, the
26 existing captioning tests, and actual packed export, dynamic-length execution,
calibration, integer audits, caption scoring, and fresh-process timing above.

## 2026-10-17 — MobileNetV4 and Stackformers encoder–decoder

The earlier tiny global Transformer results did not establish an end-to-end gain
for a mixed CNN/Transformer generation workload. Add a pretrained MobileNetV4 spatial
encoder and Stackformers visual encoder/causal cross-attention decoder for COCO captions.

Keep the encoder and decoder as separate ONNX graphs so generation reuses image
context. Compare five whole-model integer policies plus explicit encoder-only and
CNN-only policies against one FP32 checkpoint, preserve
raw ORT traces, and measure fixed-step generation alongside EOS-terminated generation.
Use PyTorch AdamW and ordinary model state dictionaries for export and evaluation.
Training and calibration stream disjoint images;
evaluation scorers have an explicit bounded corpus.

Verified numerical tests cover causal masking, visual conditioning, real MobileNetV4
and Stackformers ONNX export, integer convolution/projection execution, bounded
minibatch admission, and one image encoding per caption. The completed 300-update,
512-image smoke study measures ten variants. Encoder-only MinMax reduces representative
caption latency from 11.50 to 10.76 ms and ROM from 31.04 to 23.25 MiB, while CIDEr
falls from 0.2433 to 0.2342 and BLEU-4 from 0.1107 to 0.0948. Full-model integer
execution remains slower than FP32. Preserve these quality/latency regressions and
all raw traces; the design document contains the complete table and profiling evidence.

Final verification: `just check` passes Ruff, formatting, strict Pyrefly, and 919 tests,
including 26 new captioning cases. Encoder/CNN-only tests require the FP32 decoder's
bytes to remain unchanged and verify convolution-only selection leaves matrix
projections outside integer execution. ORT may fuse connected CNN residual additions.

Open: useful captioning quality after larger training, generated-prefix calibration,
incremental KV caching in Stackformers, reducing repeated vocabulary projection work,
and isolated inference workspace memory. Smoke-scale results cannot settle these.
## 2026-10-17 — Compact BF16 storage experiment

Static integer decoder projections regressed latency, motivating storage-only
compression with FP32 execution. Evaluated the opt-in `BFloat16StorageConfig`
against the same saved checkpoint and disjoint 32-image evaluation set; no
training or KV-cache changes. ROM falls from 31.041 to 16.287 MiB. ORT folds all
storage Casts before execution. Two fresh-process sweeps (20 warmups, 100 timed
runs, one thread) give FP32 caption medians 12.122/11.899 ms and BF16
12.194/12.104 ms, with peak RSS about 127.8 versus 133.3 MiB. These observations
do not establish a speedup or latency equivalence.

BF16 preserves 29/32 captions, CIDEr changes 0.243301 → 0.232028 and BLEU-4
0.110714 → 0.094637. The earlier FP16-storage experiment preserved 32/32 captions.
Keep BF16 optional; do not infer quality from its wider exponent range or from
unchanged teacher-forced token accuracy. Startup cost and broader held-out
quality remain unmeasured. Raw reports/traces are under ignored
`work_dir/artifacts/storage_bfloat16/`; design and scope are documented in
`docs/float16-storage.md`.
