# Mobile captioning experiment log

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
