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
