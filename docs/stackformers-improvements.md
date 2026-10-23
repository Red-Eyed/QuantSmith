# Stackformers inference improvements: evidence and next experiments

Date: 2026-10-18. Inspected Stackformers version: 4.6.0.

## Recommendation

Prioritize packed self-attention QKV, packed cross-attention KV, and packed
SwiGLU gate/value projections as experiments. Preserve existing model mathematics
and weights, then measure the complete exported block. Keep precision selection
in QuantSmith or deployment tooling, where it can depend on actual shapes and
target hardware.

There is measured performance headroom. With dynamic sequence axes, a 768-wide
causal encoder layer takes **390.42 µs in FP32 versus 134.75 µs in native ORT
dynamic INT8 at one token**. At 128 tokens, the same serialized graphs take
**1761.90 versus 3628.63 µs**. Changing sequence length reverses the advantage.
This is evidence for workload-specific optimization, not a universal INT8 recipe.

The current measurements do not establish an end-to-end speedup from modifying
Stackformers. Packed QKV has supporting isolated evidence; packed SwiGLU and
their combined effect inside the complete block remain unmeasured.

## Experimental scope

PyTorch constructs random FP32 fragments and exports ONNX. All measured execution
and quantization use ONNX Runtime CPU. There is no PyTorch quantization, training,
dataset calibration, or held-out task-accuracy evaluation in these experiments.

Three terms must remain distinct:

- **Dynamic sequence length:** symbolic query and KV axes in the exported graph.
- **Native dynamic INT8:** ORT quantizes activations and computes their scales
  during inference; those costs are included in latency.
- **Fixed-grid INT8:** declared activation scales and offline weight scales,
  used to isolate integer-kernel and conversion costs without calibration.

The fastest measured native-dynamic path is not automatically the performance
of QuantSmith's existing calibrated static pipeline. The fixed-grid paths are
diagnostics, not accuracy-qualified QuantSmith plans.

The latest dynamic-shape run uses:

| Setting | Value |
|---|---|
| Hardware/platform | arm64, macOS 27.0.1 |
| Runtime | ONNX Runtime 1.30.0, CPUExecutionProvider, ORT_ENABLE_ALL |
| Export stack | PyTorch 2.14.0, ONNX 1.23.0, Stackformers 4.6.0 |
| Geometry | batch 1, width 768, 12 heads, query length 1 or 128 |
| Context length | 128 where the fragment consumes a separate KV context |
| Threads | 1 intra-op, 1 inter-op |
| Export | independent native exports at opsets 18 and 21 |
| Precision scope | constant-weight MatMuls; dynamic attention MatMuls stay float |
| Timing | 10 warmups; 3 alternating-order sweeps of 300 calls per candidate |
| Profiling | separate 10-call profile; excluded from timing |

All 120 variants execute successfully. Every fragment/opset/scheme has an
identical serialized graph hash at query lengths 1 and 128: 60 matched pairs.
Repetitions share a process and prepared sessions. Initialization and weight
packing are excluded from steady-state latency; no inference-RAM claim is made.

The complete encoder has one causal layer with RoPE, RMSNorm and SwiGLU. Its
single-token case has no KV cache. Cross-attention recomputes key/value
projections over its context. Neither is a pretrained LLM decoding benchmark.

## Measured results

### Dynamic sequence length changes the preferred precision

Values below use opset 18 and the median of three repetition medians. Speedup
is FP32 latency divided by INT8 latency; values below 1 mean INT8 is slower.

| Fragment | Query length | FP32 µs | Native dynamic INT8 µs | Speedup |
|---|---:|---:|---:|---:|
| Square Linear | 1 | 7.25 | 9.21 | 0.79× |
| GELU MLP | 1 | 237.46 | 54.13 | 4.39× |
| Packed QKV | 1 | 66.48 | 19.08 | 3.48× |
| Cross-attention | 1 | 242.33 | 723.88 | 0.33× |
| Causal encoder layer | 1 | 390.42 | 134.75 | 2.90× |
| Square Linear | 128 | 84.58 | 115.54 | 0.73× |
| GELU MLP | 128 | 1066.58 | 946.15 | 1.13× |
| Packed QKV | 128 | 237.85 | 230.56 | 1.03× |
| Cross-attention | 128 | 595.29 | 1132.65 | 0.53× |
| Causal encoder layer | 128 | 1761.90 | 3628.63 | 0.49× |

The encoder executes seven integer projections at both lengths, but benefits
only at one token. Even an isolated Linear can lose while a larger projection
set wins. Kernel selection must account for the complete surrounding region,
batch/query/context geometry, layout, runtime and hardware.

For the single-token encoder, relative output MSE against the FP32 reference is
approximately 0.000034. For 128 tokens it is approximately 0.000021. These measure
synthetic output perturbation, not task accuracy or an accepted quality budget.

Fixed U8S8 integer projections with float outputs take 142.29 µs for the
single-token encoder and 4471.85 µs at 128 tokens. The result therefore also
supports testing static integer kernels in useful regimes, but does not justify
assuming they match native dynamic INT8 performance.

### Packed QKV has isolated supporting evidence

The earlier **static-shape** matrix compares separate and packed QKV using the
same FP32 weights, width 768, batch 1, one token and opset 18:

| Projection layout | FP32 µs | Native dynamic INT8 µs |
|---|---:|---:|
| Three separate projections | 72.94 | 30.90 |
| One packed projection | 72.46 | 19.15 |

Packing improves the isolated INT8 case by approximately **1.61×**. FP32 is
effectively unchanged in this measurement. The larger 3.48× figure in the dynamic
table compares FP32 with INT8 for an already-packed fragment; it is not the
incremental gain from packing.

Packing inside the complete dynamic-sequence attention/encoder block remains
an experiment. Per-tensor quantization of one packed weight matrix can also
change scales relative to quantizing three matrices separately.

### Native opset export can improve the floating graph too

With dynamic sequence axes, the 128-token GELU MLP has these measurements:

| Precision | Native opset 18 µs | Native opset 21 µs |
|---|---:|---:|
| FP32 | 1066.58 | 903.44 |
| Native dynamic INT8 | 946.15 | 785.83 |

The opset-21 export represents GELU as one operator instead of the older
Pow/Mul/Add/Tanh chain. Both precision paths benefit. Simply converting the old
graph's opset did not introduce this operator in the earlier control experiment.
ONNX provides native `Gelu` from opset 20, including the `tanh` approximation.
See the [operator specification](https://onnx.ai/onnx/operators/onnx__Gelu.html).

Stackformers already uses `nn.GELU(approximate="tanh")` in its GELU FFN. This
opportunity belongs primarily to the export path; it does not require replacing
the activation implementation or changing a trained model's activation choice.

## Proposed Stackformers experiments

### 1. Pack compatible attention projections

Inspected code: `stackformers/attention/self_attn.py` and
`stackformers/attention/cross_attn.py`.

Self-attention currently applies separate `to_q`, `to_k`, and `to_v` Linear
modules to the same input. Concatenating their weight rows permits one projection
whose output is split back into Q, K and V. Cross-attention can pack K and V,
which share the context input; Q consumes a different input and stays separate.

Preserve the original projection widths. GQA/MQA can have different Q and KV
widths, so splitting into three equal chunks is not generally correct. Preserve
head order, normalization placement, positional encoding, masks and padded/packed
sequence behavior. Verify checkpoint conversion and packing/unpacking explicitly.
Do not assume a larger MatMul improves every floating or integer workload.

**Evidence:** isolated static-shape packed-QKV improvement; FP32 equivalence tests.
**Open:** complete-block gains with dynamic lengths, other attention configurations,
deployment hardware, and trained-model quality after quantization.

### 2. Pack SwiGLU gate and value projections

Inspected code: `stackformers/feedforward/swiglu.py`.

The current computation applies `w1(x)` and `w2(x)` independently, then computes
`SiLU(w1(x)) * w2(x)` before `w3`. Both input projections can be represented by
one packed weight matrix and a split. Keep the activation, multiplication,
dropout placement, inner width and output projection unchanged.

The hypothesis is fewer projection calls and potentially less repeated
activation quantization. Actual reuse depends on the exported and optimized
graph. A shared per-tensor grid can change quantization error, so inspect encoding
granularity and numerical results as well as integer-kernel counts.

**Evidence:** shared-input structure in the implementation and measured gains for
some larger projection sets.
**Open:** the packed SwiGLU variant itself has not been benchmarked.

### 3. Keep export structure and dynamic axes explicit

Export native operators where the target runtime supports them, and verify the
optimized graph rather than inferring its kernels from Python module names.
Retain dynamic query/KV dimensions through reshape, transpose, masking and
projection packing. Test length one explicitly: tracing a singleton example
can otherwise specialize the exported sequence dimension.

The current benchmark validates all 19 fragments at both opsets using one graph
and session per variant at query/KV pairs `(1,1)`, `(2,7)`, `(7,2)`, and `(31,13)`.
FP32 outputs must match PyTorch; native dynamic INT8 and four fixed-grid integer
variants must return finite FP32 outputs with the correct shapes. The same checks
must cover the proposed packed implementations.

### 4. Keep workload-dependent precision outside the model architecture

Use measured full-region performance and an explicit quality requirement to
select FP32 or an integer variant. Include actual executed lengths, padding,
KV context and batch size. This selection belongs in QuantSmith/deployment
tooling rather than a universal sequence-length threshold inside Stackformers.

The current measurements justify investigating different variants for different
workloads. They do not establish a crossover threshold, a policy for arbitrary
shapes, or a validated mixed-precision plan for a trained model.

## Validation before adoption

1. Compare unchanged, packed-QKV/KV, packed-SwiGLU, and combined variants using
   identical weights and inputs. Verify FP32 forward parity and, if the training
   representation changes, gradient parity and optimizer/checkpoint behavior.
2. Exercise padded/packed sequences, causal and cross-attention masks, positional
   encodings, and supported head-sharing configurations. Preserve dynamic axes
   and reuse each saved graph at multiple independent query/KV lengths.
3. Measure complete blocks and a complete trained model in ORT FP32 and the
   relevant integer paths. Include one-token and longer-sequence workloads;
   benchmark cached decoding separately if that is the deployment target.
4. Inspect optimized operators, quantization boundaries and profiles. Preserve
   per-channel encodings where applicable and identify any new scale coupling
   caused by packing. Retain slower results and unsupported cases.
5. Evaluate task quality with an explicit acceptance budget. For static
   quantization, recalibrate the transformed graph rather than reusing stale
   activation statistics. Promote only repeatable end-to-end gains on the target
   hardware with acceptable quality.

The next implementation experiment should compare the four layout variants in
step 1 inside the same complete block. That establishes the incremental value of
Stackformers changes separately from the existing advantage of changing precision.

## Evidence and reproduction

Tracked experiment implementation:

- [Transformer fragments and fixed-grid kernels](../examples/quantization_performance/transformer_blocks.py)
- [ORT benchmark runner](../examples/quantization_performance/transformer_kernels.py)
- [Correctness and dynamic-sequence tests](../tests/test_transformer_kernels.py)
- [Complete experiment log](../examples/quantization_performance/DEVLOG.md)
- [Benchmark options and limitations](../examples/quantization_performance/README.md)

Local raw evidence, generated under ignored `work_dir/`:

- [Dynamic-shape results](../work_dir/artifacts/transformer_ort_dynamic/report.json):
  the primary measurements in this report, with graphs, hashes and traces nearby.
- [Historical static-shape matrix](../work_dir/artifacts/transformer_ort_kernels/report.json):
  supporting packed/separate QKV comparison, not dynamic-shape timing evidence.

Reproduce the dynamic-shape measurements with the locked environment:

```sh
uv run --locked --group examples \
  -m examples.quantization_performance.transformer_kernels \
  --fragments '["linear","mlp_gelu","qkv_fused","attention","transformer"]' \
  --geometries '[{"tokens":1,"context":128,"width":768,"heads":12},{"tokens":128,"context":128,"width":768,"heads":12}]' \
  --scope weights --opset-mode native --opsets '[18,21]' \
  --seed 42 --threads 1 --warmup 10 --runs 300 --repetitions 3 \
  --profile-runs 10 --output work_dir/artifacts/transformer_ort_dynamic
```

Run focused correctness checks with `just test tests/test_transformer_kernels.py`.
The completed full-project verification passed Ruff, formatting, strict Pyrefly
and all 1,052 tests. Those checks validate the existing benchmark and fragments;
they do not validate the proposed Stackformers changes, which remain unimplemented.
