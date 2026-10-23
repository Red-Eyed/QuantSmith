# Quantization performance experiment log

## 2026-10-18 — Native Stackformers to_packed comparisons

The upgraded local Stackformers checkout provides `to_packed()` for self-
attention, cross-attention and SwiGLU. QuantSmith remains pinned to 4.6.0;
benchmark an isolated uv overlay built from the local checkout, preserving the
project dependencies and the user's library worktree. Record the installed
library's Python-source SHA-256 so a dirty checkout is not identified only by
its release version.

The new `packed_stackformers.py` runner creates one random FP32 source per
workload/geometry. Every layout is an owned copy of those same weights. Use the
native bound `to_packed()` method, check FP32 parity against the unpacked source,
and assert that its parameter hash remains unchanged. Complete Transformer
comparisons include unpacked, attention-only packing, SwiGLU-only packing and
both. Isolated workloads compare unpacked and packed self-attention, cross-
attention and SwiGLU.

All execution is ORT CPU, with dynamic sequence axes, native opset-21 export,
one thread, batch one, widths 256/768, head width 64, query lengths 1/128 and
context length 128 for cross-attention. Self-attention is causal without RoPE;
the complete layer has causal attention, RoPE, RMSNorm and SwiGLU. No KV cache is
used. Compare FP32, native ORT dynamic INT8 and the four fixed U8S8/S8S8 float-
output/quantized-output kernel variants. Only learned projections are quantized.
No training or dataset calibration is involved.

Prepare all layouts before timing, then interleave their six precision variants
in three alternating-order sweeps of 300 calls, following 30 warmups. Each ORT
candidate has a separate ten-call diagnostic profile. Exclude copying, packing,
export, initialization and profiling from measured steady-state latency. These
are paired session measurements in one process, not process-RSS measurements.

Use the source checkout explicitly, refreshing its wheel snapshot:

```sh
uv run --locked --group examples \
  --with /Users/vadstup/Projects/personal/stackformers \
  --refresh-package stackformers \
  -m examples.quantization_performance.packed_stackformers
```

CLI options expose geometry, workloads, precision schemes, opsets, seed, activation range, warmups,
runs and repetitions. The runner prints progress and final JSON, supports
`--quiet`, and checkpoints each completed comparison under
`work_dir/artifacts/packed_stackformers/`. Reports retain every repetition,
optimized operators, integer counts, profiles, graph/input/source hashes and
numerical error against the common unpacked FP32 reference.

### Versions and confirmation

The initial sweep finishes 16 workload/geometry comparisons and 240 timed
variants using Stackformers **5.0.0rc12**. The checkout changes during the work,
so the confirmation uses a frozen **5.0.0rc13** installed snapshot rather than
silently mixing versions. Each paired packed/unpacked comparison uses one
version and the same source weights. Other primary versions are ORT 1.30.0,
PyTorch 2.14.0, ONNX 1.23.0 and NumPy 2.5.3, on arm64 macOS 27.0.1.

Installed Python-source hashes:

```text
rc12: b25b8f30e27e77265c98afbec73634d70b89fd1bb8d8462053a48198ace57e20
rc13: 6a3bac8920cf5206eeff12bde4091d79c10de31a6379175f37c3b3666ac22800
```

Preserve both snapshots as `stackformers-source.zip` next to their reports;
the rc13 archive's normalized Python-source hash is verified against the report.
This preserves the measured implementation even as the source checkout advances.

The initial sweep has noticeable timing drift in some repetitions. Confirm
SwiGLU and the complete width-768 layer at query lengths 1/128 using 100 warmups,
three alternating-order sweeps of **1,000 calls**, and the FP32, native dynamic
INT8 and fixed U8S8 float-output paths. All 36 confirmation variants execute.
Retain all repetitions, including slower ones. The tables below report the
median of three repetition medians from this rc13 confirmation.

### Confirmed packed versus unpacked INT8

Both columns use native ORT dynamic INT8. These are incremental packing gains,
not FP32-to-INT8 speedup claims.

| Workload, width 768 | Query tokens | Unpacked µs | Packed µs | Packing speedup |
|---|---:|---:|---:|---:|
| SwiGLU | 1 | 67.42 | 46.46 | 1.45× |
| SwiGLU | 128 | 2071.37 | 760.33 | 2.72× |
| Complete causal layer | 1 | 137.62 | 106.75 | 1.29× |
| Complete causal layer | 128 | 3670.56 | 1635.69 | 2.24× |

The long-sequence complete-layer packing speedups are approximately 2.24×,
2.24× and 1.97× in the individual sweeps. The improvement therefore survives
the observed timing drift. The single-token layer improves by approximately
1.29–1.31× across the three sweeps.

### Which complete-layer branch contributes?

Same rc13 native dynamic INT8 comparison, with all four layouts interleaved:

| Layout | Integer projections | 1 token µs | 128 tokens µs |
|---|---:|---:|---:|
| Unpacked | 7 | 137.62 | 3670.56 |
| Attention packed | 5 | 125.75 | 2947.98 |
| SwiGLU packed | 6 | 120.58 | 2362.08 |
| Both packed | 4 | 106.75 | 1635.69 |

Both transformations contribute in this width-768 layer. SwiGLU packing provides
the larger individual reduction at 128 tokens, and combining both performs best.

### Precision path still matters

Packing provides little improvement for the fixed-grid integer-to-float route:

| Complete layer | Query tokens | Unpacked µs | Both packed µs |
|---|---:|---:|---:|
| FP32 | 1 | 393.88 | 386.42 |
| Native ORT dynamic INT8 | 1 | 137.62 | 106.75 |
| Fixed U8S8, float outputs | 1 | 149.46 | 143.92 |
| FP32 | 128 | 1813.60 | 1826.98 |
| Native ORT dynamic INT8 | 128 | 3670.56 | 1635.69 |
| Fixed U8S8, float outputs | 128 | 4539.67 | 4566.62 |

The packed native-dynamic layer is about 10% faster than the unpacked FP32
control at 128 tokens in this run. That margin is smaller and more sensitive
to the observed timing variation than the 2.24× packing gain against unpacked
INT8. It is not a deployment-wide speedup guarantee. The fixed-grid path remains
slower than FP32 despite executing the same four integer projections as the
packed native-dynamic path. Runtime lowering/kernel selection matters as well
as projection packing; integer coverage alone cannot select the best result.

The fixed grids are diagnostic and were not calibrated. This experiment does
not establish the behavior of a calibrated QuantSmith plan automatically.
Task-quality evaluation on trained weights remains separate from this timing
study. Packing changes native INT8's per-tensor weight grouping and can change
quantization error even when FP32 behavior is preserved.

### Numerical and dynamic-shape checks

In the rc13 confirmation, packed and unpacked PyTorch FP32 outputs match exactly
on all measured inputs. Source parameter hashes remain unchanged. For the
128-token complete layer, native INT8 relative MSE against the common FP32
reference is approximately 0.0000205 unpacked and 0.0000179 with both branches
packed. These are synthetic output-error measurements, not task accuracy.

All 18 matching layout/scheme graph pairs have identical serialized hashes
across query lengths 1 and 128. The exporter also rejects fixed sequence axes.
Thus the packed timings use dynamic-sequence graphs, rather than a separately
specialized graph at each length.

The smaller width-256 rc12 controls are mixed: at one token, packing both
branches takes 98.98 µs versus 89.73 µs unpacked in native INT8, while packing
only SwiGLU takes 72.92 µs. These exploratory controls were not repeated in the
rc13 confirmation, and argue against enabling packing universally based on the
width-768 result alone. All initial outcomes remain in the raw report.

### Artifacts and validation

- Initial rc12 sweep: `work_dir/artifacts/packed_stackformers/report.json`.
- Frozen rc13 confirmation:
  `work_dir/artifacts/packed_stackformers_confirmation/report.json`.
- Each directory retains its exact source archive, exported/optimized graphs,
  execution profiles, source/input/graph hashes, and every median/p95 repetition.
- `just check` passes Ruff, formatting, strict Pyrefly and all 1,052 tests in the
  project's pinned environment. The upgraded-library smoke and measurement runs
  additionally execute native conversions, ownership checks, FP32 parity and
  ORT numerical/output validation. The full suite is not claimed to validate
  general project migration to Stackformers 5.

For another confirmation against the current checkout, add these options to
the command above, using a new output directory to retain historical evidence:

```sh
--workloads '["swiglu","transformer"]' \
--geometries '[{"tokens":1,"context":128,"width":768,"heads":12},{"tokens":128,"context":128,"width":768,"heads":12}]' \
--schemes '["fp32","dynamic","fixed_float_u8s8"]' \
--warmup 100 --runs 1000 --repetitions 3 \
--output work_dir/artifacts/packed_stackformers_new_confirmation
```

## 2026-10-18 — Require dynamic query and KV sequence lengths

The original synthetic matrix exported a separate fixed-shape graph for each
geometry. Those measurements do not establish dynamic-sequence performance.
Export symbolic query and context dimensions with explicit named constraints,
and use representative nonsingleton inputs during tracing to avoid length-one
specialization. Batch, head count and hidden width remain fixed.

QK and cross-attention permit independent query/context lengths. AV ties its
probability matrix's context axis to the value input's sequence axis. Full-tensor
Add now consumes a runtime residual tied to query length; a fixed-size constant
cannot represent an arbitrary sequence length. Scalar/channel Add retain their
constant broadcast operands. This changes the full-tensor Add workload relative
to the earlier fixed-shape experiment.

Reject source exports and transformed graphs with fixed sequence dimensions.
Reports explicitly declare `dynamic_sequence: true`; the new default output is
`work_dir/artifacts/transformer_ort_dynamic/`, preserving earlier static results.
For all 19 fragments at both opsets, regression tests reuse one exported graph
and session per kernel variant at query/context pairs (1,1), (2,7), (7,2), and
(31,13). FP32 must match the PyTorch fragment; native dynamic INT8 and all four
fixed-grid integer paths must retain dynamic metadata and execute with finite
FP32 outputs of the expected shape. Test setup starts with one-token examples to
cover the specialization trap explicitly.

`just check` passes Ruff, formatting, strict Pyrefly and all 1,052 tests. The
39 additional regressions cover all fragment/opset pairs plus rejection of
specialized sequence metadata.

Remeasure Linear, GELU MLP, packed QKV, cross-attention and the causal encoder at
width 768, batch one, query lengths 1/128 and context length 128. Use native
exports at opsets 18/21, projection-only quantization, ten warmups, three
alternating-order sweeps of 300 timed calls and ten separate profiling calls.
All 120 variants execute successfully. For every fragment/opset/scheme, the
serialized graph SHA-256 is identical across query lengths 1 and 128: 60 matched
pairs. These measurements therefore change runtime sequence length without
changing the model graph.

Selected opset-18 dynamic-graph results, median of three repetition medians:

| Fragment | Query length | FP32 µs | Native dynamic INT8 µs |
|---|---:|---:|---:|
| GELU MLP | 1 | 237.46 | 54.13 |
| Packed QKV | 1 | 66.48 | 19.08 |
| Causal encoder | 1 | 390.42 | 134.75 |
| Causal encoder | 128 | 1761.90 | 3628.63 |

The direction of the shape-dependent result remains: some larger single-token
projection sets benefit, while the 128-token encoder is slower in INT8. The
single-token encoder still has no KV cache. New measurements and traces are in
`work_dir/artifacts/transformer_ort_dynamic/`; the report records the complete
command configuration and numerical errors. Earlier tables below are static-
shape experiments and remain labeled historical in the README.

## 2026-10-18 — Calibration-free Transformer kernel and surrounding-operation matrix

Use PyTorch to construct and export random FP32 fragments; measure all execution
in ONNX Runtime CPU. No PyTorch quantization, trained checkpoint, dataset or
calibration pass is involved. The explicit static activation range is 4, with
step 4/127; each constant weight tensor gets its own absolute-maximum scale.
Dynamic INT8 uses ORT's native quantizer and includes inference-time scale
computation. These are kernel experiments, not accuracy-qualified model recipes.

The full matrix covers 19 fragments and six geometries: widths 64/256/768,
query lengths 1/32/128, batches 1/4, and heads 4/8/12. Fragments include Linear,
bias, GELU, SiLU, residuals, input LayerNorm/RMSNorm, scalar/channel/full-tensor
Add, noncontiguous inputs, GELU MLP, SwiGLU, separate/packed QKV, QKᵀ, AV,
Stackformers cross-attention and a causal Stackformers encoder layer.
Complete attention and Transformer implementations come from Stackformers.

Compare FP32, native dynamic INT8, fixed U8S8/S8S8 integer MatMul with FP32
outputs, and fixed U8S8/S8S8 QLinearMatMul with output dequantization. All 1,368
candidates across 114 cases and two opsets execute successfully. Each candidate
gets ten warmups, three alternating-order sweeps of 100 unprofiled calls, and
a separate ten-call diagnostic profile. Sessions and process are shared across
repetitions. ORT 1.30.0, PyTorch 2.14.0, arm64 macOS, one CPU thread.

### Shape-dependent results

Selected opset-18 results at width 768, batch one. Latency is the median of the
three repetition medians. INT8 here means ORT native dynamic quantization of
constant-weight MatMuls; it leaves dynamic attention matmuls floating.

| Fragment | Query tokens | FP32 µs | Dynamic INT8 µs | FP32 / INT8 | Relative MSE |
|---|---:|---:|---:|---:|---:|
| Linear | 1 | 7.13 | 9.08 | 0.78× | 0.000054 |
| Packed QKV | 1 | 72.46 | 19.15 | 3.78× | 0.000056 |
| GELU MLP | 1 | 237.67 | 54.13 | 4.39× | 0.000125 |
| SwiGLU | 1 | 239.04 | 67.42 | 3.55× | 0.000582 |
| Causal encoder layer | 1 | 382.06 | 126.50 | 3.02× | 0.000034 |
| Linear | 128 | 84.46 | 115.67 | 0.73× | 0.000068 |
| GELU MLP | 128 | 1071.35 | 946.33 | 1.13× | 0.000141 |
| SwiGLU | 128 | 893.69 | 2082.19 | 0.43× | 0.000691 |
| Causal encoder layer | 128 | 1723.75 | 3589.08 | 0.48× | 0.000021 |

The same seven learned projections help the single-token encoder and hurt its
128-token counterpart. An isolated square Linear is slower in INT8 even where
the larger MLP and packed QKV are faster. Integer coverage alone cannot choose a
recipe. The single-token MLP's diagnostic MatMul total drops from approximately
219.2 to 48.2 µs when the optimized graph uses two DynamicQuantizeMatMul kernels.
This is consistent with reduced weight traffic and a different packed kernel
helping the larger projection set; no bandwidth-counter evidence is collected.

The encoder's single-token case has no KV cache, and its context argument is
unused. Cross-attention includes fresh key/value projections over the context.
Neither case is a complete autoregressive decoder or pretrained HF model.

### Surroundings, boundaries and region selection

Bias, nonlinearities, normalization, residuals, transposes and broadcast geometry
are retained in each exported region and timed with its matmul. Their costs and
fusions belong in kernel selection; the report preserves optimized operators and
profiles rather than assuming an operator name selects a kernel independently.
Separate and packed QKV have identical FP32 weights and outputs in correctness
tests. Quantized variants may use different weight scales for the packed tensor.

An additional eight-case, 300-call-per-sweep comparison uses width 768 and
`--scope weights`. Quantizing every MatMul can destroy attention probabilities:
cross-attention's fixed U8S8 float-output relative MSE is about 1.01 at one query
token with six integer kernels, versus 0.030 with four projection-only integer
kernels. Native dynamic INT8 has approximately 0.000264 relative MSE, but is
still slower than FP32 for this cross-attention workload. Fixed grids are not
an accuracy substitute for calibration.

For the 128-token encoder, keeping attention floating reduces fixed float-output
relative MSE from 0.000791 to 0.000100 and executes seven integer kernels instead
of nine. It remains slower than FP32. Zero-integer dynamic QK/AV candidates are
unchanged FP32 controls and must not count as integer speedups. No automatic
quality budget, chosen plan, or library-default change follows from these results.

### Opsets and reproduction

The main matrix exports opset 18 and semantically converts the graph to 21.
All 684 matched pairs have identical optimized operator-count maps; the median
21/18 latency ratio is 1.000. This measures version conversion, which does not
introduce newer operators into an older decomposed graph. The runner also exposes
`--opset-mode native` for independent exports at each requested opset.

Direct native exports cover another eight cases with 300 calls per sweep,
projection-only quantization, width 768 and query lengths 1/128. For GELU
fragments, opset 21 replaces Pow/Mul/Add/Tanh arithmetic with one `ai.onnx::Gelu`
operator. This improves both the floating and integer paths:

| Fragment, 128 query tokens | Opset 18 µs | Native opset 21 µs |
|---|---:|---:|
| Linear + GELU, FP32 | 156.42 | 115.96 |
| Linear + GELU, dynamic INT8 | 188.38 | 147.17 |
| GELU MLP, FP32 | 1060.37 | 984.35 |
| GELU MLP, dynamic INT8 | 944.13 | 785.31 |

Linear + GELU is still faster in FP32. The complete SwiGLU encoder does not gain
from a newer Gelu operator it does not contain. Opset version is consequential
when it changes emitted operators; merely converting a decomposed graph does
not provide that benefit.

An independent process repeats ten key cases with 1,000 calls per sweep. At one
query token, dynamic INT8 versus FP32 is 57.21 versus 239.73 µs for GELU MLP,
19.08 versus 67.54 µs for packed QKV, and 126.17 versus 389.29 µs for the causal
encoder. At 128 tokens, the encoder remains slower in INT8: 3580.60 versus
1714.50 µs. These longer measurements confirm the direction of the original
results rather than a universal integer speedup.

Run the full matrix with:

```sh
uv run --locked --group examples -m examples.quantization_performance.transformer_kernels
```

All graphs, graph hashes, input shapes/strides/hash, error metrics, median/p95
repetitions, optimized operators and traces are under
`work_dir/artifacts/transformer_ort_kernels/`. Projection-only attention comparisons
are under `work_dir/artifacts/transformer_ort_weights/`. Independent confirmation
uses `work_dir/artifacts/transformer_ort_confirmation/`, 1,000 timed calls per
candidate and sweep, opset 18, widths 768, query lengths 1/128, and the Linear,
GELU MLP, SwiGLU, packed QKV and causal encoder fragments. Native-opset exports
are under `work_dir/artifacts/transformer_ort_native_opsets/`, with the GELU,
LayerNorm, GELU MLP and encoder fragments. All four runs finish without unsupported
candidates: 1,620 variant measurements, each with three timing sweeps. Runtime
files are ignored.
The [README](README.md) documents all CLI controls and measurement limitations.

Validation: `just check` passes Ruff, formatting, strict Pyrefly and all 1,013
tests. The 38 new cases cover fragment determinism, packed/separate QKV parity,
causality, noncontiguous and probability inputs, graph ownership, signedness,
operand selection, fixed grids, malformed constants/outputs, native export,
unchanged zero-integer paths and actual integer dispatch.

## 2026-10-18 — Issue 007: quantize the ORT_ENABLE_ALL snapshot

The previous experiments optimized execution sessions while supplying the original
export to QuantSmith. They did not test issue 007's requested ordering. Add an
isolated runner that saves and reloads the ORT_ENABLE_ALL FP32 graph and supplies
that actual revision to description, fresh calibration, planning, lowering, and
the final audit. No library defaults or supported-operator rules change.

### Matched experiment

Use the same two saved FP32 exports, seed 42, 128 train calibration images,
256 held-out validation images, one CPU thread, ORT 1.30.0 on the same arm64 Mac,
and function-aware MinMax as the original-graph quantization control. For each
model compare original FP32, saved optimized FP32, original quantized, and
optimized quantized. All eight candidates complete three alternating fresh-process
repetitions with 50 warmups and 300 unprofiled runs, followed by a separate
profiling session. The full project check finishes before the measured run.

Before quantization, compare optimization-disabled execution of the original
graph against optimization-disabled execution of the saved, already-fused graph
over all 256 held-out inputs. Maximum absolute FP32 differences are 3.624e-5 for
ResNet-18 and 2.420e-5 for MobileNetV2; logit MSE is approximately 1.200e-11 and
2.288e-12. These are rounding differences, not bitwise equality. The explicit
parity guard uses the existing example export tolerances, rtol 1e-3 / atol 1e-4.

### Results

Latency is the median of three fresh-process medians. RSS is the median process
inference high-water mark, including startup/runtime/serialization/buffers rather
than isolated tensor memory.

| ResNet-18 candidate | Median ms | Top-1 % | ROM MiB | Process peak RSS MiB |
|---|---:|---:|---:|---:|
| Original FP32 | 4.397 | 66.02 | 44.661 | 213.2 |
| Saved optimized FP32 | 4.474 | 66.02 | 44.588 | 213.1 |
| Quantized original graph | 9.394 | 66.80 | 11.357 | 123.1 |
| Quantized optimized graph | 4.452 | 67.19 | 43.138 | 211.0 |

| MobileNetV2 candidate | Median ms | Top-1 % | ROM MiB | Process peak RSS MiB |
|---|---:|---:|---:|---:|
| Original FP32 | 4.461 | 64.06 | 13.557 | 116.6 |
| Saved optimized FP32 | 4.441 | 64.06 | 13.333 | 115.9 |
| Quantized original graph | 2.741 | 63.67 | 3.937 | 101.4 |
| Quantized optimized graph | 4.437 | 65.23 | 8.399 | 108.3 |

ResNet's preoptimized quantization path is about 2.11 times faster than quantizing
the original graph, but it remains near rather than faster than FP32 and retains
most FP32 weight storage and process memory. MobileNet's preoptimized path is
about 1.62 times slower than its existing quantization path. Small-subset top-1
changes do not establish broader quality improvements. The saved optimized FP32
controls do not demonstrate a consistent latency benefit over the original FP32
deployment sessions, which already enable ORT optimizations.

### Why the quantized paths differ

The optimized ResNet graph contains 17 com.microsoft::NhwcFusedConv and three
com.microsoft::FusedConv nodes, with no standard Conv remaining. The optimized
MobileNet graph contains one NhwcFusedConv, 44 FusedConv, and seven standard Conv.
QuantSmith selects standard Conv/MatMul/Gemm; the custom fused nodes remain FP32.
FusedConv includes activation semantics, and NhwcFusedConv also changes layout.
Treating either as standard Conv by dropping its domain/name would not preserve
the current planning/lowering contract.

Both optimized revisions calibrate and pass the final audit. ResNet's original
quantized graph audits 20 Conv, eight Add, and one Gemm integer kernels; the
preoptimized quantized graph audits only one Add and one Gemm. MobileNet changes
from 52 Conv, ten Add, and one Gemm to seven Conv and one Gemm. Thus ResNet's speed
recovery comes from preserving fast fused FP32 convolutions, not faster integer
convolution kernels or greater integer coverage.

Separate ResNet profiles attribute about 8.43 ms per inference to original-path
QLinearConv versus about 3.75 ms to the preoptimized path's fused FP32 convolutions.
MobileNet's preoptimized path still spends about 3.75 ms in its 44 FusedConv nodes.
Mixing them with seven integer convolutions also produces 16 Transpose nodes and
eight QuantizeLinear / 13 DequantizeLinear nodes; the original quantized path has
two of each. Its all-integer convolution total is about 2.76 ms. These instrumented
profiles are attribution from separate sessions, not exact decompositions of
unprofiled latency.

### Compatibility and remaining scope

An intentionally unfused Conv/BatchNorm/Relu fixture verifies optimization-disabled
FP32 parity and caller ownership. With pre-existing inferred intermediate metadata,
the saved fused graph freshly calibrates its remaining Gemm while keeping fused
Conv floating point. The test verifies that both revisions invoke replay
independently and that the saved optimized input hash differs from the original.

Without that intermediate metadata, the valid optimized fixture fails calibration
with kind invalid_data, operation `ONNX execution`, and detail
`requested tensor lacks FP32 metadata: flat`. Standard ONNX shape inference does
not recover that type through the ORT-specific fused operator. This is a precise
remaining observability/type-inference limit, not a claim that all optimized
graphs are incompatible. The real exports preserve enough metadata for this
experiment. No fallback optimization level or library workaround is installed.

The runner preserves original and optimized snapshots, hashes, qualified operator
counts, opset/domain versions, provider/runtime/platform, source names or surviving
outputs where observable, plans, exclusions, audits, quality, size/RSS, timing
repetitions, and raw traces under ignored
`work_dir/artifacts/quantization_preoptimization/`. Absorbed identities are marked
explicitly rather than assigned invented lineage. These optimized graphs may be
hardware-specific and are not promoted as portable exports.

Issue 007 now has an initial measured answer: the requested ordering is compatible
for these exports and changes the mixed-precision result, but it is not a general
speed/compression improvement. Default adoption, custom fused-operator planning,
and missing-metadata support remain open. Greater quantized coverage is not an
acceptance criterion. Verification passes all 975 tests, Ruff, formatting, strict
Pyrefly, the functional CLI smoke run, and post-run checks of all 24 fresh-process
records, graph identities, audit inputs, and disjoint calibration/validation paths.

## 2026-10-18 — Shape-aware regions and mixed-precision proposals

The objective is lower complete-model latency at acceptable task quality, not
more integer operations. Keep FP32 in every comparison and in the final frontier.
Use local measurements only to propose regions, then test their complete-model
latency and quality. No deployment winner is chosen without an accuracy budget.

### Scope and validation

Extract the two most expensive FP32 source operators plus the classifier from each
saved ResNet-18/MobileNetV2 graph. Preserve constant ancestors and external residual
edges. Compare isolated operations with private chains of up to two following
activations, additions, multiplications, or pooling operations. The most expensive
convolution also gets equal-valued scalar/channel/full-tensor scaling controls.

The completed sweep measures 32 cases and 256 configurations: batch 1/4,
opset 18/21, and FP32/QuantSmith function-aware U8S8/native U8S8/native S8S8.
Each configuration has three alternating-order repetitions of ten warmups and
100 unprofiled runs. These microbenchmarks use separately initialized sessions
within one process. They include the ORT Python call but exclude output-array
validation, preprocessing, calibration, and session initialization from timing.

Calibration replays the same 128 train images through the original model prefix.
Only current region inputs are retained. Batch four repeats the current image,
isolating geometry rather than evaluating four independent images. The comparison
is bounded, not an enumeration of every runtime microkernel or every source node.

Numerical verification of all 256 saved candidates finds exactly zero maximum
absolute output difference between opsets 18 and 21 on the tested real inputs,
including the quantized graphs. Every candidate hash matches its recorded identity.
The aggregate median opset-21/opset-18 latency ratio is approximately 0.999;
individual timings vary, particularly for tiny operators. There is no consistent
opset speed benefit in this sweep. Both versions compile the relevant operators
to the same operator families here.

### Shape and graph context change the winner

Representative opset-18 median-of-medians times, including FP32 interfaces around
the integer operation:

| Source/context | Batch | FP32 ms | Native U8S8 ms |
|---|---:|---:|---:|
| ResNet first Conv, 3→64 channels, 7×7, 224×224 input | 1 | 0.813 | 1.087 |
| Same isolated Conv | 4 | 10.558 | 4.366 |
| ResNet 512→512 Conv, 3×3, 7×7 input | 1 | 0.165 | 0.519 |
| Same Conv plus residual/activation region | 1 | 0.192 | 0.586 |
| Same isolated Conv | 4 | 2.229 | 2.087 |
| Same Conv plus residual/activation region | 4 | 2.137 | 2.252 |
| MobileNet depthwise Conv, 96 groups, 112×112 input | 1 | 0.882 | 0.333 |
| Same depthwise Conv plus following region | 1 | 0.987 | 0.346 |
| MobileNet depthwise Conv, 144 groups, 56×56 input | 1 | 0.335 | 0.118 |

The ResNet stem's FP32 compiled graph is Transpose/NhwcFusedConv/Transpose at
batch one and plain Conv at batch four. This is concrete evidence of a shape-driven
runtime path change, not a claim that batch four universally makes INT8 preferable.
The 512-channel batch-four example also shows why an isolated-kernel win cannot
be lifted directly into a surrounding region: its U8S8 region is slower overall.

Classifiers exhibit a batch reversal too: ResNet's 512→1000 Gemm takes about
5.7 µs FP32 versus 13.2 µs U8S8 at batch one, and 25.6 versus 17.5 µs at batch four.
MobileNet's 1280→1000 projection changes from 18.9 versus 23.4 µs to 47.4 versus
35.6 µs. These very small operations include Python-call overhead and merit
complete-model confirmation rather than standalone deployment conclusions.

Broadcast layout changes cost even with identical coefficients. The batch-one
ResNet FP32 stem takes about 0.817/0.815/0.902 ms with scalar/channel/full-tensor
scaling; native U8S8 takes about 1.193/1.187/1.265 ms. MobileNet's depthwise FP32
controls take about 0.886/0.886/0.921 ms. These are artificial diagnostic contexts,
not transformations automatically inserted into the real models.

### Complete-model acceptance

For batch-one model proposals, use the measured surrounding region when available,
require at least a 5% local speed improvement, and ignore artificial broadcasts.
This proposes no quantized ResNet producers and only MobileNet's two measured
depthwise producers (`node_Conv_817`, `node_Conv_829`) for all three quantizers.
Unmeasured producers remain floating point. An empty proposal reuses FP32.

Compare these mixed candidates against FP32 and all-quantized controls at both
opsets. Calibration remains the original 128 train images; tuning uses the next
128 deterministically ranked train images. All candidates report quality on the
original 256 validation images. The frontier uses tuning top-1 and measured
complete-model latency; validation labels never select candidates. Full-model
timings use three alternating fresh-process repetitions, 50 warmups, and 300
unprofiled runs, followed by separate profiling sessions.

The completed comparison contains 22 complete-model candidates and 66 fresh-process
measurements. The following table holds opset 18 fixed; each opset-21 counterpart
has identical tuning and validation quality and less than 1% change in the
median-of-medians latency. The small timing differences do not establish an opset
performance improvement.

| Model / candidate | Selected weight-bearing producers | Median ms | Tuning top-1 % | Validation top-1 % |
|---|---:|---:|---:|---:|
| ResNet FP32 | 0 | 4.454 | 63.28 | 66.02 |
| ResNet all QuantSmith U8S8 | 21 | 9.462 | 63.28 | 66.80 |
| ResNet all native U8S8 | 21 | 8.687 | 64.06 | 67.58 |
| ResNet all native S8S8 | 21 | 8.410 | 65.62 | 67.97 |
| MobileNet FP32 | 0 | 4.447 | 63.28 | 64.06 |
| MobileNet selective QuantSmith U8S8 | 2 | 3.646 | 62.50 | 63.28 |
| MobileNet selective native U8S8 | 2 | 3.626 | 62.50 | 62.89 |
| MobileNet selective native S8S8 | 2 | 3.640 | 62.50 | 63.28 |
| MobileNet all QuantSmith U8S8 | 53 | 2.750 | 60.94 | 63.67 |
| MobileNet all native U8S8 | 53 | 2.741 | 60.94 | 64.06 |
| MobileNet all native S8S8 | 53 | 3.888 | 60.94 | 61.33 |

The selective MobileNet candidates improve latency by roughly 18% versus FP32
with most producers left floating point. All-quantized unsigned candidates are
faster still, but their tuning accuracy is lower. Quantizing the two depthwise
operators alone is therefore a useful measured tradeoff rather than an assertion
that fewer quantized operations always improve accuracy or speed. On final
validation, all native U8S8 happens to match FP32 top-1 while its selective version
does not. This reversal illustrates why small-subset quality cannot be assumed
monotonic and validation labels must not feed back into selection.

ResNet's measured proposal stays entirely FP32. Its formal tuning frontier also
retains the slower native S8S8 candidate because tuning accuracy rises by 2.34
percentage points. MobileNet's formal frontier retains FP32, selective native
U8S8, and all-native U8S8. These are exact measured Pareto sets; tiny latency
differences between statistically unseparated candidates/opsets do not establish
a unique practical winner. The report leaves deployment choice explicitly absent
until an accuracy budget is provided. Neither operator coverage nor selected-node
count participates in the frontier calculation.

Raw cases, converted graphs, compiled operator counts, verification, decisions,
and complete-model results remain under ignored
`work_dir/artifacts/quantization_kernels/` and `quantization_selection/`.

Verification: all 970 project tests, Ruff, formatting, and strict Pyrefly pass.
The 23 focused cases cover the existing comparison runner plus extraction ownership,
shared outputs, external broadcast edges, unresolved shapes, malformed runtime
responses, batch/opset parity, broadcast equivalence, and selection semantics.
Post-run checks verify all candidate hashes, all three-way disjoint sample sets,
all 66 fresh-process records, and identical paired-opset quality scores. Regression
tests explicitly reject an isolated-kernel win when its region is slower, ignore
artificial broadcast contexts for real-model proposals, retain a faster FP32
candidate over a slower more-quantized candidate, and exclude held-out labels
from frontier decisions.

## 2026-10-18 — Native ORT comparison

The project had both MobileNetV2 speedups and ResNet-18 slowdowns, despite verified
integer execution. Compare identical FP32 exports and streamed calibration data
against native ORT QDQ/QOperator graph generation, signed/unsigned activations,
and per-channel/per-tensor weights. Preserve fresh-process repetitions, raw traces,
compiled graphs, and held-out quality rather than choosing the fastest candidate.

The verified run uses QuantSmith 0.10.2, ORT 1.30.0, an arm64 Mac, CPUExecutionProvider,
ORT_ENABLE_ALL, one thread, batch one, seed 42, 128 train calibration images,
and 256 held-out validation images per model. Each candidate starts from the same
saved FP32 bytes and replays the same preprocessing and calibration selection.
The timed input is the first selected validation image, shared across candidates.

Latency below is the median of three fresh-process medians, each from 50 warmups
and 300 unprofiled runs. Repetitions alternate variant order. Profiling occurs in
a separate session after timing/RSS capture. Full project tests and experimental
setup checks finish before the verified sweep starts; earlier exploratory timings
overlapped checks and are excluded from these conclusions.

### ResNet-18

| Variant | Median ms | Top-1 % |
|---|---:|---:|
| FP32 | 4.457 | 66.02 |
| QuantSmith producer U8S8 | 9.553 | 66.41 |
| QuantSmith function-aware U8S8 | 9.428 | 66.80 |
| ORT QDQ U8S8, per-channel | 8.605 | 67.58 |
| ORT QDQ S8S8, per-channel | 8.397 | 67.97 |
| ORT QOperator U8S8, per-channel | 8.607 | 67.58 |
| ORT QDQ U8S8, per-tensor | 8.589 | 66.41 |
| ORT QDQ S8S8, per-tensor | 8.367 | 64.45 |

Native ORT reproduces the slowdown. Changing representation, activation signedness,
or weight granularity does not beat FP32 in this experiment. QDQ and QOperator
U8S8 compile to the same operator counts and have nearly identical latency.
The faster per-tensor signed candidate also loses held-out accuracy.

All quantized candidates compile all 20 convolutions to QLinearConv. Averaging
the separate profiles across repetitions gives these approximate per-inference
operator totals:

| Variant | Convolution ms | Quantize + dequantize ms | Transpose ms |
|---|---:|---:|---:|
| FP32 | 3.741 | 0 | 0.619 |
| QuantSmith producer U8S8 | 8.498 | 0.479 | 0.345 |
| QuantSmith function-aware U8S8 | 8.492 | 0.339 | 0.345 |
| ORT QDQ U8S8 | 8.516 | 0.022 | 0.022 |
| ORT QDQ S8S8 | 7.749 | 0.457 | 0.021 |

FP32 convolution time includes NhwcFusedConv and FusedConv. ORT U8S8 retains only
two QuantizeLinear, two DequantizeLinear, and two Transpose nodes, compared with
26/18/10 for QuantSmith producer and 18/10/10 for function-aware output ranges.
ORT also keeps pooling in NHWC and removes the remaining standalone Relu nodes
in the U8S8 graph. These differences explain an additional optimization opportunity
in QuantSmith. However, roughly 8.5 ms inside integer convolutions versus 3.7 ms
inside FP32 convolutions shows that conversion removal alone cannot explain or
resolve the overall slowdown here. Instrumented totals are diagnostic attribution,
not exact decompositions of the separately measured unprofiled latency.

### MobileNetV2

| Variant | Median ms | Top-1 % |
|---|---:|---:|
| FP32 | 4.454 | 64.06 |
| QuantSmith producer U8S8 | 3.766 | 58.59 |
| QuantSmith function-aware U8S8 | 2.736 | 63.67 |
| ORT QDQ U8S8, per-channel | 2.745 | 64.06 |
| ORT QDQ S8S8, per-channel | 3.905 | 61.33 |
| ORT QOperator U8S8, per-channel | 2.733 | 64.06 |
| ORT QDQ U8S8, per-tensor | 2.708 | 49.61 |
| ORT QDQ S8S8, per-tensor | 3.879 | 52.34 |

QuantSmith's function-aware path is about 1.63 times faster than FP32 and closely
matches native ORT U8S8. Its top-1 differs by one correct prediction out of 256;
this is not proof of accuracy equivalence. All quantized candidates compile all
52 convolutions to integer kernels.

Function-aware calibration reduces QuantSmith's QuantizeLinear/DequantizeLinear
counts from 36/36 to 2/2. Separate profiles put their combined time at about
1.092 ms versus 0.024 ms per inference, while convolution time stays around
2.7 ms. This reproduces a graph-boundary cost with a measured remedy.

Native signed QDQ has faster convolution totals (about 2.427 ms) than native
unsigned QDQ (2.737 ms), but retains 37 conversion pairs and 35 Clip nodes.
Conversion and Clip totals add about 1.57 ms in the signed profiles, and the
unprofiled signed graph is slower overall. A faster kernel in isolation does
not guarantee a faster graph.

Per-tensor U8S8 saves only about 0.037 ms versus native per-channel U8S8 while
top-1 drops by 14.45 percentage points. Choosing the lowest latency without
quality would therefore select a strongly regressed candidate. No default
quantization policy is changed by this investigation.

Median fresh-process inference peak RSS is approximately 116.5 MiB FP32,
107.9 MiB QuantSmith producer, 101.5 MiB function-aware, and 100.1 MiB native
per-channel U8S8 QDQ. For ResNet-18 the corresponding values are 213.2, 123.0,
123.1, and 116.2 MiB. These are process high-water marks rather than isolated
tensor/workspace usage.

### Implications and next experiments

The current blanket statement that quantization slows these models is unsupported:
the same machine/runtime speeds up MobileNetV2 substantially and slows ResNet-18
even with native ORT quantization. The result also does not establish why any
particular Hub model is fast. Specialized low-bit kernels, different hardware,
model shapes, and autoregressive cache/memory behavior are separate comparisons.

The next useful work is to profile the largest ResNet-18 integer convolutions
with their actual dimensions on the target deployment device, then compare an
alternative optimized runtime/build using identical inputs and quality checks.
Within QuantSmith, shared residual/activation boundaries and layout-preserving
pooling are concrete opportunities to approach native ORT's graph. Native ORT's
own 8.6 ms result shows why graph cleanup alone should not be promised to beat
the 4.46 ms FP32 baseline. Neither direct QOperator emission nor per-tensor
weights is supported as a general remedy by these measurements.

### Evidence and limits

The completed raw results live under ignored
`work_dir/artifacts/quantization_performance_verified/`: model/input/candidate
SHA-256 identities, selected image metadata, quality, every fresh-process timing,
compiled graphs, and raw ORT traces. Results are reproducible with the README
command plus `--output work_dir/artifacts/quantization_performance_verified`.

The runner also verifies automatic export when saved FP32 graphs are missing.
A real ORT regression test confirms one-image-stride calibration merges extrema
from both an early dark sample and a late bright sample. This avoids ORT 1.30's
`CalibMaxIntermediateOutputs=1` behavior, which clears data before range computation
and reports that no data was collected. The experiment uses `CalibStridedMinMax=1`
instead; no runtime library workaround or monkeypatch is installed.

No specific Hugging Face model/runtime pairing, GPU serving stack, other CPU,
batch-size sweep, or Transformer prefill/decode shape sweep is measured here.
The operator labels establish runtime dispatch, not the exact machine instructions
or microkernel implementation. Wider accuracy confidence intervals, isolated
tensor/workspace memory, and cold-start costs remain unmeasured. Signedness,
granularity, and clipping affect accuracy; no candidate is selected on held-out
quality for deployment, and the 256-image subset does not establish equivalence.

Verification: the project-wide `just check` passes Ruff, formatting, strict Pyrefly,
and 952 tests. After refining CLI list parsing, all six focused regression cases
pass along with changed-file Ruff/format/Pyrefly checks. The automatic-export
smoke command succeeds, and the final report validates all 48 fresh-process
measurements, candidate/input hashes, and disjoint calibration/evaluation metadata.
