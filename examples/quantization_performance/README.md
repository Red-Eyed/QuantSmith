# Quantization performance investigation

The current priority is Transformer latency. The
[global Transformer preoptimization study](../transformer/DEVLOG.md)
compares six matched candidates over three fresh-process repetitions: full
preoptimization shrinks files, but does not produce a repeatable FP32 speedup.
The [synthetic fragment study](DEVLOG.md) finds substantial single-token gains
for larger projection sets, slowdowns in other shapes, and a concrete benefit
from exporting native opset-21 Gelu instead of retaining decomposed arithmetic.
The CNN experiments below remain supporting investigations.

## Calibration-free Transformer fragments

```sh
uv run --locked --group examples -m examples.quantization_performance.transformer_kernels
```

PyTorch constructs random FP32 fragments and exports them; all measured execution
and quantization use ONNX Runtime CPU. No training, dataset, observer or
calibration pass is used. The default matrix contains 19 fragments, six geometries
and two opsets. It covers Linear with bias, GELU, SiLU, residual Add, input
LayerNorm/RMSNorm, scalar/channel/full-tensor broadcasting, noncontiguous input,
GELU MLP, SwiGLU, separate/packed QKV, isolated QKᵀ and attention×V, complete
Stackformers cross-attention, and a one-layer causal Stackformers Transformer.

Query lengths are 1, 32 and 128; widths are 64, 256 and 768; batches are 1 and 4;
head counts are 4, 8 and 12. The explicit geometry list is in the JSON report and
can be replaced with `--geometries`. Exported query and KV lengths are symbolic,
with independent axes where the operation permits them; batch and width stay
fixed. Export traces use nonsingleton representative lengths so a requested
one-token measurement cannot silently specialize the graph. Tests execute the
same FP32 and integer graphs/sessions at query/KV pairs (1,1), (2,7), (7,2) and
(31,13), with FP32 parity against PyTorch. Full-tensor Add uses a runtime residual
with the same query length; scalar/channel Add keep constant broadcast operands.
Cross-attention and isolated matmuls use the declared key/value context length.
The causal encoder uses only the query input:
a single-token encoder case is not cached autoregressive decoding. The attention
case includes key/value projections, so it does not simulate a precomputed KV cache.

Compare ORT FP32, native dynamic INT8, fixed U8S8/S8S8 MatMulIntegerToFloat and
fixed U8S8/S8S8 QLinearMatMul followed by DequantizeLinear. Static activation
grids use the declared `--activation-range` (default 4, step 4/127); weight grids
use each constant's absolute maximum. Activation quantization, output conversion
and dynamic scale computation are included in timed inference. Weight conversion,
packing, export, session initialization and profiling are outside it. This
runner does not invoke QuantSmith's calibration/planning pipeline.

The default `--scope all` inserts fixed-grid integer kernels for both constant-
weight projections and dynamic attention MatMuls. Use `--scope weights` to keep
dynamic attention matmuls floating. ORT's native dynamic quantizer only targets
constant-weight MatMuls here. Paths with `integer_matmuls: 0` are unchanged FP32
controls, irrespective of their requested scheme. Fixed scales can clip QK scores
or round small attention probabilities to zero: numerical error is reported for
every candidate, with no automatic accuracy budget or winner promotion.

Export once at opset 18, then use ONNX semantic version conversion to compare
opsets 18 and 21 without changing the Torch exporter path. Both execute with
ORT_ENABLE_ALL and one thread by default. Use `--opset-mode native` to export
each opset directly instead; this can expose newer operators such as Gelu that
version conversion does not introduce into an older decomposed graph.
Record optimized graph operators, raw traces, exact serialized graphs and hashes,
input geometry/strides/hash, relative
MSE and maximum absolute error, and all median/p95 measurements. Three timed
sweeps alternate candidate order, after ten warmups, with 100 calls per candidate.
Diagnostic profiling runs separately after session construction and before
unprofiled warmups. Repetitions share a process and sessions; these are kernel
experiments, not independent-process model benchmarks or RAM measurements.

Reports and artifacts go to `work_dir/artifacts/transformer_ort_dynamic/` and
declare `dynamic_sequence: true`. The previous `transformer_ort_kernels/` and
related studies contain static-shape exports; their timings are historical
results and must not be presented as dynamic-shape measurements.
Change budgets with `--warmup`, `--runs`, `--repetitions`, `--profile-runs`, and
`--threads`; filter fragments with, for example, `--fragments '["linear","swiglu"]'`.
The runner checkpoints JSON after each completed case, shows progress by default
and supports `--quiet`. It also prints the final report as JSON.

## CNN comparison

Compare QuantSmith with ONNX Runtime's native static quantizer on identical
pretrained ResNet-18 and MobileNetV2 FP32 exports. This experiment asks whether
slowdowns come from QuantSmith's graph boundaries or the runtime's integer kernels.

The objective is end-to-end latency at acceptable task quality. Integer operator
count is diagnostic evidence, never a reward. FP32 remains a valid result.

Hugging Face's [Optimum ONNX quantization guide](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/quantization)
uses ORT's quantization tools and exposes hardware-specific configurations. This
runner calls ORT directly; it does not benchmark a particular Hub checkpoint,
Optimum's complete optimization pipeline, or GPU/LLM serving implementations.

Some fast Hub inference paths combine packed low-bit weights with specialized
fused kernels. For example, [HQQ documents torchao and Marlin fused 4-bit kernels](https://huggingface.co/docs/transformers/main/en/quantization/hqq).
Those paths differ from this experiment's static W8A8 CPU convolution workload.
As an intuition for the memory benefit, seven billion weights occupy about 14 GB
at 16 bits versus 3.5 GB at 4 bits before metadata. Reading fewer weight bytes can
help a bandwidth-limited workload, but this comparison does not establish that
either measured CNN is bandwidth-limited. Cache policy, model dimensions,
precision, device, and kernels must match before attributing a speed difference
to quantization. Hugging Face likewise advises
[benchmarking accuracy and speed on the target task and hardware](https://huggingface.co/docs/transformers/v5.2.0/en/quantization/selecting).

## Run

```sh
uv run --locked --group examples -m examples.quantization_performance.main
```

The command downloads missing Imagenette/checkpoint inputs, reuses saved exports
under `work_dir/artifacts/integer_execution/<model>/onnx_fp32.onnx`, and exports
missing graphs automatically. A saved export must match the example checkpoint;
the runner checks PyTorch/ONNX output parity on the timed held-out input.

Each model uses its exact checkpoint preprocessing, 128 train calibration images,
256 validation images, seed 42, batch one, and one CPU thread by default. Calibration
and evaluation stream images without retaining datasets or batch activations.
ORT MinMax uses one-image strides to merge extrema with bounded intermediate
storage. A regression test verifies extrema survive across strides.

Useful controls:

```sh
# Explicit small functional check; this cannot establish task quality.
uv run --locked --group examples -m examples.quantization_performance.main \
  --models mobilenet_v2 --calibration-samples 8 --evaluation-samples 8 \
  --warmup 5 --runs 20 --repetitions 1 \
  --output work_dir/artifacts/quantization_performance_smoke

# A separate thread-count experiment; preserve results in another directory.
uv run --locked --group examples -m examples.quantization_performance.main \
  --threads 4 --output work_dir/artifacts/quantization_performance_threads4
```

`--models` and `--variants` accept lowercase values, including comma-separated
lists. `--fp32-root`, `--cache`, `--output`, sample counts, seed, warmup, runs,
repetitions, and `--quiet` are explicit. The command prints JSON to stdout and
progress to stderr. Run comparisons without competing compute workloads.

## Controls

| Variant | Activations | Weights | Graph generation |
|---|---|---|---|
| `fp32` | FP32 | FP32 | Original export |
| `qs_producer_u8s8` | Asymmetric UINT8 | Symmetric per-channel INT8 | QuantSmith producer output ranges |
| `qs_function_u8s8` | Asymmetric UINT8 | Symmetric per-channel INT8 | QuantSmith activation-function output ranges |
| `ort_qdq_u8s8` | Asymmetric UINT8 | Symmetric per-channel INT8 | Native ORT QDQ |
| `ort_qdq_s8s8` | Symmetric INT8 | Symmetric per-channel INT8 | Native ORT QDQ |
| `ort_qoperator_u8s8` | Asymmetric UINT8 | Symmetric per-channel INT8 | Native ORT QOperator |
| `ort_qdq_tensor_u8s8` | Asymmetric UINT8 | Symmetric per-tensor INT8 | Native ORT QDQ |
| `ort_qdq_tensor_s8s8` | Symmetric INT8 | Symmetric per-tensor INT8 | Native ORT QDQ |

All quantizers use MinMax and the same deterministic calibration selection.
ORT's operator-selection and fusion defaults remain intact, whereas QuantSmith
uses its built-in supported-operator policy. Differences in the optimized graph
are experiment outcomes, not a claim of equivalent clipping or rounding. No
separate ORT preprocessing optimization is applied to the shared FP32 export.

## Measurements

Each variant runs in a separate fresh process with ORT_ENABLE_ALL and
CPUExecutionProvider: 50 warmups, then 300 unprofiled timings. Three repetitions
alternate forward/reverse variant order. After timing and process peak RSS capture,
a separate session records 300 profiled inferences. No calibration, decoding,
evaluation, model loading, or profiling belongs inside the latency interval.
The existing timing adapter includes output validation and feed-dictionary creation.

The same real held-out image is timed for every variant of one model. Held-out
top-1, FP32 prediction agreement, and logit MSE use all selected validation images.
This bounded subset is useful for detecting large regressions; it does not
establish population-level accuracy equivalence.

Outputs under `work_dir/artifacts/quantization_performance/` include:

- Configuration, runtime/package versions, FP32/input/candidate SHA-256 identities,
  selected calibration/evaluation file metadata, quality scores, and every
  repetition's median/p95 latency in `report.json`.
- Each model's `comparison.json` and generated ONNX candidates.
- Fresh-process reports, compiled optimized graphs, and raw ORT traces in
  `repeat_<index>_<variant>/`.

`profiled_operator_us` is a sum across profiled inferences: divide by `runs` for
microseconds per inference. It comes from a different, instrumented session;
do not subtract it from unprofiled latency as an exact accounting identity.
Process RSS is a cumulative high-water mark including Python, runtime setup,
optimized graph serialization, and inference, rather than isolated tensor memory.
Compiled graphs are local diagnostic artifacts and may be hardware-specific.

See [DEVLOG.md](DEVLOG.md) for measured results and unresolved questions.

## Shape, broadcasting, and opset investigation

```sh
uv run --locked --group examples -m examples.quantization_performance.kernels
uv run --locked --group examples -m examples.quantization_performance.selection
```

Missing prerequisite reports are generated automatically, including their dataset,
checkpoint, and export setup. Running `selection` alone can therefore produce the
baseline report, kernel investigation, and complete-model comparison. Existing
reports and source graphs are checked by SHA-256 before reuse.

`regions.py` ranks source Conv/MatMul/Gemm operations from the original FP32 trace.
By default it takes the two most expensive operations plus the most expensive
matrix projection. It extracts each operation alone and a private chain of up to
two following Relu/Clip/Add/Mul/MaxPool nodes, retaining external residual inputs
and constant ancestors. Shared or public outputs stop chain extension. Input
activations are replayed from the original FP32 model for each calibration image;
only the current region's tensors are retained.

`kernels.py` measures four graph-generation choices: FP32, QuantSmith function-aware
U8S8, native ORT per-channel U8S8, and native ORT per-channel S8S8. Defaults compare
batches 1/4 and opsets 18/21. Batch four repeats the same current image to isolate
geometry; it is not a four-image accuracy experiment. The first selected convolution
also gets scalar, channel-wise, and full-tensor constant multiplication controls
with identical coefficients. These artificial controls measure broadcasting and
fusion costs; they are never used to rewrite the original model.

Opset changes use ONNX's version converter and model checker. Each converted graph
is checked against its own original-precision/quantized output, so an opset number
is never changed by editing metadata alone. Unsupported conversions fail explicitly.
The saved outputs record the actual optimized operator counts, shapes, grouping,
weight dimensions, strides, padding, and dilation.

Microbenchmarks use one process with separately created sessions, ten warmups,
100 unprofiled runs, and three forward/reverse-order repetitions by default.
Output dtype/finiteness is checked before timing. The timing interval contains
the normal Python ORT call, not full-output validation or image preprocessing.
Session creation, packing, calibration, and optimization are excluded. Results
are local diagnostics; the decisive full-model benchmark uses fresh processes.

Controls include `--top-k`, `--region-hops`, `--batches`, `--opsets`,
`--calibration-samples`, timing counts, `--report`, `--output`, and `--quiet`.
For example, a bounded functional check is:

```sh
uv run --locked --group examples -m examples.quantization_performance.kernels \
  --top-k 1 --batches 1 --opsets 18 --calibration-samples 2 \
  --warmup 1 --runs 2 --repetitions 1 \
  --output work_dir/artifacts/quantization_kernels_smoke
```

`selection.py` proposes quantizing only measured real regions whose median local
latency improves by at least `--minimum-speedup` (default 0.05). It prefers the
surrounding-region measurement over the isolated operation, considers batch one
for the batch-one model, and leaves unmeasured operations in FP32. An empty
proposal reuses the FP32 control without instantiating a quantizer. This bounded
proposal generator is not an exhaustive search or a guarantee of a global optimum.

Every nonempty proposal is freshly calibrated and compared with FP32 and the
all-quantized controls in complete models at both opsets. Calibration uses the
original 128 train images. Tuning uses the next 128 deterministically ranked train
images, excluding calibration. All candidates are evaluated on the same 256
validation images for reporting. The quality/latency frontier uses **tuning** top-1
and fresh-process model latency; final validation labels do not decide the frontier.
No deployment winner is declared without a user-specified accuracy budget.

Complete models use 50 warmups, 300 unprofiled runs, and three alternating
fresh-process repetitions by default, with separate profiling sessions. Use
`--tuning-samples`, `--minimum-speedup`, timing counts, `--kernels`, `--output`,
and `--quiet` to make experiment budgets explicit. Raw evidence stays under
`work_dir/artifacts/quantization_kernels/` and `quantization_selection/`.

## Quantize the ORT-preoptimized graph

```sh
uv run --locked --group examples -m examples.quantization_performance.preoptimization
```

This is the issue-007 experiment. It explicitly creates an ORT CPU session with
`ORT_ENABLE_ALL`, saves `optimized_fp32.onnx`, reloads that file, and supplies the
reloaded graph to QuantSmith. Description, planning, fresh calibration, lowering,
and the final integer audit use that revision. Optimizing only the final execution
session would not test this hypothesis.

Four complete-model candidates are compared for each source model: original FP32,
saved optimized FP32, freshly quantized original, and freshly quantized optimized.
Both quantization paths use function-aware MinMax with the same 128 train images;
the same 256 validation images measure quality. Each candidate has three
alternating fresh-process timing repetitions with 50 warmups and 300 unprofiled
runs, followed by separate profiling. ROM, process peak RSS, compiled operator
counts, and source-oriented plans/audits are saved alongside latency and quality.
Integer coverage does not rank candidates.

Before calibration, parity compares **optimization-disabled** execution of the
original graph with optimization-disabled execution of the saved, already-fused
graph. The report retains maximum absolute error and MSE over the complete held-out
selection. Default parity tolerances match the existing example export checks
(`rtol=1e-3`, `atol=1e-4`) and are explicit CLI controls. Fused FP32 arithmetic can
have rounding differences, so numerical closeness is not called bitwise parity.

The current quantizer selects standard Conv/MatMul/Gemm. ORT's
`com.microsoft::FusedConv` includes an activation and `NhwcFusedConv` also changes
layout; neither is treated as a standard Conv by dropping its domain/name.
They remain FP32. Expected quantization failures are recorded with their input
graph hash; no restricted optimization pass is substituted automatically.

The intentionally unfused Conv/BatchNorm/Relu fixture verifies that full ORT
optimization fuses nodes without changing the caller graph. It also exposes a
compatibility limit: if intermediate type metadata is absent, custom fused
operators can prevent standard ONNX shape inference from recovering a subsequent
calibration tensor's FP32 metadata. The fixture records the precise failure;
its metadata-annotated counterpart verifies successful fresh quantization of the
saved graph. The experiment does not patch the library to hide missing metadata.

Use `--report`, `--output`, sample counts, timing counts, parity tolerances, and
`--quiet` to control the experiment. Missing baseline/dataset/checkpoint setup is
automated. Artifacts default to `work_dir/artifacts/quantization_preoptimization/`.
The report includes both graph identities, qualified operator counts, ONNX domains
and versions, environment/provider, and source lineage where names or outputs
survive. Absorbed/removed identities are explicitly marked rather than guessed.
The original graph and all input/output snapshots remain available for inspection.

These fully optimized snapshots may contain hardware-specific operators or packed
weights. They are local diagnostic artifacts for the recorded ORT CPU environment,
not a new portable export format or a change to library defaults.
