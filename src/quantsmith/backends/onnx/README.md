# ONNX backend

[`DynamicQuantizationStage`](dynamic.py) provides calibration-free MatMul/Gemm
projection lowering with `DynamicQuantizeLinear`, `MatMulInteger`, and FP32
restoration. It preserves symbolic shapes and floating cache interfaces and
uses the same source-oriented execution audit. See
[dynamic W8A8](../../algorithms/dynamic/README.md) for supported layouts and
deployment limits.

`load`/`normalize` validate input and assign deterministic names to unnamed nodes.
`describe` translates connectivity, numeric attributes, and floating constants into
domain records. Custom-domain operator names are qualified to avoid treating them
as standard ONNX operators.

`lower` copies the graph and translates a plan into ordinary QDQ or paired scaling
nodes, with fresh names and edge-local weights. The input model is never mutated.
Constant inputs selected for quantization are rounded and saturated into INT8 or
UINT8 initializers at export, followed by `DequantizeLinear`. Runtime activation
inputs retain their QDQ pairs. Unused initializers are removed after each stage,
including original SmoothQuant weights; shared constants and direct graph outputs
remain live. The backend reuses the native affine quantizer used by reconstruction
methods, whose exact learned codes still bypass rounding.
Integer weight storage reduces serialized weight bytes, but total file size also
includes graph metadata and unquantized tensors. Runtime RAM depends on execution
provider fusion, weight prepacking, and activation buffers; it is not file size.
ONNX checking and shape inference run on the result. Unsupported metadata, axes,
control-flow subgraphs, and overridable initializers fail explicitly.
`load`, `describe`, and `lower` return `Result`; graph checking, inference, and I/O
failures become `quantsmith.result.Err` carrying `QuantSmithError` at this boundary.
`normalize` is an explicit validator that
retains exception semantics. Unselected mask constants can contain IEEE infinities;
selected quantizable weights still require finite numerical statistics.

`pipeline.quantize` is application wiring: domain analysis, an ORT evaluator,
planning, and lowering for each stage. It invalidates prior statistics by collecting
a fresh set at every graph revision. Built-in static methods receive calibrated
producer outputs and accumulator-safe INT32 biases through `integer_execution`.
The final graph is compiled privately for CPU and audited against selected integer
operations. `QuantizationResult.execution` contains the evidence or an explicit
`Absent` for pipelines that do not perform this audit, such as reconstruction.
See [architecture](../../../../docs/architecture.md) and
[execution restrictions](../../../../docs/integer-execution.md).

After a stage containing rescaling operations, `pipeline.quantize` invokes the
concrete `optimization.optimize` pass before describing and calibrating the next
revision. It folds positive channel-wise `Div` operations into directly preceding
Conv weights/biases or inference BatchNormalization gain/bias. Supported scales
have shape `[1, C, 1, ...]` matching the producer's rank and channel count.
Replacement constants are private; shared initializers remain unchanged.
A producer output with another consumer or exposed as a graph output is preserved.
Custom operators, nonlinearities, residual additions, and unsupported broadcasts
are skipped. Floating-point rounding can change slightly, and subsequent INT8
encodings can change; fresh calibration is essential.

The pass returns the backend's existing `Result[ModelProto, QuantSmithError]`
contract and owns its output. Concrete pass functions implement the graph rewrites without a new public
optimizer protocol. ORT remains responsible
for provider-specific QDQ fusion, kernel selection, and buffer management.

## ONNX Runtime integer scaling

ONNX Runtime is the execution target. SmoothQuant followed by built-in UINT8
`StaticW8A8` automatically uses the calibrated integer execution path where
supported. ONNX remains the serialization format; execution may use ORT contrib
operators. Signed activation stages do not acquire automatic UINT8 scaling or
residual boundaries. Static-only UINT8 stages do acquire connected residual
boundaries, independently of smoothing.

Before static calibration, it extends the rules with MinMax UINT8 boundaries on
remaining smoothing divisions and connected residual Add operands. Explicit
exclusion overrides and custom boundary algorithms remain authoritative. These
additional `QuantizeInput` and residual `QuantizeOutput` operations appear in the same stage's returned plan;
calibration streams the same replay factory without retaining prior batches.
The boundary ranges use MinMax even when the weight-bearing stage uses Percentile.

After lowering, exclusive Conv → Clip → QDQ → smoothing paths can absorb scales
into private weight encodings and biases. An integer broadcast `Min` applies the
channel-dependent ReLU6 upper bounds. Remaining eligible DQ → Div → Q boundaries
use `com.microsoft::QLinearMul` with UINT8 reciprocal coefficients. Shared producer
values, exposed outputs, unsupported encodings/layouts, and coefficients that
cannot be represented safely are preserved or use the supported fallback.
Complete Conv output/bias boundaries can make the specialized clipping pattern
ineligible; Clip then remains FP32. Improving this interaction is documented TBD.

Integer execution changes rounding and quantizes additional tensors. Plans record the
pre-rewrite calibration decisions, including the new boundaries; the final ONNX
initializers hold the rewritten weight scales and integer bounds. Evaluate the
final graph for accuracy. QLinearMul requires ORT's contrib operator domain;
other runtimes are not promised to execute these graphs. ORT still chooses and
fuses its kernels. Speed and memory improvements require measurement on the target
model and hardware; file size alone does not establish RAM savings.

## Functional rewrite planning

`ort_scaling.graph.GraphIndex` provides read-only queries on an owned normalized
revision. Recognition produces named matches, numerical helpers return validated
parameters, and `plan_clipping` / `plan_multiplication` produce complete
`RewritePatch` records or `Absent(reason=...)`. The ordered rule selection does not
modify graph nodes, constants, connectivity, or name reservations. Reserving a
patch's names returns a new index value sharing the same read-only connectivity.

Only `apply_patches` edits the existing graph, after all planning has completed.
The pipeline transfers an owned model to this function, preserving caller storage.
Local construction of new nodes and constants never rewrites indexed protobufs.
Stage progression returns typed `FloatGraph` and `QuantizedGraph` revisions.
`Pipeline.then` composes concrete transitions without inferring stage kinds. Calibration and ORT
execution remain in the shell, with explicit `Ok`/`Err` handling.

The public fluent facade and fold-only policy are documented in the
[typed pipeline design](../../../../docs/declarative-pipeline.md).

## Complete-group folding and exact quantizer reuse

`group_folding` applies only complete common-scale consumer groups; public values,
excluded consumers, incompatible scales or axes, and unsupported producer paths
retain the original region. `quantizer_reuse.plan_reuse` produces immutable aliases
from identical private QuantizeLinear inputs, constant encodings and attributes.
Application owns its graph; the final ORT audit checks the resulting model.
`ReuseQuantizersStage` is an explicit typed transition between quantized revisions.
The Joint SmoothQuant configuration declares it; existing recipes remain unchanged.
See [design and measured tradeoffs](../../../../docs/joint-smoothing.md).
