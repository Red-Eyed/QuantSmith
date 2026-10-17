# ONNX backend

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
a fresh set at every graph revision. Filesystem loading lives only at this boundary.

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
contract and owns its output. It introduces no optimizer protocol: there is one
implementation and no established substitution requirement. ORT remains responsible
for provider-specific QDQ fusion, kernel selection, and buffer management.

## Optional ONNX Runtime integer scaling

Pass `ort_integer_scaling=True` to `pipeline.quantize` to target the CPU ORT
execution path for SmoothQuant followed by built-in UINT8 `StaticW8A8`.
The default remains portable ONNX. The mode is inactive without a preceding
rescaling stage or when activation encodings are signed.

Before static calibration, it extends the rules with MinMax UINT8 boundaries on
remaining smoothing divisions and connected residual Add operands. Explicit
exclusion overrides and custom boundary algorithms remain authoritative. These
additional `QuantizeInput` operations appear in the same stage's returned plan;
calibration streams the same replay factory without retaining prior batches.
The boundary ranges use MinMax even when the weight-bearing stage uses Percentile.

After lowering, exclusive Conv → Clip → QDQ → smoothing paths can absorb scales
into private weight encodings and biases. An integer broadcast `Min` applies the
channel-dependent ReLU6 upper bounds. Remaining eligible DQ → Div → Q boundaries
use `com.microsoft::QLinearMul` with UINT8 reciprocal coefficients. Shared producer
values, exposed outputs, unsupported encodings/layouts, and coefficients that
cannot be represented safely are preserved or use the supported fallback.

This mode changes rounding and quantizes additional tensors. Plans record the
pre-rewrite calibration decisions, including the new boundaries; the final ONNX
initializers hold the rewritten weight scales and integer bounds. Evaluate the
final graph for accuracy. QLinearMul requires ORT's contrib operator domain;
other runtimes are not promised to execute these graphs. ORT still chooses and
fuses its kernels. Speed and memory improvements require measurement on the target
model and hardware; file size alone does not establish RAM savings.
