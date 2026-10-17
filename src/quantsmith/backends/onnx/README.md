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
