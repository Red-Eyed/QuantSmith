# Plans

A `QuantizationPlan` is an inspectable set of decisions before backend mutation.
`QuantizeInput` owns one consumer's encoding. `RescaleInput` describes a paired
activation/weight transformation. These are closed alternatives so new operation
kinds expose incomplete lowering consumers to static checking.

`QuantizeOutput` owns a producer's output encoding. It adds rounding for every
consumer of that value and preserves the public FP32 tensor name after
dequantization. Input and output conflict keys are separate; duplicate output
writes and transform/output combinations are rejected. Saved example JSON marks
output records with `target: "output"`.

`QuantizeConstant` owns reconstructed integer codes and an encoding for one
constant consumer edge. It lowers to an integer initializer plus dequantization,
preserving exact learned rounding and other consumers of the original weights.
It conflicts with `QuantizeInput` on the same edge.

`then` combines compatible patches and rejects duplicate writes, exclusions with
operations, and mixed transform/quantization stages. The application lowers a
transform stage before recalibrating the next revision. Plans do not contain I/O
or backend objects, and must be applied to their original graph revision.

`then` returns `quantsmith.result.Ok(plan)` or `Err(QuantSmithError(...))` with
the `conflict` category. Direct plan constructors use normal Pydantic validation
exceptions.

`FloatOutput` explicitly omits producer-output rounding after quantized input
computation. It requires activation/weight input operations and conflicts with
`QuantizeOutput` on that output. The initial ORT lowering supports integer MatMul
with FP32 output. Saved JSON marks it with `target: "float_output"` and no output
encoding. See [the execution design](../../../docs/float-matmul-outputs.md).
