# QuantSmith development log

## 2026-10-17 — Activation-function-aware integer output calibration

Add an explicit ORT pipeline opt-in that calibrates private Conv output encodings
using following Relu/Clip outputs. Isolate eligibility and the algorithm decorator
from numerical algorithms and lowering. Preserve shared/public outputs, exclusions,
unsupported functions and custom algorithms. Static calibration still follows
SmoothQuant lowering on a fresh graph revision. The strategy changes numerics;
the default remains unchanged. The MobileNetV2 example exposes the option and
persists its full configuration. Design, Mermaid and restrictions are documented
in `docs/integer-execution.md`; measured model tradeoffs are recorded in
`examples/_shared/vision/DEVLOG.md`. Integrated artifacts match the measured
experimental graphs byte-for-byte; universal latency/accuracy gains remain unproven.
Fresh integrated profiling and evaluation confirm 2.75 ms / 63.67% MinMax and
3.20 ms / 64.84% SmoothQuant, versus FP32 4.54 ms / 64.06%. All 556 tests,
Ruff, formatting and strict Pyrefly checks pass. Version 0.9.1 keeps this optional.

## 2026-09-29 — Package and repository rename

Version 0.8.0 adopts QuantSmith as the project name and `quantsmith` as the
distribution and Python import name. The former `qraft` distribution name is
already used by an unrelated PyPI project. This release renames the import
package, public `QuantSmithError` diagnostic, example report metadata, and
example environment prefixes. Existing consumers must update those references.
Older entries below retain the name and paths used when they were written.
Ruff, formatting, strict Pyrefly, and all 401 tests pass. The built wheel
contains the `quantsmith` import package and `quantsmith` 0.8.0 metadata.

## 2026-09-26 — Local result union and explicit matching

Qraft 0.6.0 replaces the external result-container dependency with frozen generic
`Ok[T]` and `Err[E]` dataclasses and the closed `Result[T, E]` union in
`qraft.result`. They carry native payloads unchanged. Pydantic still validates
external inputs and diagnostics; the containers add no parsing or copying.

Library code, examples, test helpers, and documentation now match both variants
explicitly. There are no success-by-default branches or unwrapping/composition
methods. Imports and plugin return values must migrate to the local variants.
Pyrefly verifies exhaustive matches and rejects accessing a successful value
before narrowing. Tuple payloads are unpacked inside the `Ok` branch because
nested tuple patterns do not establish exhaustiveness with the current checker.

`Result` is for expected failures that callers can meaningfully handle, not every
operation. Normal return values and exceptions remain appropriate where success
is expected. Programming errors and broken invariants propagate as exceptions;
only documented failures are translated at the boundary that expects them.

Verification: Ruff, formatting, strict Pyrefly, and all 359 tests pass. Static
cases cover payload types, immutable fields, missing variants, and unavailable
unwrapping. Existing integration tests preserve numerical behavior and diagnostic
identity. The parallel run emitted the previously recorded macOS native teardown
error after passing assertions; the serial run completed without that message.
The final pre-commit parallel gate also completed without the teardown message.

## 2026-09-26 — Initial quantization pipeline

The initial design isolates algorithms from ONNX mutation and exposes quantization
intent as a typed plan. Domain/configuration records use Pydantic `BaseModel` and
`Field`; domain constructors are strict so Pyrefly rejects coercive input mistakes.
Python is pinned to 3.12.13 to use supported native dependency wheels.

Static W8A8 and SmoothQuant compose in separate stages. Transforming a graph forces
fresh calibration. Consumer-edge operations preserve shared tensors and excluded
consumers. Histogram collection uses fixed storage with replay, rather than
retaining the calibration dataset.

Verification covers real ORT inference for Conv, MatMul, all Gemm transpose
combinations, INT8/UINT8 activation encodings, SmoothQuant floating equivalence,
shared consumers, malformed boundaries, streaming retention, and static rejection
categories. Early tests exposed NumPy scalar/array normalization and Pydantic lax
constructor typing; both were corrected and retained as regressions.

Toy-graph numerical tolerances do not establish production-model accuracy,
integer-kernel fusion, artifact compression, or deployment speed. Those remain
unproven. MSE calibration and reconstruction algorithms are deferred.

## 2026-09-26 — Explicit outcomes and Transformer demonstrations

Qraft 0.2.0 introduces Pydantic `Ok[T] | Err[Failure]` outcomes for planning,
calibration, composition, ONNX admission/lowering, execution, and evaluation.
Expected failure categories and diagnostics survive plugin and stage boundaries;
failed lowering cannot expose a partial mutation. Constructors and explicit
validators retain normal exception semantics. Known ONNX checker/inference and
ORT exception types are translated at adapters; unexpected programming errors and
interrupts propagate. Static tests reject using an unchecked result as a value.

Input contracts now admit int64 token IDs alongside FP32 arrays. Graph constants
preserve causal attention masks containing infinities; numerical calibration still
requires finite selected weights and observations. Real Transformer export exposed
this boundary mistake and the fix has mask/weight regressions.

The demonstrations now cover two Stackformers causal Transformer variants, a
tabular MLP, three pretrained CNNs, and an optional RNN. Transformers are the
default priority. Adjacent experiment logs record measured accuracy/perplexity,
training provenance, and slowdowns. Portable HTML/JSON reports show actual held-out
predictions, images or continuations, saved model artifacts, and configuration.
Task metrics remain outside the reusable quantization core.

## 2026-09-26 — Standard returns containers

Qraft 0.3.0 replaces the custom outcome containers with `returns.result.Result`,
`Success`, and `Failure`. The Pydantic diagnostic is now named `QraftError` so
library failure containers and domain errors remain distinct. Algorithms, adapters,
examples, and tests import containers directly from returns; no compatibility
Ok/Err implementation remains. Existing failure categories and exception boundaries
are preserved.

Pyrefly checks the concrete map/bind contracts used by the project. Because Result
is an abstract base, success/failure matches cannot establish closed-union
exhaustiveness. Production branches handle Failure before extracting values, and
tests document that unchecked unwrap remains statically permitted but fails at
runtime. Library callbacks retain normal programming-error propagation.

## 2026-09-26 — Readable models × methods examples

The catalog now links eight explicit model folders to MinMax, Percentile, and
SmoothQuant → MinMax commands. The previous suite names and wrapper-only entry
points hid the application steps. Each main file now shows its concrete model,
export, replayable inputs, operator selection, Qraft calls, and held-out comparison.
Downloads, training primitives, and report rendering remain supporting helpers.

The projection now covers all three methods. Method selection accepts ordinary
names and comma-separated subsets. Stage-plan JSON preserves NumPy dtype, shape,
and values. Numerical regressions cover all 24 cells using offline model exports;
real-data smoke runs verify the CLI/report paths separately.

## 2026-09-26 — Pure planning and independent algorithm tests

Qraft 0.5.0 separates rule selection and plan assembly in `algorithms/core.py`
from calibration orchestration in `algorithms/shell.py`. Algorithms still receive
graph data and statistics through the same protocol, and existing `build_plan`
imports remain compatible. Selection carries its original graph snapshot.
Statistics now detach input dictionaries and expose read-only mappings, closing
an aliasing gap in the previous frozen record.

Direct unit tests use tiny NumPy arrays and independently calculated answers for
affine encoding, Static W8A8, and SmoothQuant. Layouts include Conv, MatMul, batched
MatMul, and all Gemm transpose combinations. Core tests cover rule precedence,
exclusions, requirement sharing, composition conflicts, and diagnostics. Shell
tests inject in-memory evaluators/sources to verify execution and replay counts.
The independent `just test-algorithms` target excludes parent ONNX fixtures.

Verification: 103 independent tests pass, with no ONNX, ORT, Torch, or TorchVision
modules imported by that test process. The full suite passes 310 tests; Ruff and
strict Pyrefly pass. The 0.5.0 portable wheel includes the new planning modules.

Purity is a contract for open plugins, not a Python-enforced property. Statistics
must come from the selected graph revision; provenance is not fingerprint-checked.
These numerical tests do not establish production-model quality or speedup.

## 2026-09-26 — Parallel test command

Qraft 0.5.1 adds pytest-xdist to the development dependencies. `just test` runs
the complete suite with `pytest -n auto`, and `just check` calls that target after
linting and type checking. Additional pytest arguments pass through unchanged,
including explicit worker counts and serial debugging with `-n 0`.

Verification: `just test` passes all 310 tests in parallel. Ruff lint/format checks
and strict Pyrefly pass; manifest and lockfile both declare 0.5.1.

## 2026-09-26 — Separate tool configuration files

Qraft 0.5.2 moves Ruff, Pyrefly, and pytest settings into `ruff.toml`,
`pyrefly.toml`, and `pytest.ini`, preserving their existing values. Static contract
tests explicitly load the new Pyrefly config. Package metadata, dependency groups,
build-system declarations, and uv settings remain in `pyproject.toml`.

Verification confirms automatic config discovery, 310 collected tests, and
310 passing tests under xdist. Ruff and strict Pyrefly pass. One earlier run
emitted the intermittent native shutdown diagnostic after test completion; the
final full check completed without it.

## Version 0.5.8 — establish reviewable commit history

Split the initial implementation into seven independently checked commits: typed
core and tooling, ONNX execution, projection tutorial, Transformer/tabular
examples, pretrained vision examples, character RNN/catalog, and final docs.
Each package-changing commit advances the manifest and lockfile together.
The README leads with model quantization, layer control, and measured quality;
the mnemonic is two sentences. The exact API example was executed against a
local FP32 graph and produced valid SmoothQuant/W8A8 ONNX output. Ruff, strict
Pyrefly, all 310 tests with pytest-xdist, and the portable wheel build validate
the completed tree.

The previously observed macOS native teardown error (`recursive_mutex lock
failed`) also appeared after test completion in two checks of the full model
matrix. Both reported 310 passing tests and pytest exit status 0. The underlying
native-library lifecycle issue remains undiagnosed; these checks establish
passing assertions, not clean native teardown on every run.

The final 0.5.8 gate completed with 310 passing tests and no native teardown
message. The wheel is `qraft-0.5.8-py3-none-any.whl`; final source comparison
preserved the implementation byte-for-byte, with only package versions and
this log differing from the approved working snapshot.
