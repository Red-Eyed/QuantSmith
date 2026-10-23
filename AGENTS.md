# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project overview

QuantSmith performs post-training quantization: it uses representative calibration
inputs to choose lower-precision encodings for an already trained model. It
produces both an ONNX graph and inspectable quantization plans.

- ONNX is the serialized model graph format. ONNX Runtime executes those graphs.
- W8A8 means 8-bit weights and activations. QDQ represents quantization using
  paired QuantizeLinear/DequantizeLinear graph operations; it does not guarantee
  smaller files or faster execution.
- An encoding contains scales, integer zero points, and per-tensor or per-channel
  granularity. A plan selects consumer edges to quantize or channels to rescale.
- MinMax uses observed extrema; Percentile clips histogram tails. SmoothQuant
  balances activation and weight channels before a separate quantization stage.
- `quantsmith.result` defines the local `Result[T, E] = Ok[T] | Err[E]` union and the
  Pydantic `QuantSmithError` diagnostic payload. No result-container dependency is used.
- Stackformers supplies the attention components in Transformer examples.

## Development commands

Use uv for Python commands. The just recipes provision the pinned interpreter and
locked dependencies automatically, including example dependencies where needed.

```sh
just check                          # Ruff, formatting, strict Pyrefly, and all tests
just test                           # Full suite with pytest-xdist workers
just test tests/test_ownership.py    # One regression module
just test-algorithms                 # Numerical tests without parent ONNX fixtures
just build                          # Build a wheel; never upload package artifacts
uv run ruff format src examples tests # Apply Python formatting
```

Examples are human-readable source walkthroughs; running them is optional.
Keep run commands and options in each example’s own README, not in shared
documentation or just recipes.

Examples write graphs, JSON plans, and reports under `work_dir/artifacts/` by default.
Keep generated artifacts and downloaded data out of commits. Pytest temporary
files and per-test working directories live under `work_dir/tests/`; wheel builds
use the project-root `dist/`, which is ignored by Git. Keep runtime files in
`work_dir/`, which is ignored by Git.
Tool caches may remain in their conventional ignored locations at the root.

## Architecture and contracts

The functional core consumes graph snapshots, rules, and statistics and returns
plans. Algorithms and domain records have no ONNX execution or filesystem
dependencies. `Algorithm.requirements` declares calibration needs;
`Algorithm.plan` makes decisions from supplied statistics. Each algorithm belongs
in its own package with a short `README.md` containing an intuitive explanation,
numerical example, Mermaid diagram, limitations, and primary references.
Planning algorithms plug into `Rules[Algorithm]`; reconstruction methods use
`Rules[Reconstructor]` and their paired-replay contract.

The shell invokes an injected `Evaluator` and a replayable `Samples` factory.
Calibration streams samples and retains current activations plus bounded
statistics. Never materialize the dataset or retain activations across batches.
Histogram replay must reproduce the same inputs used to establish extrema.

The ONNX pipeline describes a graph, builds a plan, lowers it, and starts the next
stage on the resulting graph revision. Lower SmoothQuant rescaling before fresh
quantization calibration; never reuse pre-transform statistics afterward.

`QuantizationConfig` is the Pydantic configuration boundary for built-in methods.
The Python algorithm and selector protocols remain open to extensions. Example
entry points use pydantic-settings and keep model loading, export, and stage
selection visible in their own `main.py` files.

Use `Result` for expected execution, data, and planning failures that callers can
meaningfully handle. Do not require every operation to return `Result`: operations
expected to succeed can return their value normally and raise on failure.
Programming errors, broken invariants, and unexpected plugin failures remain
exceptions. Direct model construction can raise Pydantic validation errors.
Translate documented failures only at the boundary where they are expected;
never catch arbitrary exceptions just to fit a result signature. Preserve each
protocol's established failure contract. Handle results through explicit matching.

Use explicit `match` branches for `Ok(value)` and `Err(error)`, or `Err() as error`
when propagating the original container. Never interpret a wildcard as success.
Bind tuple payloads in `Ok(payload)` and unpack inside the branch; Pyrefly does
not establish exhaustiveness for nested tuple patterns. The local union supports
exhaustive checking with `assert_never` when needed. Do not add unwrapping methods,
`.map()`/`.bind()` chains, compatibility aliases, or an external result library.

`Ok` and `Err` are frozen generic dataclasses that carry native values unchanged;
they do not parse, copy, or validate their payloads. Input validation remains at
Pydantic boundaries, and domain constructors retain their ownership checks.

## Type safety and validation

- Do not use `object` or `Any` annotations anywhere in project code, including
  library adapters, examples, and tests. An external-library boundary is not an
  exception. Do not merely remove annotations and let unknown types propagate.
- Use Pydantic models or strict `TypeAdapter` schemas to validate external values
  directly at their source. Parse JSON with `model_validate_json` or
  `TypeAdapter.validate_json`. Keep validated tensors and arrays native.
- Use precise records, typed collections, and minimal callable contracts. Keep
  JSON serialization typed; do not use catch-all serialization callbacks.
- Do not use `setattr` or `object.__setattr__` to populate or rewrite records.
  Frozen models must remain frozen. Field validators return normalized values;
  model validators check relationships between fields without rewriting them.
- `arbitrary_types_allowed=True` validates the native array instance, not its
  dtype, shape, finiteness, or ownership. Preserve those numerical checks when
  adopting a Pydantic schema. Never silently coerce a runtime's wrong dtype into
  an accepted output; retain intentional dataset conversion at admission.
- Own copies of graph constants, encodings, and statistics; freeze returned
  arrays and mappings without freezing or retaining caller-owned mutable storage.
  Graph constants may contain infinities used in attention masks; selected
  quantizable data, statistics, and encodings have stricter finite-value rules.
- Do not replace validation with casts, unchecked precise annotations, checker
  suppressions, or weaker project-wide checking settings. A passing type checker
  alone does not establish runtime validity.
- Use `Absent` with a reason for missing domain information. Use `date` and
  `datetime` for dates and timestamps. Prefer exhaustive `match` handling with
  `assert_never` for closed alternatives.

## Testing and changes

Use fixtures for reusable setup and parametrization for related cases. Numerical
tests belong alongside the algorithm tests; boundary tests must cover malformed
external responses, array ownership, and preserved failure semantics. Static
rejection cases belong in `tests/test_static_contracts.py` so invalid snippets do
not require suppressions in normal project code.

Give every Python module, class, and function a docstring. Keep functions short
and flat, and document invariants and rationale rather than narrating code.
Review SOLID before non-trivial implementation and keep side effects at the edge.
Show a brief plan and obtain approval before unapproved work spanning more than
two files or changing architecture.

Before committing, run Ruff checks, Ruff formatting, and Pyrefly on the changed
Python files; re-read any formatter changes. Run the relevant behavioral tests.
On main, package changes must update both `pyproject.toml` and `uv.lock`; do not
start version-bump work on feature branches unless explicitly requested.

Use the author identity from Git configuration. Every commit message has three
sections: an imperative title of at most 72 characters, concrete change bullets,
and a paragraph explaining the motivation and design decision. Only the user
publishes package artifacts. Never force-push main.
