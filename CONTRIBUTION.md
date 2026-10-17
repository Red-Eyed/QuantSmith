# Contributing to QuantSmith

QuantSmith is built for experiments that can be inspected and reproduced. Contributions
are most useful when they make a quantization decision easier to understand,
measure, or extend.

## Find the right starting point

- **New calibration or quantization method:** Read the
  [algorithm guide](src/quantsmith/algorithms/README.md) and its existing method
  packages. Keep the decision logic in its own package and select it through
  `Rules[Algorithm]`.
- **New reconstruction method:** Read the
  [reconstruction guide](src/quantsmith/reconstruction/README.md) and its paired-replay
  contract. Reconstruction methods use `Rules[Reconstructor]`.
- **ONNX support or graph lowering:** Read the
  [backend guide](src/quantsmith/backends/onnx/README.md) and cover the new graph
  behavior with an ONNX Runtime integration test.
- **A model or task comparison:** Follow an [example](examples/README.md). Keep
  export, calibration inputs, method selection, and held-out evaluation visible
  in that example's `main.py`; put its run commands in its own README.

Planning algorithms consume graph snapshots and supplied statistics and return
plans. The shell handles ONNX execution and replayable input factories. Keep
calibration streaming: do not materialize the dataset or retain activations
between batches. When one stage changes the graph, collect fresh statistics for
the next stage.

## Develop and verify

The [justfile](justfile) uses uv to provision the pinned Python interpreter and
locked dependencies. From the project root:

```sh
just check                        # Ruff, formatting, strict Pyrefly, all tests
just test tests/test_ownership.py  # One regression module
just test-algorithms              # Numerical algorithm tests
just wheel                        # Build a local wheel
```

For a focused change, run the relevant tests first; run `just check` before
submitting. Generated files belong in the ignored `work_dir/`: examples and
downloads use `work_dir/artifacts/`, pytest temporary files and per-test working
directories use `work_dir/tests/`, and `just wheel` writes to `work_dir/dist/`.
These directories are created automatically. Pytest resets its temporary directory
on each run; do not keep reports or datasets there. Tool caches remain in their
standard ignored locations at the repository root. Only the project owner
publishes packages.

Examples and tests default `ORT_DISABLE_TELEMETRY` to `1` before importing ONNX
Runtime, preventing its import-time telemetry session files. An explicit
environment setting takes precedence; the library does not change this setting.

## Make results reviewable

For a new algorithm, provide an intuitive explanation, a small numerical
example, a Mermaid diagram, limitations, and primary references in its package
README. Test numerical decisions on a small graph with known statistics.
Boundary tests should cover malformed external responses, array ownership,
and established failure semantics. For a model comparison, report the FP32
baseline, calibration setup, held-out metric, and failure cases alongside any
improvement. Do not present a QDQ export as evidence of speedup or compression.

Use typed domain records and validate external values at the boundary. Expected
execution, data, and planning failures use QuantSmith's `Ok[T] | Err[E]` result union;
handle both variants explicitly. Unexpected plugin failures and broken
invariants remain exceptions. Keep comments focused on stable intent and
non-obvious constraints, and give every Python module, class, and function a
docstring. The repository's [AGENTS.md](AGENTS.md) records the full coding and
testing conventions.
