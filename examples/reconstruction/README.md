# Reconstruction comparison

Read [`main.py`](main.py) for the workflow. To run this example, use the
commands below from the repository root.

```sh
uv run --locked --group examples -m examples.reconstruction.main
uv run --locked --group examples -m examples.reconstruction.main --qdrop-steps 100 --json --quiet
```

The command provisions locked dependencies and compares MinMax, Percentile,
GPTQv2, and operator-level QDrop on the same deterministic FP32 projection.
Every recipe starts from the original graph; calibration and evaluation use
different seeds. The output directory is created automatically.

`work_dir/artifacts/reconstruction/report.json` records settings, version, calibration
time, and held-out MSE/maximum error. Each method saves an ONNX graph and a full
plan, including reconstructed integer codes. `--help` exposes budgets and seeds.

This is an API and numerical smoke comparison, not evidence of ImageNet accuracy,
Transformer perplexity, or inference acceleration. W8A8 methods may tie or regress
relative to static calibration. QDrop's relaxed training objective need not improve
the final hard model. See [DEVLOG.md](DEVLOG.md) for the measured default run.

PD-Quant is a future comparison for real CNN benchmarks; it is not in this runner.
The [component guide](../../src/quantsmith/reconstruction/README.md) details limitations.
