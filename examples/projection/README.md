# Small projection × quantization methods

Read [`main.py`](main.py) for the workflow. To run this example, use the
commands below from the repository root.

Minimal deterministic ONNX MatMul without downloaded data. This example measures mean squared error and maximum absolute error.

Start at [`run()` in main.py](main.py), then follow `run_variant()`, `sample_source()` in
[`model.py`](model.py), and `recipes()` in the same entry point. The example writes the workflow directly instead of delegating it to a suite.

## What the code demonstrates

`model.py` constructs `y = x @ w` directly in ONNX with an 8 × 4 FP32
weight matrix. `sample_source()` binds a seed and batch count so every pass
regenerates the same sixteen-row input batches without retaining them.

Read `recipes()` in `main.py`: this example uses `QuantizationConfig()` directly,
`PercentileConfig` for clipping, and `smoothquant=True` for two-stage rescaling
followed by MinMax. `run_variant()` uses the fluent quantizer and `evaluate()` explicitly.
Separate calibration/evaluation seeds make the measured errors held-out errors.

The ONNX input contract is `x`: float32 `[16, 8]`. Configuration is in [`schema.py`](schema.py).
All chosen recipes start from the same FP32 graph and use the same calibration
selection. The input factory is called again when fresh statistics are required.

## MinMax

```sh
uv run --locked --group examples -m examples.projection.main --methods minmax
```

Use the complete observed interval for static W8A8. The code configures
`StaticW8A8()` (or `QuantizationConfig()` in the projection).

## Percentile

```sh
uv run --locked --group examples -m examples.projection.main --methods percentile
```

Clip activation outliers before static W8A8. The code configures
`QuantizationConfig(calibration=PercentileConfig(percentile=config.percentile))`.
`--percentile 99.99` and `--histogram-bins 2048` expose the clipping controls.

## SmoothQuant

```sh
uv run --locked --group examples -m examples.projection.main --methods smoothquant
```

The stages are `(smoothing, minmax)`: `SmoothQuant(alpha=config.smoothquant_alpha)`
first balances channels, then a fresh calibration pass applies `StaticW8A8()`.
`--smoothquant-alpha` controls the balancing exponent. Smoothing alone is not the
INT8 result.

## Compare columns

```sh
uv run --locked --group examples -m examples.projection.main
uv run --locked --group examples -m examples.projection.main --methods minmax,percentile
uv run --locked --group examples -m examples.projection.main --help
```

Omitting `--methods` runs all three columns. Reports go to
`work_dir/artifacts/projection/index.html`, `results.json`, and `summary.md`. The directory
also contains FP32/INT8 ONNX graphs and full per-stage `.plan.json` decisions.
`--output` chooses another directory; `--quiet` disables progress and `--json`
prints the report as JSON. Downloads and output setup are automatic.

## Apply it to your task

Replace `example_model()` with `onnx.load()` for your FP32 graph. Replace `sample_source()` with a factory yielding your real input mappings. Keep `recipes()`, the fluent quantizer, and `evaluate()` as the minimal workflow; select operators explicitly if your graph contains unsupported/dynamic weights.

The QuantSmith boundary in the example is deliberately small:

```python
outcome = (
    Quantizer.from_config(recipes(config).select(QuantizationMethod.PERCENTILE))
    .quantize(fp32)
    .calibrate(
        Replay(sample_source(config.calibration_seed, config.calibration_samples))
    )
)
```

Model training, datasets, and HTML reports are demonstration scaffolding; QuantSmith
needs the FP32 ONNX graph, a replayable input factory, and the selected stage rules.
The result contains `model` and inspectable `plans`, or a typed `Err` diagnostic.

[Back to the model × method matrix](../README.md).
