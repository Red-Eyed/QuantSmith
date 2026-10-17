# Wine MLP × quantization methods

Read [`main.py`](main.py) for the workflow. To run this example, use the
commands below from the repository root.

Three-class classification of real chemical measurements. This example measures cultivar accuracy and actual held-out class predictions.

Start at [`run()` in main.py](main.py), then follow `export_onnx()` (for Torch
models), `calibration_samples()`, and the quantization configuration in the same
file. The example writes the workflow directly instead of delegating it to a suite.

## What the code demonstrates

`model.py` is a concrete 13 → 32 → 16 → 3 Torch MLP with ReLU between
dense layers. `run()` loads sklearn's real Wine dataset, trains on its training
rows, and exports the input name `features`. Mean and standard deviation are
fitted on training rows only and reused for calibration/evaluation.

`calibration_samples()` yields normalized float32 rows. The labels are needed for
evaluation, not calibration. All three dense projections have constant weights
and are eligible for every method. The report shows cultivar names and predicted
classes, so logit error can be compared with classification correctness.

The ONNX input contract is `features`: float32 `[1, 13]`. Configuration is in [`schema.py`](schema.py).
All chosen recipes start from the same FP32 graph and use the same calibration
selection. The input factory is called again when fresh statistics are required.

## MinMax

```sh
uv run --locked --group examples -m examples.wine_mlp.main --methods minmax
```

Use the complete observed interval for static W8A8. The code configures
`StaticW8A8()` (or `QuantizationConfig()` in the projection).

## Percentile

```sh
uv run --locked --group examples -m examples.wine_mlp.main --methods percentile
```

Clip activation outliers before static W8A8. The code configures
`StaticW8A8(calibration=Percentile(percentile=config.percentile))`.
`--percentile 99.99` and `--histogram-bins 2048` expose the clipping controls.

## SmoothQuant

```sh
uv run --locked --group examples -m examples.wine_mlp.main --methods smoothquant
```

The stages are `(smoothing, minmax)`: `SmoothQuant(alpha=config.smoothquant_alpha)`
first balances channels, then a fresh calibration pass applies `StaticW8A8()`.
`--smoothquant-alpha` controls the balancing exponent. Smoothing alone is not the
INT8 result.

## Compare columns

```sh
uv run --locked --group examples -m examples.wine_mlp.main
uv run --locked --group examples -m examples.wine_mlp.main --methods minmax,percentile
uv run --locked --group examples -m examples.wine_mlp.main --help
```

Omitting `--methods` runs all three columns. Reports go to
`work_dir/artifacts/wine_mlp/index.html`, `results.json`, and `summary.md`. The directory
also contains FP32/INT8 ONNX graphs and full per-stage `.plan.json` decisions.
`--output` chooses another directory; `--quiet` disables progress and `--json`
prints the report as JSON. Downloads and output setup are automatic.

## Apply it to your task

Replace `WineMLP` and the Wine loader with your tabular model and features. Fit preprocessing on training data only. Keep feature arrays float32 and the input name consistent with export. Replace cultivar labels/accuracy with your own target definition and metric.

The QuantSmith boundary in the example is deliberately small:

```python
outcome = quantize(
    fp32,
    calibration_samples(data),
    recipes(config)[QuantizationMethod.PERCENTILE],
    histogram_bins=config.histogram_bins,
)
```

Model training, datasets, and HTML reports are demonstration scaffolding; QuantSmith
needs the FP32 ONNX graph, a replayable input factory, and the selected stage rules.
The result contains `model` and inspectable `plans`, or a typed `Err` diagnostic.

[Back to the model × method matrix](../README.md).

The finite dataset uses 105 training, 36 calibration, and 37 evaluation rows.
The full calibration/evaluation splits are used. Normalization fits only training
rows, and the default is 300 training steps.
