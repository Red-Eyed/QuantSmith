# Windowed causal Transformer × quantization methods

Read [`main.py`](main.py) for the workflow. To run this example, use the
commands below from the repository root.

Tiny Shakespeare with a causal window of eight tokens. This example measures token accuracy, perplexity, and greedy continuations.

Start at [`run()` in main.py](main.py), then follow `export_onnx()` (for Torch
models), `calibration_samples()`, and the quantization configuration in the same
file. The example writes the workflow directly instead of delegating it to a suite.

## What the code demonstrates

`model.py` uses Stackformers' `windowed_encoder_config(window_size=8,
causal=True)`. The model keeps the same character-input/logit-output contract as
the global Transformer, but earlier attention positions outside the causal window
are masked. Fixed-context export makes the supported projection operations visible.

The local `ConstantWeights` selector leaves input-dependent attention products
and masks floating point. Compare this example with the global Transformer to see
how the same quantization methods apply to a different attention layout.

The ONNX input contract is `tokens`: int64 `[1, 32]`. Configuration is in [`schema.py`](schema.py).
All chosen recipes start from the same FP32 graph and use the same calibration
selection. The input factory is called again when fresh statistics are required.

## MinMax

```sh
uv run --locked --group examples -m examples.windowed_transformer.main --methods minmax
```

Use the complete observed interval for static W8A8. The code configures
`StaticW8A8()` (or `QuantizationConfig()` in the projection).

## Percentile

```sh
uv run --locked --group examples -m examples.windowed_transformer.main --methods percentile
```

Clip activation outliers before static W8A8. The code configures
`StaticW8A8(calibration=Percentile(percentile=config.percentile))`.
`--percentile 99.99` and `--histogram-bins 2048` expose the clipping controls.

## SmoothQuant

```sh
uv run --locked --group examples -m examples.windowed_transformer.main --methods smoothquant
```

The stages are `(smoothing, minmax)`: `SmoothQuant(alpha=config.smoothquant_alpha)`
first balances channels, then a fresh calibration pass applies `StaticW8A8()`.
`--smoothquant-alpha` controls the balancing exponent. Smoothing alone is not the
INT8 result.

## Compare columns

```sh
uv run --locked --group examples -m examples.windowed_transformer.main
uv run --locked --group examples -m examples.windowed_transformer.main --methods minmax,percentile
uv run --locked --group examples -m examples.windowed_transformer.main --help
```

Omitting `--methods` runs all three columns. Reports go to
`work_dir/artifacts/windowed_transformer/index.html`, `results.json`, and `summary.md`. The directory
also contains FP32/INT8 ONNX graphs and full per-stage `.plan.json` decisions.
`--output` chooses another directory; `--quiet` disables progress and `--json`
prints the report as JSON. Downloads and output setup are automatic.

## Apply it to your task

Replace the local windowed model with your attention architecture. Keep your real attention mask and input contract at export, and check parity before quantizing. Replace the character loader/metrics for your task; keep dynamic QK/AV products outside constant-weight selection.

The QuantSmith boundary in the example is deliberately small:

```python
outcome = (
    Quantizer.from_pipeline(
        recipes(config).select(QuantizationMethod.PERCENTILE),
        histogram_bins=config.histogram_bins,
    )
    .quantize(fp32)
    .calibrate(Replay(calibration_samples(data)))
)
```

Model training, datasets, and HTML reports are demonstration scaffolding; QuantSmith
needs the FP32 ONNX graph, a replayable input factory, and the selected stage rules.
The result contains `model` and inspectable `plans`, or a typed `Err` diagnostic.

[Back to the model × method matrix](../README.md).

Training uses disjoint 80/10/10 text spans and a training-only vocabulary. The
default is 600 CPU steps, 64 calibration contexts, and 128 held-out contexts.
These are small trained character models, not pretrained language-model benchmarks.
