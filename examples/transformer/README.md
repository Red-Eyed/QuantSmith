# Global causal Transformer × quantization methods

Read [`main.py`](main.py) for the workflow. To run this example, use the
commands below from the repository root.

Tiny Shakespeare next-character prediction. This example measures token accuracy, cross entropy, perplexity, and greedy continuations.

Start at [`run()` in main.py](main.py), then follow `export_onnx()` (for Torch
models), `calibration_samples()`, and the quantization configuration in the same
file. The example writes the workflow directly instead of delegating it to a suite.

## What the code demonstrates

`model.py` embeds character IDs, applies two Stackformers global causal
layers (dimension 64, four heads, RoPE, RMSNorm, SwiGLU), and produces vocabulary
logits. `run()` loads disjoint text spans, trains this model, and exports it with
the input name `tokens`.

In `ConstantWeights.matches()`, a MatMul is selected only when its second input
is a constant initializer. This quantizes learned projections and feedforward
weights. QK and AV attention products depend on the current tokens and stay float;
embeddings, normalization, masks, and softmax also stay float.

The ONNX input contract is `tokens`: int64 `[1, 32]`. Configuration is in [`schema.py`](schema.py).
All chosen recipes start from the same FP32 graph and use the same calibration
selection. The input factory is called again when fresh statistics are required.

## MinMax

```sh
uv run --locked --group examples -m examples.transformer.main --methods minmax
```

Use the complete observed interval for static W8A8. The code configures
`StaticW8A8()` (or `QuantizationConfig()` in the projection).

## Percentile

```sh
uv run --locked --group examples -m examples.transformer.main --methods percentile
```

Clip activation outliers before static W8A8. The code configures
`StaticW8A8(calibration=Percentile(percentile=config.percentile))`.
`--percentile 99.99` and `--histogram-bins 2048` expose the clipping controls.

## SmoothQuant

```sh
uv run --locked --group examples -m examples.transformer.main --methods smoothquant
```

The stages are `(smoothing, minmax)`: `SmoothQuant(alpha=config.smoothquant_alpha)`
first balances channels, then a fresh calibration pass applies `StaticW8A8()`.
`--smoothquant-alpha` controls the balancing exponent. Smoothing alone is not the
INT8 result.

## Compare columns

```sh
uv run --locked --group examples -m examples.transformer.main
uv run --locked --group examples -m examples.transformer.main --methods minmax,percentile
uv run --locked --group examples -m examples.transformer.main --help
```

Omitting `--methods` runs all three columns. Reports go to
`work_dir/artifacts/transformer/index.html`, `results.json`, and `summary.md`. The directory
also contains FP32/INT8 ONNX graphs and full per-stage `.plan.json` decisions.
`--output` chooses another directory; `--quiet` disables progress and `--json`
prints the report as JSON. Downloads and output setup are automatic.

## Apply it to your task

Replace `CharacterTransformer` with your language model and skip `train()` if it is pretrained. Replace the character loader with your tokenizer and replayable token batches. Preserve int64 IDs, use your exported input names, and replace the character-specific metrics/decoding with your task's evaluation.

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

Training uses disjoint 80/10/10 text spans and a training-only vocabulary. The
default is 600 CPU steps, 64 calibration contexts, and 128 held-out contexts.
These are small trained character models, not pretrained language-model benchmarks.
