# Reconstruction experiments

## 2026-10-17 — Keep static selection out of reconstruction

The new executed-output smoothing mode scores static integer graphs. Reconstruction
changes learned weight codes and currently makes no integer-audit claim. Its
entry point therefore rejects this configuration explicitly instead of silently
using an objective for a different computation. Six layout tests cover the
restriction; existing independent and joint smoothing paths remain available.

## 2026-09-28 — W8A8 projection comparison

Motivation: determine whether newer reconstruction methods improve on existing
static calibration, rather than assume their low-bit paper results transfer.

Decision: compare independently initialized MinMax, Percentile (99.99), GPTQv2,
and operator-level QDrop on the existing projection model. Use eight 16-row
calibration batches (seed 11), eight held-out batches (seed 19), and 2,000 QDrop
updates. Save exact integer plans and execute the exported ONNX graphs.

Run configuration: default reconstruction comparison with `--json --quiet`
(Qraft 0.7.0). Current run instructions are in [README.md](README.md).

| Method | Held-out MSE | Maximum absolute error |
| --- | ---: | ---: |
| MinMax | 0.0005181066453 | 0.1007266045 |
| Percentile | 0.0005181066453 | 0.1007266045 |
| GPTQv2 | 0.0005181066453 | 0.1007266045 |
| QDrop | 0.0005486007499 | 0.0981683731 |

GPTQv2 tied the static baseline; QDrop's MSE regressed about 5.9%, despite a
slightly smaller maximum error. No accuracy-based default change is justified.
Calibration took approximately 0.002 seconds for GPTQv2 and 0.93 seconds for
QDrop on this run; these are tiny-model observations, not throughput benchmarks.

Verified separately: controlled reconstruction cases improve hard-output error;
GPTQv2 matches independent constrained solves; ONNX exports preserve integer
decisions and shared-weight consumers. Real CNN/Transformer task metrics,
low-bit deployment, GPU optimizer behavior, and a PD-Quant comparison remain
unverified. No pretrained model weights were changed by this experiment.
