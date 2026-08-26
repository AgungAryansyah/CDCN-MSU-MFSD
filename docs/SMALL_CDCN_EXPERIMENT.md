# Small-CDCN experiment guide

## Purpose

This experiment measures the effect of CDCN channel width on MSU-MFSD
presentation-attack detection. It uses the existing MSU-MFSD preprocessing,
subject-disjoint folds, pseudo-depth supervision, 256×256 inputs, augmentation,
depth loss, and validation-derived threshold protocol unchanged.

No extra dataset, classifier head, attention module, input-resolution change, or
model compression method is introduced.

## Baseline and width scaling

`CDCN-a1.00` is the repository baseline: CDCN with `theta=0.7` and 2,245,016
parameters. Its channel sequence is stem `64`; three `128 → 196 → 128` blocks;
three-level concatenation `384`; then head `128 → 64 → 1`. The network predicts
a 32×32 depth map.

`model.width_multiplier` scales every base channel count with:

```text
max(1, int(round(base_channels * width_multiplier)))
```

| Architecture | Multiplier | Stem | Block | Expanded | Fusion | Head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CDCN-a1.00 | 1.00 | 64 | 128 | 196 | 128 | 64 |
| CDCN-a0.50 | 0.50 | 32 | 64 | 98 | 64 | 32 |
| CDCN-a0.25 | 0.25 | 16 | 32 | 49 | 32 | 16 |

The fusion input is calculated as three times the scaled block output, rather
than hardcoded. At `a1.00`, model construction and state-dict layout remain the
current baseline.

## Central Difference Convolution

Every CDC layer receives the same `model.theta` value. Its active model layers
are bias-free and implement:

```text
CDC(x) = Conv(x) - theta * x_center * sum(kernel_weights)
```

`theta=0.0` takes the ordinary-convolution branch with the same weights, bias,
stride, padding, dilation, and groups. The generic CDC wrapper also supports a
bias argument, but production CDCN layers set `bias=False`; this preserves the
repository baseline rather than silently changing it.

## Subject split and scores

`prepare` persists the official MSU-MFSD manifest and five deterministic,
subject-disjoint development folds in `cache/manifest.jsonl` and
`cache/folds.json`. The 20 official test subjects remain sealed. Each training
run writes `split.json` with train, validation, and test subject and video IDs,
along with its resolved configuration and provenance hashes.

Scores are mean predicted depth: higher means bona fide. A validation threshold
minimizes ACER, with accuracy and threshold as deterministic tie-breakers.
Out-of-fold reporting combines the five validation folds for one architecture
and seed. Test evaluation requires both a refit checkpoint and this frozen
validation threshold.

APCER is the fraction of attacks accepted as bona fide; BPCER is the fraction
of bona fide samples rejected as attacks; ACER is their mean. F1 treats
`attack` as the positive class, using `score < threshold` as an attack
prediction. AUC uses continuous scores with the same attack-positive convention.

## Stage 1: width ablation

The matrix fixes CDCN, pseudo-depth supervision, and `theta=0.7`. It runs
`a1.00`, `a0.50`, and `a0.25` across five folds and seeds `42`, `2026`, and
`2027` (45 cross-validation runs):

```bash
uv run --extra cu130 cdcn-msu run-matrix configs/matrices/small_width.yaml

uv run --extra cpu cdcn-msu benchmark-matrix \
  configs/matrices/small_width.yaml \
  --device cpu --warmup-runs 20 --timed-runs 100

uv run --extra cu130 cdcn-msu collect-results \
  configs/matrices/small_width.yaml \
  results/small_width.csv
```

Run the benchmark again with `--device cuda` on the same matrix when a CUDA
latency result is required. Use a new output filename for each collection;
`collect-results` refuses to replace prior result artifacts.

The detailed CSV/JSON has one OOF row per architecture and seed. Its companion
`*_summary.csv` contains mean and population standard deviation across seeds.
Rows include APCER, BPCER, ACER, F1, AUC, EER, accuracy, threshold, split
metadata, parameter counts, convolution MACs, serialized model-state size,
latency, checkpoint paths, and Git revision.

## Stage 2: selected-width theta ablation

Choose the compact width after inspecting Stage 1. The repository intentionally
does not choose it automatically. Replace `0.25` and `small-theta-a025` below
with the researcher-selected values:

```bash
uv run --extra cu130 cdcn-msu run-matrix \
  configs/matrices/small_theta.yaml \
  --set model.width_multiplier=0.25 \
  --set experiment.name=small-theta-a025

uv run --extra cpu cdcn-msu benchmark-matrix \
  configs/matrices/small_theta.yaml \
  --set model.width_multiplier=0.25 \
  --set experiment.name=small-theta-a025 \
  --device cpu --warmup-runs 20 --timed-runs 100

uv run --extra cu130 cdcn-msu collect-results \
  configs/matrices/small_theta.yaml \
  results/small_theta_a025.csv \
  --set model.width_multiplier=0.25 \
  --set experiment.name=small-theta-a025
```

This matrix holds all architecture and training settings fixed except theta and
runs `0.0`, `0.3`, `0.5`, `0.7`, and `1.0` across five folds and the same three
seeds (75 cross-validation runs).

## Single-run commands and checks

The existing configuration style remains available for individual runs:

```bash
uv run --extra cu130 cdcn-msu train \
  --config-path configs/base.yaml \
  --set model.name=cdcn \
  --set model.width_multiplier=0.50 \
  --set model.theta=0.7 \
  --set experiment.seed=42 \
  --set experiment.name=cdcn-a050-t07

uv run --extra cpu cdcn-msu benchmark \
  --config-path configs/base.yaml \
  --set model.width_multiplier=0.50 \
  --set model.theta=0.7
```

The benchmark times only model inference with batch size one; it excludes model
loading and dataset I/O. It records the exact device, CPU/GPU identity, PyTorch
and CUDA versions, warm-up count, timed count, parameter counts, MACs, and
serialized model-state size in `efficiency_cpu.json` or `efficiency_cuda.json`.

Run the project checks before an experiment campaign:

```bash
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
uv run --extra cpu mypy
uv run --extra cpu pytest
```
