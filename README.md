# CDCN-MSU-MFSD

A reproducible PyTorch 2.11 implementation of CDCN and CDCN++ for face
anti-spoofing experiments on MSU-MFSD. It includes subject-disjoint
cross-validation, binary and PRNet pseudo-depth supervision, YAML ablations,
video-level PAD metrics, resumable training, and metrics-only W&B tracking.

The authors' [`CDCN`](https://github.com/ZitongYu/CDCN) repository is an
architecture reference only. This project has no runtime dependency on it and
does not modify it.

## Environment

Python and all dependencies are locked by uv. Use exactly one PyTorch extra:

```bash
uv sync --extra cpu
uv sync --extra cu130
```

The second command installs PyTorch 2.11 and torchvision 0.26 from the official
CUDA 13.0 wheel index. On the RTX 3060 server, verify the driver and wheel before
preprocessing or training:

```bash
uv run --extra cu130 cdcn-msu doctor --require-cuda
```

The expected compute capability is 8.6 and expected VRAM is approximately
12 GiB. See [the remote runbook](docs/REMOTE_RUNBOOK.md) for the complete setup.

## W&B configuration

Copy `.env.example` to `.env`, set the private API key, and enable online mode:

```dotenv
WANDB_API_KEY=your-private-key
WANDB_ENTITY=your-account-or-team
WANDB_PROJECT=cdcn-msu-mfsd
WANDB_MODE=online
```

`.env` is ignored by Git. W&B receives resolved configuration, scalar metrics,
and system telemetry only. Videos, frames, depth maps, predictions, and model
checkpoints remain local. Set `WANDB_MODE=offline` for an unreliable connection
or `disabled` for local-only operation.

## Data preparation

The default layout expects the repository and dataset to be siblings:

```text
Skripsi/
├── CDCN-MSU-MFSD/
└── data/raw/
    ├── scene01/
    ├── train_sub_list.txt
    └── test_sub_list.txt
```

Inspect the raw data and create the manifest, folds, and 32 cached candidate
frames per video:

```bash
uv run --extra cu130 cdcn-msu inspect-data
uv run --extra cu130 cdcn-msu prepare
```

Generated data is written under ignored `cache/`. The decoder honors FFmpeg
display rotation before applying the dataset's display-space `.face` boxes.

Binary targets require no additional preprocessing. For pseudo-depth targets,
use the external PRNet contract described in [the PRNet guide](docs/PRNET_CONTRACT.md):

```bash
uv run --extra cu130 cdcn-msu export-prnet cache/prnet-export
uv run --extra cu130 cdcn-msu import-prnet \
  cache/prnet-export/contract.jsonl /path/to/prnet/outputs
```

## Training and experiments

Run one configuration with YAML overrides:

```bash
uv run --extra cu130 cdcn-msu train \
  --config-path configs/base.yaml \
  --set model.name=cdcnpp \
  --set supervision.kind=binary \
  --set experiment.fold=0 \
  --set experiment.seed=2026
```

Inspect or execute the complete 60-run comparison:

```bash
uv run --extra cu130 cdcn-msu run-matrix configs/matrices/main.yaml --dry-run
uv run --extra cu130 cdcn-msu run-matrix configs/matrices/main.yaml
```

The ablation matrices are:

- `configs/ablations/theta.yaml`: five central-difference values over five folds.
- `configs/ablations/contrastive.yaml`: three contrastive-loss weights over five folds.

Completed runs are skipped and incomplete runs resume from `last.pt`. Outputs
are organized by experiment, model, supervision, fold, and seed under ignored
`outputs/`.

## Evaluation

Every frame score is the mean predicted depth. Eight frame scores are averaged
into one video score. Validation predictions determine the threshold; the test
command refuses to run without a frozen threshold.

Aggregate one seed's five validation folds:

```bash
uv run --extra cu130 cdcn-msu aggregate-oof \
  outputs/main/cdcn-binary/fold-0/seed-2026/validation_videos.jsonl \
  outputs/main/cdcn-binary/fold-1/seed-2026/validation_videos.jsonl \
  outputs/main/cdcn-binary/fold-2/seed-2026/validation_videos.jsonl \
  outputs/main/cdcn-binary/fold-3/seed-2026/validation_videos.jsonl \
  outputs/main/cdcn-binary/fold-4/seed-2026/validation_videos.jsonl \
  outputs/main/cdcn-binary/oof/seed-2026
```

Refit on all 15 official training subjects for the median best fold epoch and
evaluate the sealed 20-subject test set once:

```bash
uv run --extra cu130 cdcn-msu refit-test \
  outputs/main/cdcn-binary/oof/seed-2026/threshold.json \
  --summary outputs/main/cdcn-binary/fold-0/seed-2026/summary.json \
  --summary outputs/main/cdcn-binary/fold-1/seed-2026/summary.json \
  --summary outputs/main/cdcn-binary/fold-2/seed-2026/summary.json \
  --summary outputs/main/cdcn-binary/fold-3/seed-2026/summary.json \
  --summary outputs/main/cdcn-binary/fold-4/seed-2026/summary.json \
  --set experiment.name=main \
  --set model.name=cdcn \
  --set supervision.kind=binary \
  --set experiment.seed=2026
```

Reports include ACER, APCER, BPCER, ROC-AUC, EER, accuracy, confusion counts,
and attack-type/device breakdowns. The full methodology is fixed in
[the experiment protocol](docs/EXPERIMENT_PROTOCOL.md).

## Quality checks

```bash
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
uv run --extra cpu mypy
uv run --extra cpu pytest
```

