# RTX 3060 remote runbook

## Preflight

1. Install uv and FFmpeg/FFprobe.
2. Clone or transfer this repository beside the `data/` directory.
3. Run `uv sync --extra cu130`.
4. Run `uv run --extra cu130 cdcn-msu doctor --require-cuda`.
5. Confirm the report names the RTX 3060, compute capability 8.6, approximately
   12 GiB VRAM, and a CUDA 13-enabled PyTorch build.

If the doctor command fails because the CUDA wheel cannot initialize, update the
NVIDIA driver. Do not silently replace the locked CUDA/PyTorch stack.

## Tracking and storage

Create an ignored `.env` from `.env.example`, insert the private W&B key, and use
`WANDB_MODE=online`. Use `offline` when the server connection is unreliable.
Only metrics and telemetry are eligible for W&B upload.

Ensure sufficient local storage for the 11GB raw dataset, extracted candidate
frames, PRNet exports/maps, checkpoints, predictions, and logs. All generated
paths are ignored by Git.

## Execution order

1. `cdcn-msu inspect-data`
2. `cdcn-msu prepare`
3. Export/run/import PRNet if pseudo-depth experiments are required.
4. Dry-run each matrix and review its resolved run count.
5. Run the main matrix and ablations sequentially on the single GPU.
6. Aggregate five validation folds separately for each model/supervision/seed.
7. Freeze thresholds and median best epoch counts.
8. Run refit/test once per final condition.

The default batch size is 8 with effective batch size 16. W&B and the local
epoch log record peak reserved GPU memory. The acceptance ceiling is 11 GiB;
reduce only `training.batch_size` and increase `training.gradient_accumulation`
proportionally if the remote process exceeds it.

## Recovery

Each CV run writes `last.pt` after every epoch and `best.pt` on improvement.
Re-running a matrix skips completed runs and resumes incomplete runs. Local
`metrics.jsonl`, `summary.json`, resolved `config.yaml`, and `provenance.json`
remain authoritative if W&B is unavailable.
