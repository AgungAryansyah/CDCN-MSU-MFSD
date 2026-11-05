# Thesis experiment pipeline

This pipeline runs the remaining experiments in Chapter 3. It excludes the
manualisasi worksheet. It reuses the 45 saved width runs and starts 60 missing
cross-validation fits. It then creates 21 refit models and evaluates each model
once on the official test subjects.

## Run the full campaign

Run this command from the repository root on `labkc-comm3`:

```bash
bash scripts/pipelines/run_all.sh --detach
```

The script starts a `tmux` session named `thesis-completion-2026`. It uses GPU
0 for training. It measures CPU and GPU inference latency in separate processes.
It requires online W&B settings in `.env`.

Check the stage status with:

```bash
bash scripts/pipelines/run_all.sh --status
```

Read the log at
`outputs/pipelines/thesis-completion-2026/run.log`. The status file is at
`outputs/pipelines/thesis-completion-2026/status.json`. The state file records
each completed or failed stage.

Run the same command after an interruption. The pipeline skips complete CV runs.
It resumes incomplete CV and refit checkpoints. Test evaluation reuses cached
frame predictions. A frozen final plan prevents changes to thresholds or refit
epochs after test evaluation starts.

## Separate stages

Each stage has a separate command and a separate implementation module under
`src/cdcn_msu_mfsd/pipelines/`. Each stage also has a launcher under
`scripts/pipelines/`.

The master script runs these stages in order:

1. Check the data, folds, pseudo-depth maps, CUDA device, and W&B settings.
2. Run a one-epoch online W&B smoke fit.
3. Check model tensor shapes and the central-difference operator.
4. Reuse or complete the width runs, then aggregate their out-of-fold scores.
5. Measure model size and benchmark the three width conditions.
6. Select the compact width by the Chapter 3 rule.
7. Train the missing theta conditions and aggregate their out-of-fold scores.
8. Benchmark the remaining theta conditions and freeze thresholds and epochs.
9. Refit each of the seven conditions with all 15 training subjects.
10. Evaluate each refit checkpoint once on all 20 official test subjects.
11. Write per-seed and mean ± population standard-deviation reports.

Run one stage from the repository root with:

```bash
uv run --extra cu130 cdcn-msu pipeline width-oof \
  --campaign configs/pipelines/thesis_completion.yaml
```

The campaign outputs are under
`outputs/pipelines/thesis-completion-2026/`. Cross-validation outputs remain
under their experiment names. Refit outputs use condition-specific directories.
W&B stores training metrics and summaries. Checkpoints, images, and individual
predictions stay on the server.

The width phase uses multipliers 1.0, 0.5, and 0.25 at θ = 0.7. The theta
phase uses θ = 0, 0.3, 0.5, 0.7, and 1.0 at width 0.5. Each condition uses
seeds 42, 2026, and 2027. Each cross-validation condition uses five folds.
