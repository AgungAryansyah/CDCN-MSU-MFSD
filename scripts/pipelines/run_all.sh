#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
campaign="configs/pipelines/thesis_completion.yaml"
session="thesis-completion-2026"
resume=0
detach=0

for argument in "$@"; do
  case "$argument" in
    --resume) resume=1 ;;
    --detach) detach=1 ;;
    --status)
      cd "$repo_root"
      uv run --extra cu130 cdcn-msu pipeline-status --campaign "$campaign"
      exit 0
      ;;
    *) printf 'Unknown option: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

if ((detach)); then
  if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Campaign is already running in tmux session %s\n' "$session"
    exit 0
  fi
  resume_arg=""
  if ((resume)); then resume_arg=" --resume"; fi
  tmux new-session -d -s "$session" -c "$repo_root" \
    "bash $repo_root/scripts/pipelines/run_all.sh$resume_arg"
  printf 'Started tmux session %s\n' "$session"
  printf 'Log: %s/outputs/pipelines/thesis-completion-2026/run.log\n' "$repo_root"
  exit 0
fi

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
fit_gpus="${FIT_GPUS:-0,1}"
mkdir -p outputs/pipelines/thesis-completion-2026
exec 9>outputs/pipelines/thesis-completion-2026/campaign.lock
if ! flock -n 9; then
  printf 'Another process holds the campaign lock.\n' >&2
  exit 1
fi
export CDCN_PIPELINE_LOCK_HELD=1
exec > >(tee -a outputs/pipelines/thesis-completion-2026/run.log) 2>&1

run_stage() {
  local stage="$1"
  printf '\n[%s] Starting stage: %s\n' "$(date -Is)" "$stage"
  if [[ "$stage" == "theta" || "$stage" == "refit" ]]; then
    uv run --extra cu130 python scripts/pipelines/run_parallel_fits.py \
      --stage "$stage" --gpus "$fit_gpus" --campaign "$campaign"
  else
    uv run --extra cu130 cdcn-msu pipeline "$stage" --campaign "$campaign"
  fi
  printf '[%s] Completed stage: %s\n' "$(date -Is)" "$stage"
}

stages=(
  preflight
  wandb-smoke
  verify
  width
  width-oof
  model-stats
  width-efficiency
  selection
  theta
  theta-oof
  theta-efficiency
  freeze
  refit
  test
  report
)
state_path="outputs/pipelines/thesis-completion-2026/state.json"

stage_is_completed() {
  python3 - "$state_path" "$1" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1]))
sys.exit(0 if state.get("stages", {}).get(sys.argv[2], {}).get("status") == "completed" else 1)
PY
}

for stage in "${stages[@]}"; do
  if ((resume)) && stage_is_completed "$stage"; then
    printf '\n[%s] Skipping completed stage: %s\n' "$(date -Is)" "$stage"
    continue
  fi
  run_stage "$stage"
done

printf '\n[%s] Campaign completed.\n' "$(date -Is)"
