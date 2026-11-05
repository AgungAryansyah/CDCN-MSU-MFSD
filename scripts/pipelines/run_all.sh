#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
campaign="configs/pipelines/thesis_completion.yaml"
session="thesis-completion-2026"

if [[ "${1:-}" == "--detach" ]]; then
  if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Campaign is already running in tmux session %s\n' "$session"
    exit 0
  fi
  tmux new-session -d -s "$session" -c "$repo_root" \
    "bash $repo_root/scripts/pipelines/run_all.sh"
  printf 'Started tmux session %s\n' "$session"
  printf 'Log: %s/outputs/pipelines/thesis-completion-2026/run.log\n' "$repo_root"
  exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
  cd "$repo_root"
  uv run --extra cu130 cdcn-msu pipeline-status --campaign "$campaign"
  exit 0
fi

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
  uv run --extra cu130 cdcn-msu pipeline "$stage" --campaign "$campaign"
  printf '[%s] Completed stage: %s\n' "$(date -Is)" "$stage"
}

run_stage preflight
run_stage wandb-smoke
run_stage verify
run_stage width
run_stage width-oof
run_stage model-stats
run_stage width-efficiency
run_stage selection
run_stage theta
run_stage theta-oof
run_stage theta-efficiency
run_stage freeze
run_stage refit
run_stage test
run_stage report

printf '\n[%s] Campaign completed.\n' "$(date -Is)"
