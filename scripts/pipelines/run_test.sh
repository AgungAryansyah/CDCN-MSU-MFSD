#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
uv run --extra cu130 cdcn-msu pipeline test --campaign configs/pipelines/thesis_completion.yaml
