#!/bin/bash
# Run the held-out baseline evaluation through ask.sh --single.
#
# Usage:
#   scripts/baseline_eval.sh --out results/baseline
#   scripts/baseline_eval.sh --runs 5 --out results/baseline
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
P7_PROJECT_DIR="${P7_PROJECT_DIR:-$REPO_DIR}"
export P7_PROJECT_DIR

cd "$REPO_DIR"
exec "$P7_PROJECT_DIR/envs/harness/bin/python" -u "$REPO_DIR/harness/eval_baseline.py" "$@"
