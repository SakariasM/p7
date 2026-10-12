#!/bin/bash
#SBATCH --job-name=ais-clean
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/ais_clean_%A_%a.out
#
# Clean the processed Parquet year (Oct 2025 - Sep 2026) into $AIS_DATA_DIR/clean/, one
# month per array task, at most 4 at a time. Days already clean are skipped, so a failed or
# timed-out task can simply be resubmitted. CPU only.
#
# Usage, from the repo root:
#     sbatch slurm/clean_processed.slurm
#     sbatch --dependency=afterok:<job id> slurm/derive.slurm

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export AIS_DATA_DIR="${AIS_DATA_DIR:-/ceph/project/P7-Agent-Harness/data}"
FIRST_MONTH="${FIRST_MONTH:-2025-10-01}"
UV="${UV:-$HOME/.local/bin/uv}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
START=$(date -d "$FIRST_MONTH + $TASK month" +%F)
END=$(date -d "$START + 1 month - 1 day" +%F)

echo "=== AIS clean job ==="
echo "Node:     $(hostname)"
echo "Job ID:   ${SLURM_ARRAY_JOB_ID:-none}_${TASK}"
echo "Repo:     $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Data:     $AIS_DATA_DIR"
echo "Days:     $START .. $END"
echo

cd "$REPO_DIR"
# ais.duckdb is rebuilt once by the derive job, not by 12 tasks at the same time.
"$UV" run --frozen ais clean-processed "$START" "$END" --no-build
