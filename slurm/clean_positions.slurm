#!/bin/bash
#SBATCH --job-name=clean-positions
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%6
#SBATCH --exclude=ailab-l4-03,ailab-l4-04
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/clean_positions_%A_%a.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/clean_positions_%A_%a.err
#
# Builds data/processed/ais_clean (harness/clean_positions.py), the cleaned data that the
# vessel_positions view reads. 12 array tasks, each takes every 12th day; finished days are
# skipped, so a failed run is just submitted again. CPU only. At most 6 tasks run at once
# (%6): the QOS allows 8 running jobs per user and llm-serve/llm-student need two of them.
#
# Usage, from the repo root:  sbatch slurm/clean_positions.slurm

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
PYTHON="${PYTHON:-$P7_PROJECT_DIR/envs/harness/bin/python}"
SPILL="/tmp/duckdb_spill_${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}"

echo "=== Clean positions, task ${SLURM_ARRAY_TASK_ID:-0} of 12 on $(hostname) ==="
cd "$REPO_DIR/harness"
"$PYTHON" -u clean_positions.py --task "${SLURM_ARRAY_TASK_ID:-0}" --tasks 12 \
    --threads "${SLURM_CPUS_PER_TASK:-8}" --memory 24GB --spill "$SPILL"
