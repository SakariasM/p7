#!/bin/bash
#SBATCH --job-name=vessel-days
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%6
#SBATCH --exclude=ailab-l4-03,ailab-l4-04
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/vessel_days_%A_%a.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/vessel_days_%A_%a.err
#
# Builds data/derived/vessel_days (harness/build_vessel_days.py): one row per vessel per day,
# from the cleaned positions. 12 array tasks, each takes every 12th day (~6 s a day); built
# days are skipped, so a failed run is just submitted again. At most 6 tasks at once (%6):
# the QOS allows 8 running jobs per user and llm-serve/llm-student need two.
#
# Usage, from the repo root:  sbatch slurm/build_vessel_days.slurm

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
PYTHON="${PYTHON:-$P7_PROJECT_DIR/envs/harness/bin/python}"
SPILL="$P7_PROJECT_DIR/data/tmp/duckdb_spill_vd_${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}"

echo "=== vessel_days, task ${SLURM_ARRAY_TASK_ID:-0} of 12 on $(hostname) ==="
cd "$REPO_DIR/harness"
"$PYTHON" -u build_vessel_days.py --task "${SLURM_ARRAY_TASK_ID:-0}" --tasks 12 \
    --threads "${SLURM_CPUS_PER_TASK:-8}" --memory 40GB --spill "$SPILL"
