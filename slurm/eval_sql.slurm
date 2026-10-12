#!/bin/bash
#SBATCH --job-name=eval-sql
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/eval_sql_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/eval_sql_%j.err
#
# Runs harness/eval_sql.py on a compute node (CPU only), so query timings for the points
# score are measured on dedicated cores. The models are reached through their endpoints.
#
# Usage, from the repo root (arguments go to eval_sql.py):
#     sbatch slurm/eval_sql.slurm run --out /ceph/project/P7-Agent-Harness/logs/eval/x.jsonl
#     sbatch slurm/eval_sql.slurm run --model sql-student \
#         --base-url http://$(cat /ceph/project/P7-Agent-Harness/serving/student_endpoint.txt)/v1
#     sbatch slurm/eval_sql.slurm agent --mode planner --out ...

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
PYTHON="${PYTHON:-$P7_PROJECT_DIR/envs/harness/bin/python}"

echo "=== eval_sql job ==="
echo "Node:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Repo:     $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Args:     $*"
echo

mkdir -p "$P7_PROJECT_DIR/logs/eval"
cd "$REPO_DIR/harness"
"$PYTHON" -u eval_sql.py "$@"
