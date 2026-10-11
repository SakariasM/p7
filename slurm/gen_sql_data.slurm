#!/bin/bash
#SBATCH --job-name=gen-sql-data
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/gen_sql_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/gen_sql_%j.err
#
# Text-to-SQL training data (harness/gen_sql_data.py). CPU only: the teacher is the
# llm-serve job, reached through serving/current_endpoint.txt, and DuckDB runs here.
#
# *** BEFORE FIRST SUBMIT *** build the harness venv (host Python, like envs/ais):
#     python3 -m venv /ceph/project/P7-Agent-Harness/envs/harness
#     /ceph/project/P7-Agent-Harness/envs/harness/bin/pip install -r requirements-harness.txt
#
# Usage, from the repo root (arguments go to gen_sql_data.py):
#     sbatch slurm/gen_sql_data.slurm templates --per-template 60
#     sbatch slurm/gen_sql_data.slurm freeform --prompts 300
#     sbatch slurm/gen_sql_data.slurm export

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
PYTHON="${PYTHON:-$P7_PROJECT_DIR/envs/harness/bin/python}"

echo "=== Text-to-SQL data job ==="
echo "Node:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Repo:     $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Teacher:  $(cat "$P7_PROJECT_DIR/serving/current_endpoint.txt")"
echo "Args:     $*"
echo

cd "$REPO_DIR/harness"
"$PYTHON" -u gen_sql_data.py "$@"
