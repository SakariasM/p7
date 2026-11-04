#!/bin/bash
#SBATCH --job-name=eval-agent
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/eval_agent_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/eval_agent_%j.err
#
# End-to-end eval of a whole agent (harness/eval_sql.py agent) against the running model
# jobs: llm-serve for both modes, plus llm-student for planner mode. CPU only.
# Results go to logs/eval/agent_<mode>_<job id>.jsonl.
#
# Usage, from the repo root:
#     sbatch slurm/eval_agent.slurm planner
#     sbatch slurm/eval_agent.slurm single

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

MODE="${1:?usage: sbatch slurm/eval_agent.slurm planner|single}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
PYTHON="${PYTHON:-$P7_PROJECT_DIR/envs/harness/bin/python}"
OUT="$P7_PROJECT_DIR/logs/eval/agent_${MODE}_${SLURM_JOB_ID:-local}.jsonl"

echo "=== Agent eval job ==="
echo "Node:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Repo:     $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Mode:     $MODE"
echo "Planner:  $(cat "$P7_PROJECT_DIR/serving/current_endpoint.txt")"
echo "Student:  $(cat "$P7_PROJECT_DIR/serving/student_endpoint.txt" 2>/dev/null || echo none)"
echo

mkdir -p "$P7_PROJECT_DIR/logs/eval"
cd "$REPO_DIR/harness"
"$PYTHON" -u eval_sql.py agent --mode "$MODE" --out "$OUT"
