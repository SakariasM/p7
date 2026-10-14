#!/bin/bash
#SBATCH --job-name=ais-derive
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/ais_derive_%j.out
#
# Rebuild trips, stops and trip geometry from all of $AIS_DATA_DIR/clean/, in MMSI-range
# chunks (see src/ais/derive.py), then ais.duckdb. CPU only.
#
# Usage, from the repo root:
#     sbatch slurm/derive.slurm
#     sbatch --dependency=afterok:<clean job id> slurm/derive.slurm

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export AIS_DATA_DIR="${AIS_DATA_DIR:-/ceph/project/P7-Agent-Harness/data}"
UV="${UV:-$HOME/.local/bin/uv}"

echo "=== AIS derive job ==="
echo "Node:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-none}"
echo "Repo:     $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Data:     $AIS_DATA_DIR"
echo

cd "$REPO_DIR"
"$UV" run --frozen ais derive
