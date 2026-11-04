#!/bin/bash
#SBATCH --job-name=llm-student
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --exclude=ailab-l4-04
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/student_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/student_%j.err
#
# Serves the text-to-SQL student for harness/planner.py, next to llm-serve (Qwen3-8B), on
# its own GPU. Writes host:port to serving/student_endpoint.txt; the model is served as
# "sql-student" whatever its path. Same STOP file and libcuda workaround as serve_llm.slurm.
#
# Usage (from anywhere; paths come from P7_PROJECT_DIR, not the submit directory):
#     sbatch slurm/serve_student.slurm                       # models/qwen2.5-coder-1.5b-sql-v6
#     sbatch slurm/serve_student.slurm /path/to/other/model

set -uo pipefail

P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/llm_venv"
MODEL_DIR="${1:-$P7_PROJECT_DIR/models/qwen2.5-coder-1.5b-sql-v6}"
SERVING_DIR="$P7_PROJECT_DIR/serving"
STOP_FILE="$SERVING_DIR/STOP"
ENDPOINT_FILE="$SERVING_DIR/student_endpoint.txt"
PORT=8001

mkdir -p "$SERVING_DIR" "$P7_PROJECT_DIR/logs"

echo "=== SQL Student Serve Job ==="
echo "Node:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Model:   $MODEL_DIR"
echo

if [ -f "$STOP_FILE" ]; then
    echo "STOP file present ($STOP_FILE) - not starting. Delete it to resume."
    exit 0
fi

echo "$(hostname):$PORT" > "$ENDPOINT_FILE"

REAL_LIBCUDA=$(find /usr/lib/x86_64-linux-gnu /usr/lib -name 'libcuda.so.1' 2>/dev/null | head -1)
RESOLVED_LIBCUDA=$(readlink -f "$REAL_LIBCUDA" 2>/dev/null)
echo "Host libcuda.so.1 resolved to: ${RESOLVED_LIBCUDA:-<not found>}"

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${P7_PROJECT_DIR}:${P7_PROJECT_DIR}" \
    -B "${RESOLVED_LIBCUDA}:/usr/local/cuda/compat/lib/libcuda.so.1" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        vllm serve '${MODEL_DIR}' --served-model-name sql-student --port ${PORT} --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager
    "

echo "Server exited (walltime limit or crash). It will be restarted at the next 05:00/17:00 cron cycle."
