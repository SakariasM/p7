#!/bin/bash
#SBATCH --job-name=llm-serve
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=ailab-l4-04
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/serve_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/serve_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/ceph/project/P7-Agent-Harness}}"
CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/llm_venv"
MODEL_DIR="$PROJECT_DIR/models/qwen3-8b"
SERVING_DIR="$PROJECT_DIR/serving"
STOP_FILE="$SERVING_DIR/STOP"
ENDPOINT_FILE="$SERVING_DIR/current_endpoint.txt"
PORT=8000

mkdir -p "$SERVING_DIR" "$PROJECT_DIR/logs"

echo "=== LLM Serve Job ==="
echo "Node:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Model:   $MODEL_DIR"
echo

if [ -f "$STOP_FILE" ]; then
    echo "STOP file present ($STOP_FILE) - not starting. Delete it to resume."
    exit 0
fi

echo "$(hostname):$PORT" > "$ENDPOINT_FILE"

REAL_LIBCUDA=$(find /usr/lib/x86_64-linux-gnu -name 'libcuda.so.1' 2>/dev/null | head -1)
if [ -z "$REAL_LIBCUDA" ]; then
    REAL_LIBCUDA=$(find /usr/lib -name 'libcuda.so.1' 2>/dev/null | head -1)
fi
if [ -z "$REAL_LIBCUDA" ]; then
    echo "WARNING: could not find libcuda.so.1 on the host - vLLM will likely fail to start"
fi
RESOLVED_LIBCUDA=$(readlink -f "$REAL_LIBCUDA" 2>/dev/null)
echo "Host libcuda.so.1 resolved to: ${RESOLVED_LIBCUDA:-<not found>}"

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:${PROJECT_DIR}" \
    -B "${RESOLVED_LIBCUDA}:/usr/local/cuda/compat/lib/libcuda.so.1" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        vllm serve '${MODEL_DIR}' --port ${PORT} --max-model-len 8192 --gpu-memory-utilization 0.85 --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes
    "

echo "Server exited (walltime limit or crash). It will be restarted at the next 05:00/17:00 cron cycle."
