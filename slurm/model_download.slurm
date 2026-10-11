#!/bin/bash
#SBATCH --job-name=model-download
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/model_dl_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/model_dl_%j.err
#
# Downloads a Hugging Face model into $PROJECT_DIR/models/<name>.
# No GPU needed for a plain download (--gres=gpu:0), which also keeps it
# out of the max8gpus QOS count.
#
# *** BEFORE FIRST SUBMIT ***
# logs/ must already exist, or SLURM can't create --output/--error and the
# job dies instantly with no log at all:
#     mkdir -p /ceph/project/P7-Agent-Harness/logs
#
# Usage (submit from the project root):
#     cd /ceph/project/P7-Agent-Harness
#     sbatch slurm/model_download.slurm
#
# For a different model, export MODEL_ID and LOCAL_NAME before submitting, e.g.
#     MODEL_ID=Qwen/Qwen2.5-Coder-1.5B-Instruct LOCAL_NAME=qwen2.5-coder-1.5b-instruct \
#         sbatch slurm/model_download.slurm
# For a gated model (e.g. Llama), export HF_TOKEN before submitting and add
# --token "$HF_TOKEN" to the hf download line below.

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/llm_venv"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/ceph/project/P7-Agent-Harness}}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
LOCAL_NAME="${LOCAL_NAME:-qwen3-8b}"
MODEL_DIR="$PROJECT_DIR/models/$LOCAL_NAME"

echo "=== Model Download Job ==="
echo "Node:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-none}"
echo "PROJECT_DIR: ${PROJECT_DIR}"
echo "Model:       ${MODEL_ID}"
echo "Target:      ${MODEL_DIR}"
echo

mkdir -p "$MODEL_DIR" "$PROJECT_DIR/logs"

singularity exec \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:${PROJECT_DIR}" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        command -v hf >/dev/null || { echo 'ERROR: hf CLI not found in venv - was setup_llm_env.sh run?'; exit 1; }
        hf version
        echo
        hf download '${MODEL_ID}' --local-dir '${MODEL_DIR}'
        echo
        echo '=== Download complete ==='
        ls -lh '${MODEL_DIR}'
    "

echo
du -sh "$MODEL_DIR"
