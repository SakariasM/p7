#!/bin/bash
#SBATCH --job-name=train-student
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=ailab-l4-04
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/train_student_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/train_student_%j.err
#
# LoRA fine-tune of the text-to-SQL student (training/train_student.py) in the PyTorch
# container. Score the result with slurm/eval_student.slurm.
#
# *** BEFORE FIRST SUBMIT ***
# 1. Base model:
#     MODEL_ID=Qwen/Qwen2.5-Coder-1.5B-Instruct LOCAL_NAME=qwen2.5-coder-1.5b-instruct \
#         PROJECT_DIR=/ceph/project/P7-Agent-Harness sbatch slurm/model_download.slurm
# 2. envs/train, built inside the container (no singularity on the login node):
#     srun --partition=l4 --gres=gpu:0 --mem=8G --time=00:30:00 \
#         singularity exec -B /ceph/project/P7-Agent-Harness:/ceph/project/P7-Agent-Harness \
#         /ceph/container/pytorch/pytorch_25.12.sif bash -c \
#         "python -m venv --system-site-packages /ceph/project/P7-Agent-Harness/envs/train &&
#          /ceph/project/P7-Agent-Harness/envs/train/bin/pip install -r requirements-train.txt"
# 3. Data: sbatch slurm/gen_sql_data.slurm export
#
# Usage, from the repo root (arguments go to train_student.py):
#     sbatch slurm/train_student.slurm
#     sbatch slurm/train_student.slurm --max-steps 20 --out /ceph/project/P7-Agent-Harness/models/smoke

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
PYTHON="$P7_PROJECT_DIR/envs/train/bin/python"

echo "=== Student training job ==="
echo "Node:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Repo:    $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Args:    $*"
echo

# Same libcuda workaround as serve_llm.slurm.
REAL_LIBCUDA=$(find /usr/lib/x86_64-linux-gnu /usr/lib -name 'libcuda.so.1' 2>/dev/null | head -1)
RESOLVED_LIBCUDA=$(readlink -f "$REAL_LIBCUDA")
echo "Host libcuda.so.1 resolved to: $RESOLVED_LIBCUDA"

singularity exec --nv \
    -B "${P7_PROJECT_DIR}:${P7_PROJECT_DIR}" \
    -B "${RESOLVED_LIBCUDA}:/usr/local/cuda/compat/lib/libcuda.so.1" \
    "${CONTAINER}" \
    "$PYTHON" -u "$REPO_DIR/training/train_student.py" "$@"
