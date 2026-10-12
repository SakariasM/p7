#!/bin/bash
#SBATCH --job-name=eval-student
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=ailab-l4-04
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/eval_student_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/eval_student_%j.err
#
# Serves one model with vLLM for the length of this job and scores it on the held-out
# text-to-SQL questions with harness/eval_sql.py (needs envs/harness). Results go to
# logs/eval/<model name>_<job id>.jsonl unless --out is given.
#
# Usage, from the repo root (arguments after MODEL_DIR go to eval_sql.py run):
#     sbatch slurm/eval_student.slurm /ceph/project/P7-Agent-Harness/models/qwen2.5-coder-1.5b-sql-v1
#     sbatch slurm/eval_student.slurm MODEL_DIR --dynamic-places --out .../x.jsonl

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

MODEL_DIR="${1:?usage: sbatch slurm/eval_student.slurm MODEL_DIR [eval_sql.py run options]}"
shift
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/llm_venv"
NAME="$(basename "$MODEL_DIR")"
PORT=$((8100 + ${SLURM_JOB_ID:-0} % 800))

echo "=== Student eval job ==="
echo "Node:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Repo:    $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Model:   $MODEL_DIR on port $PORT"
echo

# Same libcuda workaround as serve_llm.slurm.
REAL_LIBCUDA=$(find /usr/lib/x86_64-linux-gnu /usr/lib -name 'libcuda.so.1' 2>/dev/null | head -1)
RESOLVED_LIBCUDA=$(readlink -f "$REAL_LIBCUDA")

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${P7_PROJECT_DIR}:${P7_PROJECT_DIR}" \
    -B "${RESOLVED_LIBCUDA}:/usr/local/cuda/compat/lib/libcuda.so.1" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        vllm serve '${MODEL_DIR}' --port ${PORT} --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager
    " > "$P7_PROJECT_DIR/logs/eval_student_${SLURM_JOB_ID:-x}_vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
    curl -sf "http://localhost:${PORT}/v1/models" > /dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited, see the _vllm.log"; exit 1; }
    sleep 10
done
curl -sf "http://localhost:${PORT}/v1/models" > /dev/null || { echo "vLLM not up after 15 min"; exit 1; }
echo "vLLM up"

mkdir -p "$P7_PROJECT_DIR/logs/eval"
cd "$REPO_DIR/harness"
OUT_ARGS=(--out "$P7_PROJECT_DIR/logs/eval/${NAME}_${SLURM_JOB_ID:-local}.jsonl")
[[ " $* " == *" --out "* ]] && OUT_ARGS=()
"$P7_PROJECT_DIR/envs/harness/bin/python" -u eval_sql.py run \
    --model "$MODEL_DIR" --base-url "http://localhost:${PORT}/v1" "${OUT_ARGS[@]}" "$@"
