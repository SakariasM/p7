#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=ailab-l4-03,ailab-l4-04
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/baselines_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/baselines_%j.err
#
# Baselines for the report: an untrained model on every test set, in two ways
# (harness/eval_sql.py):
#   bare  the question alone: no schema, no data, no tools, no planner; a text answer
#   run   zero-shot SQL: the same schema prompt as the student (only the places a question
#         names), one query, run with the usual guardrails; no training, planner or router
# Results: logs/eval/baseline_<TAG>_<bare|run>_<eval|places|multi|list|group>.jsonl.
# BASELINE_SETS="list" runs only those sets.
#
# Usage, from the repo root:
#     sbatch slurm/baselines.slurm /ceph/project/P7-Agent-Harness/models/qwen2.5-coder-1.5b-instruct coder1.5b
#     sbatch --gres=gpu:0 slurm/baselines.slurm 8b qwen3-8b   # the running llm-serve (Qwen3-8B)
#     BASELINE_SETS=list sbatch slurm/baselines.slurm MODEL_DIR TAG

set -euo pipefail
trap 'echo "FAILED at line $LINENO (exit code $?)"' ERR

MODEL_DIR="${1:?usage: sbatch slurm/baselines.slurm MODEL_DIR|8b TAG}"
TAG="${2:?usage: sbatch slurm/baselines.slurm MODEL_DIR|8b TAG}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
PYTHON="$P7_PROJECT_DIR/envs/harness/bin/python"
CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/llm_venv"
PORT=$((8100 + ${SLURM_JOB_ID:-0} % 800))

echo "=== Baselines: $TAG ($MODEL_DIR) on $(hostname), job ${SLURM_JOB_ID:-none}, repo $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?') ==="

if [ "$MODEL_DIR" = 8b ]; then
    MODEL_ARGS=()  # eval_sql defaults: the llm-serve endpoint and model
else
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
        " > "$P7_PROJECT_DIR/logs/baselines_${SLURM_JOB_ID:-x}_vllm.log" 2>&1 &
    VLLM_PID=$!
    trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
    for _ in $(seq 1 90); do
        curl -sf "http://localhost:${PORT}/v1/models" > /dev/null && break
        kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited, see the _vllm.log"; exit 1; }
        sleep 10
    done
    curl -sf "http://localhost:${PORT}/v1/models" > /dev/null || { echo "vLLM not up after 15 min"; exit 1; }
    MODEL_ARGS=(--model "$MODEL_DIR" --base-url "http://localhost:${PORT}/v1")
fi

mkdir -p "$P7_PROJECT_DIR/logs/eval"
cd "$REPO_DIR/harness"
declare -A SETS=([eval]="eval/questions.toml eval/answers.json"
                 [places]="eval/places_questions.json eval/places_answers.json"
                 [multi]="eval/multistep_questions.json eval/multistep_answers.json"
                 [list]="eval/list_questions.json eval/list_answers.json"
                 [group]="eval/group_questions.json eval/group_answers.json")
for set in ${BASELINE_SETS:-eval places multi list group}; do
    read -r questions answers <<< "${SETS[$set]}"
    for cmd in bare run; do
        echo; echo "=== $TAG $cmd $set ==="
        "$PYTHON" -u eval_sql.py "$cmd" "${MODEL_ARGS[@]}" --dynamic-places \
            --questions "$questions" --answers "$answers" \
            --out "$P7_PROJECT_DIR/logs/eval/baseline_${TAG}_${cmd}_${set}.jsonl" || echo "($cmd $set failed)"
    done
done
