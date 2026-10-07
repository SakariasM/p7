#!/bin/bash
#SBATCH --job-name=llm-watchdog
#SBATCH --partition=l4
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/ceph/project/P7-Agent-Harness/logs/watchdog_%j.out
#SBATCH --error=/ceph/project/P7-Agent-Harness/logs/watchdog_%j.err
#
# Runs once daily at 05:00. Checks whether the llm-serve job is currently
# running or queued; if not, starts it. Then reschedules itself for
# tomorrow at 05:00, so this repeats forever until stopped. No GPU needed
# for the check itself (--gres=gpu:0).
#
# *** BEFORE FIRST SUBMIT ***
#     mkdir -p /ceph/project/P7-Agent-Harness/logs
#     mkdir -p /ceph/project/P7-Agent-Harness/serving
#
# Kick off the daily chain (aligns to the next 05:00, whatever time it
# happens to be right now):
#     cd /ceph/project/P7-Agent-Harness
#     NEXT_RUN=$(date -d "today 05:00" +%Y-%m-%dT%H:%M:%S)
#     if [ "$(date -d "$NEXT_RUN" +%s)" -le "$(date +%s)" ]; then
#         NEXT_RUN=$(date -d "tomorrow 05:00" +%Y-%m-%dT%H:%M:%S)
#     fi
#     sbatch --begin="$NEXT_RUN" slurm/llm_watchdog.slurm
#
# *** STOP THE DAILY CHAIN ***
#     touch /ceph/project/P7-Agent-Harness/serving/STOP
#     # the next scheduled watchdog run will see this and stop
#     # rescheduling itself. To also stop the server right now:
#     scancel --name=llm-serve
#     # to resume the daily chain later:
#     rm /ceph/project/P7-Agent-Harness/serving/STOP
#     NEXT_RUN=$(date -d "tomorrow 05:00" +%Y-%m-%dT%H:%M:%S)
#     sbatch --begin="$NEXT_RUN" slurm/llm_watchdog.slurm

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/ceph/project/P7-Agent-Harness}}"
STOP_FILE="$PROJECT_DIR/serving/STOP"

mkdir -p "$PROJECT_DIR/serving" "$PROJECT_DIR/logs"

echo "=== LLM Watchdog Check ==="
echo "Time: $(date)"

if [ -f "$STOP_FILE" ]; then
    echo "STOP file present - not starting the server, and not rescheduling. Chain ends here."
    exit 0
fi

RUNNING=$(squeue --me --name=llm-serve --noheader --format=%i)

if [ -n "$RUNNING" ]; then
    echo "llm-serve already active (job ${RUNNING}) - nothing to do."
else
    echo "llm-serve not found in the queue - starting it."
    (cd "$PROJECT_DIR" && sbatch --exclude=ailab-l4-04 slurm/serve_llm.slurm)
fi

NEXT_RUN=$(date -d "tomorrow 05:00" +%Y-%m-%dT%H:%M:%S)
echo "Rescheduling next check for $NEXT_RUN"
(cd "$PROJECT_DIR" && sbatch --begin="$NEXT_RUN" slurm/llm_watchdog.slurm)
