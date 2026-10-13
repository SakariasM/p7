#!/bin/bash
[ -f ~/.llm_auto ] || exit 0

EXCLUDE="ailab-l4-03,ailab-l4-04"  # nodes with repeated job failures
cd "${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"

scancel -u "$(whoami)" --name=llm-serve
scancel -u "$(whoami)" --name=llm-student

for i in $(seq 1 30); do
    [ "$(squeue -u "$(whoami)" --name=llm-serve,llm-student -h | wc -l)" -eq 0 ] && break
    sleep 5
done

submit() {
    case "$1" in
        llm-serve)
            sbatch --exclude="$EXCLUDE" --job-name=llm-serve slurm/serve_llm.slurm ;;
        llm-student)  # SQL student for harness/planner.py; job script from the repo checkout
            sbatch --exclude="$EXCLUDE" --job-name=llm-student p7/slurm/serve_student.slurm ;;
    esac
}

# Failed = gone from the queue, or running without having written its endpoint file.
# A job still waiting in the queue (PENDING) is not a failure.
failed() {
    local file=serving/current_endpoint.txt state
    [ "$1" = llm-student ] && file=serving/student_endpoint.txt
    state=$(squeue -u "$(whoami)" --name="$1" -h -o %T | head -1)
    [ -z "$state" ] || { [ "$state" = RUNNING ] && [ ! -s "$file" ]; }
}

NAMES="llm-serve"
[ -f p7/slurm/serve_student.slurm ] && NAMES="$NAMES llm-student"
for name in $NAMES; do submit "$name"; done

# A job that dies right after starting would leave nothing served until the next run:
# check after 3 minutes and resubmit what failed, at most twice.
for attempt in 1 2; do
    sleep 180
    for name in $NAMES; do
        if failed "$name"; then
            echo "$(date '+%F %T') $name failed after start, resubmitting (retry $attempt)"
            scancel -u "$(whoami)" --name="$name"
            submit "$name"
        fi
    done
done
