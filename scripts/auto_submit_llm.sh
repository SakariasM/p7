#!/bin/bash
[ -f ~/.llm_auto ] || exit 0

scancel -u "$(whoami)" --name=llm-serve

for i in $(seq 1 30); do
    [ "$(squeue -u "$(whoami)" --name=llm-serve -h | wc -l)" -eq 0 ] && break
    sleep 5
done

cd "${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
sbatch --exclude=ailab-l4-04 --job-name=llm-serve slurm/serve_llm.slurm
