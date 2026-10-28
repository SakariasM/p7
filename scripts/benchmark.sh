#!/bin/bash
# The nightly benchmark for the usage report (harness/usage_report.py): the five eval sets
# through the harness as it runs (router, learned-question store as set by P7_QUERY_STORE),
# with the LLM calls per question, to logs/benchmark/<date>_<set>.jsonl. A set already run
# that day is skipped. Then the usage tables and graphs are refreshed. On the hosting machine
# p7-benchmark.timer runs it after the store's nightly distill.
#   scripts/benchmark.sh
P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
export P7_PROJECT_DIR
cd "$(dirname "$(readlink -f "$0")")/../harness" || exit 1
OUT="$P7_PROJECT_DIR/logs/benchmark"
mkdir -p "$OUT"
DAY=$(date +%F)
for set in eval:questions.toml:answers.json places:places_questions.json:places_answers.json \
           multi:multistep_questions.json:multistep_answers.json \
           list:list_questions.json:list_answers.json group:group_questions.json:group_answers.json; do
    IFS=: read -r name questions answers <<< "$set"
    [ -e "$OUT/${DAY}_${name}.jsonl" ] && continue
    echo "=== $(date +%T) $name"
    "$P7_PROJECT_DIR/envs/harness/bin/python" -u eval_sql.py agent --mode routed --repeats 1 \
        --questions "eval/$questions" --answers "eval/$answers" --out "$OUT/${DAY}_${name}.jsonl"
done
# The per-day tables and graphs for the report (logs/usage_*.csv, .pdf, .png).
"$P7_PROJECT_DIR/envs/harness/bin/python" usage_report.py > /dev/null \
    && "$P7_PROJECT_DIR/envs/harness/bin/python" usage_plot.py || echo "(report or graphs failed)"
echo "=== $(date +%T) done"
