#!/bin/bash
# Ask the agent harness a question and see every step (harness/ask.py). Works from any
# directory; needs the llm-serve and llm-student jobs running.
#   scripts/ask.sh "How many tankers were in Aalborg on 3 March 2026?"
#   scripts/ask.sh --single "..."              Qwen3-8B alone, for comparison
#   scripts/ask.sh --feedback ID correct|wrong "note"
P7_PROJECT_DIR="${P7_PROJECT_DIR:-/ceph/project/P7-Agent-Harness}"
export P7_PROJECT_DIR
cd "$(dirname "$(readlink -f "$0")")/../harness" || exit 1
exec "$P7_PROJECT_DIR/envs/harness/bin/python" -B ask.py "$@"
