"""
Ask the harness a question from the command line and see every step: the planner's plan,
each tool call (place lookups, data requests with the SQL the student wrote, port_calls),
the rows or errors, and the final answer. Needs the llm-serve and llm-student jobs.

    scripts/ask.sh "How many cruise ships visited Denmark in August?"
    scripts/ask.sh --mode planner "..."      # always through the planner, for comparison
    scripts/ask.sh --single "..."            # Qwen3-8B alone (agent.py), for comparison
    scripts/ask.sh --feedback ID wrong "counted ferries too"
    scripts/ask.sh --feedback ID correct

Every run is appended to logs/questions/<date>.jsonl with an id; feedback is appended to
logs/questions/feedback.jsonl. Both files are append-only (nothing is rewritten), and they
are the raw material for test questions and, later, for learning from experience. api.py
logs the questions asked on the map the same way (log_run, save_feedback).
"""

import argparse
import contextlib
import getpass
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path

import llm_usage
from ais_tool import PROJECT_DIR

# P7_QUESTION_LOG_DIR: somewhere else for tests, so they do not teach the store.
LOG_DIR = Path(os.environ.get("P7_QUESTION_LOG_DIR", PROJECT_DIR / "logs" / "questions"))
FEEDBACK_FILE = LOG_DIR / "feedback.jsonl"
SHOW_ROWS = 10
LOG_ROWS = 20  # result rows kept per step in the log
_append_lock = threading.Lock()  # api.py answers in several threads


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _append_lock, path.open("a") as f:
        f.write(line)


def _store_snapshot() -> dict | None:
    """The learned-question store when a question is answered: on or off, how many patterns."""
    try:
        import query_store  # here: it loads the planner
        return query_store.snapshot()
    except Exception:  # a log line must not fail over the store
        return None


def log_run(question: str, trace: dict, mode: str, seconds: float, source: str,
            thinking: bool = False, user: str | None = None, llm: dict | None = None) -> str:
    """Append one answered (or failed) question to today's log, with its LLM calls (llm_usage
    summary) and the store's size; returns its run id."""
    run_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    append(LOG_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl", {
        "id": run_id, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source,
        "user": user, "mode": mode, "route": trace.get("route"),
        "route_reason": trace.get("route_reason"), "thinking": thinking,
        "question": question, "question_as_read": trace.get("question_as_read"),
        "plan": trace.get("plan"), "answer": trace.get("answer"), "error": trace.get("error"),
        "seconds": seconds, "llm": llm, "store": _store_snapshot(),
        "place_calls": trace.get("place_calls", []),
        "steps": [{"request": s.get("request"), "places": s.get("places"), "sql": s.get("sql"),
                   "error": s["result"].get("error"), "columns": s["result"].get("columns"),
                   "row_count": s["result"].get("row_count"),
                   "rows": s["result"].get("rows", [])[:LOG_ROWS]}
                  for s in trace.get("tool_calls", [])],
    })
    return run_id


def save_feedback(run_id: str, verdict: str, note: str, by: str | None) -> bool:
    """Append a verdict (correct or wrong) for a logged run; False if no run has that id."""
    key = f'"id": "{run_id}"'
    known = any(key in line for f in LOG_DIR.glob("20*.jsonl") for line in f.open())
    if known:
        append(FEEDBACK_FILE, {"id": run_id, "verdict": verdict, "note": note, "by": by,
                               "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return known


def show_rows(result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    rows = result.get("rows", [])
    lines = [f"{len(rows)} row(s)" + (f", first {SHOW_ROWS}:" if len(rows) > SHOW_ROWS else ":")]
    if result.get("columns"):
        lines.append("  " + " | ".join(map(str, result["columns"])))
    lines += ["  " + " | ".join(map(str, r)) for r in rows[:SHOW_ROWS]]
    return "\n".join(lines)


def ask(question: str, mode: str, thinking: bool) -> None:
    if mode == "single":
        from agent import run
    elif mode == "planner":
        from planner import run
    else:
        from planner import run_routed as run
    from openai import APIConnectionError

    start = time.time()
    try:
        with (open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet),  # debug lines
              llm_usage.track() as llm_calls):
            trace = run(question, enable_thinking=thinking, return_trace=True)
    except APIConnectionError:
        sys.exit("A model server is not reachable. It may be restarting (05:00 and 17:00) or "
                 "still waiting in the queue; it takes a few minutes to load the model.\n"
                 "Check with: squeue -u pr55oh@student.aau.dk -n llm-serve,llm-student")
    seconds = round(time.time() - start, 1)

    print("\n=== Question\n" + question)
    if trace.get("route"):
        print(f"\n=== Route: {trace['route']} ({trace.get('route_reason', '')})")
    if trace.get("plan"):
        print("\n=== Plan\n" + trace["plan"])
    for call in trace.get("place_calls", []):
        print(f"\n--- {call['tool']}({json.dumps(call['arguments'], ensure_ascii=False)})")
        print(f"    -> {json.dumps(call['reply'], ensure_ascii=False)[:300]}")
    for i, step in enumerate(trace["tool_calls"], 1):
        print(f"\n--- step {i}: {step.get('request') or '(SQL written by the 8B)'}")
        if step.get("places"):
            print(f"    places in the student's prompt: {', '.join(step['places'])}")
        if step.get("sql") and step["sql"] != "(port_calls)":
            print("    SQL:\n      " + step["sql"].strip().replace("\n", "\n      "))
        print("    " + show_rows(step["result"]).replace("\n", "\n    "))
    print(f"\n=== Answer ({seconds} s)\n{trace['answer']}")

    run_id = log_run(question, trace, mode, seconds, "cli", thinking, getpass.getuser(),
                     llm_usage.summary(llm_calls))
    print(f"\n(run id {run_id}; mark it with: scripts/ask.sh --feedback {run_id} correct|wrong "
          f"\"note\")")


def feedback(run_id: str, verdict: str, note: str) -> None:
    if verdict not in ("correct", "wrong"):
        sys.exit("verdict must be correct or wrong")
    if not save_feedback(run_id, verdict, note, getpass.getuser()):
        sys.exit(f"no run with id {run_id} in {LOG_DIR}")
    print(f"feedback saved for {run_id}: {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("question", nargs="*")
    p.add_argument("--mode", choices=["routed", "planner", "single"], default="routed",
                   help="routed (default): simple questions straight to the student, others via "
                        "the planner; planner: always the planner; single: Qwen3-8B alone")
    p.add_argument("--single", action="store_true", help="same as --mode single")
    p.add_argument("--thinking", action="store_true", help="let Qwen3-8B reason (slower)")
    p.add_argument("--feedback", nargs="+", metavar=("ID", "VERDICT"),
                   help="mark a run: ID correct|wrong [note]")
    args = p.parse_args()
    if args.feedback:
        if len(args.feedback) < 2:
            sys.exit("usage: --feedback ID correct|wrong [note]")
        feedback(args.feedback[0], args.feedback[1], " ".join(args.feedback[2:]))
        return
    if not args.question:
        p.error("give a question")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    ask(" ".join(args.question), "single" if args.single else args.mode, args.thinking)


if __name__ == "__main__":
    main()
