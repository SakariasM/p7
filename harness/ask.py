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
are the raw material for test questions and, later, for learning from experience.
"""

import argparse
import contextlib
import getpass
import json
import os
import secrets
import sys
import time
from pathlib import Path

from ais_tool import PROJECT_DIR

LOG_DIR = PROJECT_DIR / "logs" / "questions"
FEEDBACK_FILE = LOG_DIR / "feedback.jsonl"
SHOW_ROWS = 10


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


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
        with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):  # debug lines
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

    run_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    append(LOG_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl", {
        "id": run_id, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": getpass.getuser(),
        "mode": mode, "route": trace.get("route"), "thinking": thinking,
        "question": question, "plan": trace.get("plan"), "answer": trace["answer"],
        "seconds": seconds, "place_calls": trace.get("place_calls", []),
        "steps": [{"request": s.get("request"), "places": s.get("places"), "sql": s.get("sql"),
                   "error": s["result"].get("error"),
                   "rows": s["result"].get("rows", [])[:20]} for s in trace["tool_calls"]],
    })
    print(f"\n(run id {run_id}; mark it with: scripts/ask.sh --feedback {run_id} correct|wrong "
          f"\"note\")")


def feedback(run_id: str, verdict: str, note: str) -> None:
    if verdict not in ("correct", "wrong"):
        sys.exit("verdict must be correct or wrong")
    known = any(run_id in line for f in LOG_DIR.glob("20*.jsonl") for line in f.open())
    if not known:
        sys.exit(f"no run with id {run_id} in {LOG_DIR}")
    append(FEEDBACK_FILE, {"id": run_id, "verdict": verdict, "note": note,
                           "by": getpass.getuser(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
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
