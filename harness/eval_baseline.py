"""Repeatable baseline evaluation through scripts/ask.sh --single.

The evaluator deliberately uses the CLI rather than importing the agent.  Every question is
therefore a fresh process and a fresh question-log directory, while the existing ask.py logging
remains the source for LLM-call counts and tool validation outcomes.

By default, all question and answer files in ``harness/eval`` are included.

Example on the Linux host:
    python3 harness/eval_baseline.py --runs 5 --out results/baseline
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EVAL_DIR = HERE / "eval"
QUESTIONS = tuple(sorted(EVAL_DIR.glob("*_questions.json"))) + (EVAL_DIR / "questions.toml",)
ANSWERS = tuple(sorted(EVAL_DIR.glob("*_answers.json"))) + (EVAL_DIR / "answers.json",)
ASK = PROJECT / "scripts" / "ask.sh"

sys.path.insert(0, str(HERE))
from eval_sql import answer_has_reference, results_match


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(paths: Iterable[Path]) -> list[dict]:
    questions = []
    for path in paths:
        if path.suffix == ".json":
            questions.extend(json.loads(path.read_text(encoding="utf-8")))
        else:
            with path.open("rb") as stream:
                questions.extend(tomllib.load(stream)["question"])
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return questions


def load_answers(paths: Iterable[Path]) -> dict:
    answers = {}
    for path in paths:
        path_answers = json.loads(path.read_text(encoding="utf-8"))
        duplicate_ids = answers.keys() & path_answers.keys()
        if duplicate_ids:
            raise ValueError("duplicate answer ids: " + ", ".join(sorted(duplicate_ids)))
        answers.update(path_answers)
    return answers


def latest_log(log_dir: Path) -> dict | None:
    records = []
    for path in log_dir.glob("20*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records[-1] if records else None


def score_record(record: dict, expected: dict, question: str) -> dict:
    reference = expected["rows"]
    steps = record.get("steps", [])
    successful = [s for s in steps if not s.get("error")]
    data_match = any(results_match(reference, s.get("rows", [])) for s in successful)
    answer_match = answer_has_reference(record.get("answer", ""), reference, question)
    return {
        "answer_correct": bool(data_match or answer_match),
        "data_match": data_match,
        "answer_match": answer_match,
        "validation": "accepted" if successful else "rejected",
        "tool_calls": len(steps),
        "llm_calls": (record.get("llm") or {}).get("calls", 0),
        "latency_seconds": record.get("seconds"),
    }


def run_question(question: str, expected: dict | None, env: dict) -> dict:
    started = time.monotonic()
    completed = subprocess.run(
        [str(ASK), "--single", question],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.monotonic() - started, 3)
    record = latest_log(Path(env["P7_QUESTION_LOG_DIR"]))
    result = {
        "question": question,
        "exit_code": completed.returncode,
        "latency_seconds": elapsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "log": record,
    }
    if expected is not None and record is not None:
        result["score"] = score_record(record, expected, question)
    else:
        steps = (record or {}).get("steps", [])
        successful = [step for step in steps if not step.get("error")]
        result["score"] = {
        "answer_correct": False,
        "data_match": False,
        "answer_match": False,
        "validation": "rejected" if not successful else "accepted",
        "tool_calls": len(steps),
        "llm_calls": ((record or {}).get("llm") or {}).get("calls", 0),
        }
    return result


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(records: list[dict], question_count: int) -> dict:
    valid = [r for r in records if r.get("kind") == "valid"]
    valid_scores = [r["score"] for r in valid]
    return {
        "questions": question_count,
        "answer_accuracy": mean([s["answer_correct"] for s in valid_scores]),
        "validation_effectiveness_valid": mean(
            [s["validation"] == "accepted" and s["answer_correct"] for s in valid_scores]
        ),
        "full_llm_calls_per_request": mean([s["llm_calls"] for s in valid_scores]),
        "end_to_end_latency_seconds": mean([r["latency_seconds"] for r in valid]),
        "safe_reuse_rate": None,
        "safe_reuse_rate_status": "not_applicable_baseline_single",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=None,
                        help="question file (default: all files in eval)")
    parser.add_argument("--answers", type=Path, default=None,
                        help="answer file (default: all files in eval)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    question_files = (args.questions,) if args.questions else QUESTIONS
    answer_files = (args.answers,) if args.answers else ANSWERS
    missing_files = [str(path) for path in (*question_files, *answer_files) if not path.exists()]
    if missing_files:
        parser.error("evaluation files do not exist: " + ", ".join(missing_files))

    questions = load_questions(question_files)
    answers = load_answers(answer_files)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": str(PROJECT),
        "questions": [str(path) for path in question_files],
        "questions_sha256": {str(path): sha256(path) for path in question_files},
        "answers": [str(path) for path in answer_files],
        "answers_sha256": {str(path): sha256(path) for path in answer_files},
        "runs": args.runs,
        "mode": "scripts/ask.sh --single",
        "safe_reuse_rate": None,
        "safe_reuse_rate_status": "not_applicable_baseline_single",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summaries = []
    for run_number in range(1, args.runs + 1):
        run_dir = args.out / f"run-{run_number:02d}"
        run_dir.mkdir(exist_ok=True)
        records = []
        for question in questions:
            with tempfile.TemporaryDirectory(prefix="baseline-question-") as log_dir:
                env = os.environ.copy()
                env.setdefault("P7_PROJECT_DIR", str(PROJECT))
                env["P7_QUESTION_LOG_DIR"] = log_dir
                result = run_question(question["question"], answers.get(question["id"]), env)
            result.update({"id": question["id"], "kind": "valid", "run": run_number})
            records.append(result)
        (run_dir / "records.json").write_text(json.dumps(records, indent=2, default=str) + "\n")
        summary = summarize(records, len(questions))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(f"run {run_number}/{args.runs}: {json.dumps(summary, sort_keys=True)}", flush=True)

    aggregate = {
        key: mean([s[key] for s in summaries if isinstance(s.get(key), (int, float))])
        for key in summaries[0]
        if key != "questions"
    }
    aggregate.update({"runs": args.runs, "questions": len(questions)})
    (args.out / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()