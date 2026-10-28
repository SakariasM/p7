"""
LLM use and accuracy over time, for the report: does LLM use go down as the learned-question
store grows (query_store.py), while accuracy stays? Two views, one row per day:

  live       the questions asked on the map, API and CLI (logs/questions): questions, LLM calls
             per question, the share answered from the store, seconds per question, the
             map's verdicts (correct / wrong) and the store's approved patterns
  benchmark  the five eval sets run each night (scripts/benchmark.sh, logs/benchmark): the
             same fixed questions every day, so accuracy, LLM calls and seconds per question
             are comparable from day to day

    cd harness
    python3 usage_report.py             # prints both, writes logs/usage_live.csv and
                                        # logs/usage_benchmark.csv for the graphs
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from ais_tool import PROJECT_DIR
from ask import FEEDBACK_FILE, LOG_DIR
from report_table import EVAL_DIR, SETS, questions_by_id, rescore

BENCHMARK_DIR = PROJECT_DIR / "logs" / "benchmark"
REPORT_DIR = PROJECT_DIR / "logs"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _mean(xs: list) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


def live() -> list[dict]:
    verdicts = {}
    if FEEDBACK_FILE.exists():
        for v in _jsonl(FEEDBACK_FILE):
            verdicts[v["id"]] = v["verdict"]  # the latest per question
    days = defaultdict(list)
    for f in sorted(LOG_DIR.glob("20*.jsonl")):
        for r in _jsonl(f):
            days[r["at"][:10]].append(r)
    rows = []
    for day, runs in sorted(days.items()):
        llm = [r["llm"]["calls"] for r in runs if r.get("llm")]
        judged = [verdicts[r["id"]] for r in runs if r["id"] in verdicts]
        rows.append({
            "day": day, "questions": len(runs),
            "llm_calls_per_question": _mean(llm),
            "from_store": round(sum(str(r.get("route")).startswith("learned") for r in runs)
                                / len(runs), 3),
            "seconds_per_question": _mean([r["seconds"] for r in runs]),
            "verdicts": len(judged),
            "correct_share": round(judged.count("correct") / len(judged), 3) if judged else None,
            "approved_patterns": max((r["store"]["approved"] for r in runs if r.get("store")),
                                     default=None),
        })
    return rows


def benchmark() -> list[dict]:
    answers = {}
    for f in SETS.values():
        answers.update(json.loads((EVAL_DIR / f).read_text()))
    meta = questions_by_id()
    days = defaultdict(list)
    for f in sorted(BENCHMARK_DIR.glob("20*_*.jsonl")) if BENCHMARK_DIR.exists() else []:
        days[f.name[:10]] += _jsonl(f)
    rows = []
    for day, records in sorted(days.items()):
        ok = []
        for r in records:
            expect = r.get("expect") or meta.get(r["id"], {}).get("expect", "answer")
            if expect != "skip":
                ok.append(rescore(r, expect, answers.get(r["id"], {}).get("rows")))
        llm = [r["llm"]["calls"] for r in records if r.get("llm")]
        rows.append({
            "day": day, "questions": len(ok), "accuracy": round(sum(ok) / len(ok), 3),
            "llm_calls_per_question": _mean(llm),
            "from_store": round(sum(str(r.get("route")).startswith("learned") for r in records)
                                / len(records), 3),
            "seconds_per_question": _mean([r["seconds"] for r in records]),
        })
    return rows


def show(name: str, rows: list[dict]) -> None:
    out = REPORT_DIR / f"usage_{name}.csv"
    print(f"\n{name} ({out})")
    if not rows:
        print("  no data yet")
        return
    cols = list(rows[0])
    print("  " + "  ".join(cols))
    for r in rows:
        print("  " + "  ".join(f"{'-' if r[c] is None else r[c]!s:>{len(c)}}" for c in cols))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    show("live", live())
    show("benchmark", benchmark())
