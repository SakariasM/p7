"""
The comparison table for the report, from the result files in logs/eval: every setup on
every test set, all scored the same way (rescored here from the stored answers, so older
runs use today's answer check: numbers from the question and list markers do not count).

    cd harness
    python3 report_table.py

Setups (rows): the untrained models with no harness ("bare": the question only) and with
zero-shot SQL (schema prompt, one query), the trained students alone (one query), and the
full harness (router + planner + student). A cell is correct/total over all runs found,
with the mean per run; "-" where a setup was not run on that set.
"""

import json
import sys
from pathlib import Path

from ais_tool import PROJECT_DIR
from eval_sql import CANNOT_RE, CLARIFY_RE, EVAL_DIR, answer_has_reference

LOGS = PROJECT_DIR / "logs" / "eval"
SETS = {"eval": "answers.json", "places": "places_answers.json",
        "multi": "multistep_answers.json", "list": "list_answers.json",
        "group": "group_answers.json"}
# The group set is shown as two columns: its data questions, and the ones to decline or
# clarify (a one-query SQL setup cannot decline, so those count as wrong for it).
COLUMNS = ["eval", "places", "multi", "list", "group data", "group decline"]
# setup -> {set: [result files]}; a file per run
SETUPS = {
    "Qwen3-8B, bare (question only)": {s: [f"baseline_qwen3-8b_bare_{s}.jsonl"] for s in SETS},
    "Coder-1.5B base, bare": {s: [f"baseline_coder1.5b_bare_{s}.jsonl"] for s in SETS},
    "Qwen3-8B, zero-shot SQL": {s: [f"baseline_qwen3-8b_run_{s}.jsonl"] for s in SETS},
    "Coder-1.5B base, zero-shot SQL": {s: [f"baseline_coder1.5b_run_{s}.jsonl"] for s in SETS},
    "sql-v3 alone": {"eval": ["v3_eval_single.jsonl"], "places": ["v3_places_single.jsonl"]},
    "sql-v4 alone": {s: [f"v4_{s}_single.jsonl"] for s in SETS},
    "sql-v5 alone": {s: [f"v5_{s}_single.jsonl"] for s in SETS},
    "sql-v6 alone": {s: [f"v6_{s}_single.jsonl"] for s in SETS},
    "Harness, router + sql-v4": {s: [f"r4_{k}_{s}.jsonl" for k in (1, 2, 3)]
                                 for s in ("eval", "places", "multi")}
                                | {"group": ["group_r_3.jsonl", "group_r_4.jsonl"]},
    "Harness, router + sql-v5": {s: [f"r10_{k}_{s}.jsonl" for k in (1, 2)]
                                 for s in ("eval", "places", "multi")}
                                | {"group": ["group_r_5.jsonl", "group_r_6.jsonl"]},
    "Harness, router + sql-v6": {s: [f"r11_{k}_{s}.jsonl" for k in (1, 2)]
                                 for s in ("eval", "places", "multi", "list")}
                                | {"group": ["group_r_7.jsonl", "group_r_8.jsonl"]},
    # The same harness on the hosting machine (RTX 4080 SUPER 16 GB): planner Qwen3-8B-FP8.
    "Harness, router + sql-v6, FP8 planner": {s: [f"host_fp8_{k}_{s}.jsonl" for k in (1, 2)]
                                              for s in SETS},
}


def questions_by_id() -> dict:
    out = {}
    for f in EVAL_DIR.glob("*_questions.json"):
        out.update({q["id"]: q for q in json.loads(f.read_text())})
    return out


def rescore(record: dict, expect: str, ref_rows: list | None) -> bool:
    answer = record.get("answer") or ""
    if expect == "reply":
        return record.get("route") == "reply" or bool(CANNOT_RE.search(answer))
    if expect == "clarify":
        return bool(CLARIFY_RE.search(answer))
    if "sql" in record and "answer" not in record:  # one-query run: the result rows decide
        return bool(record["correct"])
    data_ok = bool(record.get("correct")) and "answer" in record and record.get("calls") is not None
    return data_ok or answer_has_reference(answer, ref_rows or [], record["question"])


def main() -> None:
    answers = {}
    for f in SETS.values():
        answers.update(json.loads((EVAL_DIR / f).read_text()))
    meta = questions_by_id()
    decline_ids = [q for q, m in meta.items() if m.get("expect") in ("reply", "clarify")]
    print(f"{'setup':34}" + "".join(f"{c:>18}" for c in COLUMNS))
    for setup, files in SETUPS.items():
        cells = []
        for column in COLUMNS:
            s = column.split()[0]
            runs = [LOGS / f for f in files.get(s, []) if (LOGS / f).exists()]
            if not runs:
                cells.append("-")
                continue
            ok = n = 0
            for path in runs:
                records = [json.loads(line) for line in path.open()]
                seen = {r["id"] for r in records}
                for r in records:
                    expect = r.get("expect") or meta.get(r["id"], {}).get("expect", "answer")
                    if expect == "skip" or (column == "group data") != (expect == "answer") \
                            and s == "group":
                        continue
                    ok += rescore(r, expect, answers.get(r["id"], {}).get("rows"))
                    n += 1
                if column == "group decline":  # not asked at all (one-query SQL): wrong
                    n += sum(q not in seen for q in decline_ids)
            per_run = n // len(runs)
            cells.append(f"{ok / len(runs):.1f}/{per_run} ({100 * ok / max(n, 1):.0f}%)")
        print(f"{setup:34}" + "".join(f"{c:>18}" for c in cells))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
