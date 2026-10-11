"""
Text-to-SQL execution accuracy on the held-out questions in eval/questions.toml.

    cd harness
    python3 eval_sql.py answers                       # reference results -> eval/answers.json
    python3 eval_sql.py run                           # current vLLM model (Qwen3-8B)
    python3 eval_sql.py run --model NAME --base-url http://host:port/v1 --out results.jsonl

`run` gives the model schema_context.md as the system prompt and each question (and each
paraphrase) as the user message, takes one SQL answer at temperature 0, runs it through
query_ais_data with the production guardrails, and compares the result with the reference.
A prediction counts as correct when, for some choice of its columns, it has the same rows
as the reference (any row order, extra columns allowed, numbers within rounding).
"""

import argparse
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 on the login node
    import tomli as tomllib

from ais_tool import query_ais_data

HERE = Path(__file__).parent
QUESTIONS_FILE = HERE / "eval" / "questions.toml"
ANSWERS_FILE = HERE / "eval" / "answers.json"
CONTEXT_FILE = HERE / "schema_context.md"


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open("rb") as f:
        questions = tomllib.load(f)["question"]
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        sys.exit("duplicate question ids")
    return questions


def run_sql(sql: str) -> dict:
    """query_ais_data, with values passed through JSON so stored and fresh results compare."""
    return json.loads(json.dumps(query_ais_data(sql), default=str))


def _same_value(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-2)
    return str(a).strip().lower() == str(b).strip().lower()


def _sort_key(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (0, round(float(v), 2), "")
    return (1, 0.0, "" if v is None else str(v).strip().lower())


def _same_rows(ref: list[list], pred: list[list]) -> bool:
    if len(ref) != len(pred):
        return False
    ref_sorted = sorted(ref, key=lambda r: [_sort_key(v) for v in r])
    pred_sorted = sorted(pred, key=lambda r: [_sort_key(v) for v in r])
    return all(
        all(_same_value(a, b) for a, b in zip(r, p, strict=True))
        for r, p in zip(ref_sorted, pred_sorted, strict=True)
    )


def results_match(ref_rows: list[list], pred_rows: list[list]) -> bool:
    if len(ref_rows) != len(pred_rows):
        return False
    if not ref_rows:
        return True
    n_ref, n_pred = len(ref_rows[0]), len(pred_rows[0])
    # Candidate prediction columns for each reference column: same values as a multiset.
    candidates = [
        [
            j
            for j in range(n_pred)
            if _same_rows([[r[i]] for r in ref_rows], [[p[j]] for p in pred_rows])
        ]
        for i in range(n_ref)
    ]
    for cols in itertools.islice(itertools.product(*candidates), 1000):
        if len(set(cols)) == n_ref and _same_rows(
            ref_rows, [[p[j] for j in cols] for p in pred_rows]
        ):
            return True
    return False


def extract_sql(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    fenced = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fenced[-1] if fenced else text).strip()


def cmd_answers(_args) -> None:
    answers = {}
    for q in load_questions():
        result = run_sql(q["sql"])
        if "error" in result:
            sys.exit(f"{q['id']}: {result['error']}")
        if "check_sql" in q:
            check = run_sql(q["check_sql"])
            if "error" in check or not results_match(result["rows"], check["rows"]):
                sys.exit(f"{q['id']}: check_sql disagrees: {check.get('rows', check)}")
        answers[q["id"]] = {"columns": result["columns"], "rows": result["rows"]}
        print(f"{q['id']:24} {result['elapsed_seconds']:6.2f}s  {result['rows'][:3]}")
    ANSWERS_FILE.write_text(json.dumps(answers, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {ANSWERS_FILE}")


def cmd_run(args) -> None:
    from openai import OpenAI

    from agent import MODEL_PATH, get_base_url

    client = OpenAI(base_url=args.base_url or get_base_url(), api_key="not-needed")
    model = args.model or MODEL_PATH
    system = CONTEXT_FILE.read_text()
    answers = json.loads(ANSWERS_FILE.read_text())

    records = []
    for q in load_questions():
        for i, text in enumerate([q["question"], *q.get("paraphrases", [])]):
            t = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
                temperature=0,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            sql = extract_sql(response.choices[0].message.content)
            result = run_sql(sql)
            correct = "error" not in result and results_match(
                answers[q["id"]]["rows"], result["rows"]
            )
            records.append({
                "id": q["id"], "variant": i, "question": text, "tags": q["tags"], "sql": sql,
                "correct": correct, "error": result.get("error"),
                "rows": result.get("rows", [])[:5], "seconds": round(time.time() - t, 1),
            })
            mark = "ok " if correct else ("ERR" if "error" in result else "-- ")
            print(f"{mark} {q['id']:24} v{i}  {result.get('error', '')[:80]}", flush=True)

    n = len(records)
    n_ok = sum(r["correct"] for r in records)
    canonical = [r for r in records if r["variant"] == 0]
    print(f"\nmodel {model}")
    print(f"all variants: {n_ok}/{n} = {n_ok / n:.0%}")
    print(f"canonical:    {sum(r['correct'] for r in canonical)}/{len(canonical)}")
    print(f"errors:       {sum(r['error'] is not None for r in records)}/{n}")
    if args.out:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("answers", help="compute reference results").set_defaults(func=cmd_answers)
    run = sub.add_parser("run", help="score a model")
    run.add_argument("--model", help="served model name (default: agent.MODEL_PATH)")
    run.add_argument("--base-url", help="OpenAI-compatible URL (default: current endpoint)")
    run.add_argument("--out", help="write per-question results as JSONL")
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
