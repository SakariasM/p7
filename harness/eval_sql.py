"""
Text-to-SQL execution accuracy on the held-out questions in eval/questions.toml.

    cd harness
    python3 eval_sql.py answers                       # reference results -> eval/answers.json
    python3 eval_sql.py run                           # current vLLM model (Qwen3-8B)
    python3 eval_sql.py run --model NAME --base-url http://host:port/v1 --out results.jsonl
    python3 eval_sql.py run --dynamic-places ...   # place list as at run time (planner.py)
    python3 eval_sql.py agent --mode planner --out planner.jsonl   # or --mode single
    python3 eval_sql.py bare --model NAME --base-url URL    # baseline: question only, no harness

`run` gives the model schema_context.md as the system prompt and each question (and each
paraphrase) as the user message, takes one SQL answer at temperature 0, runs it through
query_ais_data with the production guardrails, and compares the result with the reference.
A prediction counts as correct when, for some choice of its columns, it has the same rows
as the reference (any row order, extra columns allowed, numbers within rounding).

`agent` runs a whole agent (planner.py, or the single-model agent.py) on each question and
counts it as answered when any query the agent ran returned the reference rows ("data").
Agents often fetch a broader table and pick the answer themselves, so it also checks whether
every reference value appears in the final answer text ("answer": numbers within ~1%,
strings case-insensitive, at most 10 reference rows). Answers are in the JSONL to review.

Points (0-100, both commands):
  correctness  CORRECT_POINTS * F1, where F1 is 1 for a correct result and otherwise the row
               overlap between prediction and reference (best column matching, numbers to 2
               decimals; fewer columns than the reference scale it down); 0 on error/timeout
  speed        SPEED_POINTS * min(1, work_reference / work_query), only when F1 == 1, with
               work = rows scanned + rows through all operators (DuckDB profile). Work does
               not vary between runs or machines, unlike time; it catches a missing date
               filter (rows scanned x365) and join blow-ups (cardinality).
Wall-clock time (median of --repeats runs after a warm-up, floored at MIN_SECONDS) is still
recorded, and VES (BIRD's valid efficiency score, correct * sqrt(t_reference / t_query)) is
computed from it for comparison with the literature. In agent mode the query side is the
sum over all the agent's successful calls; F1 is 1 when the data or answer check passes,
else the best F1 of its calls; a timeout in any call forfeits the speed points.
"""

import argparse
import itertools
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 on the login node
    import tomli as tomllib

import places
from ais_tool import QUERY_TIMEOUT_SECONDS, extract_sql, query_ais_data, schema_context

HERE = Path(__file__).parent
EVAL_DIR = HERE / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.toml"
ANSWERS_FILE = HERE / "eval" / "answers.json"

CORRECT_POINTS = 70
SPEED_POINTS = 30
MIN_SECONDS = 0.05


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    """Questions from a TOML file ([[question]] tables) or a JSON list (e.g. the place set)."""
    if path.suffix == ".json":
        questions = json.loads(path.read_text())
    else:
        with path.open("rb") as f:
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


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


_NONE_RE = re.compile(r"\b(no|none|zero|not any|nobody)\b")
# Questions whose right outcome is not a number (group test set, eval/group_questions.json):
#   expect "reply"   the router answers without a query (off-topic, maps, not in AIS data)
#   expect "clarify" the answer asks for what is missing (e.g. placeholder coordinates)
#   expect "skip"    not scored (route questions, to be handled by another part of the project)
# A reply question is also answered right when another route says the data cannot answer it.
CANNOT_RE = re.compile(r"\b(?:not (?:in|included|include|contain|available|supported)|does not "
                       r"(?:include|contain|have)|cannot|can't|only answer|no information)\b",
                       re.IGNORECASE)
CLARIFY_RE = re.compile(r"\b(specify|provide|clarify|which|unknown|not (?:properly )?defined)\b",
                        re.IGNORECASE)


def answer_has_reference(answer: str, ref_rows: list[list]) -> bool:
    """Every reference value appears in the answer text (heuristic, see the module doc)."""
    if not answer or not ref_rows or len(ref_rows) > 10:
        return False
    numbers = [float(n.replace(",", "")) for n in _NUMBER_RE.findall(answer)]
    text = answer.lower()
    for value in (v for row in ref_rows for v in row):
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value == 0 and _NONE_RE.search(text):
            continue  # "there were no vessels" states a zero count
        if isinstance(value, (int, float)):
            # A negative change is usually written as "decreased by 2", so |value| also counts.
            targets = {value, abs(value)}
            if not any(math.isclose(v, n, rel_tol=1e-2, abs_tol=1e-2)
                       for v in targets for n in numbers):
                return False
        elif str(value).strip().lower() not in text:
            return False
    return True


def _row_key(row: list) -> tuple:
    return tuple(_sort_key(v) for v in row)


def result_f1(ref_rows: list[list], pred_rows: list[list]) -> float:
    """1.0 for a matching result, else the best row-overlap F1 over column matchings."""
    if results_match(ref_rows, pred_rows):
        return 1.0
    if not ref_rows or not pred_rows:
        return 0.0
    n_ref, n_pred = len(ref_rows[0]), len(pred_rows[0])
    k = min(n_ref, n_pred)
    best = 0.0
    for ref_cols in itertools.combinations(range(n_ref), k):
        ref = Counter(_row_key([r[i] for i in ref_cols]) for r in ref_rows)
        for cols in itertools.islice(itertools.permutations(range(n_pred), k), 500):
            pred = Counter(_row_key([p[j] for j in cols]) for p in pred_rows)
            matched = sum((ref & pred).values())
            f1 = 2 * matched / (len(ref_rows) + len(pred_rows)) * k / n_ref
            best = max(best, f1)
    return round(min(best, 0.99), 3)  # 1.0 is reserved for results_match


def measure(sql: str, repeats: int) -> dict:
    """Deterministic work (one profiled run) and median wall-clock time after a warm-up."""
    profiled = query_ais_data(sql, profile=True)
    if "error" in profiled:
        return {"work": None, "seconds": None}
    times = sorted(query_ais_data(sql).get("elapsed_seconds", QUERY_TIMEOUT_SECONDS)
                   for _ in range(repeats))
    return {"work": profiled["rows_scanned"] + profiled["cardinality"],
            "seconds": times[len(times) // 2]}


def score(f1: float, ref: dict | None, query: dict | None) -> dict:
    """Points (correctness * F1 + speed by work) and VES (by wall-clock), see module doc."""
    points = CORRECT_POINTS * f1
    speed_ratio, ves = None, 0.0
    if f1 == 1.0 and ref and query and ref["work"] and query["work"]:
        speed_ratio = ref["work"] / query["work"]
        points += SPEED_POINTS * min(1.0, speed_ratio)
        ves = math.sqrt(max(ref["seconds"], MIN_SECONDS) / max(query["seconds"], MIN_SECONDS))
    return {"points": round(points, 1), "f1": f1, "ves": round(ves, 3),
            "work_ratio": None if speed_ratio is None else round(speed_ratio, 4)}


class ReferenceCosts:
    """Work and time of each question's reference SQL, measured once per eval run."""

    def __init__(self, questions: list[dict], repeats: int):
        self._sql = {q["id"]: q["sql"] for q in questions if "sql" in q}
        self._repeats = repeats
        self._costs: dict[str, dict] = {}

    def __getitem__(self, qid: str) -> dict:
        if qid not in self._costs:
            self._costs[qid] = measure(self._sql[qid], self._repeats)
        return self._costs[qid]


def print_points(records: list[dict]) -> None:
    n = len(records)
    canonical = [r for r in records if r["variant"] == 0]
    print(f"points:       mean {sum(r['points'] for r in records) / n:.1f} / 100 "
          f"(canonical {sum(r['points'] for r in canonical) / len(canonical):.1f}), "
          f"mean F1 {sum(r['f1'] for r in records) / n:.3f}, "
          f"VES {sum(r['ves'] for r in records) / n:.3f}")
    tags = sorted({t for r in records for t in r["tags"]})
    for tag in tags:
        group = [r for r in records if tag in r["tags"]]
        if len(group) >= 3:
            ok = sum(r.get("correct") or r.get("answer_ok", False) for r in group)
            print(f"  {tag:22} {ok:3}/{len(group):<3} correct, "
                  f"{sum(r['points'] for r in group) / len(group):5.1f} pts")


def cmd_answers(args) -> None:
    if args.answers.exists() and not args.overwrite:
        sys.exit(f"{args.answers} exists; use --overwrite to recompute it")
    answers = {}
    for q in load_questions(args.questions):
        if q.get("expect", "answer") != "answer":
            continue
        result = run_sql(q["sql"])
        if "error" in result:
            sys.exit(f"{q['id']}: {result['error']}")
        if "check_sql" in q:
            check = run_sql(q["check_sql"])
            if "error" in check or not results_match(result["rows"], check["rows"]):
                sys.exit(f"{q['id']}: check_sql disagrees: {check.get('rows', check)}")
        answers[q["id"]] = {"columns": result["columns"], "rows": result["rows"]}
        print(f"{q['id']:24} {result['elapsed_seconds']:6.2f}s  {result['rows'][:3]}")
    args.answers.write_text(json.dumps(answers, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.answers}")


def cmd_run(args) -> None:
    from openai import OpenAI

    from agent import MODEL_PATH, get_base_url

    client = OpenAI(base_url=args.base_url or get_base_url(), api_key="not-needed")
    model = args.model or MODEL_PATH
    context = schema_context()
    known = places.load(include_learned=False)  # eval must not drift with learned places
    answers = json.loads(args.answers.read_text())

    questions = [q for q in load_questions(args.questions) if q.get("expect", "answer") == "answer"]
    ref_costs = ReferenceCosts(questions, args.repeats)
    records = []
    for q in questions:
        for i, text in enumerate([q["question"], *q.get("paraphrases", [])]):
            t = time.time()
            system = (places.with_places(context, places.mentioned(text, known))
                      if args.dynamic_places else context)
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
            f1 = 0.0 if "error" in result else result_f1(answers[q["id"]]["rows"], result["rows"])
            ref = ref_costs[q["id"]] if correct else None
            cost = measure(sql, args.repeats) if correct else None
            records.append({
                "id": q["id"], "variant": i, "question": text, "tags": q["tags"], "sql": sql,
                "correct": correct, "error": result.get("error"),
                "ref_cost": ref, "query_cost": cost, **score(f1, ref, cost),
                "rows": result.get("rows", [])[:5], "seconds": round(time.time() - t, 1),
            })
            mark = "ok " if correct else ("ERR" if "error" in result else "-- ")
            detail = (f"work x{1 / records[-1]['work_ratio']:.2f} of ref, "
                      f"{cost['seconds']:.3f}s vs {ref['seconds']:.3f}s"
                      if correct and records[-1]["work_ratio"] else f"F1 {f1}")
            print(f"{mark} {q['id']:24} v{i}  {records[-1]['points']:5.1f} pts  {detail}  "
                  f"{result.get('error', '')[:60]}", flush=True)

    n = len(records)
    n_ok = sum(r["correct"] for r in records)
    canonical = [r for r in records if r["variant"] == 0]
    print(f"\nmodel {model}")
    print(f"all variants: {n_ok}/{n} = {n_ok / n:.0%}")
    print(f"canonical:    {sum(r['correct'] for r in canonical)}/{len(canonical)}")
    print(f"errors:       {sum(r['error'] is not None for r in records)}/{n}")
    print_points(records)
    if args.out:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {args.out}")


def cmd_agent(args) -> None:
    if args.mode == "planner":
        from planner import run
    elif args.mode == "planner-8b":
        import functools

        from planner import run as planner_run
        run = functools.partial(planner_run, sql_writer="8b")
    elif args.mode == "student":
        from planner import run_student_only as run
    elif args.mode == "routed":
        from planner import run_routed as run
    else:
        from agent import run
    answers = json.loads(args.answers.read_text())
    questions = load_questions(args.questions)
    ref_costs = ReferenceCosts(questions, args.repeats)

    records = []
    for q in questions:
        expect = q.get("expect", "answer")
        if expect == "skip":
            continue
        for i, text in enumerate([q["question"], *q.get("paraphrases", [])]):
            t = time.time()
            try:
                trace = run(text, return_trace=True)
            except Exception as e:  # endpoint down, context too long, ...
                trace = {"answer": f"agent error: {e}", "tool_calls": []}
            if expect != "answer":
                text = trace["answer"] or ""
                ok = ((trace.get("route") == "reply" or bool(CANNOT_RE.search(text)))
                      if expect == "reply" else bool(CLARIFY_RE.search(text)))
                records.append({
                    "id": q["id"], "variant": i, "question": text, "tags": q["tags"],
                    "expect": expect, "correct": ok, "answer_ok": ok, "sql_seconds": 0.0,
                    "route": trace.get("route"), "ref_cost": None, "query_cost": None,
                    "points": 100.0 if ok else 0.0, "f1": 1.0 if ok else 0.0, "ves": 0.0,
                    "work_ratio": None,
                    "answer": trace["answer"], "plan": trace.get("plan"),
                    "calls": [{"request": c.get("request"), "sql": c["sql"],
                               "error": c["result"].get("error")} for c in trace["tool_calls"]],
                    "seconds": round(time.time() - t, 1),
                })
                print(f"{expect} {'ok' if ok else '--'}  {q['id']:24} v{i}  route "
                      f"{trace.get('route')}, {records[-1]['seconds']}s", flush=True)
                continue
            results = [
                json.loads(json.dumps(c["result"], default=str)) for c in trace["tool_calls"]
            ]
            correct = any(
                "error" not in r and results_match(answers[q["id"]]["rows"], r["rows"])
                for r in results
            )
            answer_ok = answer_has_reference(trace["answer"], answers[q["id"]]["rows"])
            sql_seconds = sum(
                r.get("elapsed_seconds", QUERY_TIMEOUT_SECONDS if "timeout" in r.get("error", "")
                      else 0.0)
                for r in results
            )
            ok_results = [r for r in results if "error" not in r]
            if correct or answer_ok:
                f1 = 1.0
            else:
                f1 = max((result_f1(answers[q["id"]]["rows"], r["rows"]) for r in ok_results),
                         default=0.0)
            timed_out = any("timeout" in r.get("error", "") for r in results)
            ref = query = None
            if f1 == 1.0 and not timed_out:
                ref = ref_costs[q["id"]]
                works = [query_ais_data(c["sql"], profile=True) for c in trace["tool_calls"]
                         if "error" not in c["result"]]
                query = {"work": sum(w.get("rows_scanned", 0) + w.get("cardinality", 0)
                                     for w in works),
                         "seconds": sql_seconds}
            records.append({
                "id": q["id"], "variant": i, "question": text, "tags": q["tags"],
                "correct": correct, "answer_ok": answer_ok, "sql_seconds": round(sql_seconds, 3),
                "route": trace.get("route"),
                "ref_cost": ref, "query_cost": query, **score(f1, ref, query),
                "answer": trace["answer"], "plan": trace.get("plan"),
                "calls": [{"request": c.get("request"), "sql": c["sql"],
                           "error": c["result"].get("error")} for c in trace["tool_calls"]],
                "seconds": round(time.time() - t, 1),
            })
            n_err = sum(c["error"] is not None for c in records[-1]["calls"])
            print(f"data {'ok' if correct else '--'}  answer {'ok' if answer_ok else '--'}  "
                  f"{q['id']:24} v{i}  "
                  f"{len(results)} calls, {n_err} errors, {records[-1]['seconds']}s", flush=True)

    n = len(records)
    canonical = [r for r in records if r["variant"] == 0]
    print(f"\nagent {args.mode}")
    for key in ("correct", "answer_ok"):
        label = "data  " if key == "correct" else "answer"
        n_ok = sum(r[key] for r in records)
        print(f"{label}: all {n_ok}/{n} = {n_ok / n:.0%}, "
              f"canonical {sum(r[key] for r in canonical)}/{len(canonical)}")
    either = sum(r["correct"] or r["answer_ok"] for r in records)
    print(f"either: {either}/{n}")
    print_points(records)
    print(f"calls/question: {sum(len(r['calls']) for r in records) / n:.1f}, "
          f"seconds/question: {sum(r['seconds'] for r in records) / n:.0f}")
    if args.out:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {args.out}")


def cmd_bare(args) -> None:
    """Baseline without any harness: the model gets only the question - no schema, no data,
    no tools, no planner - and answers in text. Scored like the agents: an answer question
    is right when every reference value appears in the answer, a reply question when the
    answer says it cannot answer from the data (or declines), a clarify question when it
    asks for what is missing."""
    from openai import OpenAI

    from agent import MODEL_PATH, get_base_url

    client = OpenAI(base_url=args.base_url or get_base_url(), api_key="not-needed")
    model = args.model or MODEL_PATH
    answers = json.loads(args.answers.read_text())
    records = []
    for q in load_questions(args.questions):
        expect = q.get("expect", "answer")
        if expect == "skip":
            continue
        for i, text in enumerate([q["question"], *q.get("paraphrases", [])]):
            t = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text}],
                temperature=0,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            answer = strip_think(response.choices[0].message.content or "")
            if expect == "answer":
                ok = answer_has_reference(answer, answers[q["id"]]["rows"])
            elif expect == "reply":
                ok = bool(CANNOT_RE.search(answer))
            else:
                ok = bool(CLARIFY_RE.search(answer))
            records.append({"id": q["id"], "variant": i, "question": text, "tags": q["tags"],
                            "expect": expect, "correct": ok, "answer": answer,
                            "points": 100.0 if ok else 0.0,
                            "seconds": round(time.time() - t, 1)})
            print(f"{'ok ' if ok else '-- '} {q['id']:24} v{i}  {expect:7} "
                  f"{answer[:80]!r}", flush=True)
    n = len(records)
    print(f"\nbare model {model}\nall: {sum(r['correct'] for r in records)}/{n}")
    for expect in ("answer", "reply", "clarify"):
        group = [r for r in records if r["expect"] == expect]
        if group:
            print(f"  {expect:8} {sum(r['correct'] for r in group)}/{len(group)}")
    tags = sorted({t for r in records for t in r["tags"]})
    for tag in tags:
        group = [r for r in records if tag in r["tags"]]
        if len(group) >= 3:
            print(f"  {tag:24} {sum(r['correct'] for r in group)}/{len(group)}")
    if args.out:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {args.out}")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    sub = parser.add_subparsers(dest="cmd", required=True)
    answers = sub.add_parser("answers", help="compute reference results")
    answers.add_argument("--overwrite", action="store_true")
    answers.set_defaults(func=cmd_answers)
    run = sub.add_parser("run", help="score a model")
    run.add_argument("--model", help="served model name (default: agent.MODEL_PATH)")
    run.add_argument("--base-url", help="OpenAI-compatible URL (default: current endpoint)")
    run.add_argument("--out", help="write per-question results as JSONL")
    run.add_argument("--repeats", type=int, default=3, help="timed runs per query for points")
    run.add_argument("--dynamic-places", action="store_true",
                     help="only the places a question names in the prompt, as planner.py does")
    run.set_defaults(func=cmd_run)
    bare = sub.add_parser("bare", help="baseline: the model answers from the question alone")
    bare.add_argument("--model", help="served model name (default: agent.MODEL_PATH)")
    bare.add_argument("--base-url", help="OpenAI-compatible URL (default: current endpoint)")
    bare.add_argument("--out", help="write per-question results as JSONL")
    bare.add_argument("--dynamic-places", action="store_true", help="ignored (no prompt)")
    bare.set_defaults(func=cmd_bare)
    agent = sub.add_parser("agent", help="score a whole agent end to end")
    agent.add_argument("--mode", choices=["planner", "planner-8b", "student", "single", "routed"],
                       default="planner",
                       help="planner+student, planner+8B as SQL writer, student alone, 8B alone, "
                            "router (student directly or planner)")
    agent.add_argument("--out", help="write per-question results as JSONL")
    agent.add_argument("--repeats", type=int, default=3, help="timed runs of reference SQL")
    agent.set_defaults(func=cmd_agent)
    for p in sub.choices.values():
        p.add_argument("--questions", type=Path, default=QUESTIONS_FILE,
                       help="question set (TOML or JSON; default eval/questions.toml)")
        p.add_argument("--answers", type=Path, default=ANSWERS_FILE,
                       help="its reference answers (default eval/answers.json)")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
