"""
Text-to-SQL training data for the student model, with the served Qwen3-8B as teacher.

    cd harness
    python3 gen_sql_data.py templates --per-template 60   # -> OUT/templates.jsonl
    python3 gen_sql_data.py freeform --prompts 300         # -> OUT/freeform.jsonl
    python3 gen_sql_data.py planner --per-template 60      # -> OUT/planner.jsonl
    python3 gen_sql_data.py export [--export-dir DIR]      # -> DIR/train.jsonl, DIR/val.jsonl

OUT is $P7_PROJECT_DIR/data/text2sql unless --out-dir is given (before the subcommand).
On the cluster run it as a CPU job:
`sbatch slurm/gen_sql_data.slurm templates --per-template 60` from the repo root.

templates: sql_templates.py fills each template with sampled real values; the SQL must run
through query_ais_data and give a non-trivial answer, then the teacher writes three
rewrites (casual, keyword, analyst instruction) that must keep every date, place, name,
ship type and number.

freeform: the teacher writes questions around sampled values, then answers each with
several SQL samples; a question is kept when enough samples give the same usable result.

planner: the questions the student really gets come from the planner (planner.py), so each
template question goes to the real planner prompt and tool; when the planner sends exactly
one ask_data request that keeps every specific and condition, the pair (that request,
template SQL) is kept. Splits and drifted requests are counted, not kept.

All three append, so a run can be resubmitted with another --seed. export deduplicates, drops
anything touching the eval dates or repeating an eval question, splits train/val by SQL
(rewrites of one query stay together), and writes chat-format JSONL with
schema_context.md as the system prompt. It reads OUT and writes to --export-dir (default
OUT) and will not overwrite an existing train/val there without --overwrite.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from agent import MODEL_PATH, get_base_url
from ais_tool import PROJECT_DIR, connect, extract_sql
from build_context import PLACES
from eval_sql import CONTEXT_FILE, load_questions, results_match, run_sql
from sql_templates import EXCLUDED_DATE_RE, TEMPLATES, Sampler, d_text, place_text

OUT_DIR = PROJECT_DIR / "data" / "text2sql"
MAX_ANSWER_ROWS = 50
# Rewrites that read like SQL teach the student nothing about real questions.
SQL_LEAK_RE = re.compile(
    r"vessel_positions|\bais\b table|ship_type|nav_status|haversine|DATE '|\bselect\b|"
    r"group by|order by|\bwhere\b.*=|\btable\b",
    re.IGNORECASE,
)

PARAPHRASE_PROMPT = """\
Question: {question}

SQL that answers it:
{sql}

Write 3 rewrites of the question that the same SQL answers exactly:
1. how a casual user would ask it,
2. a short keyword-style version,
3. a precise instruction as an analyst would give it (e.g. "Count the distinct vessels that ...").
Keep every date, time, place, vessel name, ship type, number and condition (such as
"underway", "moving" or "on every day") from the question, and add none. Do not mention SQL, tables,
columns or functions. Reply with only a JSON array of 3 strings."""

QUESTION_SYSTEM = "You write questions for testing a text-to-SQL system on this database.\n\n"

QUESTION_PROMPT = """\
Real values from the data:
- dates: {d0} and {d1}
- vessels seen on {d0}: {v0} ({t0}), {v1} ({t1})
- place: {place}
Focus: {focus}

Write 5 different questions a maritime analyst could ask about this data. Each must be
answerable by one SQL query on the tables above, cover at most a few days, and have exactly
one correct answer. Use the values above and always give the day and month. Reply with only
a JSON array of 5 strings."""

FOCUS = [
    "counting vessels in a time window",
    "speeds of a ship type or in an area",
    "one named vessel's movements during a day",
    "busiest times or places",
    "comparisons between ship types",
    "vessel size, draught and declared destination",
    "navigational status",
    "rankings: top N vessels, ship types or hours",
    "changes across several consecutive days",
    "distance travelled, or distances between vessels",
    "when vessels first or last appeared",
    "raw data quality: duplicates, base stations, bad positions",
]


class Teacher:
    """Chat calls to the served model. Waits out the twice-daily vLLM restart."""

    def __init__(self, base_url: str | None, model: str | None):
        self.base_url = base_url
        self.model = model or MODEL_PATH
        self._client: OpenAI | None = None
        self._lock = threading.Lock()

    def _client_now(self) -> OpenAI:
        with self._lock:
            if self._client is None:
                url = self.base_url or get_base_url()
                self._client = OpenAI(base_url=url, api_key="not-needed", timeout=300)
            return self._client

    def chat(self, messages: list[dict], n: int = 1, temperature: float = 0.7) -> list[str]:
        response = self.create(messages=messages, n=n, temperature=temperature)
        return [c.message.content or "" for c in response.choices]

    def create(self, **kwargs):
        """chat.completions.create on the served model, thinking off."""
        for _ in range(60):
            try:
                return self._client_now().chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    **kwargs,
                )
            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                if isinstance(e, APIStatusError) and e.status_code < 500:
                    raise
                print(f"teacher unavailable ({type(e).__name__}), retrying in 60 s",
                      file=sys.stderr, flush=True)
            time.sleep(60)
            with self._lock:
                self._client = None  # the endpoint file may point at a new node
        raise RuntimeError("teacher unavailable for an hour")


def parse_list(text: str) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    try:
        items = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def usable(result: dict) -> bool:
    """Ran, has rows, not too many, and not only zeros and NULLs."""
    if "error" in result or not result["rows"] or result["row_count"] > MAX_ANSWER_ROWS:
        return False
    return any(v not in (None, 0, 0.0) for row in result["rows"] for v in row)


def append(path, records: list[dict]) -> None:
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def make_instances(args) -> list:
    """per_template instances of every template, shuffled, from values sampled with args.seed."""
    rng = random.Random(args.seed)
    sampler = Sampler(connect(), rng, args.days)
    instances = []
    for template in TEMPLATES:
        made = 0
        for _ in range(args.per_template * 3):
            if made == args.per_template:
                break
            inst = template(sampler)
            if inst is None:
                continue
            if not inst.keeps_specifics(inst.question) or EXCLUDED_DATE_RE.search(inst.sql):
                sys.exit(f"template bug in {inst.family}: {inst.question}\n{inst.sql}")
            instances.append(inst)
            made += 1
    rng.shuffle(instances)
    print(f"{len(instances)} instances from {len(TEMPLATES)} templates", flush=True)
    return instances


def cmd_templates(args) -> None:
    instances = make_instances(args)
    stats: Counter = Counter()
    teacher = Teacher(args.base_url, args.model)

    def work(inst):
        result = run_sql(inst.sql)
        if not usable(result):
            return inst, None, 0, []
        prompt = PARAPHRASE_PROMPT.format(question=inst.question, sql=inst.sql)
        reply = teacher.chat([{"role": "user", "content": prompt}], temperature=0.8)[0]
        rewrites = parse_list(reply)
        kept = [p for p in rewrites if inst.keeps_specifics(p) and not SQL_LEAK_RE.search(p)]
        return inst, result, len(rewrites), kept

    out = args.out_dir / "templates.jsonl"
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, inst) for inst in instances]
        for done, future in enumerate(as_completed(futures), 1):
            inst, result, n_rewrites, kept = future.result()
            stats[inst.family, "instances"] += 1
            if result is None:
                continue
            stats[inst.family, "usable"] += 1
            stats[inst.family, "rewrites"] += n_rewrites
            stats[inst.family, "kept"] += len(kept)
            append(out, [
                {"question": q, "sql": inst.sql, "source": "template", "family": inst.family,
                 "variant": k, "answer": result["rows"][:5]}
                for k, q in enumerate([inst.question, *kept])
            ])
            if done % 100 == 0:
                print(f"{done}/{len(instances)} done", flush=True)

    print(f"\n{'family':26} inst usable rewrites kept")
    for family in sorted({f for f, _ in stats}):
        print(f"{family:26} {stats[family, 'instances']:4} {stats[family, 'usable']:6} "
              f"{stats[family, 'rewrites']:8} {stats[family, 'kept']:4}")
    print(f"appended to {out}")


def agreeing(sqls: list[str], results: list[dict], min_agree: int) -> tuple[str, dict] | None:
    """The shortest SQL from the largest group of samples with the same usable result."""
    groups: list[list[int]] = []
    for i, res in enumerate(results):
        if not usable(res):
            continue
        for group in groups:
            ref = results[group[0]]["rows"]
            if results_match(ref, res["rows"]) or results_match(res["rows"], ref):
                group.append(i)
                break
        else:
            groups.append([i])
    best = max(groups, key=len, default=[])
    if len(best) < min_agree:
        return None
    i = min(best, key=lambda k: len(sqls[k]))
    return sqls[i], results[i]


def cmd_freeform(args) -> None:
    rng = random.Random(args.seed)
    sampler = Sampler(connect(), rng, args.days)
    context = CONTEXT_FILE.read_text()
    question_system = QUESTION_SYSTEM + context.split("\n\n", 1)[1]

    prompts = []
    while len(prompts) < args.prompts:
        d0, d1 = sampler.day(), sampler.day()
        v0, v1 = sampler.seed(d0), sampler.seed(d0)
        if v0 is None or v1 is None or v0.mmsi == v1.mmsi:
            continue
        prompts.append(QUESTION_PROMPT.format(
            d0=d_text(d0, rng), d1=d_text(d1, rng),
            v0=v0.name, t0=v0.ship_type, v1=v1.name, t1=v1.ship_type,
            place=place_text(rng.choice(list(PLACES)))[0],
            focus=rng.choice(FOCUS),
        ))
    print(f"{len(prompts)} prompts", flush=True)

    teacher = Teacher(args.base_url, args.model)

    def work(prompt):
        reply = teacher.chat([{"role": "system", "content": question_system},
                              {"role": "user", "content": prompt}], temperature=0.9)[0]
        records, n_questions = [], 0
        for question in parse_list(reply)[:5]:
            n_questions += 1
            samples = teacher.chat([{"role": "system", "content": context},
                                    {"role": "user", "content": question}],
                                   n=args.samples, temperature=0.7)
            sqls = [extract_sql(s) for s in samples]
            if any(EXCLUDED_DATE_RE.search(s) for s in sqls):
                continue
            picked = agreeing(sqls, [run_sql(s) for s in sqls], args.min_agree)
            if picked:
                sql, result = picked
                records.append({"question": question, "sql": sql, "source": "freeform",
                                "family": "freeform", "variant": 0, "answer": result["rows"][:5]})
        return n_questions, records

    out = args.out_dir / "freeform.jsonl"
    n_questions = n_kept = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, p) for p in prompts]
        for done, future in enumerate(as_completed(futures), 1):
            n, records = future.result()
            n_questions += n
            n_kept += len(records)
            append(out, records)
            if done % 20 == 0:
                print(f"{done}/{len(prompts)} prompts, kept {n_kept}/{n_questions}", flush=True)
    print(f"kept {n_kept}/{n_questions} questions; appended to {out}")


# "How many ships" vs "which ship": a request must ask the same kind of thing as its question.
COUNT_RE = re.compile(
    r"\bhow many\b(?!\s+(kilomet|km\b|met|nautical|hours?\b|minutes?\b))|\bnumber of\b|"
    r"\bcount\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b20\d\d\b")
SQL_YEAR_RE = re.compile(r"DATE '(\d{4})-")


def request_drift(question: str, request: str, sql: str) -> str | None:
    """Why a planner request no longer matches the template SQL, or None."""
    if bool(COUNT_RE.search(question)) != bool(COUNT_RE.search(request)):
        return "asks a different kind of answer"
    if set(YEAR_RE.findall(request)) - set(SQL_YEAR_RE.findall(sql)):
        return "year does not match the data"
    return None


def cmd_planner(args) -> None:
    from planner import ASK_DATA_SCHEMA, PLANNER_PROMPT

    instances = make_instances(args)
    teacher = Teacher(args.base_url, args.model)

    def planner_requests(question: str) -> list[str]:
        """The ask_data requests of the planner's first tool-calling turn (one nudge)."""
        messages = [{"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": question}]
        for _ in range(2):
            # No temperature: the served model's default, as planner.run uses it.
            message = teacher.create(messages=messages, tools=[ASK_DATA_SCHEMA],
                                     tool_choice="auto").choices[0].message
            if message.tool_calls:
                requests = []
                for call in message.tool_calls:
                    try:
                        requests.append(json.loads(call.function.arguments).get("request", ""))
                    except json.JSONDecodeError:
                        requests.append("")
                return requests
            messages += [message.model_dump(exclude_none=True),
                         {"role": "user", "content": "Call ask_data now for the next step."}]
        return []

    def work(inst):
        result = run_sql(inst.sql)
        if not usable(result):
            return inst, "unusable", None
        requests = planner_requests(inst.question)
        if len(requests) != 1:
            return inst, "no_call" if not requests else "split", None
        request = requests[0].strip()
        if (not inst.keeps_specifics(request) or SQL_LEAK_RE.search(request)
                or request_drift(inst.question, request, inst.sql)):
            return inst, "drifted", request
        return inst, "kept", {
            "question": request, "sql": inst.sql, "source": "planner", "family": inst.family,
            "variant": 0, "planner_of": inst.question, "answer": result["rows"][:5],
        }

    out = args.out_dir / "planner.jsonl"
    stats: Counter = Counter()
    drifted = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, inst) for inst in instances]
        for done, future in enumerate(as_completed(futures), 1):
            inst, outcome, payload = future.result()
            stats[inst.family, outcome] += 1
            if outcome == "kept":
                append(out, [payload])
            elif outcome == "drifted" and len(drifted) < 15:
                drifted.append((inst.question, payload))
            if done % 100 == 0:
                print(f"{done}/{len(instances)} done", flush=True)

    outcomes = ["kept", "drifted", "split", "no_call", "unusable"]
    print(f"\n{'family':26} " + " ".join(f"{o:>8}" for o in outcomes))
    for family in sorted({f for f, _ in stats}):
        print(f"{family:26} " + " ".join(f"{stats[family, o]:8}" for o in outcomes))
    print("total" + " " * 22 + " ".join(
        f"{sum(v for (_, o), v in stats.items() if o == oc):8}" for oc in outcomes))
    print("\nexamples of drifted requests (question -> planner request):")
    for question, request in drifted:
        print(f"  {question}\n    -> {request}")
    print(f"appended to {out}")


def cmd_export(args) -> None:
    context = CONTEXT_FILE.read_text()
    eval_texts = {
        " ".join(t.lower().split())
        for q in load_questions() for t in [q["question"], *q.get("paraphrases", [])]
    }
    export_dir = args.export_dir or args.out_dir
    targets = [export_dir / "train.jsonl", export_dir / "val.jsonl"]
    if any(t.exists() for t in targets) and not args.overwrite:
        sys.exit(f"{export_dir} already has train/val; pick another --export-dir or --overwrite")
    export_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("templates.jsonl", "freeform.jsonl", "planner.jsonl"):
        path = args.out_dir / name
        if path.exists():
            records += [json.loads(line) for line in path.open()]

    seen, kept = set(), []
    for r in records:
        key = " ".join(r["question"].lower().split())
        if key in seen or key in eval_texts or EXCLUDED_DATE_RE.search(r["sql"]):
            continue
        seen.add(key)
        kept.append(r)

    def is_val(r) -> bool:
        sql = " ".join(r["sql"].split())
        return int(hashlib.sha1(sql.encode()).hexdigest(), 16) % 100 < args.val_pct

    splits = {"train": [r for r in kept if not is_val(r)], "val": [r for r in kept if is_val(r)]}
    for split, rows in splits.items():
        with (export_dir / f"{split}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["sql"]},
                ], "family": r["family"]}, ensure_ascii=False) + "\n")

    print(f"{len(records)} records, {len(kept)} after dedup/filters: "
          f"train {len(splits['train'])}, val {len(splits['val'])}")
    for source, n in sorted(Counter(r["source"] for r in kept).items()):
        print(f"  source {source:19} {n}")
    for family, n in sorted(Counter(r["family"] for r in kept).items()):
        print(f"  {family:26} {n}")
    print(f"wrote {export_dir}/train.jsonl and val.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (("templates", cmd_templates), ("freeform", cmd_freeform),
                       ("planner", cmd_planner)):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--days", type=int, default=120, help="distinct days to sample from")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--model", help="served model name (default: agent.MODEL_PATH)")
        p.add_argument("--base-url", help="OpenAI-compatible URL (default: current endpoint)")
        p.set_defaults(func=func)
    sub.choices["templates"].add_argument("--per-template", type=int, default=60)
    sub.choices["planner"].add_argument("--per-template", type=int, default=60)
    sub.choices["freeform"].add_argument("--prompts", type=int, default=300,
                                         help="teacher prompts, 5 questions each")
    sub.choices["freeform"].add_argument("--samples", type=int, default=4)
    sub.choices["freeform"].add_argument("--min-agree", type=int, default=3)
    export = sub.add_parser("export")
    export.add_argument("--val-pct", type=int, default=5)
    export.add_argument("--export-dir", type=Path, help="where train/val go (default: --out-dir)")
    export.add_argument("--overwrite", action="store_true", help="replace an existing train/val")
    export.set_defaults(func=cmd_export)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
