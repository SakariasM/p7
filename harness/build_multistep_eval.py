"""
Write eval/multistep_questions.json: questions that need several data steps or arithmetic,
to test whether the planner earns its place (compare eval_sql.py agent --mode planner,
planner-8b, student and single). Interim set until the group's own test questions exist.

Patterns, all on the held-out eval dates (22-24 September 2026, never in training):
  day_change      a place on two days: both counts and the difference
  compare_places  which of two places had more vessels of a type, with both counts
  union_ports     distinct vessels in any of three ports (a vessel in two counts once)
  type_share      percentage of a place's vessels that were of a type
  daily_average   average daily number of vessels of a type in a place over three days
  country_calls   passenger ships >= 250 m (cruise ships) that called at a country's ports
Each reference is one SQL query combining the parts, so eval_sql.py answers can compute it;
grade with the "answer" check (every reference value must appear in the final answer).

    cd harness
    python3 build_multistep_eval.py
    python3 eval_sql.py answers --questions eval/multistep_questions.json \\
        --answers eval/multistep_answers.json
"""

import json
import random
from datetime import date
from pathlib import Path

import functions
import places
import sql_templates as t
from ais_tool import query_ais_data

OUT = Path(__file__).parent / "eval" / "multistep_questions.json"
DAYS = [date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
PER_PATTERN = 5
TYPES = ["Cargo", "Fishing", "Passenger", "Tanker", "Sailing", "Tug", "Pleasure"]


def usable(sql: str) -> bool:
    r = query_ais_data(sql)
    return ("error" not in r and len(r["rows"]) == 1
            and all(v not in (None, 0, 0.0) for v in r["rows"][0]))


def count_sql(day: date, place: str, ship_type: str | None = None) -> str:
    extra = f" AND ship_type = '{ship_type}'" if ship_type else ""
    return (f"SELECT count(DISTINCT mmsi) AS n FROM vessel_positions "
            f"WHERE date = {t.d_sql(day)} AND {t.box_sql(place)}{extra}")


def day_change(rng):
    p = rng.choice(t.BIG_PLACES)
    d1, d2 = sorted(rng.sample(DAYS, 2))
    pt, _ = t.place_text(p)
    q = (f"How did the number of vessels in {pt} change from {t.d_text(d1, rng)} to "
         f"{t.d_text(d2, rng)}? Give both counts and the difference.")
    sql = (f"WITH a AS ({count_sql(d1, p)}), b AS ({count_sql(d2, p)})\n"
           f"SELECT a.n AS first_day, b.n AS second_day, b.n - a.n AS change FROM a, b")
    return q, sql


def compare_places(rng):
    p1, p2 = rng.sample(t.BIG_PLACES, 2)
    st, d = rng.choice(TYPES), rng.choice(DAYS)
    plural = t.SHIP_TYPES[st][0]
    q = (f"Were there more {plural} in {t.place_text(p1)[0]} or in {t.place_text(p2)[0]} on "
         f"{t.d_text(d, rng)}? Give the number in each.")
    sql = (f"WITH a AS ({count_sql(d, p1, st)}), b AS ({count_sql(d, p2, st)})\n"
           f"SELECT CASE WHEN a.n > b.n THEN '{p1}' WHEN b.n > a.n THEN '{p2}' ELSE NULL END "
           f"AS more_in, a.n AS n_first, b.n AS n_second FROM a, b")
    return q, sql


def union_ports(rng):
    ports = [p for p in t.BIG_PLACES if places.find(p).kind == "port"]
    chosen, d = rng.sample(ports, 3), rng.choice(DAYS)
    names = [t.place_text(p)[0] for p in chosen]
    q = (f"How many different vessels were in {names[0]}, {names[1]} or {names[2]} on "
         f"{t.d_text(d, rng)}? Count a vessel that was in more than one of them once.")
    boxes = " OR ".join(f"({t.box_sql(p)})" for p in chosen)
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_positions "
           f"WHERE date = {t.d_sql(d)} AND ({boxes})")
    return q, sql


def type_share(rng):
    p, st, d = rng.choice(t.BIG_PLACES), rng.choice(TYPES), rng.choice(DAYS)
    q = (f"What percentage of the vessels in {t.place_text(p)[0]} on {t.d_text(d, rng)} were "
         f"{t.SHIP_TYPES[st][0]}?")
    sql = (f"SELECT 100.0 * count(DISTINCT mmsi) FILTER (WHERE ship_type = '{st}') "
           f"/ count(DISTINCT mmsi) AS pct FROM vessel_positions "
           f"WHERE date = {t.d_sql(d)} AND {t.box_sql(p)}")
    return q, sql


def daily_average(rng):
    p, st = rng.choice(t.BIG_PLACES), rng.choice(TYPES)
    q = (f"On average, how many {t.SHIP_TYPES[st][0]} were in {t.place_text(p)[0]} per day "
         f"from 22 to 24 September 2026?")
    parts = " UNION ALL ".join(f"({count_sql(d, p, st)})" for d in DAYS)
    sql = f"SELECT avg(n) AS per_day FROM ({parts}) WHERE n > 0 HAVING count(*) = 3"
    return q, sql


def country_calls(country: str):
    code = functions.country_code(country)
    ports = [p for p in places.load(include_learned=False)
             if p.kind == "port" and p.country == code]
    boxes = " OR ".join(f"({t.box_sql(p.name)})" for p in ports)
    q = (f"How many cruise ships called at ports in {country} from 22 to 24 September 2026? "
         f"Count passenger ships of 250 m or longer as cruise ships.")
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_positions "
           f"WHERE date BETWEEN {t.d_sql(DAYS[0])} AND {t.d_sql(DAYS[-1])} "
           f"AND ship_type = 'Passenger' AND length >= {functions.CRUISE_MIN_LENGTH_M} "
           f"AND sog < {functions.PORT_CALL_SOG_KN} AND ({boxes})")
    return q, sql


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it yourself if you want to rebuild it")
    rng = random.Random(11)
    questions = []
    for pattern in (day_change, compare_places, union_ports, type_share, daily_average):
        made, attempts = 0, 0
        while made < PER_PATTERN and attempts < PER_PATTERN * 20:
            attempts += 1
            q, sql = pattern(rng)
            if any(x["question"] == q for x in questions) or not usable(sql):
                continue
            made += 1
            questions.append({"id": f"{pattern.__name__}_{made:02d}", "question": q,
                              "tags": [pattern.__name__, "multistep"], "sql": sql})
        print(f"{pattern.__name__:16} {made} questions ({attempts} attempts)", flush=True)
    for i, country in enumerate(("Denmark", "Sweden", "Germany"), 1):
        q, sql = country_calls(country)
        if usable(sql):
            questions.append({"id": f"country_calls_{i:02d}", "question": q,
                              "tags": ["country_calls", "multistep"], "sql": sql})
    print(f"country_calls    {sum('country_calls' in x['tags'] for x in questions)} questions")
    OUT.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(questions)} questions to {OUT}")


if __name__ == "__main__":
    main()
