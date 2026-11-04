"""
Write eval/list_questions.json: "which ships ..." questions, which sql-v5 answered with one ship
or a sum (2026-10-22), on the held-out eval dates (22-24 September 2026, never in training).
The reference is the set of MMSIs (a prediction may add names or other columns: names vary
between reports, so they are not compared).

Patterns: at a moment in a place, in a time window, on a day, docked at a moment, docked on a
day, passing under a bridge on a day, longer than some length in a place over the three days.

    cd harness
    python3 build_list_eval.py
    python3 eval_sql.py answers --questions eval/list_questions.json --answers eval/list_answers.json
"""

import json
import random
from datetime import date
from pathlib import Path

import sql_templates as t
from ais_tool import query_ais_data

OUT = Path(__file__).parent / "eval" / "list_questions.json"
DAYS = [date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
PER_PATTERN = 3
MAX_ROWS = 1000


def usable(sql: str) -> bool:
    r = query_ais_data(sql)
    return "error" not in r and 1 <= len(r["rows"]) < MAX_ROWS


def window(d: date, hhmm: str) -> str:
    return (f"ts > {t.ts_sql(d, hhmm)} - INTERVAL 10 MINUTE AND ts <= {t.ts_sql(d, hhmm)}")


def at_moment(rng):
    p, d, hhmm = rng.choice(t.ALL_PLACES), rng.choice(DAYS), f"{rng.randint(6, 20):02d}:00"
    q = f"Which vessels were in {t.place_text(p)[0]} at {hhmm} UTC on {t.d_text(d, rng)}?"
    sql = (f"SELECT DISTINCT mmsi FROM vessel_positions WHERE date = {t.d_sql(d)} "
           f"AND {t.box_sql(p)} AND {window(d, hhmm)} ORDER BY mmsi")
    return q, sql


def in_window(rng):
    p, d, h = rng.choice(t.ALL_PLACES), rng.choice(DAYS), rng.randint(6, 19)
    st = rng.choice(["Cargo", "Tanker", "Passenger", "Fishing"])
    q = (f"Which {t.SHIP_TYPES[st][0]} were in {t.place_text(p)[0]} between {h:02d}:00 and "
         f"{h + 2:02d}:00 UTC on {t.d_text(d, rng)}?")
    sql = (f"SELECT DISTINCT mmsi FROM vessel_positions WHERE date = {t.d_sql(d)} "
           f"AND ts >= {t.ts_sql(d, f'{h:02d}:00')} AND ts < {t.ts_sql(d, f'{h + 2:02d}:00')} "
           f"AND {t.box_sql(p)} AND ship_type = '{st}' ORDER BY mmsi")
    return q, sql


def on_day(rng):
    p, d = rng.choice(t.PORTS), rng.choice(DAYS)
    q = f"What ships were seen in {t.place_text(p)[0]} on {t.d_text(d, rng)}?"
    sql = (f"SELECT DISTINCT mmsi FROM vessel_positions WHERE date = {t.d_sql(d)} "
           f"AND {t.box_sql(p)} ORDER BY mmsi")
    return q, sql


def docked_moment(rng):
    p, d, hhmm = rng.choice(t.PORTS), rng.choice(DAYS), f"{rng.randint(0, 23):02d}:30"
    q = f"Which ships were docked in {t.place_text(p)[0]} at {hhmm} on {t.d_text(d, rng)}?"
    sql = (f"SELECT mmsi FROM (SELECT mmsi, arg_max(sog, ts) AS sog FROM vessel_positions "
           f"WHERE date = {t.d_sql(d)} AND {t.box_sql(p)} AND {window(d, hhmm)} GROUP BY mmsi) "
           f"WHERE sog < 0.5 ORDER BY mmsi")
    return q, sql


def docked_day(rng):
    p, d = rng.choice(t.PORTS), rng.choice(DAYS)
    q = f"List the vessels that docked in {t.place_text(p)[0]} on {t.d_text(d, rng)}."
    sql = (f"SELECT DISTINCT mmsi FROM vessel_positions WHERE date = {t.d_sql(d)} "
           f"AND {t.box_sql(p)} AND sog < 0.5 ORDER BY mmsi")
    return q, sql


def under_bridge(rng):
    p, d = rng.choice(t.BRIDGES), rng.choice(DAYS)
    st = rng.choice(["Tanker", "Cargo", "Passenger"])
    q = f"Which {t.SHIP_TYPES[st][0]} passed under the {p} on {t.d_text(d, rng)}?"
    sql = (f"SELECT DISTINCT mmsi FROM vessel_positions WHERE date = {t.d_sql(d)} "
           f"AND {t.box_sql(p)} AND sog >= 1 AND ship_type = '{st}' ORDER BY mmsi")
    return q, sql


def long_ships(rng):
    p, length = rng.choice(t.BIG_PLACES), rng.randrange(250, 360, 10)
    q = (f"Which ships longer than {length} m were in {t.place_text(p)[0]} from 22 to 24 "
         f"September 2026?")
    # The ship's own length (its longest over the days): a position report can lack it, and
    # GREENFIELD (272 m) was left out of Cuxhaven over one such report.
    days = f"date BETWEEN {t.d_sql(DAYS[0])} AND {t.d_sql(DAYS[-1])}"
    sql = (f"SELECT DISTINCT mmsi FROM vessel_positions WHERE {days} AND {t.box_sql(p)} "
           f"AND mmsi IN (SELECT mmsi FROM vessel_days WHERE {days} GROUP BY mmsi "
           f"HAVING max(length) > {length} AND max(length) <= {t.MAX_REAL_LENGTH_M}) ORDER BY mmsi")
    return q, sql


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it yourself if you want to rebuild it")
    rng = random.Random(21)
    questions = []
    for pattern in (at_moment, in_window, on_day, docked_moment, docked_day, under_bridge,
                    long_ships):
        made, attempts = 0, 0
        while made < PER_PATTERN and attempts < PER_PATTERN * 30:
            attempts += 1
            q, sql = pattern(rng)
            if any(x["question"] == q for x in questions) or not usable(sql):
                continue
            made += 1
            questions.append({"id": f"{pattern.__name__}_{made:02d}", "question": q,
                              "tags": [pattern.__name__, "list"], "sql": sql})
        print(f"{pattern.__name__:14} {made} questions ({attempts} attempts)", flush=True)
    OUT.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(questions)} questions to {OUT}")


if __name__ == "__main__":
    main()
