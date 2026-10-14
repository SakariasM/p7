"""
Write eval/places_questions.json: a held-out test of place handling. Single-day template
questions about many of the places in places.json, on the eval dates (22-24 September 2026),
which are never in the training data. About two thirds name a place outside the original ten
(tag new_place), the rest one of the ten (tag original_place), so models trained with the
fixed ten-place prompt can be compared with ones trained on the full place list. Reference
SQL is the template SQL; only questions with a usable answer are kept.

    cd harness
    python3 build_place_eval.py
    python3 eval_sql.py answers --questions eval/places_questions.json \\
        --answers eval/places_answers.json
    python3 eval_sql.py run --dynamic-places --questions eval/places_questions.json \\
        --answers eval/places_answers.json ...
"""

import json
import random
from datetime import date
from pathlib import Path

import sql_templates as t
from ais_tool import connect, query_ais_data

OUT = Path(__file__).parent / "eval" / "places_questions.json"
EVAL_DAYS = [date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
PER_FAMILY = 12
NEW_SHARE = 2 / 3
FAMILIES = [
    t.vessels_place_window, t.hourly_counts_place, t.busiest_hour_place,
    t.avg_speed_type_place, t.fastest_vessel_place, t.status_at_time_place,
    t.closest_pair_place,
]
ORIGINAL = {"Øresund", "Kattegat", "Copenhagen", "Aarhus", "Esbjerg", "Frederikshavn", "Skagen",
            "Gothenburg", "Helsingør", "Helsingborg"}


def usable(result: dict) -> bool:
    return ("error" not in result and bool(result["rows"]) and result["row_count"] <= 50
            and any(v not in (None, 0, 0.0) for row in result["rows"] for v in row))


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it yourself if you want to rebuild it")
    rng = random.Random(42)
    sampler = t.Sampler(connect(), rng, n_days=1)
    sampler.days = EVAL_DAYS  # the held-out eval dates, normally excluded from sampling

    questions = []
    for family in FAMILIES:
        made, attempts = 0, 0
        while made < PER_FAMILY and attempts < PER_FAMILY * 10:
            attempts += 1
            new = made < round(PER_FAMILY * NEW_SHARE)
            big_pool = [p for p in t.BIG_PLACES if (p not in ORIGINAL) == new]
            all_pool = [p for p in t.ALL_PLACES if (p not in ORIGINAL) == new]
            sampler.place = lambda big=True, b=big_pool, a=all_pool: rng.choice(b if big else a)
            inst = family(sampler)
            if inst is None or any(q["question"] == inst.question for q in questions):
                continue
            if not usable(query_ais_data(inst.sql)):
                continue
            made += 1
            questions.append({
                "id": f"{inst.family}_{made:02d}",
                "question": inst.question,
                "tags": [inst.family, "new_place" if new else "original_place"],
                "sql": inst.sql,
            })
        print(f"{family.__name__:24} {made:3} questions ({attempts} attempts)", flush=True)

    OUT.write_text(json.dumps(questions, ensure_ascii=False, indent=1) + "\n")
    n_new = sum("new_place" in q["tags"] for q in questions)
    print(f"wrote {len(questions)} questions ({n_new} new places) to {OUT}")


if __name__ == "__main__":
    main()
