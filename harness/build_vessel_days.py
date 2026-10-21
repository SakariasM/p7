"""
Write the vessel_days table: one row per vessel per day, from the cleaned positions
(vessel_positions), so questions over months or the whole year (ships per day, the largest
ship in May, port calls at Bornholm in 2025) are small queries instead of scans of ~15
million rows per day. data/derived/vessel_days/date=YYYY-MM-DD/vessel_days.parquet.

Per vessel and day:
  name, ship_type, imo, destination   most common value that day (missing text left out)
  length, width                       largest plausible value (length 1-450 m, width 1-80 m;
                                      longer values are AIS errors), draught the largest
  reports, first_ts, last_ts          distinct report times, first and last report
  distance_km                         sum of the steps between consecutive reports at most
                                      an hour apart
  max_sog, avg_sog                    highest and average speed over the reports (knots)
  places                              named places (places.json) with at least one report
  still_in                            ... where it lay still (sog < 0.5): docked / harbored
  moving_in                           ... where it moved (sog >= 1): e.g. under a bridge

The place lists follow places.json at build time (PLACES_VERSION in _build.json); after
places change, the lists only know the new places once the days are built again. Days
whose file exists are skipped, so a failed run is started again; files are written to
.partial and renamed when complete.

    python3 build_vessel_days.py [--task I --tasks N]   # day i where i % N == I (SLURM array)
"""

import argparse
import hashlib
import json
import time

import duckdb

import places
from ais_tool import _DATES, CLEAN_DIR, PROJECT_DIR

OUT_DIR = PROJECT_DIR / "data" / "derived" / "vessel_days"
MAX_LENGTH_M, MAX_WIDTH_M = 450, 80
MAX_STEP = "INTERVAL 1 HOUR"

HAVERSINE = """
CREATE MACRO haversine_m(lat1, lon1, lat2, lon2) AS
  2 * 6371008.8 * asin(sqrt(
      pow(sin(radians(lat2 - lat1) / 2), 2)
    + cos(radians(lat1)) * cos(radians(lat2)) * pow(sin(radians(lon2 - lon1) / 2), 2)));
"""


def places_table(con: duckdb.DuckDBPyConnection) -> str:
    known = places.load(include_learned=False)
    rows = [(p.name, *(float(places.coord(x)) for x in p.box)) for p in known]
    con.execute("CREATE OR REPLACE TEMP TABLE pl (name VARCHAR, lon0 DOUBLE, lat0 DOUBLE, "
                "lon1 DOUBLE, lat1 DOUBLE)")
    con.executemany("INSERT INTO pl VALUES (?, ?, ?, ?, ?)", rows)
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:12]


def build_day(con: duckdb.DuckDBPyConnection, day: str) -> dict:
    source = CLEAN_DIR / f"date={day}" / "ais.parquet"
    target = OUT_DIR / f"date={day}" / "vessel_days.parquet"
    partial = target.with_suffix(".parquet.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    missing = "('Undefined', 'Unknown value', '')"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE p AS
        SELECT mmsi, ts, lat, lon, sog, name, ship_type, imo, destination, length, width,
               draught FROM read_parquet('{source}')
    """)
    con.execute(f"""
        COPY (
          WITH steps AS (
            SELECT mmsi, ts, lat, lon, lag(lat) OVER w AS plat, lag(lon) OVER w AS plon,
                   lag(ts) OVER w AS pts
            FROM p WINDOW w AS (PARTITION BY mmsi ORDER BY ts)
          ), dist AS (
            SELECT mmsi, round(sum(haversine_m(plat, plon, lat, lon))
                         FILTER (WHERE pts IS NOT NULL AND ts - pts <= {MAX_STEP}) / 1000, 2)
                         AS distance_km
            FROM steps GROUP BY mmsi
          ), base AS (
            SELECT mmsi,
                   mode(name) FILTER (WHERE name NOT IN {missing}) AS name,
                   mode(ship_type) FILTER (WHERE ship_type NOT IN {missing}) AS ship_type,
                   mode(imo) FILTER (WHERE imo > 0) AS imo,
                   mode(destination) FILTER (WHERE destination NOT IN {missing}) AS destination,
                   max(length) FILTER (WHERE length BETWEEN 1 AND {MAX_LENGTH_M}) AS length,
                   max(width) FILTER (WHERE width BETWEEN 1 AND {MAX_WIDTH_M}) AS width,
                   max(draught) FILTER (WHERE draught > 0) AS draught,
                   count(DISTINCT ts) AS reports, min(ts) AS first_ts, max(ts) AS last_ts,
                   max(sog) AS max_sog, round(avg(sog), 2) AS avg_sog
            FROM p GROUP BY mmsi
          ), inside AS (
            SELECT p.mmsi, pl.name AS place, bool_or(p.sog < 0.5) AS still,
                   bool_or(p.sog >= 1) AS moving
            FROM p JOIN pl ON p.lon BETWEEN pl.lon0 AND pl.lon1
                          AND p.lat BETWEEN pl.lat0 AND pl.lat1
            GROUP BY 1, 2
          ), lists AS (
            SELECT mmsi, list(place ORDER BY place) AS places,
                   list(place ORDER BY place) FILTER (WHERE still) AS still_in,
                   list(place ORDER BY place) FILTER (WHERE moving) AS moving_in
            FROM inside GROUP BY mmsi
          )
          SELECT b.*, d.distance_km,
                 coalesce(l.places, []) AS places, coalesce(l.still_in, []) AS still_in,
                 coalesce(l.moving_in, []) AS moving_in
          FROM base b LEFT JOIN dist d USING (mmsi) LEFT JOIN lists l USING (mmsi)
          ORDER BY mmsi
        ) TO '{partial}' (FORMAT parquet, COMPRESSION zstd)
    """)
    partial.rename(target)
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{target}')").fetchone()[0]
    return {"date": day, "vessels": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--tasks", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory", default="24GB", help="DuckDB memory limit")
    p.add_argument("--spill", default="/tmp/duckdb_spill", help="DuckDB spill directory")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}; SET enable_progress_bar = false; "
                f"SET memory_limit = '{args.memory}'; SET temp_directory = '{args.spill}'; "
                "SET preserve_insertion_order = false")
    con.execute(HAVERSINE)
    version = places_table(con)
    build_file = OUT_DIR / "_build.json"
    if args.task == 0:
        build_file.write_text(json.dumps({"places_version": version,
                                          "places": con.execute("SELECT count(*) FROM pl")
                                          .fetchone()[0]}) + "\n")
    days = [d for i, d in enumerate(_DATES) if i % args.tasks == args.task]
    for day in days:
        if (OUT_DIR / f"date={day}" / "vessel_days.parquet").exists():
            print(f"[skip] {day}", flush=True)
            continue
        start = time.time()
        stats = build_day(con, day)
        stats["seconds"] = round(time.time() - start, 1)
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
