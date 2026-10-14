"""
Write the cleaned position data that the vessel_positions view reads:
data/processed/ais_clean/date=YYYY-MM-DD/ais.parquet, one file per day, sorted by mmsi, ts.

Per day, from the raw DMA rows (data/processed/ais):
  1. ships only: Class A/B, ship MMSI (2xx-7xx), inside the DMA coverage box
     (the rules vessel_positions applied on the fly before)
  2. exact duplicate rows removed (the same report received by several stations)
  3. sog above MAX_SOG_KN set to NULL, row kept (incl. the 102.2 'not available' value)
  4. bad positions dropped, with the rules of src/ais/derive.py: a point farther from the
     median position of its +-MEDIAN_WINDOW neighbours than the vessel could travel at
     MAX_SOG_KN (+ MEDIAN_SLACK_M), then single spikes with an implied speed above
     MAX_SOG_KN both into and out of the point
Counts per day go to data/processed/ais_clean/_quality/<date>.json. Days whose file already
exists are skipped, so a failed run is simply started again. Files are written to .partial and
renamed when complete.

    python3 clean_positions.py [--task I --tasks N]   # day i where i % N == I (SLURM array)
"""

import argparse
import json
import time
from pathlib import Path

import duckdb

from ais_tool import _DATES, DATA_DIR, PROJECT_DIR

CLEAN_DIR = PROJECT_DIR / "data" / "processed" / "ais_clean"
QUALITY_DIR = CLEAN_DIR / "_quality"
MAX_SOG_KN = 50.0  # = MAX_IMPLIED_KN in src/ais/derive.py
MEDIAN_WINDOW = 10
MEDIAN_SLACK_M = 500.0
KN_PER_MPS = 1.943844

HAVERSINE = """
CREATE MACRO haversine_m(lat1, lon1, lat2, lon2) AS
  2 * 6371008.8 * asin(sqrt(
      pow(sin(radians(lat2 - lat1) / 2), 2)
    + cos(radians(lat1)) * cos(radians(lat2)) * pow(sin(radians(lon2 - lon1) / 2), 2)));
"""


def clean_day(con: duckdb.DuckDBPyConnection, day: str) -> dict:
    raw = DATA_DIR / f"date={day}" / "ais.parquet"
    target = CLEAN_DIR / f"date={day}" / "ais.parquet"
    partial = target.with_suffix(".parquet.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    mps = MAX_SOG_KN / KN_PER_MPS

    def count(table: str) -> int:
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ships AS
        SELECT * FROM read_parquet('{raw}')
        WHERE mobile_type IN ('Class A', 'Class B') AND mmsi BETWEEN 200000000 AND 799999999
          AND lat BETWEEN 50 AND 64 AND lon BETWEEN -5 AND 32
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dedup AS
        SELECT * REPLACE (CASE WHEN sog > {MAX_SOG_KN} THEN NULL ELSE sog END AS sog)
        FROM (SELECT DISTINCT * FROM ships)
    """)
    sog_nulled = con.execute(
        f"SELECT count(*) FROM (SELECT DISTINCT * FROM ships) WHERE sog > {MAX_SOG_KN}"
    ).fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE p0 AS
        SELECT * EXCLUDE (mlat, mlon, wstart, wend) FROM (
          SELECT *, median(lat) OVER w AS mlat, median(lon) OVER w AS mlon,
                 min(ts) OVER w AS wstart, max(ts) OVER w AS wend
          FROM dedup
          WINDOW w AS (PARTITION BY mmsi ORDER BY ts
                       ROWS BETWEEN {MEDIAN_WINDOW} PRECEDING AND {MEDIAN_WINDOW} FOLLOWING)
        )
        WHERE haversine_m(lat, lon, mlat, mlon) <= {mps} * epoch(wend - wstart) + {MEDIAN_SLACK_M}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE p1 AS
        SELECT * EXCLUDE (plat, plon, pts, nlat, nlon, nts) FROM (
          SELECT *, lag(lat) OVER w AS plat, lag(lon) OVER w AS plon, lag(ts) OVER w AS pts,
                 lead(lat) OVER w AS nlat, lead(lon) OVER w AS nlon, lead(ts) OVER w AS nts
          FROM p0 WINDOW w AS (PARTITION BY mmsi ORDER BY ts)
        )
        WHERE NOT (
          coalesce(haversine_m(plat, plon, lat, lon) / epoch(ts - pts) * {KN_PER_MPS}
                   > {MAX_SOG_KN}, false)
          AND coalesce(haversine_m(lat, lon, nlat, nlon) / epoch(nts - ts) * {KN_PER_MPS}
                       > {MAX_SOG_KN}, false)
        )
    """)
    con.execute(f"COPY (SELECT * FROM p1 ORDER BY mmsi, ts) TO '{partial}' "
                "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)")
    stats = {
        "date": day,
        "raw_rows": con.execute(f"SELECT count(*) FROM read_parquet('{raw}')").fetchone()[0],
        "ship_rows": count("ships"), "after_dedup": count("dedup"), "sog_nulled": sog_nulled,
        "after_outliers": count("p0"), "after_spikes": count("p1"),
    }
    stats["outliers"] = stats["after_dedup"] - stats["after_outliers"]
    stats["spikes"] = stats["after_outliers"] - stats["after_spikes"]
    partial.rename(target)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--tasks", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory", default="24GB", help="DuckDB memory limit")
    p.add_argument("--spill", default="/tmp/duckdb_spill", help="DuckDB spill directory "
                                                                "(node-local, not Ceph)")
    args = p.parse_args()
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}; SET enable_progress_bar = false; "
                f"SET memory_limit = '{args.memory}'; SET temp_directory = '{args.spill}'")
    con.execute(HAVERSINE)
    days = [d for i, d in enumerate(_DATES) if i % args.tasks == args.task]
    for day in days:
        if (CLEAN_DIR / f"date={day}" / "ais.parquet").exists():
            print(f"[skip] {day}", flush=True)
            continue
        start = time.time()
        stats = clean_day(con, day)
        stats["seconds"] = round(time.time() - start, 1)
        (QUALITY_DIR / f"{day}.json").write_text(json.dumps(stats) + "\n")
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
