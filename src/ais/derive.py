"""Phase 2: clean positions -> trips, stops and simplified trip geometry in derived/.

Segmentation, per vessel, in time order:
1. Drop bad fixes. First drop points outside DMA receiver coverage (COVERAGE). Corrupt
   positions also come in short runs (e.g. a longitude off by 80 degrees for up to ~8
   messages), so then drop points farther from the median position of
   their +-MEDIAN_WINDOW neighbours than the vessel could travel at MAX_IMPLIED_KN, then drop
   remaining single spikes (implied speed too high both into and out of the point).
2. A point is stationary when SOG (or implied speed if SOG is missing) < STOP_SOG_KN.
   A run of stationary points with no gap over MAX_GAP that lasts >= MIN_STOP is a stop.
3. Everything else is split into trips at stops, at gaps over MAX_GAP and at impossible
   jumps (> MAX_IMPLIED_KN over > JUMP_MIN_M) that step 1 could not remove, such as a
   45-minute run with a dropped longitude digit. No trip contains such a step. Short pauses
   stay inside the trip. Trips need >= MIN_TRIP_POINTS points and >= MIN_TRIP_M metres.

Derived data spans days (trips cross midnight), so it is rebuilt from all of clean/ at once.
"""

import shutil
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import duckdb

from ais.paths import DataPaths

MAX_IMPLIED_KN = 50.0
MEDIAN_WINDOW = 10
# lat/lon bounds DMA's shore stations can plausibly hear; 46 of 33M dev-slice points fall outside.
# TODO: move to clean.STEPS the next time clean/ is rebuilt from raw.
COVERAGE = {"min_lat": 50.0, "max_lat": 64.0, "min_lon": -5.0, "max_lon": 32.0}
MEDIAN_SLACK_M = 500.0
JUMP_MIN_M = 500.0
STOP_SOG_KN = 0.5
MAX_GAP = timedelta(minutes=30)
MIN_STOP = timedelta(minutes=10)
MIN_TRIP_POINTS = 10
MIN_TRIP_M = 1000.0
SIMPLIFY_DEG = 0.0002  # ~20 m in latitude

DERIVED_TABLES = ("trips", "stops", "trip_geometry", "quality")
KN_PER_MPS = 1.943844

MACROS = """
CREATE OR REPLACE MACRO haversine_m(lat1, lon1, lat2, lon2) AS
  2 * 6371008.8 * asin(sqrt(
      pow(sin(radians(lat2 - lat1) / 2), 2)
    + cos(radians(lat1)) * cos(radians(lat2)) * pow(sin(radians(lon2 - lon1) / 2), 2)));
"""


@dataclass(frozen=True)
class DeriveResult:
    positions: int
    outside_coverage: int
    outliers_removed: int
    spikes_removed: int
    jump_splits: int
    stops: int
    trips: int


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    assert row is not None
    return int(row[0])


def derive(paths: DataPaths) -> DeriveResult:
    """Rebuild derived/{trips,stops,trip_geometry,quality} from clean/positions."""
    gap_s = MAX_GAP.total_seconds()
    paths.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.root, prefix=".tmp-derive-") as tmp_str:
        tmp = Path(tmp_str)
        con = duckdb.connect(str(tmp / "work.duckdb"))
        con.execute("SET enable_progress_bar = false")
        con.execute("INSTALL spatial; LOAD spatial")
        con.execute(MACROS)
        con.execute(
            f"CREATE VIEW positions AS SELECT * FROM read_parquet('{paths.glob('positions')}')"
        )
        n_positions = _count(con, "SELECT count(*) FROM positions")
        c = COVERAGE
        con.execute(f"""
            CREATE VIEW covered AS SELECT * FROM positions
            WHERE lat BETWEEN {c["min_lat"]} AND {c["max_lat"]}
              AND lon BETWEEN {c["min_lon"]} AND {c["max_lon"]}
        """)
        outside = n_positions - _count(con, "SELECT count(*) FROM covered")

        # 1a. Outliers: too far from the rolling median position for the time elapsed.
        mps = MAX_IMPLIED_KN / KN_PER_MPS
        con.execute(f"""
            CREATE TABLE p0 AS
            WITH m AS (
              SELECT mmsi, ts, lat, lon, sog,
                median(lat) OVER w AS mlat, median(lon) OVER w AS mlon,
                min(ts) OVER w AS wstart, max(ts) OVER w AS wend
              FROM covered
              WINDOW w AS (PARTITION BY mmsi ORDER BY ts
                           ROWS BETWEEN {MEDIAN_WINDOW} PRECEDING AND {MEDIAN_WINDOW} FOLLOWING)
            )
            SELECT mmsi, ts, lat, lon, sog FROM m
            WHERE haversine_m(lat, lon, mlat, mlon)
                  <= {mps} * epoch(wend - wstart) + {MEDIAN_SLACK_M}
        """)
        outliers = n_positions - outside - _count(con, "SELECT count(*) FROM p0")

        # 1b. Spikes: implied speed too high both into and out of the point.
        con.execute(f"""
            CREATE TABLE p1 AS
            WITH n AS (
              SELECT mmsi, ts, lat, lon, sog,
                lag(lat) OVER w AS plat, lag(lon) OVER w AS plon, lag(ts) OVER w AS pts,
                lead(lat) OVER w AS nlat, lead(lon) OVER w AS nlon, lead(ts) OVER w AS nts
              FROM p0 WINDOW w AS (PARTITION BY mmsi ORDER BY ts)
            )
            SELECT mmsi, ts, lat, lon, sog FROM n
            WHERE NOT (
              coalesce(haversine_m(plat, plon, lat, lon)
                       / epoch(ts - pts) * {KN_PER_MPS} > {MAX_IMPLIED_KN}, false)
              AND coalesce(haversine_m(lat, lon, nlat, nlon)
                       / epoch(nts - ts) * {KN_PER_MPS} > {MAX_IMPLIED_KN}, false)
            )
        """)
        spikes = n_positions - outside - outliers - _count(con, "SELECT count(*) FROM p1")

        # 2. Per-point step to previous point, stationary flag and stationary-run ids.
        con.execute(f"""
            CREATE TABLE p2 AS
            WITH a AS (
              SELECT *,
                epoch(ts - lag(ts) OVER w) AS dt_s,
                haversine_m(lag(lat) OVER w, lag(lon) OVER w, lat, lon) AS step_m
              FROM p1 WINDOW w AS (PARTITION BY mmsi ORDER BY ts)
            ), b AS (
              SELECT *,
                coalesce(sog, step_m / nullif(dt_s, 0) * {KN_PER_MPS}, 0) < {STOP_SOG_KN}
                  AS still,
                coalesce(step_m > {JUMP_MIN_M}
                  AND (dt_s = 0 OR step_m / dt_s * {KN_PER_MPS} > {MAX_IMPLIED_KN}), false)
                  AS jump
              FROM a
            ), c AS (
              SELECT *,
                dt_s IS NULL OR dt_s > {gap_s} OR jump
                  OR still IS DISTINCT FROM lag(still) OVER w AS new_run
              FROM b WINDOW w AS (PARTITION BY mmsi ORDER BY ts)
            )
            SELECT *, sum(new_run::INT) OVER (PARTITION BY mmsi ORDER BY ts) AS run
            FROM c
        """)

        # Stationary runs long enough to be stops.
        con.execute(f"""
            CREATE TABLE stop_runs AS
            SELECT mmsi, run, min(ts) AS start_ts, max(ts) AS end_ts,
                   median(lat) AS lat, median(lon) AS lon, count(*) AS point_count
            FROM p2 WHERE still
            GROUP BY mmsi, run
            HAVING max(ts) - min(ts) >= INTERVAL '{int(MIN_STOP.total_seconds())} seconds'
        """)

        # 3. Trip segments: break at stop membership changes and at gaps.
        con.execute(f"""
            CREATE TABLE p3 AS
            WITH a AS (
              SELECT p.*, s.run IS NOT NULL AS in_stop
              FROM p2 p LEFT JOIN stop_runs s USING (mmsi, run)
            ), b AS (
              SELECT *,
                dt_s IS NULL OR dt_s > {gap_s} OR jump
                  OR in_stop IS DISTINCT FROM lag(in_stop) OVER w AS new_seg
              FROM a WINDOW w AS (PARTITION BY mmsi ORDER BY ts)
            )
            SELECT *, sum(new_seg::INT) OVER (PARTITION BY mmsi ORDER BY ts) AS seg
            FROM b WHERE NOT in_stop
        """)
        con.execute(f"""
            CREATE TABLE trips AS
            SELECT
              mmsi || '-' || strftime(min(ts), '%Y%m%dT%H%M%S') AS trip_id,
              mmsi,
              min(ts) AS start_ts, max(ts) AS end_ts,
              arg_min(lat, ts) AS start_lat, arg_min(lon, ts) AS start_lon,
              arg_max(lat, ts) AS end_lat, arg_max(lon, ts) AS end_lon,
              epoch(max(ts) - min(ts)) AS duration_s,
              -- step_m of a segment's first point links to the previous segment; exclude it
              sum(step_m) FILTER (NOT new_seg) AS distance_m,
              count(*) AS point_count,
              mmsi AS _mmsi, seg AS _seg
            FROM p3
            GROUP BY mmsi, seg
            HAVING count(*) >= {MIN_TRIP_POINTS}
               AND coalesce(sum(step_m) FILTER (NOT new_seg), 0) >= {MIN_TRIP_M}
        """)

        con.execute(f"""
            CREATE TABLE trip_geometry AS
            SELECT t.trip_id, t.mmsi,
              ST_AsGeoJSON(ST_Simplify(
                ST_MakeLine(list(ST_Point(p.lon, p.lat) ORDER BY p.ts)), {SIMPLIFY_DEG}
              )) AS geojson
            FROM trips t JOIN p3 p ON p.mmsi = t._mmsi AND p.seg = t._seg
            GROUP BY t.trip_id, t.mmsi
        """)

        jumps = _count(con, "SELECT count(*) FROM p2 WHERE jump")
        n_stops = _count(con, "SELECT count(*) FROM stop_runs")
        n_trips = _count(con, "SELECT count(*) FROM trips")
        out = {t: tmp / t for t in DERIVED_TABLES}
        for d in out.values():
            d.mkdir()
        con.execute(
            "COPY (SELECT * EXCLUDE (_mmsi, _seg) FROM trips ORDER BY mmsi, start_ts)"
            f" TO '{out['trips'] / 'part.parquet'}' (FORMAT parquet, COMPRESSION zstd)"
        )
        con.execute(
            "COPY (SELECT mmsi, start_ts, end_ts, lat, lon,"
            " epoch(end_ts - start_ts) AS duration_s, point_count"
            " FROM stop_runs ORDER BY mmsi, start_ts)"
            f" TO '{out['stops'] / 'part.parquet'}' (FORMAT parquet, COMPRESSION zstd)"
        )
        con.execute(
            "COPY (SELECT * FROM trip_geometry ORDER BY trip_id)"
            f" TO '{out['trip_geometry'] / 'part.parquet'}' (FORMAT parquet, COMPRESSION zstd)"
        )
        con.execute(
            "COPY (SELECT * FROM (VALUES"
            f" ('positions_in', {n_positions}), ('outside_coverage', {outside}),"
            f" ('outliers_removed', {outliers}),"
            f" ('spikes_removed', {spikes}), ('jump_splits', {jumps}),"
            f" ('stops', {n_stops}), ('trips', {n_trips})) AS q(metric, value))"
            f" TO '{out['quality'] / 'part.parquet'}' (FORMAT parquet)"
        )
        con.close()

        paths.derived.mkdir(parents=True, exist_ok=True)
        for table, built in out.items():
            target = paths.derived / table
            if target.exists():
                shutil.rmtree(target)
            built.rename(target)
    return DeriveResult(n_positions, outside, outliers, spikes, jumps, n_stops, n_trips)
