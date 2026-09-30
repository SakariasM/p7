"""AisStore over the cleaned Parquet partitions, queried in-process with DuckDB.

Timestamps are naive UTC throughout. Every query filters on the `date` partition column
so DuckDB only opens the days it needs.
"""

import json
import math
from datetime import datetime, timedelta
from typing import Any

import duckdb

from ais.derive import DERIVED_TABLES
from ais.models import (
    BBox,
    LineString,
    Snapshot,
    SnapshotItem,
    Stop,
    Stops,
    Track,
    TrackPoint,
    Trip,
    TripGeometry,
    Vessel,
    VesselSummary,
)
from ais.paths import DataPaths
from ais.store.base import DerivedDataMissing


def has_derived(paths: DataPaths) -> bool:
    return all(any((paths.derived / t).glob("*.parquet")) for t in DERIVED_TABLES)


def create_views(con: duckdb.DuckDBPyConnection, paths: DataPaths) -> None:
    """Define `positions`, `vessels_daily`, `vessels` and `quality` views over clean/, and
    `trips`, `stops`, `trip_geometry`, `derive_quality` over derived/ when it exists."""
    if has_derived(paths):
        for table in DERIVED_TABLES:
            name = "derive_quality" if table == "quality" else table
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS"
                f" SELECT * FROM '{paths.derived / table / '*.parquet'}'"
            )
    for table in ("positions", "quality"):
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet("
            f"'{paths.glob(table)}', hive_partitioning = true, hive_types = {{'date': DATE}})"
        )
    con.execute(
        "CREATE OR REPLACE VIEW vessels_daily AS SELECT * FROM read_parquet("
        f"'{paths.glob('vessels')}', hive_partitioning = true, hive_types = {{'date': DATE}})"
    )
    # Latest known static data per vessel across all loaded days.
    con.execute("""
        CREATE OR REPLACE VIEW vessels AS
        SELECT mmsi,
               arg_max(name, last_seen)        FILTER (name IS NOT NULL)        AS name,
               arg_max(imo, last_seen)         FILTER (imo IS NOT NULL)         AS imo,
               arg_max(callsign, last_seen)    FILTER (callsign IS NOT NULL)    AS callsign,
               arg_max(ship_type, last_seen)   FILTER (ship_type IS NOT NULL)   AS ship_type,
               arg_max(cargo_type, last_seen)  FILTER (cargo_type IS NOT NULL)  AS cargo_type,
               arg_max(mobile_type, last_seen)                                  AS mobile_type,
               arg_max(length, last_seen)      FILTER (length IS NOT NULL)      AS length,
               arg_max(width, last_seen)       FILTER (width IS NOT NULL)       AS width,
               arg_max(draught, last_seen)     FILTER (draught IS NOT NULL)     AS draught,
               arg_max(destination, last_seen) FILTER (destination IS NOT NULL) AS destination,
               arg_max(eta, last_seen)         FILTER (eta IS NOT NULL)         AS eta,
               min(first_seen) AS first_seen,
               max(last_seen)  AS last_seen
        FROM vessels_daily
        GROUP BY mmsi
    """)


def _rows(cur: duckdb.DuckDBPyConnection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    cur.execute(sql, params)
    names = [d[0] for d in cur.description or []]
    return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]


class DuckDBStore:
    def __init__(self, paths: DataPaths) -> None:
        if not any(paths.clean.glob("positions/date=*/*.parquet")):
            raise FileNotFoundError(
                f"No cleaned data under {paths.clean}. Run `uv run ais dev` first."
            )
        self._con = duckdb.connect()
        self._has_derived = has_derived(paths)
        create_views(self._con, paths)

    def _cursor(self) -> duckdb.DuckDBPyConnection:
        # One cursor per call: safe to use from FastAPI's worker threads.
        return self._con.cursor()

    def close(self) -> None:
        self._con.close()

    def vessel(self, mmsi: int) -> Vessel | None:
        rows = _rows(self._cursor(), "SELECT * FROM vessels WHERE mmsi = ?", [mmsi])
        return Vessel.model_validate(rows[0]) if rows else None

    def search_vessels(self, q: str, limit: int) -> list[VesselSummary]:
        q = q.strip()
        rows = _rows(
            self._cursor(),
            """
            SELECT mmsi, name, callsign, ship_type FROM vessels
            WHERE CAST(mmsi AS VARCHAR) LIKE ? || '%'
               OR name ILIKE '%' || ? || '%'
               OR callsign ILIKE ?
            ORDER BY CAST(mmsi AS VARCHAR) = ? DESC, name NULLS LAST, mmsi
            LIMIT ?
            """,
            [q, q, q, q, limit],
        )
        return [VesselSummary.model_validate(r) for r in rows]

    def track(self, mmsi: int, start: datetime, end: datetime, max_points: int) -> Track:
        where = "mmsi = ? AND date BETWEEN ?::DATE AND ?::DATE AND ts BETWEEN ? AND ?"
        params: list[Any] = [mmsi, start.date(), end.date(), start, end]
        cur = self._cursor()
        total = _rows(cur, f"SELECT count(*) AS n FROM positions WHERE {where}", params)[0]["n"]
        step = max(1, math.ceil(total / max_points))
        rows = _rows(
            cur,
            f"""
            SELECT ts, lat, lon, sog, cog, heading FROM positions WHERE {where}
            QUALIFY (row_number() OVER (ORDER BY ts) - 1) % ? = 0
            ORDER BY ts
            """,
            [*params, step],
        )
        return Track(
            mmsi=mmsi,
            start=start,
            end=end,
            total_points=total,
            downsampled=step > 1,
            points=[TrackPoint.model_validate(r) for r in rows],
        )

    def snapshot(self, bbox: BBox, at: datetime, lookback: timedelta, max_items: int) -> Snapshot:
        since = at - lookback
        rows = _rows(
            self._cursor(),
            """
            WITH latest AS (
                SELECT mmsi, ts, lat, lon, sog, cog, heading FROM positions
                WHERE date BETWEEN ?::DATE AND ?::DATE AND ts > ? AND ts <= ?
                QUALIFY row_number() OVER (PARTITION BY mmsi ORDER BY ts DESC) = 1
            )
            SELECT l.*, v.name, v.ship_type
            FROM latest l LEFT JOIN vessels v USING (mmsi)
            WHERE l.lon BETWEEN ? AND ? AND l.lat BETWEEN ? AND ?
            ORDER BY l.mmsi
            LIMIT ?
            """,
            [
                since.date(),
                at.date(),
                since,
                at,
                bbox.min_lon,
                bbox.max_lon,
                bbox.min_lat,
                bbox.max_lat,
                max_items + 1,
            ],
        )
        return Snapshot(
            at=at,
            lookback_min=int(lookback.total_seconds() // 60),
            truncated=len(rows) > max_items,
            items=[SnapshotItem.model_validate(r) for r in rows[:max_items]],
        )

    def _require_derived(self) -> None:
        if not self._has_derived:
            raise DerivedDataMissing("trips and stops are not built; run `uv run ais derive`")

    def trips(self, mmsi: int, start: datetime, end: datetime) -> list[Trip]:
        self._require_derived()
        rows = _rows(
            self._cursor(),
            "SELECT * FROM trips WHERE mmsi = ? AND start_ts <= ? AND end_ts >= ?"
            " ORDER BY start_ts",
            [mmsi, end, start],
        )
        return [Trip.model_validate(r) for r in rows]

    def trip_geometry(self, trip_id: str) -> TripGeometry | None:
        self._require_derived()
        rows = _rows(self._cursor(), "SELECT * FROM trip_geometry WHERE trip_id = ?", [trip_id])
        if not rows:
            return None
        geom = json.loads(rows[0]["geojson"])
        # Simplifying a line whose points coincide can collapse it to a Point.
        coords = geom["coordinates"] if geom["type"] == "LineString" else [geom["coordinates"]] * 2
        return TripGeometry(
            trip_id=trip_id,
            mmsi=rows[0]["mmsi"],
            geometry=LineString(coordinates=[(c[0], c[1]) for c in coords]),
        )

    def stops(
        self, bbox: BBox, start: datetime, end: datetime, mmsi: int | None, max_items: int
    ) -> Stops:
        self._require_derived()
        rows = _rows(
            self._cursor(),
            """
            SELECT * FROM stops
            WHERE start_ts <= ? AND end_ts >= ?
              AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?
              AND (? IS NULL OR mmsi = ?)
            ORDER BY start_ts, mmsi
            LIMIT ?
            """,
            [
                end,
                start,
                bbox.min_lon,
                bbox.max_lon,
                bbox.min_lat,
                bbox.max_lat,
                mmsi,
                mmsi,
                max_items + 1,
            ],
        )
        return Stops(
            truncated=len(rows) > max_items,
            items=[Stop.model_validate(r) for r in rows[:max_items]],
        )
