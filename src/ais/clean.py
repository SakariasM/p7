"""Phase 1: typed raw rows -> cleaned positions + per-day vessel static data + quality log.

Each cleaning rule is a named step. Steps run in order and every step's effect is counted,
so the quality partition doubles as the data-quality table for the report.
"""

import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import duckdb

from ais.ingest import CLEAN_TEXT_MACRO, csv_source, typed_select
from ais.paths import CLEAN_TABLES, DataPaths

MMSI_MIN, MMSI_MAX = 200_000_000, 799_999_999
SOG_SENTINEL = 102.2

POSITION_COLUMNS = "mmsi, ts, lat, lon, sog, cog, heading, rot, nav_status"


@dataclass(frozen=True)
class Step:
    name: str
    action: Literal["drop", "null"]
    sql: str  # SELECT over the previous step's view, which is available as `prev`
    description: str


STEPS: tuple[Step, ...] = (
    Step(
        "non_vessel",
        "drop",
        "SELECT * FROM prev WHERE mobile_type IN ('Class A', 'Class B')",
        "Base stations, AtoN, SAR aircraft and other non-ship transmitters",
    ),
    Step(
        "invalid_mmsi",
        "drop",
        f"SELECT * FROM prev WHERE mmsi BETWEEN {MMSI_MIN} AND {MMSI_MAX}",
        "MMSI outside the ship range (MID 2xx-7xx)",
    ),
    Step(
        "invalid_position",
        "drop",
        "SELECT * FROM prev WHERE lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180",
        "Missing or out-of-range lat/lon, incl. the 91/181 'not available' sentinels",
    ),
    Step(
        "sog_sentinel",
        "null",
        f"SELECT * REPLACE (CASE WHEN sog >= {SOG_SENTINEL} THEN NULL ELSE sog END AS sog)"
        " FROM prev",
        "SOG >= 102.2 means 'not available'; SOG set to NULL, row kept",
    ),
    Step(
        "exact_duplicate",
        "drop",
        "SELECT mmsi, ts, lat, lon, sog, cog, heading, min(rot) AS rot,"
        " min(nav_status) AS nav_status FROM prev GROUP BY ALL",
        "Same position report received by several stations; one copy kept",
    ),
    Step(
        "conflicting_same_second",
        "drop",
        "SELECT * FROM prev QUALIFY count(*) OVER (PARTITION BY mmsi, ts) = 1",
        "Different positions for one MMSI in the same second; whole group dropped",
    ),
)

STATIC_STEP = "sog_sentinel"  # vessel static data is taken from rows after this step

VESSELS_SQL = """
SELECT mmsi,
       arg_max(name, ts)        FILTER (name IS NOT NULL)        AS name,
       arg_max(imo, ts)         FILTER (imo IS NOT NULL)         AS imo,
       arg_max(callsign, ts)    FILTER (callsign IS NOT NULL)    AS callsign,
       arg_max(ship_type, ts)   FILTER (ship_type IS NOT NULL)   AS ship_type,
       arg_max(cargo_type, ts)  FILTER (cargo_type IS NOT NULL)  AS cargo_type,
       arg_max(mobile_type, ts)                                  AS mobile_type,
       arg_max(length, ts)      FILTER (length IS NOT NULL)      AS length,
       arg_max(width, ts)       FILTER (width IS NOT NULL)       AS width,
       arg_max(draught, ts)     FILTER (draught IS NOT NULL)     AS draught,
       arg_max(destination, ts) FILTER (destination IS NOT NULL) AS destination,
       arg_max(eta, ts)         FILTER (eta IS NOT NULL)         AS eta,
       min(ts) AS first_seen,
       max(ts) AS last_seen
FROM {src}
GROUP BY mmsi
ORDER BY mmsi
"""


@dataclass(frozen=True)
class StepResult:
    name: str
    action: str
    rows_in: int
    rows_affected: int
    description: str


def _view(i: int) -> str:
    return f"step{i}"


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    assert row is not None
    return int(row[0])


def clean_day(source: Path, day: date, paths: DataPaths) -> list[StepResult]:
    """Clean one day's raw file (.zip or .csv) into clean/{positions,vessels,quality}/date=DAY/.

    Idempotent per day: outputs are built in a temp dir and swapped in; other days are untouched.
    """
    paths.clean.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.clean, prefix=".tmp-") as tmp_str:
        tmp = Path(tmp_str)
        # File-backed so a full day (~17M rows) spills to disk instead of exhausting RAM.
        con = duckdb.connect(str(tmp / "work.duckdb"))
        con.execute(CLEAN_TEXT_MACRO)

        staged = tmp / "staged.parquet"
        with csv_source(source) as src:
            con.execute(
                f"COPY ({typed_select(src)}) TO '{staged}' (FORMAT parquet, COMPRESSION zstd)"
            )

        bad_day = _count(
            con, f"SELECT count(*) FROM '{staged}' WHERE ts IS NULL OR ts::DATE <> DATE '{day}'"
        )
        if bad_day:
            raise ValueError(f"{source}: {bad_day} rows have a timestamp outside {day}")

        con.execute(f"CREATE VIEW {_view(0)} AS SELECT * FROM '{staged}'")
        results: list[StepResult] = []
        n_prev = _count(con, f"SELECT count(*) FROM {_view(0)}")
        static_view = _view(0)
        for i, step in enumerate(STEPS, start=1):
            sql = step.sql.replace("FROM prev", f"FROM {_view(i - 1)}")
            # Materialize each step so later counts don't re-run the whole chain.
            con.execute(f"CREATE TABLE {_view(i)} AS {sql}")
            n = _count(con, f"SELECT count(*) FROM {_view(i)}")
            if step.action == "drop":
                affected = n_prev - n
            else:
                affected = _count(
                    con, f"SELECT count(*) FROM {_view(i - 1)} WHERE sog >= {SOG_SENTINEL}"
                )
            results.append(StepResult(step.name, step.action, n_prev, affected, step.description))
            if step.name == STATIC_STEP:
                static_view = _view(i)
            elif i > 1 and _view(i - 1) != static_view:
                con.execute(f"DROP TABLE {_view(i - 1)}")
            n_prev = n

        final = _view(len(STEPS))
        out = {t: tmp / t for t in CLEAN_TABLES}
        for d in out.values():
            d.mkdir()
        con.execute(
            f"COPY (SELECT {POSITION_COLUMNS} FROM {final} ORDER BY mmsi, ts)"
            f" TO '{out['positions'] / 'part.parquet'}' (FORMAT parquet, COMPRESSION zstd)"
        )
        con.execute(
            f"COPY ({VESSELS_SQL.format(src=static_view)})"
            f" TO '{out['vessels'] / 'part.parquet'}' (FORMAT parquet, COMPRESSION zstd)"
        )
        con.execute(
            "CREATE TABLE quality (step_order INTEGER, step VARCHAR, action VARCHAR,"
            " rows_in BIGINT, rows_affected BIGINT, description VARCHAR)"
        )
        con.executemany(
            "INSERT INTO quality VALUES (?, ?, ?, ?, ?, ?)",
            [
                [i, r.name, r.action, r.rows_in, r.rows_affected, r.description]
                for i, r in enumerate(results, start=1)
            ],
        )
        con.execute(
            f"COPY (SELECT * FROM quality ORDER BY step_order)"
            f" TO '{out['quality'] / 'part.parquet'}' (FORMAT parquet)"
        )
        con.close()

        for table, built in out.items():
            target = paths.partition(table, day)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            built.rename(target)
    return results
