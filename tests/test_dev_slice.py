"""Data-quality checks on the real dev slice. Skipped unless `uv run ais dev` has been run.

Run with: uv run pytest -m data
"""

import duckdb
import pytest

from ais.clean import MMSI_MAX, MMSI_MIN, SOG_SENTINEL
from ais.paths import DataPaths

paths = DataPaths.from_env()
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not any(paths.clean.glob("positions/date=*/*.parquet")), reason="no dev slice"
    ),
]


def _one(sql: str) -> object:
    row = duckdb.sql(sql).fetchone()
    assert row is not None
    return row[0]


POS = f"read_parquet('{paths.glob('positions')}', hive_partitioning = true)"


def test_no_invalid_values() -> None:
    assert (
        _one(f"""
        SELECT count(*) FROM {POS}
        WHERE mmsi NOT BETWEEN {MMSI_MIN} AND {MMSI_MAX}
           OR sog >= {SOG_SENTINEL}
           OR lat IS NULL OR lon IS NULL OR abs(lat) > 90 OR abs(lon) > 180
           OR ts::DATE <> date
    """)
        == 0
    )


def test_timestamps_strictly_increasing_per_vessel() -> None:
    assert (
        _one(f"""
        SELECT count(*) FROM (
            SELECT ts <= lag(ts) OVER (PARTITION BY mmsi ORDER BY ts) AS bad FROM {POS}
        ) WHERE bad
    """)
        == 0
    )


def test_every_position_has_a_vessel_row() -> None:
    vessels = f"read_parquet('{paths.glob('vessels')}', hive_partitioning = true)"
    assert (
        _one(f"""
        SELECT count(*) FROM (SELECT DISTINCT date, mmsi FROM {POS})
        ANTI JOIN {vessels} USING (date, mmsi)
    """)
        == 0
    )


@pytest.mark.skipif(not (paths.derived / "trips").exists(), reason="no derived data")
def test_derived_trips_are_plausible() -> None:
    trips = f"'{paths.derived}/trips/*.parquet'"
    # Average speed over a whole trip: fast ferries reach ~40 kn; nothing should near 60.
    assert _one(f"SELECT count(*) FROM {trips} WHERE distance_m / duration_s * 1.9438 > 60") == 0
    assert _one(f"SELECT count(*) FROM {trips} WHERE point_count < 10 OR distance_m < 1000") == 0
    geoms = f"'{paths.derived}/trip_geometry/*.parquet'"
    assert _one(f"SELECT count(*) FROM {trips} ANTI JOIN {geoms} USING (trip_id)") == 0
