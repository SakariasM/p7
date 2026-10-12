from datetime import date
from pathlib import Path

import duckdb
import pytest

from ais.clean import MMSI_MAX, MMSI_MIN, SOG_SENTINEL, StepResult, clean_day
from ais.ingest import CLEAN_TEXT_MACRO, processed_select, typed_select
from ais.paths import DataPaths

from .conftest import DAY, PREV_DAY, fixture_for


def _one(sql: str) -> object:
    row = duckdb.sql(sql).fetchone()
    assert row is not None
    return row[0]


def test_each_filter_removes_the_planted_rows(cleaned: dict[date, list[StepResult]]) -> None:
    counts = {r.name: r.rows_affected for r in cleaned[DAY]}
    assert counts == {
        "non_vessel": 4,
        "invalid_mmsi": 2,
        "invalid_position": 2,
        "sog_sentinel": 1,
        "exact_duplicate": 2,
        "conflicting_same_second": 2,
    }
    assert cleaned[DAY][0].rows_in == 27


def test_positions_are_clean(paths: DataPaths, cleaned: dict[date, list[StepResult]]) -> None:
    pos = f"read_parquet('{paths.glob('positions')}')"
    assert _one(f"SELECT count(*) FROM {pos}") == 2 * 15
    assert _one(f"SELECT count(*) FROM {pos} WHERE mmsi NOT BETWEEN {MMSI_MIN} AND {MMSI_MAX}") == 0
    assert _one(f"SELECT count(*) FROM {pos} WHERE sog >= {SOG_SENTINEL}") == 0
    assert (
        _one(f"SELECT count(*) FROM {pos} WHERE lat IS NULL OR lon IS NULL OR abs(lat) > 90") == 0
    )
    non_increasing = f"""
        SELECT count(*) FROM (
            SELECT ts <= lag(ts) OVER (PARTITION BY mmsi ORDER BY ts) AS bad FROM {pos}
        ) WHERE bad"""
    assert _one(non_increasing) == 0


def test_vessel_static_data_uses_latest_non_null(
    paths: DataPaths, cleaned: dict[date, list[StepResult]]
) -> None:
    row = duckdb.sql(
        f"SELECT name, ship_type, imo, length FROM '{paths.partition('vessels', DAY)}/*.parquet'"
        " WHERE mmsi = 219000001"
    ).fetchone()
    assert row == ("ALPHA", "Cargo", 9214006, 120.0)


def _digest(paths: DataPaths, table: str, day: date) -> object:
    return _one(
        f"SELECT md5(string_agg(t::VARCHAR, '|')) FROM"
        f" '{paths.partition(table, day)}/*.parquet' AS t"
    )


def test_rerun_is_identical_and_leaves_other_days_alone(
    paths: DataPaths, tmp_path: Path, cleaned: dict[date, list[StepResult]]
) -> None:
    before = {t: _digest(paths, t, DAY) for t in ("positions", "vessels", "quality")}
    other = paths.partition("positions", PREV_DAY) / "part.parquet"
    other_mtime = other.stat().st_mtime_ns

    clean_day(fixture_for(DAY, tmp_path), DAY, paths)

    assert {t: _digest(paths, t, DAY) for t in before} == before
    assert other.stat().st_mtime_ns == other_mtime
    assert not list(paths.clean.glob(".tmp-*"))


def _as_processed(csv: Path, out: Path) -> Path:
    """Write the fixture in the layout of AI-LAB's processed Parquet: renamed columns,
    FLOAT/SMALLINT where that dump narrows them, and a `date` column."""
    con = duckdb.connect()
    con.execute(CLEAN_TEXT_MACRO)
    con.execute(f"""
        COPY (
          SELECT ts, mobile_type, mmsi, lat, lon, nav_status, rot::FLOAT AS rot,
                 sog::FLOAT AS sog, cog::FLOAT AS cog, heading, imo, callsign, name, ship_type,
                 cargo_type, width::SMALLINT AS width, length::SMALLINT AS length,
                 pos_fix_device, draught::FLOAT AS draught, destination, eta,
                 data_source_type AS data_source, a::SMALLINT AS dim_a, b::SMALLINT AS dim_b,
                 c::SMALLINT AS dim_c, d::SMALLINT AS dim_d, ts::DATE AS date
          FROM ({typed_select(str(csv))})
        ) TO '{out}' (FORMAT parquet)""")
    con.close()
    return out


def test_processed_parquet_cleans_like_the_csv(
    paths: DataPaths, tmp_path: Path, cleaned: dict[date, list[StepResult]]
) -> None:
    other = DataPaths(tmp_path / "from-parquet")
    parquet = _as_processed(fixture_for(DAY, tmp_path), tmp_path / "ais.parquet")

    results = clean_day(parquet, DAY, other)

    assert results == cleaned[DAY]  # incl. the SOG sentinel stored as FLOAT 102.2
    for table in ("positions", "vessels", "quality"):
        assert _digest(other, table, DAY) == _digest(paths, table, DAY), table


def test_processed_floats_keep_their_decimal_value(tmp_path: Path) -> None:
    # FLOAT 102.2 is 102.19999694824219 as a DOUBLE, below the sentinel; mini.csv only has
    # 102.3, which rounds up and would pass either way.
    src = tmp_path / "floats.parquet"
    duckdb.sql(f"""
        COPY (SELECT 102.2::FLOAT AS sog, 12.3::FLOAT AS draught, 'x' AS name,
                     * EXCLUDE (sog, draught, name)
              FROM '{_as_processed(fixture_for(DAY, tmp_path), tmp_path / "f.parquet")}'
              LIMIT 1) TO '{src}' (FORMAT parquet)""")
    con = duckdb.connect()
    con.execute(CLEAN_TEXT_MACRO)
    row = con.sql(f"SELECT sog, draught FROM ({processed_select(str(src))})").fetchone()
    assert row == (SOG_SENTINEL, 12.3)


def test_rejects_rows_from_another_day(paths: DataPaths, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        clean_day(fixture_for(DAY, tmp_path), PREV_DAY, paths)
    assert not paths.partition("positions", PREV_DAY).exists()
