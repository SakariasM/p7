"""Trip/stop segmentation on synthetic tracks, one scenario per vessel."""

from datetime import date, datetime, timedelta

import duckdb
import pytest

from ais.clean import VESSELS_SQL
from ais.derive import DeriveResult, derive
from ais.models import BBox
from ais.paths import DataPaths
from ais.store.duckdb_store import DuckDBStore

DAY = date(2026, 9, 24)
T0 = datetime(2026, 9, 24)
STEP_LAT = 0.005  # ~556 m per minute = ~18 kn

STOP_THEN_GO = 219000011
GAP = 219000012
ONE_OUTLIER = 219000013
BAD_RUN = 219000014
OUT_OF_COVERAGE = 219000015
TOO_SHORT = 219000016

Row = tuple[int, datetime, float, float, float]


def moving(mmsi: int, lon: float, start_min: int, n: int, lat0: float = 55.0) -> list[Row]:
    return [
        (mmsi, T0 + timedelta(minutes=start_min + i), lat0 + i * STEP_LAT, lon, 18.0)
        for i in range(n)
    ]


def scenario() -> list[Row]:
    rows: list[Row] = []
    # 20 moving, 15 min stationary, 20 moving -> 2 trips around 1 stop
    rows += moving(STOP_THEN_GO, 8.0, 0, 20)
    stop_lat = 55.0 + 19 * STEP_LAT
    rows += [(STOP_THEN_GO, T0 + timedelta(minutes=20 + i), stop_lat, 8.0, 0.0) for i in range(15)]
    rows += moving(STOP_THEN_GO, 8.0, 35, 20, lat0=stop_lat + STEP_LAT)
    # 15 moving, 40 min gap, 15 moving -> 2 trips
    rows += moving(GAP, 9.0, 0, 15) + moving(GAP, 9.0, 55, 15, lat0=56.0)
    # one point ~127 km off -> removed, 1 trip
    track = moving(ONE_OUTLIER, 10.0, 0, 30)
    m, ts, lat, lon, sog = track[15]
    track[15] = (m, ts, lat, lon + 2.0, sog)
    rows += track
    # 20 consecutive points with a dropped longitude digit (11.x -> 1.x)
    track = moving(BAD_RUN, 11.0, 0, 60)
    rows += [
        (m, ts, lat, lon - 10 if 20 <= i < 40 else lon, s)
        for i, (m, ts, lat, lon, s) in enumerate(track)
    ]
    # one point at longitude 89
    track = moving(OUT_OF_COVERAGE, 12.0, 0, 20)
    m, ts, lat, lon, sog = track[5]
    track[5] = (m, ts, lat, 89.0, sog)
    rows += track
    rows += moving(TOO_SHORT, 13.0, 0, 5)
    return rows


def write_clean(paths: DataPaths, rows: list[Row]) -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE p (mmsi BIGINT, ts TIMESTAMP, lat DOUBLE, lon DOUBLE, sog DOUBLE)")
    con.executemany("INSERT INTO p VALUES (?, ?, ?, ?, ?)", [list(r) for r in rows])
    for table, sql in {
        "positions": "SELECT mmsi, ts, lat, lon, sog, NULL::DOUBLE AS cog,"
        " NULL::SMALLINT AS heading, NULL::DOUBLE AS rot, NULL::VARCHAR AS nav_status"
        " FROM p ORDER BY mmsi, ts",
        "vessels": VESSELS_SQL.format(
            src="(SELECT *, 'V' || mmsi AS name, NULL::BIGINT AS imo, NULL::VARCHAR AS callsign,"
            " NULL::VARCHAR AS ship_type, NULL::VARCHAR AS cargo_type, 'Class A' AS mobile_type,"
            " NULL::DOUBLE AS length, NULL::DOUBLE AS width, NULL::DOUBLE AS draught,"
            " NULL::VARCHAR AS destination, NULL::TIMESTAMP AS eta FROM p)"
        ),
        "quality": "SELECT 1 AS step_order, 'exact_duplicate' AS step, 'drop' AS action,"
        " 10::BIGINT AS rows_in, 2::BIGINT AS rows_affected, '' AS description",
    }.items():
        d = paths.partition(table, DAY)
        d.mkdir(parents=True)
        con.execute(f"COPY ({sql}) TO '{d / 'part.parquet'}' (FORMAT parquet)")
    con.close()


def trips_of(store: DuckDBStore, mmsi: int) -> list[tuple[int, float]]:
    return [(t.point_count, t.distance_m) for t in store.trips(mmsi, T0, T0 + timedelta(days=1))]


def test_bad_fix_counts(derived: DeriveResult) -> None:
    assert derived.outside_coverage == 1
    assert derived.outliers_removed >= 1  # ONE_OUTLIER, plus edges of BAD_RUN
    assert derived.jump_splits >= 2  # into and out of what remains of BAD_RUN


def test_stop_splits_trips(dstore: DuckDBStore) -> None:
    assert trips_of(dstore, STOP_THEN_GO) == [
        (20, pytest.approx(19 * 556, rel=0.01)),
        (20, pytest.approx(19 * 556, rel=0.01)),
    ]
    stops = dstore.stops(
        BBox(min_lon=7.9, min_lat=54.9, max_lon=8.1, max_lat=55.3),
        T0,
        T0 + timedelta(days=1),
        None,
        10,
    )
    assert not stops.truncated
    [stop] = stops.items
    assert (stop.mmsi, stop.duration_s, stop.point_count) == (STOP_THEN_GO, 14 * 60, 15)


def test_gap_splits_trips(dstore: DuckDBStore) -> None:
    assert [n for n, _ in trips_of(dstore, GAP)] == [15, 15]


def test_single_outlier_is_removed(dstore: DuckDBStore) -> None:
    assert trips_of(dstore, ONE_OUTLIER) == [(29, pytest.approx(29 * 556, rel=0.01))]


def test_no_trip_contains_an_impossible_jump(dstore: DuckDBStore) -> None:
    trips = trips_of(dstore, BAD_RUN)
    assert len(trips) >= 2
    assert all(d < 50_000 for _, d in trips)  # the jump itself is ~640 km


def test_out_of_coverage_point_removed_and_short_tracks_dropped(dstore: DuckDBStore) -> None:
    assert [n for n, _ in trips_of(dstore, OUT_OF_COVERAGE)] == [19]
    assert trips_of(dstore, TOO_SHORT) == []


def test_trip_geometry_is_simplified_line(dstore: DuckDBStore) -> None:
    [first, _] = dstore.trips(STOP_THEN_GO, T0, T0 + timedelta(days=1))
    g = dstore.trip_geometry(first.trip_id)
    assert g is not None
    assert g.geometry.coordinates == [
        (8.0, 55.0),
        (8.0, pytest.approx(55.0 + 19 * STEP_LAT)),
    ]
    assert dstore.trip_geometry("nope") is None


def test_trip_window_overlap(dstore: DuckDBStore) -> None:
    # Window inside the first trip only.
    trips = dstore.trips(STOP_THEN_GO, T0 + timedelta(minutes=5), T0 + timedelta(minutes=6))
    assert len(trips) == 1 and trips[0].start_ts == T0


def test_derive_is_deterministic(paths: DataPaths, derived: DeriveResult) -> None:
    def digest() -> object:
        row = duckdb.sql(
            f"SELECT md5(string_agg(t::VARCHAR, '|')) FROM '{paths.derived}/trips/*.parquet' t"
        ).fetchone()
        assert row is not None
        return row[0]

    before = digest()
    assert derive(paths) == derived
    assert digest() == before
