"""Store contract tests. Parametrize `store` over new implementations (e.g. Postgres)."""

from datetime import datetime, timedelta

import pytest

from ais.models import BBox
from ais.store.base import AisStore, DerivedDataMissing

A, B = 219000001, 219000002
DAY_START = datetime(2026, 9, 24)
DK = BBox(min_lon=7.5, min_lat=54.5, max_lon=15.5, max_lat=58.0)


def test_vessel(store: AisStore) -> None:
    v = store.vessel(A)
    assert v is not None
    assert (v.name, v.ship_type, v.first_seen) == ("ALPHA", "Cargo", datetime(2026, 9, 23))
    assert store.vessel(219999999) is None


def test_search(store: AisStore) -> None:
    assert [v.mmsi for v in store.search_vessels("brav", 10)] == [B]
    assert [v.mmsi for v in store.search_vessels("21900000", 10)] == [A, B]
    assert [v.mmsi for v in store.search_vessels("oxab", 10)] == [A]
    assert len(store.search_vessels("21900000", 1)) == 1


def test_track_within_one_day(store: AisStore) -> None:
    t = store.track(A, DAY_START, DAY_START + timedelta(hours=1), max_points=100)
    assert t.total_points == 10 and not t.downsampled
    assert [p.ts.minute for p in t.points] == list(range(10))
    assert t.points[-1].lat == pytest.approx(55.09)


def test_track_spans_days_and_downsamples(store: AisStore) -> None:
    t = store.track(A, datetime(2026, 9, 23), DAY_START + timedelta(hours=1), max_points=5)
    assert t.total_points == 20 and t.downsampled
    assert len(t.points) <= 5
    assert t.points == sorted(t.points, key=lambda p: p.ts)


def test_sog_sentinel_is_null(store: AisStore) -> None:
    t = store.track(B, DAY_START, DAY_START + timedelta(hours=1), max_points=100)
    assert [p.sog for p in t.points] == [5.0, 5.0, None, 5.0, 5.0]


def test_snapshot_latest_position_in_window(store: AisStore) -> None:
    s = store.snapshot(DK, DAY_START + timedelta(minutes=4), timedelta(minutes=10), 100)
    by_mmsi = {i.mmsi: i for i in s.items}
    assert set(by_mmsi) == {A, B} and not s.truncated
    assert by_mmsi[A].ts == DAY_START + timedelta(minutes=4)
    assert by_mmsi[B].name == "BRAVO"


def test_snapshot_bbox_limit_and_truncation(store: AisStore) -> None:
    at, lb = DAY_START + timedelta(minutes=9), timedelta(minutes=10)
    only_a = BBox(min_lon=9.9, min_lat=54.9, max_lon=10.1, max_lat=55.2)
    assert [i.mmsi for i in store.snapshot(only_a, at, lb, 100).items] == [A]
    s = store.snapshot(DK, at, lb, 1)
    assert s.truncated and len(s.items) == 1
    # Nothing reported in the lookback window before the day starts.
    assert store.snapshot(DK, DAY_START - timedelta(hours=2), lb, 100).items == []


def test_trips_and_stops_need_derive(store: AisStore) -> None:
    window = (DAY_START, DAY_START + timedelta(hours=1))
    with pytest.raises(DerivedDataMissing):
        store.trips(A, *window)
    with pytest.raises(DerivedDataMissing):
        store.stops(DK, *window, None, 10)
    with pytest.raises(DerivedDataMissing):
        store.trip_geometry("x")
