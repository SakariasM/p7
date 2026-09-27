from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ais.api import create_app
from ais.store.duckdb_store import DuckDBStore

A = 219000001
BBOX = "7.5,54.5,15.5,58"


@pytest.fixture
def client(store: DuckDBStore) -> Iterator[TestClient]:
    with TestClient(create_app(store)) as c:
        yield c


def test_vessel_and_404(client: TestClient) -> None:
    assert client.get(f"/vessels/{A}").json()["name"] == "ALPHA"
    assert client.get("/vessels/219999999").status_code == 404
    assert client.get("/vessels/123").status_code == 422


def test_search_limit(client: TestClient) -> None:
    assert [v["mmsi"] for v in client.get("/vessels", params={"q": "alpha"}).json()] == [A]
    assert client.get("/vessels", params={"q": "a", "limit": 101}).status_code == 422
    assert client.get("/vessels", params={"q": ""}).status_code == 422


def test_track(client: TestClient) -> None:
    r = client.get(
        f"/vessels/{A}/track",
        params={"start": "2026-09-24T00:00:00Z", "end": "2026-09-24T01:00:00Z"},
    )
    assert r.status_code == 200
    assert r.json()["total_points"] == 10


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"start": "2026-09-24T01:00:00", "end": "2026-09-24T00:00:00"}, 400),
        ({"start": "2026-09-22T00:00:00", "end": "2026-09-24T00:00:00"}, 400),
        ({"start": "2026-09-24T00:00:00", "end": "2026-09-24T01:00:00", "max_points": 5001}, 422),
        ({"start": "not-a-date", "end": "2026-09-24T01:00:00"}, 422),
    ],
)
def test_track_limits(client: TestClient, params: dict[str, str | int], status: int) -> None:
    assert client.get(f"/vessels/{A}/track", params=params).status_code == status


def test_timezone_is_converted_to_utc(client: TestClient) -> None:
    r = client.get(
        f"/vessels/{A}/track",
        params={"start": "2026-09-24T02:00:00+02:00", "end": "2026-09-24T02:05:00+02:00"},
    )
    assert r.json()["total_points"] == 6


def test_snapshot(client: TestClient) -> None:
    r = client.get("/snapshot", params={"bbox": BBOX, "at": "2026-09-24T00:05:00"})
    assert r.status_code == 200
    assert {i["mmsi"] for i in r.json()["items"]} == {219000001, 219000002}


@pytest.mark.parametrize(
    "params",
    [
        {"bbox": "1,2,3", "at": "2026-09-24T00:05:00"},
        {"bbox": "15,54,7,58", "at": "2026-09-24T00:05:00"},
        {"bbox": BBOX, "at": "2026-09-24T00:05:00", "lookback_min": 61},
        {"at": "2026-09-24T00:05:00"},
    ],
)
def test_snapshot_limits(client: TestClient, params: dict[str, str | int]) -> None:
    assert client.get("/snapshot", params=params).status_code == 422


def test_trips_and_stops_are_501(client: TestClient) -> None:
    window = {"start": "2026-09-24T00:00:00", "end": "2026-09-24T01:00:00"}
    assert client.get("/trips", params={"mmsi": A, **window}).status_code == 501
    assert client.get("/stops", params={"bbox": BBOX, **window}).status_code == 501


def test_openapi_contract_is_committed_and_current() -> None:
    import json

    committed = Path(__file__).parents[1] / "api" / "openapi.json"
    current = json.dumps(create_app(None).openapi(), indent=2, sort_keys=True) + "\n"
    assert committed.read_text() == current, "run `uv run ais openapi`"
