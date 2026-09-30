"""HTTP API over an AisStore. All limits are enforced here, server-side."""

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ais.models import BBox, Snapshot, Stops, Track, Trip, TripGeometry, Vessel, VesselSummary
from ais.store.base import AisStore, DerivedDataMissing

MAX_SEARCH_RESULTS = 100
MAX_TRACK_WINDOW = timedelta(hours=24)
MAX_TRACK_POINTS = 5000
DEFAULT_TRACK_POINTS = 2000
MAX_LOOKBACK_MIN = 60
DEFAULT_LOOKBACK_MIN = 10
MAX_SNAPSHOT_ITEMS = 5000
MAX_TRIPS_WINDOW = timedelta(days=31)
MAX_STOPS_ITEMS = 5000

BBOX_DOC = "min_lon,min_lat,max_lon,max_lat (WGS84), e.g. 7.5,54.5,15.5,58"
MMSI = Annotated[int, Path(ge=200_000_000, le=799_999_999)]
MMSIQuery = Annotated[int, Query(ge=200_000_000, le=799_999_999)]


def to_utc_naive(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def parse_bbox(bbox: Annotated[str, Query(description=BBOX_DOC)]) -> BBox:
    try:
        return BBox.parse(bbox)
    except (ValueError, ValidationError) as e:
        raise HTTPException(422, f"invalid bbox: {e}") from e


def check_window(
    start: datetime, end: datetime, max_window: timedelta
) -> tuple[datetime, datetime]:
    start, end = to_utc_naive(start), to_utc_naive(end)
    if end <= start:
        raise HTTPException(400, "end must be after start")
    if end - start > max_window:
        raise HTTPException(
            400, f"time window exceeds the maximum of {max_window.total_seconds() / 3600:g} hours"
        )
    return start, end


def get_store(request: Request) -> AisStore:
    store: AisStore | None = request.app.state.store
    if store is None:
        raise HTTPException(503, "no data store configured")
    return store


StoreDep = Annotated[AisStore, Depends(get_store)]
BBoxDep = Annotated[BBox, Depends(parse_bbox)]


def create_app(store: AisStore | None) -> FastAPI:
    app = FastAPI(
        title="AIS API",
        version="0.1.0",
        description="Vessel positions, tracks and snapshots from AIS data. "
        "All timestamps are UTC; naive timestamps are interpreted as UTC.",
    )
    app.state.store = store

    @app.exception_handler(DerivedDataMissing)
    def _derived_missing(_: Request, e: DerivedDataMissing) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=503)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/vessels", response_model=list[VesselSummary], tags=["vessels"])
    def search_vessels(
        s: StoreDep,
        q: Annotated[
            str, Query(min_length=1, max_length=100, description="MMSI prefix, name or callsign")
        ],
        limit: Annotated[int, Query(ge=1, le=MAX_SEARCH_RESULTS)] = 20,
    ) -> list[VesselSummary]:
        return s.search_vessels(q, limit)

    @app.get(
        "/vessels/{mmsi}",
        response_model=Vessel,
        tags=["vessels"],
        responses={404: {"description": "Unknown MMSI"}},
    )
    def vessel(s: StoreDep, mmsi: MMSI) -> Vessel:
        v = s.vessel(mmsi)
        if v is None:
            raise HTTPException(404, f"vessel {mmsi} not found")
        return v

    @app.get(
        "/vessels/{mmsi}/track",
        response_model=Track,
        tags=["vessels"],
        responses={400: {"description": "Window invalid or longer than 24 h"}},
    )
    def track(
        s: StoreDep,
        mmsi: MMSI,
        start: datetime,
        end: datetime,
        max_points: Annotated[int, Query(ge=2, le=MAX_TRACK_POINTS)] = DEFAULT_TRACK_POINTS,
    ) -> Track:
        start, end = check_window(start, end, MAX_TRACK_WINDOW)
        return s.track(mmsi, start, end, max_points)

    @app.get("/snapshot", response_model=Snapshot, tags=["map"])
    def snapshot(
        s: StoreDep,
        bbox: BBoxDep,
        at: Annotated[datetime, Query(description="Snapshot time (UTC)")],
        lookback_min: Annotated[
            int,
            Query(ge=1, le=MAX_LOOKBACK_MIN, description="Include vessels seen this recently"),
        ] = DEFAULT_LOOKBACK_MIN,
    ) -> Snapshot:
        return s.snapshot(
            bbox, to_utc_naive(at), timedelta(minutes=lookback_min), MAX_SNAPSHOT_ITEMS
        )

    derived_503 = {503: {"description": "Trips/stops not built yet (`uv run ais derive`)"}}

    @app.get(
        "/trips",
        response_model=list[Trip],
        tags=["trajectories"],
        responses={400: {"description": "Window invalid or longer than 31 days"}, **derived_503},
    )
    def trips(s: StoreDep, mmsi: MMSIQuery, start: datetime, end: datetime) -> list[Trip]:
        """Trips of one vessel that overlap the window."""
        start, end = check_window(start, end, MAX_TRIPS_WINDOW)
        return s.trips(mmsi, start, end)

    @app.get(
        "/trips/{trip_id}/geometry",
        response_model=TripGeometry,
        tags=["trajectories"],
        responses={404: {"description": "Unknown trip"}, **derived_503},
    )
    def trip_geometry(s: StoreDep, trip_id: Annotated[str, Path(max_length=64)]) -> TripGeometry:
        """Simplified trip line as GeoJSON, for drawing on a map."""
        g = s.trip_geometry(trip_id)
        if g is None:
            raise HTTPException(404, f"trip {trip_id} not found")
        return g

    @app.get(
        "/stops",
        response_model=Stops,
        tags=["trajectories"],
        responses={400: {"description": "Window invalid or longer than 31 days"}, **derived_503},
    )
    def stops(
        s: StoreDep,
        bbox: BBoxDep,
        start: datetime,
        end: datetime,
        mmsi: Annotated[int | None, Query(ge=200_000_000, le=799_999_999)] = None,
    ) -> Stops:
        """Stops located in the bbox that overlap the window (ports, anchorages)."""
        start, end = check_window(start, end, MAX_TRIPS_WINDOW)
        return s.stops(bbox, start, end, mmsi, MAX_STOPS_ITEMS)

    return app


def app_from_env() -> FastAPI:
    """uvicorn factory: `uvicorn ais.api:app_from_env --factory`."""
    from ais.paths import DataPaths
    from ais.store.duckdb_store import DuckDBStore

    return create_app(DuckDBStore(DataPaths.from_env()))
