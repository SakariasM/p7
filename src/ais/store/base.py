"""The store interface. The API and tools depend only on this, so a Postgres/PostGIS
implementation is a drop-in replacement verified by the same contract tests."""

from datetime import datetime, timedelta
from typing import Protocol

from ais.models import (
    BBox,
    Route,
    Snapshot,
    Stops,
    Track,
    Trip,
    TripGeometry,
    Vessel,
    VesselSummary,
)


class DerivedDataMissing(RuntimeError):
    """Trips/stops have not been built yet (`uv run ais derive`)."""


class AisStore(Protocol):
    def vessel(self, mmsi: int) -> Vessel | None: ...

    def search_vessels(self, q: str, limit: int) -> list[VesselSummary]: ...

    def track(self, mmsi: int, start: datetime, end: datetime, max_points: int) -> Track: ...

    def snapshot(
        self,
        bbox: BBox,
        at: datetime,
        lookback: timedelta,
        max_items: int,
        name: str | None = None,
        ship_type: str | None = None,
    ) -> Snapshot:
        """Latest position per vessel in the window; `name` matches a substring, any case."""
        ...

    def trips(self, mmsi: int, start: datetime, end: datetime) -> list[Trip]:
        """Trips of one vessel overlapping [start, end], ordered by start."""
        ...

    def trip_geometry(self, trip_id: str) -> TripGeometry | None: ...

    def route(self, mmsi: int, start: datetime, end: datetime, max_legs: int) -> Route:
        """Trips (with lines) and stops of one vessel overlapping [start, end], in time order."""
        ...

    def stops(
        self, bbox: BBox, start: datetime, end: datetime, mmsi: int | None, max_items: int
    ) -> Stops:
        """Stops located in bbox overlapping [start, end], ordered by start."""
        ...
