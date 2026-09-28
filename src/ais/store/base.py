"""The store interface. The API and tools depend only on this, so a Postgres/PostGIS
implementation is a drop-in replacement verified by the same contract tests."""

from datetime import datetime, timedelta
from typing import Protocol

from ais.models import BBox, Snapshot, Stops, Track, Trip, TripGeometry, Vessel, VesselSummary


class DerivedDataMissing(RuntimeError):
    """Trips/stops have not been built yet (`uv run ais derive`)."""


class AisStore(Protocol):
    def vessel(self, mmsi: int) -> Vessel | None: ...

    def search_vessels(self, q: str, limit: int) -> list[VesselSummary]: ...

    def track(self, mmsi: int, start: datetime, end: datetime, max_points: int) -> Track: ...

    def snapshot(
        self, bbox: BBox, at: datetime, lookback: timedelta, max_items: int
    ) -> Snapshot: ...

    def trips(self, mmsi: int, start: datetime, end: datetime) -> list[Trip]:
        """Trips of one vessel overlapping [start, end], ordered by start."""
        ...

    def trip_geometry(self, trip_id: str) -> TripGeometry | None: ...

    def stops(
        self, bbox: BBox, start: datetime, end: datetime, mmsi: int | None, max_items: int
    ) -> Stops:
        """Stops located in bbox overlapping [start, end], ordered by start."""
        ...
