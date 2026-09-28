"""Typed records shared by the store and the API (and the OpenAPI contract)."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    min_lon: float = Field(ge=-180, le=180)
    min_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.min_lon >= self.max_lon or self.min_lat >= self.max_lat:
            raise ValueError("bbox must be min_lon,min_lat,max_lon,max_lat with min < max")
        return self

    @classmethod
    def parse(cls, s: str) -> "BBox":
        parts = s.split(",")
        if len(parts) != 4:
            raise ValueError("bbox must be four comma-separated numbers")
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
        return cls(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)

    @property
    def area_deg2(self) -> float:
        return (self.max_lon - self.min_lon) * (self.max_lat - self.min_lat)


class Vessel(BaseModel):
    mmsi: int
    name: str | None
    imo: int | None
    callsign: str | None
    ship_type: str | None
    cargo_type: str | None
    mobile_type: str | None
    length: float | None
    width: float | None
    draught: float | None
    destination: str | None
    eta: datetime | None
    first_seen: datetime
    last_seen: datetime


class VesselSummary(BaseModel):
    mmsi: int
    name: str | None
    callsign: str | None
    ship_type: str | None


class TrackPoint(BaseModel):
    ts: datetime
    lat: float
    lon: float
    sog: float | None
    cog: float | None
    heading: int | None


class Track(BaseModel):
    mmsi: int
    start: datetime
    end: datetime
    total_points: int
    downsampled: bool
    points: list[TrackPoint]


class SnapshotItem(BaseModel):
    mmsi: int
    ts: datetime
    lat: float
    lon: float
    sog: float | None
    cog: float | None
    heading: int | None
    name: str | None
    ship_type: str | None


class Snapshot(BaseModel):
    at: datetime
    lookback_min: int
    truncated: bool
    items: list[SnapshotItem]


class Trip(BaseModel):
    trip_id: str
    mmsi: int
    start_ts: datetime
    end_ts: datetime
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    duration_s: float
    distance_m: float
    point_count: int


class LineString(BaseModel):
    """GeoJSON LineString; coordinates are [lon, lat] pairs."""

    type: Literal["LineString"] = "LineString"
    coordinates: list[tuple[float, float]]


class TripGeometry(BaseModel):
    trip_id: str
    mmsi: int
    geometry: LineString


class Stop(BaseModel):
    mmsi: int
    start_ts: datetime
    end_ts: datetime
    lat: float
    lon: float
    duration_s: float
    point_count: int


class Stops(BaseModel):
    truncated: bool
    items: list[Stop]
