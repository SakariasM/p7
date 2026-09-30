# Data

Everything here is git-ignored and reproducible with `uv run ais dev`.
Source: Danish Maritime Authority open AIS data, http://aisdata.ais.dk/ (one zip per day).

```
raw/aisdk-YYYY-MM-DD.zip                  downloaded dump (deleted after cleaning unless --keep-raw)
clean/positions/date=YYYY-MM-DD/*.parquet ts, mmsi, lat, lon, sog, cog, heading, rot, nav_status
clean/vessels/date=YYYY-MM-DD/*.parquet   latest static info per MMSI that day
clean/quality/date=YYYY-MM-DD/*.parquet   rows affected by each cleaning step
derived/{trips,stops,trip_geometry,quality}/part.parquet   rebuilt from all of clean/ by `ais derive`
ais.duckdb                                views over clean/; open from the repo root
```

Positions are sorted by `mmsi, ts`, zstd-compressed, about 130 MB per day.
Each day is rebuilt independently: `uv run ais clean 2026-09-24` replaces only that day.

## Cleaning steps

Applied in order; `uv run ais quality` prints the counts per day.

| Step | Action | Rule |
|---|---|---|
| `non_vessel` | drop | Keep only `Class A` / `Class B` (removes base stations, AtoN, SAR aircraft) |
| `invalid_mmsi` | drop | MMSI outside 200000000-799999999 |
| `invalid_position` | drop | Missing or out-of-range lat/lon (incl. 91/181 sentinels) |
| `sog_sentinel` | null | SOG >= 102.2 set to NULL, row kept |
| `exact_duplicate` | drop | Same report from several receivers; one kept |
| `conflicting_same_second` | drop | Different positions for one MMSI in one second; whole group dropped |

Text sentinels (`Unknown`, `Undefined`, `Unknown value`, empty) are NULL from the start.

2026-09-24: 17,612,827 raw rows -> 10,394,187 positions for 4,230 vessels.
Most removed rows (5.7M) are exact duplicates.

## Trips and stops

`uv run ais derive` rebuilds derived/ from every cleaned day at once (trips cross midnight).
Thresholds are constants at the top of `src/ais/derive.py`.

1. Bad fixes removed: outside DMA coverage (lat 50-64, lon -5..32), farther from the
   rolling median of +-10 neighbours than 50 kn allows, and single-point spikes.
2. Stop: stationary (SOG < 0.5 kn) for >= 10 min with no gap over 30 min.
3. Trip: movement between stops, split at gaps over 30 min and at impossible jumps
   (> 50 kn over > 500 m). Needs >= 10 points and >= 1 km.
4. Geometry: trip line simplified at ~20 m tolerance (DuckDB spatial), as GeoJSON.

Dev slice: 33.0M positions -> 16,526 trips (median 86 min, 16 km) and 32,079 stops.
Removed 46 points outside coverage, 315 outliers, 469 spikes; 1,241 jump splits.
Known limit: a long run of corrupt fixes (e.g. 45 min with a dropped longitude digit)
becomes its own short trip at the wrong place rather than being removed.

## Ad-hoc queries

```sh
duckdb data/ais.duckdb -c "select * from vessels order by last_seen desc limit 10"
duckdb data/ais.duckdb -c "select * from quality order by date, step_order"
duckdb data/ais.duckdb -c "select * from trips order by distance_m desc limit 10"
```
