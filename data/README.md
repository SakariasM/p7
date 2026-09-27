# Data

Everything here is git-ignored and reproducible with `uv run ais dev`.
Source: Danish Maritime Authority open AIS data, http://aisdata.ais.dk/ (one zip per day).

```
raw/aisdk-YYYY-MM-DD.zip                  downloaded dump (deleted after cleaning unless --keep-raw)
clean/positions/date=YYYY-MM-DD/*.parquet ts, mmsi, lat, lon, sog, cog, heading, rot, nav_status
clean/vessels/date=YYYY-MM-DD/*.parquet   latest static info per MMSI that day
clean/quality/date=YYYY-MM-DD/*.parquet   rows affected by each cleaning step
derived/                                  trips, stops (Phase 2)
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

## Ad-hoc queries

```sh
duckdb data/ais.duckdb -c "select * from vessels order by last_seen desc limit 10"
duckdb data/ais.duckdb -c "select * from quality order by date, step_order"
```
