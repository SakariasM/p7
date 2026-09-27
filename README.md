# p7

AIS data pipeline and API. Stand-in data comes from the Danish Maritime Authority's
open daily dumps (http://aisdata.ais.dk/) while the project database is unavailable.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```sh
uv sync
uv run ais dev      # download + clean the dev slice (2026-09-22..24), ~6 min, ~400 MB
uv run ais serve    # API on http://127.0.0.1:8000, interactive docs at /docs
```

`ais dev` streams each day's 600 MB zip straight into DuckDB and deletes it afterwards,
so peak disk use stays around 2 GB. Re-running skips days that are already clean.

## API

The contract is [`api/openapi.json`](api/openapi.json). Generate frontend types with:

```sh
npx openapi-typescript api/openapi.json -o src/api-types.ts
```

| Endpoint | Purpose | Limits |
|---|---|---|
| `GET /vessels?q=` | Search by MMSI prefix, name or callsign | `limit` ≤ 100 |
| `GET /vessels/{mmsi}` | Latest static info for one vessel | |
| `GET /vessels/{mmsi}/track?start&end` | Positions in a time window | window ≤ 24 h, `max_points` ≤ 5000 (evenly downsampled) |
| `GET /snapshot?bbox&at` | Latest position per vessel in a bbox | `lookback_min` ≤ 60, ≤ 5000 vessels (`truncated` flag) |
| `GET /trips`, `GET /stops` | Trajectories | 501 until Phase 2 |

Timestamps are UTC. Out-of-range parameters return 422; invalid time windows return 400.

## Layout

```
src/ais/
  ingest.py        fetch DMA zips, typed parse of the raw CSV
  clean.py         named cleaning steps, positions/vessels split, quality log
  store/base.py    AisStore Protocol (DuckDB now, Postgres/PostGIS later)
  store/duckdb_store.py
  api.py           FastAPI app, server-side limits
  cli.py           `uv run ais --help`
data/              raw/ -> clean/ -> derived/ (git-ignored, see data/README.md)
api/openapi.json   committed contract; CI fails if it is stale
```

## Checks

```sh
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
uv run ais openapi   # after changing models or endpoints
```

The unit tests run on a 27-row fixture (`tests/fixtures/mini.csv`) with every data defect
planted. `tests/test_dev_slice.py` adds data-quality checks on the real dev slice when present.
