# p7

AIS data pipeline and API. Stand-in data comes from the Danish Maritime Authority's
open daily dumps (http://aisdata.ais.dk/) while the project database is unavailable.

## Use the shared API (start here)

The full year (2025-10-01..2026-09-30) is served from AI-LAB: an `ais serve` process on
`ailab-fe01`, port 8200, listening only on that machine. You reach it through an SSH tunnel,
so nothing is exposed to the internet and you need no local data.

**1. SSH access.** You need an AI-LAB account and AAU MFA. Add this to `~/.ssh/config`,
with your own AAU login:

```
Host ailab-vpn
    HostName ailab-fe01.srv.aau.dk
    User abc12@student.aau.dk
    ProxyJump abc12@student.aau.dk@sshgw.aau.dk
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

`ProxyJump` goes through AAU's SSH gateway, which works off campus without the VPN. The
`Control*` lines reuse one login for later commands, so you type your password and MFA once.
Don't retry a failing login in a loop: the gateway locks the account after repeated failures.

**2. Open a tunnel** and leave it running:

```sh
ssh -N -L 8200:127.0.0.1:8200 ailab-vpn
```

If 8200 is taken on your machine, use `-L 8300:127.0.0.1:8200` and port 8300 below.

**3. Use it.**

- Map: <http://localhost:8200> — ships at any moment, filters by name or type, vessel
  details, routes over up to 31 days with playback.
- Interactive API docs: <http://localhost:8200/docs>.
- From code, e.g. Python:

  ```python
  import httpx

  api = httpx.Client(base_url="http://localhost:8200")
  ferry = api.get("/vessels", params={"q": "TYCHO BRAHE"}).json()[0]
  route = api.get(f"/vessels/{ferry['mmsi']}/route",
                  params={"start": "2026-01-01T00:00:00", "end": "2026-01-08T00:00:00"}).json()
  print(len(route["legs"]), "trips,", len(route["stops"]), "stops")
  ```

**If the page doesn't load** (the tunnel connects but `curl localhost:8200/health` fails),
the API isn't running. Start it on AI-LAB from your own clone; the data is shared in the
project folder:

```sh
ssh ailab-vpn
tmux ls                                   # is a p7-api session already there?
curl -s 127.0.0.1:8200/health             # {"status":"ok"} means it's up; stop here
git clone https://github.com/SakariasM/p7 ~/p7 && cd ~/p7    # first time only
curl -LsSf https://astral.sh/uv/install.sh | sh              # first time only
tmux new -d -s p7-api "cd ~/p7 && AIS_DATA_DIR=/ceph/project/P7-Agent-Harness/data ~/.local/bin/uv run ais serve --port 8200"
```

It runs on the login node in tmux, so it keeps going after you log out. Only one process can
use port 8200; `tmux attach -t p7-api` shows its log, Ctrl-b d leaves it running.

## Develop locally

Needs [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```sh
uv sync
uv run ais dev      # download, clean and derive the dev slice (2026-09-22..24), ~7 min, ~400 MB
uv run ais serve    # API on http://127.0.0.1:8000, interactive docs at /docs
```

`ais dev` streams each day's 600 MB zip straight into DuckDB and deletes it afterwards,
so peak disk use stays around 2 GB. Re-running skips days that are already clean.

## Full year on AI-LAB

The project folder holds a year of DMA data (2025-10-01..2026-09-30) as typed Parquet in
`/ceph/project/P7-Agent-Harness/data/processed/ais/date=YYYY-MM-DD/ais.parquet`.
`ais clean-processed START END` cleans from there instead of downloading; on the dev days
the output is identical to the CSV route. From the repo root on AI-LAB:

```sh
sbatch slurm/clean_processed.slurm                         # one array task per month, ~1 h each
sbatch --dependency=afterok:<job id> slurm/derive.slurm    # trips and stops, then ais.duckdb
AIS_DATA_DIR=/ceph/project/P7-Agent-Harness/data uv run ais serve --port 8200
```

Both jobs write to `$AIS_DATA_DIR` (default: the project's `data/`), logs go to its `logs/`.
`derive` processes vessels in MMSI-range chunks of ~60M positions, so a year fits in memory;
the result matches a single pass.

## API

`ais serve` also serves a prototype map page at `/` (`src/ais/ui/index.html`, plain HTML and
Leaflet, no build step). It only calls the endpoints below and is not part of the contract.

The contract is [`api/openapi.json`](api/openapi.json). Generate frontend types with:

```sh
npx openapi-typescript api/openapi.json -o src/api-types.ts
```

| Endpoint | Purpose | Limits |
|---|---|---|
| `GET /vessels?q=` | Search by MMSI prefix, name or callsign | `limit` ≤ 100 |
| `GET /vessels/{mmsi}` | Latest static info for one vessel | |
| `GET /vessels/{mmsi}/track?start&end` | Positions in a time window | window ≤ 24 h, `max_points` ≤ 5000 (evenly downsampled) |
| `GET /snapshot?bbox&at` | Latest position per vessel in a bbox; optional `name` (substring, any case) and `ship_type` | `lookback_min` ≤ 60, ≤ 5000 vessels (`truncated` flag) |
| `GET /trips?mmsi&start&end` | Trips of one vessel overlapping the window | window ≤ 31 days |
| `GET /trips/{trip_id}/geometry` | Simplified trip line as GeoJSON `LineString` | |
| `GET /vessels/{mmsi}/route?start&end` | Trips as simplified lines plus the stops between them, in time order | window ≤ 31 days, ≤ 2000 trips (`truncated` flag) |
| `GET /stops?bbox&start&end` | Stops (ports, anchorages) in a bbox, optional `mmsi` | window ≤ 31 days, ≤ 5000 (`truncated` flag) |

Timestamps are UTC. Out-of-range parameters return 422; invalid time windows return 400.
Trip and stop endpoints return 503 until `uv run ais derive` has run.

## Layout

```
src/ais/
  ingest.py        fetch DMA zips, typed parse of the raw CSV
  clean.py         named cleaning steps, positions/vessels split, quality log
  derive.py        bad-fix removal, stop detection, trip segmentation, simplified geometry
  store/base.py    AisStore Protocol (DuckDB now, Postgres/PostGIS later)
  store/duckdb_store.py
  api.py           FastAPI app, server-side limits
  ui/index.html    prototype map page served at /
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
planted; trip segmentation is tested on synthetic tracks (`tests/test_derive.py`), one
scenario per rule. `tests/test_dev_slice.py` adds checks on the real dev slice when present.

To eyeball segmentation (bugs show up visually, not in tables):
`uv run python scripts/plot_trips.py` writes `data/derived/plots/trips.png`.
