# p7

AIS data pipeline and API. Stand-in data comes from the Danish Maritime Authority's
open daily dumps (http://aisdata.ais.dk/) while the project database is unavailable.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```sh
uv sync
uv run ais dev      # download, clean and derive the dev slice (2026-09-22..24), ~7 min, ~400 MB
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
| `GET /trips?mmsi&start&end` | Trips of one vessel overlapping the window | window ≤ 31 days |
| `GET /trips/{trip_id}/geometry` | Simplified trip line as GeoJSON `LineString` | |
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

## Agent harness (AI-LAB)

LLM agents that answer questions about AIS data through SQL. They run on AAU's AI-LAB
cluster (SLURM + Singularity + Ceph) and is separate from the `src/ais` pipeline above: it
reads the full DMA months at `data/processed/ais/date=YYYY-MM-DD/ais.parquet` under the
project folder, not `data/clean/`.

```
vLLM (Qwen3-8B, hermes tool calling, GPU job)      slurm/serve_llm.slurm
  -> agent loop: chat, tool calls, final answer     harness/agent.py
     -> query_ais_data: DuckDB over Hive-partitioned harness/ais_tool.py
        Parquet, read-only SQL (SELECT/WITH only, banned keywords, LIMIT capped at 20000,
        30 s timeout)
  -> HTTP: GET /health, POST /ask                    harness/api.py
```

Two-model mode (default for `/ask`), with the single agent above kept as a baseline:

```
Qwen3-8B planner: plan, ask_data(request), check, answer   harness/planner.py
  -> SQL student (Qwen2.5-Coder-1.5B, fine-tuned)          slurm/serve_student.slurm
     request + schema_context.md -> SQL
  -> query_ais_data (same guardrails) -> rows or error back to the planner
```

The planner never sees the schema or writes SQL. Each `ask_data` request must stand alone
(full dates, UTC times, named places, vessel names in capitals), the way the student was
trained; the planner does the arithmetic across requests and rephrases failed ones (an
identical failed request is refused, since the student is deterministic).

Places: the student's prompt lists only the places a request names, looked up in
`harness/places.json` (hand-made sea areas plus ports located from AIS: where moored
vessels that declare the port as destination lie). The planner checks names with
`find_place`; for an unknown place it proposes a box with `propose_place`, which must lie in
the coverage area, have a sensible size and contain vessels in the AIS data, and is then
appended to `data/places_learned.jsonl` (project folder) as unverified. SQL whose lon/lat
box belongs to no place in the request is not run. Rebuild the list with
`python3 build_places.py /ceph/project/P7-Agent-Harness/data/external/un-locode/code-list.csv`
(UN/LOCODE from github.com/datasets/un-locode); edit `places_manual.json` for fixed boxes.

```
harness/           agent.py, planner.py, ais_tool.py, api.py (flat imports, run from here)
                   places.py + places.json (named places; built by build_places.py from
                   places_manual.json, UN/LOCODE and AIS, checked against
                   places_reference_ports.json)
                   build_context.py -> schema_context.md, eval_sql.py + eval/ (text-to-SQL),
                   gen_sql_data.py + sql_templates.py (training data)
slurm/             serve_llm.slurm (vLLM server), serve_student.slurm (SQL student server),
                   model_download.slurm (Hugging Face -> models/), gen_sql_data.slurm
                   (training data, CPU), train_student.slurm, eval_student.slurm
training/          train_student.py (LoRA fine-tune of the SQL student)
scripts/           auto_submit_llm.sh (cron restart), setup_llm_env.sh (builds ~/llm_venv)
legacy/            superseded: llm_watchdog.slurm (self-rescheduling job, replaced by cron),
                   fix_cuda_libs.sh (libcuda workaround, replaced by a bind mount in serve_llm)
```

Paths default to `/ceph/project/P7-Agent-Harness`. Set `P7_PROJECT_DIR` to override them in
the harness and `auto_submit_llm.sh`; the SLURM scripts read `PROJECT_DIR` (their
`#SBATCH --output` log paths are fixed).

Each query connection gets two views and a macro (`ais_tool.SETUP_SQL`): `ais` is every raw
row; `vessel_positions` keeps Class A/B ships with a ship MMSI inside the DMA coverage box and
sets the 102.2 kn "not available" SOG to NULL, the same rules as `src/ais/clean.py`. Raw data
is about half exact duplicates, which the view keeps, hence `count(DISTINCT mmsi)`.
`haversine_m(lat1, lon1, lat2, lon2)` gives metres.

### Text-to-SQL eval

The planned SQL student (Qwen2.5-Coder-1.5B, distilled from Qwen3-8B) is scored on
`harness/eval/questions.toml`: 14 questions plus paraphrases ported from the Phase 3 set
(the trip and stop questions need derived tables the harness lacks). Never train on them.

```sh
cd harness
python3 build_context.py      # schema_context.md, the prompt for teacher, student and eval
python3 eval_sql.py answers   # reference results -> eval/answers.json
python3 eval_sql.py run --out ../../logs/eval/run.jsonl    # served model, see --help
```

Training data (step 3) comes from `harness/gen_sql_data.py`, run as a CPU job against the
llm-serve teacher; output goes to `data/text2sql/` in the project folder, not the repo:

```sh
sbatch slurm/gen_sql_data.slurm templates --per-template 60   # from the repo root
sbatch slurm/gen_sql_data.slurm freeform --prompts 600
sbatch slurm/gen_sql_data.slurm export                        # train.jsonl, val.jsonl
```

`templates` fills the 30 templates in `sql_templates.py` with real sampled values, keeps SQL
that runs with a non-trivial answer, and adds teacher rewrites that keep every date, place,
name, number and condition (checked by regex, SQL-like wording rejected). `freeform` keeps a
teacher-written question when 3 of 4 sampled SQL answers agree. Dates 20-26 Sep 2026 are
never sampled, and `export` drops eval questions, so the eval stays held out. The job needs
`envs/harness` (see the header of `slurm/gen_sql_data.slurm`).

A prediction is correct when its result has the reference rows (extra columns, row order
and rounding are ignored). Qwen3-8B, single shot, no thinking, on 2026-10-11: 12/22 (8/14
canonical) with the current context, 6/22 with the first draft, before the rules on
`count(DISTINCT ts)`, timestamp windows and upper-case names. It still ignores the
`sog >= 1` rule, and fails proximity questions and the duplicate share.

### Restart chain

The vLLM job has a 12 h walltime, so cron restarts it twice a day:

```
cron 05:00 and 17:00
  -> ~/auto_submit_llm.sh       exits unless ~/.llm_auto exists; scancels llm-serve, waits
                                for it to leave the queue, then sbatch from the project folder
  -> slurm/serve_llm.slurm      exits if serving/STOP exists; writes host:port to
                                serving/current_endpoint.txt, then starts vllm serve on :8000
  -> harness/agent.py           reads serving/current_endpoint.txt on every request
  -> p7/slurm/serve_student.slurm   (repo version of auto_submit_llm.sh only) the SQL
                                student on :8001, writes serving/student_endpoint.txt,
                                read by harness/planner.py
```

The endpoint file is written when the job starts, before the model has loaded, so expect a
few minutes of connection errors after a restart. Always submit `serve_llm.slurm` from the
project folder: it falls back to `$SLURM_SUBMIT_DIR` to find `models/` and `serving/`.

The live chain runs `~/auto_submit_llm.sh` and the project folder's `slurm/serve_llm.slurm`,
not the copies in this repo. Copy changes over deliberately. Example crontab (`crontab -e`):

```
0 5,17 * * * ~/auto_submit_llm.sh >> ~/auto_submit_llm.log 2>&1
```

Pause: `rm ~/.llm_auto` (cron) or `touch serving/STOP` (jobs), then
`scancel --name=llm-serve --name=llm-student`. Start the student by hand with
`sbatch slurm/serve_student.slurm [MODEL_DIR]` (default `models/qwen2.5-coder-1.5b-sql-v4`).

### Running the agent and the API

The harness needs its own venv. Never install into `~/llm_venv`: vLLM 0.29.0 pins
`fastapi[standard]>=0.133.0,<0.137.0`.

```sh
python3 -m venv ~/harness_venv
~/harness_venv/bin/pip install -r requirements-harness.txt
source ~/harness_venv/bin/activate
cd harness

python3 planner.py "How many tankers were in the Kattegat on 3 March 2026?"   # two models
python3 agent.py "How many distinct vessels were seen on 2025-12-01?"          # 8B only

uvicorn api:app --host 0.0.0.0 --port 8100     # on the login node
curl -X POST http://ailab-fe01:8100/ask -H "Content-Type: application/json" \
    -d '{"question": "How many distinct vessels were seen on 2025-12-01?"}'
```

`/ask` also takes `mode`: `"routed"` (default: one short Qwen3-8B call decides whether the
SQL student can answer the question with one query; if so, it does and the 8B words the
answer from the rows, otherwise or if that query fails the planner takes over), `"planner"`
(always the planner) or `"single"` (Qwen3-8B alone). It also takes `enable_thinking` (Qwen3
reasoning, default false) and `max_turns` (default 10, 6 single), and returns the answer,
the plan if any, and every call with its request, SQL and result. `scripts/ask.sh` does the
same from the command line and shows every step (`--mode planner`, `--single`).
