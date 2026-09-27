"""Command line: `uv run ais --help`."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import duckdb
import typer

from ais.clean import clean_day
from ais.ingest import fetch as fetch_zip
from ais.paths import CLEAN_TABLES, DataPaths

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEV_DAYS = (date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24))
OPENAPI_FILE = Path("api/openapi.json")

Day = Annotated[datetime, typer.Argument(formats=["%Y-%m-%d"], help="YYYY-MM-DD")]


def _paths() -> DataPaths:
    return DataPaths.from_env()


def _is_clean(paths: DataPaths, day: date) -> bool:
    return all((paths.partition(t, day) / "part.parquet").exists() for t in CLEAN_TABLES)


def _process(paths: DataPaths, day: date, keep_raw: bool) -> None:
    typer.echo(f"{day}: fetching")
    raw = fetch_zip(day, paths.raw_zip(day))
    typer.echo(f"{day}: cleaning")
    for r in clean_day(raw, day, paths):
        typer.echo(f"  {r.name:<24} {r.action:<4} {r.rows_affected:>10,} of {r.rows_in:>11,}")
    if not keep_raw:
        raw.unlink()


@app.command()
def fetch(day: Day) -> None:
    """Download one DMA daily zip into data/raw/."""
    typer.echo(fetch_zip(day.date(), _paths().raw_zip(day.date())))


@app.command()
def clean(
    day: Day,
    keep_raw: Annotated[bool, typer.Option(help="Keep the 600 MB raw zip afterwards")] = False,
) -> None:
    """Fetch (if needed) and clean one day. Re-running replaces only that day."""
    _process(_paths(), day.date(), keep_raw)
    build_duckdb()


@app.command()
def dev(
    force: Annotated[bool, typer.Option(help="Re-clean days that already exist")] = False,
    keep_raw: bool = False,
) -> None:
    """Reproduce the dev slice (2026-09-22..24) and build data/ais.duckdb."""
    paths = _paths()
    for day in DEV_DAYS:
        if _is_clean(paths, day) and not force:
            typer.echo(f"{day}: already clean")
            continue
        _process(paths, day, keep_raw)
    build_duckdb()


@app.command("build-duckdb")
def build_duckdb() -> None:
    """(Re)create data/ais.duckdb with views over data/clean/. Open it from the repo root."""
    from ais.store.duckdb_store import create_views

    paths = _paths()
    paths.duckdb_file.unlink(missing_ok=True)
    with duckdb.connect(str(paths.duckdb_file)) as con:
        create_views(con, paths)
    typer.echo(f"wrote {paths.duckdb_file} (views: positions, vessels, vessels_daily, quality)")


@app.command()
def quality() -> None:
    """Print the per-filter row counts for every cleaned day."""
    paths = _paths()
    duckdb.sql(
        "SELECT date, step, action, rows_affected, rows_in,"
        " round(100.0 * rows_affected / rows_in, 2) AS pct"
        f" FROM read_parquet('{paths.glob('quality')}', hive_partitioning = true)"
        " ORDER BY date, step_order"
    ).show(max_rows=1000)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API (docs at /docs)."""
    import uvicorn

    uvicorn.run("ais.api:app_from_env", factory=True, host=host, port=port, reload=reload)


@app.command()
def openapi(
    out: Path = OPENAPI_FILE,
    check: Annotated[bool, typer.Option(help="Fail if the file is out of date")] = False,
) -> None:
    """Write the OpenAPI contract the frontend generates its types from."""
    from ais.api import create_app

    text = json.dumps(create_app(None).openapi(), indent=2, sort_keys=True) + "\n"
    if check:
        if not out.exists() or out.read_text() != text:
            typer.echo(f"{out} is out of date; run `uv run ais openapi`", err=True)
            raise typer.Exit(1)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    typer.echo(f"wrote {out}")
