"""Optional DuckDB limits for the pipeline, for a machine it shares with the services."""

import os

import duckdb


def apply_limits(con: duckdb.DuckDBPyConnection) -> None:
    """Apply AIS_THREADS and AIS_MEMORY_LIMIT (e.g. 8 and 16GB) when they are set. Unset,
    DuckDB takes every core and up to 80% of the memory."""
    if threads := os.environ.get("AIS_THREADS"):
        con.execute(f"SET threads = {int(threads)}")
    if memory := os.environ.get("AIS_MEMORY_LIMIT"):
        con.execute("SET memory_limit = ?", [memory])
