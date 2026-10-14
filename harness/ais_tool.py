"""
Read-only SQL query tool against the AIS trajectory Parquet store.

Data layout (produced by the ingestion pipeline, not this harness):
  /ceph/project/P7-Agent-Harness/data/processed/ais/date=YYYY-MM-DD/ais.parquet

Set P7_PROJECT_DIR to point at a different project root.
"""

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import duckdb

PROJECT_DIR = Path(os.environ.get("P7_PROJECT_DIR", "/ceph/project/P7-Agent-Harness"))
DATA_DIR = PROJECT_DIR / "data" / "processed" / "ais"
PARQUET_GLOB = f"{DATA_DIR}/*/ais.parquet"

# Date coverage, read from the date=YYYY-MM-DD partition folders so the tool
# description stays correct as more days are ingested.
_DATES = sorted(p.parent.name.removeprefix("date=") for p in DATA_DIR.glob("date=*/ais.parquet"))
DATE_RANGE = f"{_DATES[0]} through {_DATES[-1]}" if _DATES else "no dates (store is empty)"

# Cleaned ship positions (clean_positions.py): ships in coverage, exact duplicates removed,
# sog > 50 kn set to NULL, bad GPS positions (outliers and spikes) removed. Used only when
# every raw day has been cleaned; otherwise vessel_positions falls back to the on-the-fly
# filters below (same ship/coverage rules, duplicates and bad positions kept).
CLEAN_DIR = PROJECT_DIR / "data" / "processed" / "ais_clean"
CLEAN_GLOB = f"{CLEAN_DIR}/*/ais.parquet"
_CLEAN_DATES = {p.parent.name.removeprefix("date=") for p in CLEAN_DIR.glob("date=*/ais.parquet")}
USE_CLEAN = bool(_DATES) and set(_DATES) <= _CLEAN_DATES
if _DATES and not USE_CLEAN:
    print(f"ais_tool: {len(set(_DATES) - _CLEAN_DATES)} days not cleaned yet, vessel_positions "
          "uses the raw data with on-the-fly filters", file=sys.stderr)

_VESSEL_POSITIONS = (
    f"SELECT * FROM read_parquet('{CLEAN_GLOB}', hive_partitioning=1)" if USE_CLEAN else
    """SELECT * REPLACE (CASE WHEN sog >= 102.2 THEN NULL ELSE sog END AS sog) FROM ais
  WHERE mobile_type IN ('Class A', 'Class B')
    AND mmsi BETWEEN 200000000 AND 799999999
    AND lat BETWEEN 50 AND 64 AND lon BETWEEN -5 AND 32"""
)

# Every query connection gets these.
SETUP_SQL = f"""
CREATE MACRO haversine_m(lat1, lon1, lat2, lon2) AS
  2 * 6371008.8 * asin(sqrt(
      pow(sin(radians(lat2 - lat1) / 2), 2)
    + cos(radians(lat1)) * cos(radians(lat2)) * pow(sin(radians(lon2 - lon1) / 2), 2)));
CREATE VIEW ais AS SELECT * FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=1);
CREATE VIEW vessel_positions AS {_VESSEL_POSITIONS};
"""

DEFAULT_ROW_LIMIT = 1000
MAX_ROW_LIMIT = 20000
QUERY_TIMEOUT_SECONDS = 30

_BANNED_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "detach", "copy", "pragma", "install", "load", "export", "import",
    "call", "set", "execute", "vacuum", "checkpoint", "rename",
]
_BANNED_RE = re.compile(r"\b(" + "|".join(_BANNED_KEYWORDS) + r")\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit\s+\d+\b", re.IGNORECASE)
_LEADING_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class QueryRejected(Exception):
    pass


def _validate(sql: str) -> str:
    stripped = sql.strip().rstrip(";")

    if ";" in stripped:
        raise QueryRejected("Only a single statement is allowed (no ';' inside the query).")

    if not _LEADING_RE.match(stripped):
        raise QueryRejected("Query must start with SELECT or WITH - this tool is read-only.")

    if _BANNED_RE.search(stripped):
        raise QueryRejected("Query contains a disallowed keyword (only read-only SELECT is permitted).")

    if not _LIMIT_RE.search(stripped):
        stripped = f"{stripped}\nLIMIT {DEFAULT_ROW_LIMIT}"
    else:
        def _cap(match):
            n = int(re.search(r"\d+", match.group(0)).group(0))
            return f"LIMIT {min(n, MAX_ROW_LIMIT)}"
        stripped = _LIMIT_RE.sub(_cap, stripped, count=1)

    return stripped


def extract_sql(text: str) -> str:
    """The SQL in a model reply: drops <think> blocks, takes the last fenced block if any, and
    removes full-line -- comments and the trailing ';' (models add notes after the query)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    fenced = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = fenced[-1] if fenced else text
    lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    return "\n".join(lines).strip().rstrip(";").strip()


def connect() -> duckdb.DuckDBPyConnection:
    """An in-memory connection with the views and macros from SETUP_SQL."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    if _DATES:
        con.execute(SETUP_SQL)
    return con


# Deterministic work counters (DuckDB profiling, kept in memory) for the eval's speed score.
_PROFILE_METRICS = json.dumps({"CUMULATIVE_ROWS_SCANNED": "true", "CUMULATIVE_CARDINALITY": "true"})


def query_ais_data(sql: str, profile: bool = False) -> dict:
    """Run one read-only query with the guardrails. profile=True also returns rows_scanned
    and cardinality (rows through all operators), which do not vary between runs."""
    try:
        safe_sql = _validate(sql)
    except QueryRejected as e:
        return {"error": str(e), "sql_attempted": sql}

    con = connect()
    if profile:
        con.execute("PRAGMA enable_profiling='no_output'")
        con.execute(f"PRAGMA custom_profiling_settings='{_PROFILE_METRICS}'")
    result_holder = {}
    error_holder = {}

    def _run():
        try:
            result_holder["df"] = con.execute(safe_sql).fetchdf()
        except Exception as e:
            error_holder["error"] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    start = time.time()
    thread.start()
    thread.join(QUERY_TIMEOUT_SECONDS)

    if thread.is_alive():
        try:
            con.interrupt()
        except Exception:
            pass
        thread.join(5)
        return {
            "error": f"Query exceeded {QUERY_TIMEOUT_SECONDS}s timeout and was cancelled.",
            "sql_attempted": safe_sql,
        }

    if "error" in error_holder:
        return {"error": error_holder["error"], "sql_attempted": safe_sql}

    df = result_holder["df"]
    truncated = len(df) >= MAX_ROW_LIMIT
    result = {
        "columns": list(df.columns),
        "rows": df.astype(object).where(df.notnull(), None).values.tolist(),
        "row_count": len(df),
        "truncated": truncated,
        "elapsed_seconds": round(time.time() - start, 3),
        "sql_run": safe_sql,
    }
    if profile:
        info = json.loads(con.get_profiling_information(format="json"))
        result["rows_scanned"] = info["cumulative_rows_scanned"]
        result["cardinality"] = info["cumulative_cardinality"]
    return result


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_ais_data",
        "description": (
            "Run a read-only DuckDB SQL SELECT query against Danish AIS vessel data "
            "and return the matching rows. Query the view vessel_positions: ship "
            "position reports (Class A/B transponders, ship MMSIs, positions within "
            "50-64N 5W-32E; sog is NULL when not available"
            + ("; exact duplicates, bad GPS positions and speeds above 50 kn removed"
               if USE_CLEAN else "") + "). The view ais has every "
            "raw row incl. base stations and bad fixes; use it only for data-quality "
            "questions. "
            "Columns: ts (TIMESTAMP, UTC), mobile_type, mmsi (BIGINT), lat, lon (DOUBLE), "
            "nav_status, rot, sog (speed over ground, knots), cog (course over ground), "
            "heading, imo, callsign, name, ship_type, cargo_type, width, length, "
            "pos_fix_device, draught, destination, eta (TIMESTAMP), data_source, "
            "dim_a, dim_b, dim_c, dim_d, date (DATE, partition column - always filter "
            f"on this first for performance; data covers {DATE_RANGE}, "
            "about 19 million rows per day). The same report is often received several "
            "times: count vessels with count(DISTINCT mmsi). Missing text values are "
            "strings such as 'Undefined' or 'Unknown value', not NULL. "
            "haversine_m(lat1, lon1, lat2, lon2) returns the distance in metres. "
            "Queries are cancelled after "
            f"{QUERY_TIMEOUT_SECONDS}s: aggregates over more than about a month of "
            "rows will time out unless they also filter on mmsi or another narrow "
            "condition, so split long periods into smaller date ranges. "
            "Only SELECT/WITH statements are allowed; a LIMIT is enforced automatically "
            "if you don't include one (max 20000 rows)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single read-only SQL SELECT statement.",
                }
            },
            "required": ["sql"],
        },
    },
}
