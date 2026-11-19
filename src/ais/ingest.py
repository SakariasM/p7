"""Fetch DMA daily dumps and parse the raw CSV into typed columns."""

import os
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import duckdb

from ais.limits import apply_limits

DMA_URL = "http://aisdata.ais.dk/aisdk-{day}.zip"
N_RAW_COLUMNS = 26


def fetch(day: date, dest: Path) -> Path:
    """Download one day's zip to `dest` unless it is already there."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".zip.part")
    url = DMA_URL.format(day=day.isoformat())
    # The timeout is per read, so a stalled connection fails instead of hanging a nightly run.
    with urllib.request.urlopen(url, timeout=120) as resp, part.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    part.rename(dest)
    return dest


@contextmanager
def csv_source(path: Path) -> Generator[str]:
    """Yield a path DuckDB can read_csv from. A .zip is streamed through a FIFO, never extracted."""
    if path.suffix != ".zip":
        yield str(path)
        return
    with tempfile.TemporaryDirectory() as tmp:
        fifo = os.path.join(tmp, "raw.csv")
        os.mkfifo(fifo)
        error: list[BaseException] = []

        def pump() -> None:
            try:
                with (
                    zipfile.ZipFile(path) as zf,
                    zf.open(zf.namelist()[0]) as src,
                    open(fifo, "wb") as dst,
                ):
                    shutil.copyfileobj(src, dst, length=1 << 20)
            except BaseException as e:
                error.append(e)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            yield fifo
        finally:
            if t.is_alive():
                # Reader stopped early (or never opened): open and close our own read end so the
                # writer's open() returns and its next write fails with BrokenPipeError.
                os.close(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))
            t.join()
        if error and not isinstance(error[0], BrokenPipeError):
            raise error[0]


def raw_csv(src: str) -> str:
    """read_csv over a raw DMA CSV: 26 text columns, column00..column25.

    The dialect is stated in full and DuckDB's sniffer is off. Left to guess, it settles the
    quote escape on the first 2,048 rows: a quoted name without a doubled quote there ("DEEP
    8,1") makes it read the file with no escape, and the first doubled quote further down
    (2026-10-02, line 32,960, the name UST-113 "AGAT") fails with 'unterminated quote'."""
    columns = ", ".join(f"'column{i:02d}': 'VARCHAR'" for i in range(N_RAW_COLUMNS))
    src_sql = src.replace("'", "''")
    return (
        f"read_csv('{src_sql}', auto_detect = false, header = false, skip = 1, delim = ',',"
        f" quote = '\"', escape = '\"', all_varchar = true, columns = {{{columns}}})"
    )


def typed_select(src: str) -> str:
    """SELECT over a raw DMA CSV with typed, snake_case columns; sentinels become NULL."""
    return f"""
    SELECT
      strptime(column00, '%d/%m/%Y %H:%M:%S')      AS ts,
      clean_text(column01)                         AS mobile_type,
      TRY_CAST(column02 AS BIGINT)                 AS mmsi,
      TRY_CAST(column03 AS DOUBLE)                 AS lat,
      TRY_CAST(column04 AS DOUBLE)                 AS lon,
      clean_text(column05)                         AS nav_status,
      TRY_CAST(column06 AS DOUBLE)                 AS rot,
      TRY_CAST(column07 AS DOUBLE)                 AS sog,
      TRY_CAST(column08 AS DOUBLE)                 AS cog,
      TRY_CAST(column09 AS SMALLINT)               AS heading,
      TRY_CAST(column10 AS BIGINT)                 AS imo,
      clean_text(column11)                         AS callsign,
      clean_text(column12)                         AS name,
      clean_text(column13)                         AS ship_type,
      clean_text(column14)                         AS cargo_type,
      TRY_CAST(column15 AS DOUBLE)                 AS width,
      TRY_CAST(column16 AS DOUBLE)                 AS length,
      clean_text(column17)                         AS pos_fix_device,
      TRY_CAST(column18 AS DOUBLE)                 AS draught,
      clean_text(column19)                         AS destination,
      try_strptime(column20, '%d/%m/%Y %H:%M:%S')  AS eta,
      clean_text(column21)                         AS data_source_type,
      TRY_CAST(column22 AS DOUBLE)                 AS a,
      TRY_CAST(column23 AS DOUBLE)                 AS b,
      TRY_CAST(column24 AS DOUBLE)                 AS c,
      TRY_CAST(column25 AS DOUBLE)                 AS d
    FROM {raw_csv(src)}
    """


def processed_select(src: str) -> str:
    """The same columns as typed_select, over one day of the project's processed Parquet
    (data/processed/ais/date=YYYY-MM-DD/ais.parquet on AI-LAB).

    Floats go through VARCHAR so FLOAT columns get the CSV's decimal values back: a direct
    cast turns sog 102.2 into 102.19999694824219, which slips past the SOG sentinel check."""
    src_sql = src.replace("'", "''")

    def num(col: str) -> str:
        return f"TRY_CAST({col}::VARCHAR AS DOUBLE)"

    return f"""
    SELECT
      ts,
      clean_text(mobile_type)                      AS mobile_type,
      mmsi::BIGINT                                 AS mmsi,
      {num("lat")}                                 AS lat,
      {num("lon")}                                 AS lon,
      clean_text(nav_status)                       AS nav_status,
      {num("rot")}                                 AS rot,
      {num("sog")}                                 AS sog,
      {num("cog")}                                 AS cog,
      heading::SMALLINT                            AS heading,
      imo::BIGINT                                  AS imo,
      clean_text(callsign)                         AS callsign,
      clean_text(name)                             AS name,
      clean_text(ship_type)                        AS ship_type,
      clean_text(cargo_type)                       AS cargo_type,
      {num("width")}                               AS width,
      {num("length")}                              AS length,
      clean_text(pos_fix_device)                   AS pos_fix_device,
      {num("draught")}                             AS draught,
      clean_text(destination)                      AS destination,
      eta,
      clean_text(data_source)                      AS data_source_type,
      {num("dim_a")}                               AS a,
      {num("dim_b")}                               AS b,
      {num("dim_c")}                               AS c,
      {num("dim_d")}                               AS d
    FROM read_parquet('{src_sql}')
    """


def processed_layout_select(src: str, day: date) -> str:
    """SELECT over a raw DMA CSV in the layout of the project's processed Parquet
    (data/processed/ais/date=YYYY-MM-DD/ais.parquet), which processed_select and the agent
    harness read: every row, the DMA's own text values ('Unknown', 'Undefined'; an empty field
    is NULL), FLOAT/SMALLINT where that dump narrows them, its column names and a `date` column.

    Checked against AI-LAB's dump: 2026-09-30 built this way has the same 24,677,317 rows,
    value for value."""
    return f"""
    SELECT
      strptime(column00, '%d/%m/%Y %H:%M:%S')      AS ts,
      column01                                     AS mobile_type,
      TRY_CAST(column02 AS BIGINT)                 AS mmsi,
      TRY_CAST(column03 AS DOUBLE)                 AS lat,
      TRY_CAST(column04 AS DOUBLE)                 AS lon,
      column05                                     AS nav_status,
      TRY_CAST(column06 AS FLOAT)                  AS rot,
      TRY_CAST(column07 AS FLOAT)                  AS sog,
      TRY_CAST(column08 AS FLOAT)                  AS cog,
      TRY_CAST(column09 AS SMALLINT)               AS heading,
      TRY_CAST(column10 AS BIGINT)                 AS imo,
      column11                                     AS callsign,
      column12                                     AS name,
      column13                                     AS ship_type,
      column14                                     AS cargo_type,
      TRY_CAST(column15 AS SMALLINT)               AS width,
      TRY_CAST(column16 AS SMALLINT)               AS length,
      column17                                     AS pos_fix_device,
      TRY_CAST(column18 AS FLOAT)                  AS draught,
      column19                                     AS destination,
      try_strptime(column20, '%d/%m/%Y %H:%M:%S')  AS eta,
      column21                                     AS data_source,
      TRY_CAST(column22 AS SMALLINT)               AS dim_a,
      TRY_CAST(column23 AS SMALLINT)               AS dim_b,
      TRY_CAST(column24 AS SMALLINT)               AS dim_c,
      TRY_CAST(column25 AS SMALLINT)               AS dim_d,
      DATE '{day.isoformat()}'                     AS date
    FROM {raw_csv(src)}
    """


def write_processed(source: Path, day: date, dest: Path) -> int:
    """Write one day's DMA .zip or .csv as processed Parquet at `dest`, sorted by mmsi, ts,
    and return its row count.

    The file gets its name only when it is complete. Raises ValueError, and leaves no file,
    when the source is empty or has a row whose timestamp is outside `day`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".parquet.partial")
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".tmp-") as tmp:
        # File-backed so the sort of a full day (~25M rows) spills to disk next to the output.
        con = duckdb.connect(os.path.join(tmp, "work.duckdb"))
        apply_limits(con)
        con.execute("SET enable_progress_bar = false; SET preserve_insertion_order = false")
        with csv_source(source) as src:
            con.execute(
                f"COPY ({processed_layout_select(src, day)} ORDER BY mmsi, ts) TO '{partial}'"
                " (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)"
            )
        row = con.execute(
            f"SELECT count(*), count(*) FILTER (ts::DATE IS DISTINCT FROM DATE '{day}')"
            f" FROM read_parquet('{partial}')"
        ).fetchone()
        con.close()
    rows, bad_day = row if row else (0, 0)
    if not rows or bad_day:
        partial.unlink()
        problem = f"{bad_day} rows have a timestamp outside {day}" if rows else "no rows"
        raise ValueError(f"{source}: {problem}")
    partial.rename(dest)
    return rows


@contextmanager
def typed_source(path: Path) -> Generator[str]:
    """Yield a typed SELECT over one day of raw data: a DMA .zip/.csv or a processed .parquet."""
    if path.suffix == ".parquet":
        yield processed_select(str(path))
        return
    with csv_source(path) as src:
        yield typed_select(src)


CLEAN_TEXT_MACRO = """
CREATE OR REPLACE MACRO clean_text(s) AS
  CASE WHEN trim(s) IN ('', 'Unknown', 'Undefined', 'Unknown value') THEN NULL ELSE trim(s) END
"""
