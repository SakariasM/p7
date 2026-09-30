from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from ais.clean import StepResult, clean_day
from ais.derive import DeriveResult, derive
from ais.paths import DataPaths
from ais.store.duckdb_store import DuckDBStore

FIXTURE = Path(__file__).parent / "fixtures" / "mini.csv"
DAY = date(2026, 9, 24)
PREV_DAY = date(2026, 9, 23)


def fixture_for(day: date, tmp: Path) -> Path:
    """mini.csv is dated 2026-09-24; write a copy shifted to `day`."""
    text = FIXTURE.read_text().replace("24/09/2026", day.strftime("%d/%m/%Y"))
    out = tmp / f"mini-{day}.csv"
    out.write_text(text)
    return out


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    return DataPaths(tmp_path / "data")


@pytest.fixture
def cleaned(paths: DataPaths, tmp_path: Path) -> dict[date, list[StepResult]]:
    return {d: clean_day(fixture_for(d, tmp_path), d, paths) for d in (PREV_DAY, DAY)}


@pytest.fixture
def store(paths: DataPaths, cleaned: dict[date, list[StepResult]]) -> Iterator[DuckDBStore]:
    s = DuckDBStore(paths)
    yield s
    s.close()


@pytest.fixture
def derived(paths: DataPaths) -> DeriveResult:
    from .test_derive import scenario, write_clean

    write_clean(paths, scenario())
    return derive(paths)


@pytest.fixture
def dstore(paths: DataPaths, derived: DeriveResult) -> Iterator[DuckDBStore]:
    s = DuckDBStore(paths)
    yield s
    s.close()
