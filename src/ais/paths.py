"""Data directory layout: raw/ -> clean/ -> derived/, each stage reading only the one before."""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

CLEAN_TABLES = ("positions", "vessels", "quality")


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def from_env(cls) -> "DataPaths":
        return cls(Path(os.environ.get("AIS_DATA_DIR", "data")))

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def clean(self) -> Path:
        return self.root / "clean"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def duckdb_file(self) -> Path:
        return self.root / "ais.duckdb"

    def raw_zip(self, day: date) -> Path:
        return self.raw / f"aisdk-{day.isoformat()}.zip"

    def partition(self, table: str, day: date) -> Path:
        return self.clean / table / f"date={day.isoformat()}"

    def glob(self, table: str) -> str:
        return str(self.clean / table / "*" / "*.parquet")
