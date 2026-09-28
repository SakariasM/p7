"""Phase 3: the question set (questions/questions.toml) and its reference answers."""

import json
import tomllib
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import duckdb
from pydantic import BaseModel, Field

from ais.derive import MACROS
from ais.paths import DataPaths
from ais.store.duckdb_store import create_views

QUESTIONS_DIR = Path("questions")
QUESTIONS_FILE = QUESTIONS_DIR / "questions.toml"
ANSWERS_FILE = QUESTIONS_DIR / "answers.json"

Tag = Literal[
    "spatial_range",
    "time_window",
    "per_trip",
    "proximity",
    "stops",
    "vessel_lookup",
    "ship_type",
    "data_quality",
]


class Question(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    question: str
    paraphrases: list[str] = []
    tags: list[Tag] = Field(min_length=1)
    sql: str
    check_sql: str | None = None


class Answer(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    questions = [Question.model_validate(q) for q in data["question"]]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return questions


def connect(paths: DataPaths) -> duckdb.DuckDBPyConnection:
    """A connection with the same views, spatial extension and macros the SQL expects."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute("INSTALL spatial; LOAD spatial")
    con.execute(MACROS)
    create_views(con, paths)
    return con


def _json_value(v: Any) -> Any:
    if isinstance(v, datetime | date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def run(con: duckdb.DuckDBPyConnection, sql: str) -> Answer:
    cur = con.cursor()
    cur.execute(sql)
    columns = [d[0] for d in cur.description or []]
    rows = [[_json_value(v) for v in row] for row in cur.fetchall()]
    return Answer(columns=columns, rows=rows)


def answer_all(con: duckdb.DuckDBPyConnection, questions: list[Question]) -> dict[str, Answer]:
    return {q.id: run(con, q.sql) for q in questions}


def dump_answers(answers: dict[str, Answer]) -> str:
    data = {k: v.model_dump() for k, v in answers.items()}
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def load_answers(path: Path = ANSWERS_FILE) -> dict[str, Answer]:
    data: dict[str, Any] = json.loads(path.read_text())
    return {k: Answer.model_validate(v) for k, v in data.items()}
