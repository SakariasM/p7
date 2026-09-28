"""The question set is well-formed, every query runs, and answers.json covers it."""

import pytest

from ais.derive import DeriveResult
from ais.paths import DataPaths
from ais.questions import ANSWERS_FILE, Question, connect, load_answers, load_questions, run

QUESTIONS = load_questions()


def test_size_and_coverage() -> None:
    assert 30 <= len(QUESTIONS) <= 50
    assert sum(1 for q in QUESTIONS if q.paraphrases) >= 10
    tags = {t for q in QUESTIONS for t in q.tags}
    assert {"spatial_range", "time_window", "per_trip", "proximity", "stops"} <= tags
    assert sum(1 for q in QUESTIONS if q.check_sql) >= 3


def test_every_question_has_a_stored_answer() -> None:
    assert set(load_answers(ANSWERS_FILE)) == {q.id for q in QUESTIONS}


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q.id)
def test_sql_runs_on_synthetic_data(paths: DataPaths, derived: DeriveResult, q: Question) -> None:
    """Catches typos and schema drift in CI, where the real dev slice is absent."""
    with connect(paths) as con:
        run(con, q.sql)
        if q.check_sql:
            run(con, q.check_sql)
