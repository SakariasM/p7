"""
FastAPI wrapper around the AIS agents, so the harness can be called over HTTP instead of
only from the CLI. mode "routed" (default) sends simple questions straight to the SQL student
and the rest through the planner (planner.run_routed); "planner" always uses the planner;
"single" is the Qwen3-8B-only agent in agent.py. routed and planner need the student job too.

Run (dev, on the login node):
    source ~/llm_venv/bin/activate
    cd /ceph/project/P7-Agent-Harness/harness
    nohup python3 -m uvicorn api:app --host 0.0.0.0 --port 8100 > ../logs/api.log 2>&1 &

August's map page (src/ais/ui, `ais serve`) asks through its own /ask, which forwards here
with "map": true; the response then also carries map hints (map_hints.py) that the page uses
to show the ships, places, time or route behind the answer.

Every question is logged with an id to logs/questions/<date>.jsonl (ask.log_run, as for the
CLI), with the LLM calls it took (llm_usage.py) and the size of the learned-question store;
POST /feedback {"id", "verdict": "correct"|"wrong", "note"} records a verdict on it.

Then from anywhere on the AAU network:
    curl -X POST http://ailab-fe01:8100/ask -H "Content-Type: application/json" \
        -d '{"question": "How many distinct vessels were seen on 2025-12-01?"}'
    (add "mode": "planner" or "single" to compare)
"""

import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

import agent
import llm_usage
import planner
from ask import log_run, save_feedback
from map_hints import describe, map_hints

app = FastAPI(title="P7 AIS Agent Harness")


class SelectedArea(BaseModel):
    """Exact WGS84 rectangle selected by a user on the map."""

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def ordered_bounds(self):
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("area must have west < east and south < north")
        return self

    def add_to_question(self, question: str) -> str:
        return (
            f"{question}\n\n"
            "Use the exact map rectangle selected by the user for this question. "
            f"Its west longitude is {self.west:.6f}, south latitude is {self.south:.6f}, "
            f"east longitude is {self.east:.6f}, and north latitude is {self.north:.6f}. "
            "Include every position inside or on this rectangle. This rectangle overrides any "
            "place name in the question. Repeat all four coordinates in every data request and "
            "do not substitute a named place."
        )

    def box(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


class AskRequest(BaseModel):
    question: str
    mode: Literal["routed", "planner", "single"] = "routed"
    enable_thinking: bool = False
    max_turns: int | None = None  # default: 10 for routed/planner, 6 for single
    map: bool = False  # asked from the map page: draw map/route requests, return map hints
    selected_area: SelectedArea | None = None


class AskResponse(BaseModel):
    answer: str
    plan: str | None = None
    tool_calls: list[dict]
    place_calls: list[dict] = []
    route: str | None = None
    route_reason: str | None = None
    question_as_read: str | None = None
    map: dict | None = None
    id: str | None = None  # run id in the question log, for /feedback


class FeedbackRequest(BaseModel):
    id: str = Field(pattern=r"^\d{8}-\d{6}-[0-9a-f]{4,6}$")
    verdict: Literal["correct", "wrong"]
    note: str = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    question = req.selected_area.add_to_question(req.question) if req.selected_area else req.question
    source = "map" if req.map else "api"
    start = time.time()
    with llm_usage.track() as llm_calls:
        try:
            run = {"routed": planner.run_routed, "planner": planner.run,
                   "single": agent.run}[req.mode]
            max_turns = req.max_turns or (6 if req.mode == "single" else 10)
            extra = {"map_view": True} if req.map and req.mode == "routed" else {}
            result = run(
                question,
                max_turns=max_turns,
                enable_thinking=req.enable_thinking,
                return_trace=True,
                **extra,
            )
            if req.map:
                result["map"] = map_hints(
                    question,
                    result,
                    selected_area=req.selected_area.box() if req.selected_area else None,
                )
                if result.get("answer") == planner.MAP_REPLY:  # a map request: say what is shown
                    result["answer"] = describe(result["map"])
        except Exception as e:
            log_run(question, {"error": str(e)}, req.mode, round(time.time() - start, 1),
                    source, req.enable_thinking, llm=llm_usage.summary(llm_calls))
            raise HTTPException(status_code=502, detail=f"agent error: {e}")
    result["id"] = log_run(question, result, req.mode, round(time.time() - start, 1), source,
                           req.enable_thinking, llm=llm_usage.summary(llm_calls))
    return result


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if not save_feedback(req.id, req.verdict, req.note.strip(), "map"):
        raise HTTPException(status_code=404, detail=f"no logged question with id {req.id}")
    return {"saved": True}
