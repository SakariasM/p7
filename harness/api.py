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

Then from anywhere on the AAU network:
    curl -X POST http://ailab-fe01:8100/ask -H "Content-Type: application/json" \
        -d '{"question": "How many distinct vessels were seen on 2025-12-01?"}'
    (add "mode": "planner" or "single" to compare)
"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import agent
import planner
from map_hints import describe, map_hints

app = FastAPI(title="P7 AIS Agent Harness")


class AskRequest(BaseModel):
    question: str
    mode: Literal["routed", "planner", "single"] = "routed"
    enable_thinking: bool = False
    max_turns: int | None = None  # default: 10 for routed/planner, 6 for single
    map: bool = False  # asked from the map page: draw map/route requests, return map hints


class AskResponse(BaseModel):
    answer: str
    plan: str | None = None
    tool_calls: list[dict]
    place_calls: list[dict] = []
    route: str | None = None
    route_reason: str | None = None
    question_as_read: str | None = None
    map: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        run = {"routed": planner.run_routed, "planner": planner.run, "single": agent.run}[req.mode]
        max_turns = req.max_turns or (6 if req.mode == "single" else 10)
        extra = {"map_view": True} if req.map and req.mode == "routed" else {}
        result = run(
            req.question,
            max_turns=max_turns,
            enable_thinking=req.enable_thinking,
            return_trace=True,
            **extra,
        )
        if req.map:
            result["map"] = map_hints(req.question, result)
            if result.get("answer") == planner.MAP_REPLY:  # a map request: say what is shown
                result["answer"] = describe(result["map"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"agent error: {e}")
    return result
