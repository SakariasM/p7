"""
FastAPI wrapper around the AIS agent (agent.run), so the harness can be
called over HTTP instead of only from the CLI.

Run (dev, on the login node):
    source ~/llm_venv/bin/activate
    cd /ceph/project/P7-Agent-Harness/harness
    nohup python3 -m uvicorn api:app --host 0.0.0.0 --port 8100 > ../logs/api.log 2>&1 &

Then from anywhere on the AAU network:
    curl -X POST http://ailab-fe01:8100/ask -H "Content-Type: application/json" \
        -d '{"question": "How many distinct vessels were seen on 2025-12-01?"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import run

app = FastAPI(title="P7 AIS Agent Harness")


class AskRequest(BaseModel):
    question: str
    enable_thinking: bool = False
    max_turns: int = 6


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        result = run(
            req.question,
            max_turns=req.max_turns,
            enable_thinking=req.enable_thinking,
            return_trace=True,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"agent error: {e}")
    return result
