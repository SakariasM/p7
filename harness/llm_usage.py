"""
Counts the LLM calls the harness makes while it answers a question: which model, tokens in and
out, and seconds per call. Every OpenAI client in the harness is made with client(base_url);
its HTTP client records each chat completion in the list of the question being answered, set
with track(). api.py and ask.py put summary() of it in the question log, eval_sql.py in its
results, so the report can show how LLM use changes as the learned-question store grows.
"""

import contextlib
import contextvars
import json

import httpx
from openai import DefaultHttpxClient, OpenAI

_calls: contextvars.ContextVar[list | None] = contextvars.ContextVar("llm_calls", default=None)


def _record(response: httpx.Response) -> None:
    calls = _calls.get()
    if calls is None or not response.request.url.path.endswith("/chat/completions"):
        return
    response.read()
    try:
        usage = response.json().get("usage") or {}
        model = json.loads(response.request.content or b"{}").get("model")
    except ValueError:
        usage, model = {}, None
    calls.append({"model": model, "prompt_tokens": usage.get("prompt_tokens") or 0,
                  "completion_tokens": usage.get("completion_tokens") or 0,
                  "seconds": round(response.elapsed.total_seconds(), 3)})


def client(base_url: str) -> OpenAI:
    """An OpenAI client for a vLLM server whose calls are counted (with the default timeouts
    and connection limits)."""
    return OpenAI(base_url=base_url, api_key="not-needed",
                  http_client=DefaultHttpxClient(event_hooks={"response": [_record]}))


@contextlib.contextmanager
def track():
    """Collect the LLM calls made inside the block (in this thread) in the list it yields."""
    calls: list[dict] = []
    token = _calls.set(calls)
    try:
        yield calls
    finally:
        _calls.reset(token)


def summary(calls: list[dict]) -> dict:
    """Calls, tokens and seconds, in total and for the student and the 8B planner."""
    out = {"calls": len(calls), "student_calls": 0, "planner_calls": 0, "prompt_tokens": 0,
           "completion_tokens": 0, "seconds": 0.0}
    for c in calls:
        role = "student" if "student" in str(c["model"]) else "planner"
        out[f"{role}_calls"] += 1
        out["prompt_tokens"] += c["prompt_tokens"]
        out["completion_tokens"] += c["completion_tokens"]
        out["seconds"] = round(out["seconds"] + c["seconds"], 3)
    return out
