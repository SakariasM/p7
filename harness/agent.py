"""
Minimal agent loop against the vLLM OpenAI-compatible endpoint + query_ais_data tool.

CLI usage:
    source ~/llm_venv/bin/activate
    cd /ceph/project/P7-Agent-Harness/harness
    python3 agent.py "How many distinct vessels were seen on 2025-12-01?"
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from ais_tool import PROJECT_DIR, TOOL_SCHEMA, query_ais_data

ENDPOINT_FILE = PROJECT_DIR / "serving" / "current_endpoint.txt"
# vLLM serves the model under the path it was started with (see slurm/serve_llm.slurm).
# The planner model as vLLM serves it (its path or --served-model-name). P7_PLANNER_MODEL
# sets it on hosts that serve another build, e.g. Qwen3-8B-FP8 on a 16 GB GPU.
MODEL_PATH = os.environ.get("P7_PLANNER_MODEL", str(PROJECT_DIR / "models" / "qwen3-8b"))

SYSTEM_PROMPT = (
    "You are a data analyst assistant for Danish AIS vessel-trajectory data. "
    "Answer questions by calling the query_ais_data tool to run SQL against the "
    "real data - never guess numbers. Always filter on the `date` partition "
    "column when you know the date range, to keep queries fast. After you get "
    "tool results, give a concise, direct answer to the user's question."
)
# Same schema, rules and places as the SQL student gets (minus its "reply with SQL only"
# line), so single mode is a like-for-like baseline for planner.py.
_SCHEMA_CONTEXT = Path(__file__).with_name("schema_context.md").read_text()
SYSTEM_PROMPT += "\n\n" + _SCHEMA_CONTEXT.split("\n\n", 1)[1]


def get_base_url() -> str:
    endpoint = ENDPOINT_FILE.read_text().strip()
    return f"http://{endpoint}/v1"


def run(question: str, max_turns: int = 6, enable_thinking: bool = False, return_trace: bool = False):
    client = OpenAI(base_url=get_base_url(), api_key="not-needed")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_call_trace = []

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        choice = response.choices[0]
        messages.append(choice.message.model_dump(exclude_none=True))

        if not choice.message.tool_calls:
            answer = choice.message.content
            if return_trace:
                return {"answer": answer, "tool_calls": tool_call_trace}
            return answer

        for tool_call in choice.message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            sql = args.get("sql")
            print(f"[tool call] {tool_call.function.name}({sql})", file=sys.stderr)
            result = query_ais_data(sql)
            tool_call_trace.append({"sql": sql, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    final = "Gave up after max_turns without a final answer."
    if return_trace:
        return {"answer": final, "tool_calls": tool_call_trace}
    return final


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many distinct vessels (MMSI) were seen on 2025-12-01?"
    print(run(question))
