"""
Two-model agent: Qwen3-8B plans and answers, the fine-tuned SQL student writes the SQL.

The planner never sees the schema or writes SQL. It writes a short plan, then calls
ask_data(request) with one self-contained question per data step; the student turns the
request into SQL (schema_context.md as its prompt), query_ais_data runs it with the usual
guardrails, and the SQL plus rows (or the error) go back to the planner, which can rephrase,
add steps, do the arithmetic and write the answer. agent.py is the single-model baseline.

CLI usage (needs both llm-serve and the student job, see slurm/serve_student.slurm):
    cd harness
    python3 planner.py "How many tankers were in the Kattegat on 3 March 2026?"
"""

import json
import re
import sys
from pathlib import Path

from openai import OpenAI

from agent import MODEL_PATH, get_base_url
from ais_tool import DATE_RANGE, PROJECT_DIR, extract_sql, query_ais_data
from build_context import PLACES

STUDENT_ENDPOINT_FILE = PROJECT_DIR / "serving" / "student_endpoint.txt"
STUDENT_MODEL = "sql-student"  # --served-model-name in slurm/serve_student.slurm
SCHEMA_CONTEXT = Path(__file__).with_name("schema_context.md").read_text()
MAX_ROWS_FOR_PLANNER = 30  # the planner has an 8192-token context

PLANNER_PROMPT = f"""\
You answer questions about Danish AIS ship-tracking data ({DATE_RANGE}): position reports
with time, position, speed, course, navigational status, ship type, name, size and
destination. You cannot query the data yourself. A SQL assistant answers data requests
through the tool ask_data: send it one plain-language request at a time and it returns the
SQL it ran and the rows, or an error.

How to work:
1. Start with "Plan:" and a short numbered list of the data you need.
2. Call ask_data for each step. Every request must stand on its own:
   - one question that one query can answer, e.g. "How many tankers were in the Kattegat
     between 08:00 and 09:00 UTC on 3 March 2026?"
   - full dates with day, month and year; times in UTC
   - places by name from this list, or as a lon/lat box: {", ".join(PLACES)}
   - vessel names in capitals as broadcast (e.g. TYCHO BRAHE); ship types such as Cargo,
     Fishing, Passenger, Tanker, Sailing, Tug, Pleasure, Pilot, HSC, SAR, Military
   - at most a few days per request; split longer periods (one request per day or week)
     and combine the results yourself
   - no SQL, table or column names
3. Do the arithmetic yourself: differences, ratios, percentages, totals across requests.
4. Check each result against its request and SQL. If it is an error, empty or implausible,
   rephrase the request more simply or split it, at most twice per step. Sending the same
   request again gives the same result.
5. Answer concisely with the numbers, dates and places, describing each number by what the
   SQL actually computed (e.g. distinct vessels, not reports). Say if a step failed or data
   was missing. Never invent numbers."""

ASK_DATA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_data",
        "description": (
            "Ask the SQL assistant one self-contained question about the AIS data. Returns "
            "the SQL it ran and the resulting rows, or an error."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "One plain-language data question with explicit dates, "
                                   "times, places and names.",
                }
            },
            "required": ["request"],
        },
    },
}


def student_base_url() -> str:
    return f"http://{STUDENT_ENDPOINT_FILE.read_text().strip()}/v1"


def strip_thinking(text: str | None) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def ask_data(student: OpenAI, request: str) -> dict:
    """The student writes SQL for the request; it runs through query_ais_data."""
    response = student.chat.completions.create(
        model=STUDENT_MODEL,
        messages=[{"role": "system", "content": SCHEMA_CONTEXT},
                  {"role": "user", "content": request}],
        temperature=0,
        max_tokens=512,
    )
    sql = extract_sql(response.choices[0].message.content)
    return {"request": request, "sql": sql, "result": query_ais_data(sql)}


def for_planner(step: dict) -> dict:
    """What the planner sees of a step: the SQL and at most MAX_ROWS_FOR_PLANNER rows."""
    result = step["result"]
    if "error" in result:
        return {"sql": step["sql"], "error": result["error"]}
    rows = result["rows"]
    return {
        "sql": step["sql"],
        "columns": result["columns"],
        "rows": rows[:MAX_ROWS_FOR_PLANNER],
        "row_count": len(rows),
        "rows_shown": min(len(rows), MAX_ROWS_FOR_PLANNER),
    }


def run(question: str, max_turns: int = 10, enable_thinking: bool = False,
        return_trace: bool = False):
    planner = OpenAI(base_url=get_base_url(), api_key="not-needed")
    student = OpenAI(base_url=student_base_url(), api_key="not-needed")

    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": question},
    ]
    plan = None
    steps = []
    failed = set()  # requests that already failed; the student is deterministic

    for _ in range(max_turns):
        response = planner.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            tools=[ASK_DATA_SCHEMA],
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if plan is None and strip_thinking(message.content):
            plan = strip_thinking(message.content)

        if not message.tool_calls:
            answer = strip_thinking(message.content)
            if return_trace:
                return {"answer": answer, "plan": plan, "tool_calls": steps}
            return answer

        for tool_call in message.tool_calls:
            try:
                request = json.loads(tool_call.function.arguments).get("request", "")
            except json.JSONDecodeError:
                request = ""
            key = " ".join(request.lower().split())
            if tool_call.function.name != "ask_data" or not key:
                step = {"request": request, "sql": "",
                        "result": {"error": "Call ask_data with a non-empty 'request' string."}}
            elif key in failed:
                step = {"request": request, "sql": "",
                        "result": {"error": "This exact request already failed and would fail "
                                            "again. Rephrase it more simply or split it."}}
            else:
                step = ask_data(student, request)
                if "error" in step["result"]:
                    failed.add(key)
            print(f"[ask_data] {request}\n    -> {step['sql']}", file=sys.stderr)
            steps.append(step)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(for_planner(step), default=str),
            })

    final = "Gave up after max_turns without a final answer."
    if return_trace:
        return {"answer": final, "plan": plan, "tool_calls": steps}
    return final


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many tankers were in the Kattegat on 3 March 2026?"
    print(run(question))
