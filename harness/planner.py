"""
Two-model agent: Qwen3-8B plans and answers, the fine-tuned SQL student writes the SQL.

The planner never sees the schema or writes SQL. It writes a short plan, then calls
ask_data(request) with one self-contained question per data step; the student turns the
request into SQL (schema_context.md as its prompt), query_ais_data runs it with the usual
guardrails, and the SQL plus rows (or the error) go back to the planner, which can rephrase,
add steps, do the arithmetic and write the answer. agent.py is the single-model baseline.

Places (places.py): the student's prompt lists only the places the request names, from
places.json and the approved learned places. The planner checks names with find_place and,
for an unknown place, may propose one box with propose_place: validated against the AIS
data, used for this question only and recorded as pending until a person approves it
(places.py approve). SQL whose lon/lat box belongs to no place in the request, or that
compares ship_type/nav_status/mobile_type with a value the data does not have, is not run
(the student otherwise swaps in a known place's box, e.g. Aarhus for Aalborg, or invents a
type such as 'Cruise').

CLI usage (needs both llm-serve and the student job, see slurm/serve_student.slurm):
    cd harness
    python3 planner.py "How many tankers were in the Kattegat on 3 March 2026?"
"""

import json
import re
import sys
from pathlib import Path

from openai import OpenAI

import places
from agent import MODEL_PATH, get_base_url
from ais_tool import DATE_RANGE, PROJECT_DIR, extract_sql, query_ais_data

STUDENT_ENDPOINT_FILE = PROJECT_DIR / "serving" / "student_endpoint.txt"
STUDENT_MODEL = "sql-student"  # --served-model-name in slurm/serve_student.slurm
SCHEMA_CONTEXT = Path(__file__).with_name("schema_context.md").read_text()
MAX_ROWS_FOR_PLANNER = 100  # the planner has an 8192-token context; enough for vessel lists

PLANNER_PROMPT = f"""\
You answer questions about Danish AIS ship-tracking data ({DATE_RANGE}): position reports
with time, position, speed, course, navigational status, ship type, name, size and
destination. You cannot query the data yourself. A SQL assistant answers data requests
through the tool ask_data: send it one plain-language request at a time and it returns the
SQL it ran and the rows, or an error.

How to work:
1. Start with "Plan:" and a short numbered list of the data you need.
2. Call ask_data for each step (call the tool; do not just announce it). Every request
   must stand on its own:
   - one question that one query can answer, e.g. "How many tankers were in the Kattegat
     between 08:00 and 09:00 UTC on 3 March 2026?"
   - keep the user's form: "which ships" asks for the ships, not a count; how busy or how
     many ships means distinct vessels, not reports
   - ask for the answer itself (e.g. "Which hour had the most vessels ...?") rather than a
     table to pick from
   - do not split a question whose parts cannot be combined afterwards, e.g. vessels seen
     on all of several days, top N over a period, the closest pair; send it as one request
   - full dates with day, month and year (a date without a year is in the data range
     above); times in UTC
   - places by name, checked first with find_place (see 3), or as a lon/lat box
   - vessel names in capitals as broadcast (e.g. TYCHO BRAHE); ship types such as Cargo,
     Fishing, Passenger, Tanker, Sailing, Tug, Pleasure, Pilot, HSC, SAR, Military
   - a request may cover up to a month only if it also names a ship type, a place or a
     vessel; otherwise ask per day or per week. To count distinct vessels over several
     requests, ask each for the list of vessels (MMSI) and combine the lists yourself (a
     vessel in two weeks counts once); never add up distinct counts
   - no SQL, table or column names
3. Call find_place for every place before your first request about it, and use the name it
   returns. If it is unknown you may call propose_place once for that place, with its kind
   (port or area) and your best estimate of its box (a port: a few km around the harbour).
   If accepted it is used for this question only: say in the answer that its location is
   estimated. If rejected, do not try another box; tell the user the place is unknown.
   Countries and regions (e.g. Denmark) are not places: ask about their ports or sea areas,
   or tell the user.
4. Do the arithmetic yourself: differences, ratios, percentages, totals across requests.
5. Check each result against its request and SQL. If it is an error, empty or implausible,
   rephrase the request more simply or split it, at most twice per step. Sending the same
   request again gives the same result.
6. Answer concisely with the numbers, dates and places, describing each number by what the
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


FIND_PLACE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_place",
        "description": "Look up a place (port, sea area) the data requests can use. Returns its "
                       "name, kind and lon/lat box, or says it is unknown.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Place name"}},
            "required": ["name"],
        },
    },
}

PROPOSE_PLACE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "propose_place",
        "description": "Propose a box for a place find_place does not know. It is checked "
                       "against the AIS data and, if plausible, saved as unverified.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "kind": {"type": "string", "enum": ["port", "area"]},
                "lon_min": {"type": "number"}, "lat_min": {"type": "number"},
                "lon_max": {"type": "number"}, "lat_max": {"type": "number"},
            },
            "required": ["name", "kind", "lon_min", "lat_min", "lon_max", "lat_max"],
        },
    },
}

TOOLS = [ASK_DATA_SCHEMA, FIND_PLACE_SCHEMA, PROPOSE_PLACE_SCHEMA]


def find_place(name: str, session: list[places.Place]) -> dict:
    place = places.find(name, places.load() + session)
    if place is None:
        return {"unknown": name, "next": "you may call propose_place once for it"}
    return place.as_dict()


def propose_place(args: dict, question: str, session: list[places.Place],
                  proposed: set[str]) -> dict:
    """One proposal per place name per question; an accepted place joins `session` (used for
    this question only) and is recorded as pending for a person to approve."""
    try:
        name, kind = args["name"].strip(), args["kind"]
        box = tuple(float(args[k]) for k in ("lon_min", "lat_min", "lon_max", "lat_max"))
    except (KeyError, TypeError, ValueError) as e:
        return {"error": f"bad arguments: {e}"}
    key = places.normalise(name)
    if key in proposed:
        return {"error": f"{name} was already proposed for this question; do not try another "
                         "box, tell the user the place is unknown"}
    proposed.add(key)
    ok, evidence = places.validate(name, kind, box)
    if not ok:
        return {"error": f"rejected: {evidence}"}
    place = places.learn(name, kind, box, evidence, context=question)
    session.append(place)
    return {**place.as_dict(), "evidence": evidence,
            "note": "used for this question only; say in the answer that this location is "
                    "estimated"}


def _context_values(column: str) -> set[str]:
    """The values schema_context.md lists for a categorical column."""
    m = re.search(rf"^\s+{column} VARCHAR\s+(.*)$", SCHEMA_CONTEXT, re.MULTILINE)
    return set(re.findall(r"'([^']*)'", m.group(1))) if m else set()


VALID_VALUES = {c: _context_values(c) for c in ("ship_type", "nav_status", "mobile_type")}


def invalid_values(sql: str) -> list[tuple[str, str]]:
    """(column, value) pairs the SQL compares a categorical column with that the data does
    not have, e.g. ship_type = 'Cruise' (cruise ships are 'Passenger' in AIS)."""
    bad = []
    for column, valid in VALID_VALUES.items():
        col = rf"(?:\w+\.)?{column}"
        values = re.findall(rf"{col}\s*(?:=|<>|!=)\s*'([^']*)'", sql, re.IGNORECASE)
        for group in re.findall(rf"{col}\s+(?:NOT\s+)?IN\s*\(([^)]*)\)", sql, re.IGNORECASE):
            values += re.findall(r"'([^']*)'", group)
        bad += [(column, v) for v in values if valid and v not in valid]
    return bad


_ANNOUNCED_RE = re.compile(r"ask_data|find_place|propose_place|\blet me\b|\bI will\b|\bI'll\b",
                           re.IGNORECASE)
NUDGE = "Call the tool you announced now, or give your final answer if you cannot continue."


def announced_step(content: str | None) -> bool:
    """The planner described a next step instead of calling the tool for it."""
    return bool(_ANNOUNCED_RE.search(strip_thinking(content)))


def student_base_url() -> str:
    return f"http://{STUDENT_ENDPOINT_FILE.read_text().strip()}/v1"


def strip_thinking(text: str | None) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def ask_data(student: OpenAI, request: str, session: list[places.Place] | None = None) -> dict:
    """The student writes SQL for the request (its prompt lists only the places the request
    names, including places accepted for this question). Unless the SQL uses an unexplained
    box or a value the data does not have, it runs through query_ais_data."""
    named = places.mentioned(request, places.load() + (session or []))
    response = student.chat.completions.create(
        model=STUDENT_MODEL,
        messages=[{"role": "system", "content": places.with_places(SCHEMA_CONTEXT, named)},
                  {"role": "user", "content": request}],
        temperature=0,
        max_tokens=512,
    )
    sql = extract_sql(response.choices[0].message.content)
    stray = places.unexplained_boxes(sql, named, request)
    bad = invalid_values(sql)
    if bad:
        column, value = bad[0]
        result = {"error": f"'{value}' is not a {column} value in the data. Valid values: "
                           f"{', '.join(sorted(VALID_VALUES[column]))}. Rephrase the request "
                           f"with one of them."}
    elif stray:
        result = {"error": f"The SQL used an area {list(stray[0])} that is not a place named in "
                           f"the request ({', '.join(p.name for p in named) or 'none'}). Check "
                           f"places with find_place and name them, or give the box as numbers."}
    else:
        result = query_ais_data(sql)
    return {"request": request, "places": [p.name for p in named], "sql": sql, "result": result}


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
    place_calls = []
    session: list[places.Place] = []  # places accepted for this question only
    proposed: set[str] = set()
    failed = set()  # requests that already failed; the student is deterministic
    nudges = 0

    for _ in range(max_turns):
        response = planner.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if plan is None and strip_thinking(message.content):
            plan = strip_thinking(message.content)

        if not message.tool_calls and announced_step(message.content) and nudges < 2:
            nudges += 1
            messages.append({"role": "user", "content": NUDGE})
            continue

        if not message.tool_calls:
            answer = strip_thinking(message.content)
            if return_trace:
                return {"answer": answer, "plan": plan, "tool_calls": steps,
                        "place_calls": place_calls}
            return answer

        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            if tool_call.function.name in ("find_place", "propose_place"):
                if tool_call.function.name == "find_place":
                    reply = find_place(str(arguments.get("name", "")), session)
                else:
                    reply = propose_place(arguments, question, session, proposed)
                print(f"[{tool_call.function.name}] {arguments} -> {reply}", file=sys.stderr)
                place_calls.append({"tool": tool_call.function.name, "arguments": arguments,
                                    "reply": reply})
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": json.dumps(reply, default=str)})
                continue
            request = str(arguments.get("request", ""))
            key = " ".join(request.lower().split())
            if tool_call.function.name != "ask_data" or not key:
                step = {"request": request, "sql": "",
                        "result": {"error": "Call ask_data with a non-empty 'request' string."}}
            elif key in failed:
                step = {"request": request, "sql": "",
                        "result": {"error": "This exact request already failed and would fail "
                                            "again. Rephrase it more simply or split it."}}
            else:
                step = ask_data(student, request, session)
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
        return {"answer": final, "plan": plan, "tool_calls": steps, "place_calls": place_calls}
    return final


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many tankers were in the Kattegat on 3 March 2026?"
    print(run(question))
