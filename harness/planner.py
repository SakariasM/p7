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
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

import functions
import places
from agent import MODEL_PATH, get_base_url
from ais_tool import DATE_RANGE, PROJECT_DIR, extract_sql, query_ais_data

STUDENT_ENDPOINT_FILE = PROJECT_DIR / "serving" / "student_endpoint.txt"
STUDENT_MODEL = "sql-student"  # --served-model-name in slurm/serve_student.slurm
SCHEMA_CONTEXT = Path(__file__).with_name("schema_context.md").read_text()
# Sampling temperature of the planner turns. 0 by default: in 3+3 repeated runs on 134
# questions (2026-10-14) it scored 127.3 on average (range 127-128) against 125.7 (124-127)
# at the model default 0.6. P7_PLANNER_TEMPERATURE overrides it ("" = model default).
_T = os.environ.get("P7_PLANNER_TEMPERATURE", "0")
PLANNER_TEMPERATURE = float(_T) if _T != "" else None
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
     Fishing, Passenger, Tanker, Sailing, Tug, Pleasure, Pilot, HSC, SAR, Military.
     Cruise ships are not an AIS type: ask for "passenger ships at least 250 m long"
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
   Countries and regions (e.g. Denmark) are not places. For vessels visiting or calling at a
   country (e.g. "visited Denmark in August") call port_calls instead of ask_data (cruise
   ships: ship_type Passenger, min_length_m 250); otherwise ask about its ports or sea
   areas, or tell the user.
4. Do the arithmetic yourself: differences, ratios, percentages, totals across requests.
5. Check each result against its request and SQL. If it is an error, empty or implausible,
   rephrase the request more simply or split it, at most twice per step. Sending the same
   request again gives the same result.
6. Answer concisely with the numbers, dates and places, describing each number by what the
   SQL actually computed (e.g. distinct vessels, not reports). Say if a step failed or data
   was missing. Never invent numbers.
7. If you counted cruise ships, say clearly in the answer that you assumed passenger ships
   of 250 m or longer are cruise ships, and that AIS has no cruise category, so the number
   cannot be certain (large ferries may be included, small cruise ships missed)."""

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

TOOLS = [ASK_DATA_SCHEMA, FIND_PLACE_SCHEMA, PROPOSE_PLACE_SCHEMA, functions.PORT_CALLS_SCHEMA]


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


CRUISE_NOTE = (
    "Note: AIS has no cruise ship category. I assumed passenger ships of 250 m or longer are "
    "cruise ships, so this number cannot be certain: large ferries may be included and smaller "
    "cruise ships missed.")
_CRUISE_SQL_RE = re.compile(r"ship_type\s*=\s*'Passenger'.*length\s*>=?\s*2[5-9]\d",
                            re.IGNORECASE | re.DOTALL)


def with_cruise_note(question: str, answer: str, steps: list[dict]) -> str:
    """Make sure an answer about cruise ships states the length assumption, whatever the
    planner wrote (the prompt asks for it, but the model does not always say it)."""
    used = any(_CRUISE_SQL_RE.search(st["sql"]) or
               ("port_calls" in st["request"] and '"Passenger"' in st["request"])
               for st in steps)
    if ("cruise" in question.lower() or used) and "assum" not in answer.lower():
        return f"{answer}\n\n{CRUISE_NOTE}"
    return answer


def student_base_url() -> str:
    return f"http://{STUDENT_ENDPOINT_FILE.read_text().strip()}/v1"


def strip_thinking(text: str | None) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def ask_data(student: OpenAI, request: str, session: list[places.Place] | None = None,
             model: str = STUDENT_MODEL) -> dict:
    """The SQL writer (the student, or with model=MODEL_PATH the 8B, for comparison) writes SQL
    for the request; its prompt lists only the places the request names, including places
    accepted for this question. Unless the SQL uses an unexplained box or a value the data
    does not have, it runs through query_ais_data."""
    named = places.mentioned(request, places.load() + (session or []))
    response = student.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": places.with_places(SCHEMA_CONTEXT, named)},
                  {"role": "user", "content": request}],
        temperature=0,
        max_tokens=1024 if model == MODEL_PATH else 512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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


def run_student_only(question: str, return_trace: bool = False, **_):
    """Baseline without a planner: the student gets the question itself, one SQL query, and
    the answer is the query result."""
    student = OpenAI(base_url=student_base_url(), api_key="not-needed")
    step = ask_data(student, question)
    result = step["result"]
    answer = (f"Error: {result['error']}" if "error" in result else
              f"Result ({', '.join(map(str, result['columns']))}): {result['rows'][:20]}")
    if return_trace:
        return {"answer": answer, "plan": None, "tool_calls": [step], "place_calls": []}
    return answer


def run(question: str, max_turns: int = 10, enable_thinking: bool = False,
        return_trace: bool = False, sql_writer: str = "student"):
    """sql_writer="8b" lets the 8B write the SQL for its own requests instead of the student
    (same prompt, place list and guards), to measure what the student adds."""
    planner = OpenAI(base_url=get_base_url(), api_key="not-needed")
    if sql_writer == "8b":
        student, sql_model = planner, MODEL_PATH
    else:
        student = OpenAI(base_url=student_base_url(), api_key="not-needed")
        sql_model = STUDENT_MODEL

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
            **({} if PLANNER_TEMPERATURE is None else {"temperature": PLANNER_TEMPERATURE}),
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
            answer = with_cruise_note(question, strip_thinking(message.content), steps)
            if return_trace:
                return {"answer": answer, "plan": plan, "tool_calls": steps,
                        "place_calls": place_calls}
            return answer

        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            if tool_call.function.name == "port_calls":
                reply = functions.port_calls(
                    str(arguments.get("country", "")), str(arguments.get("date_from", "")),
                    str(arguments.get("date_to", "")), arguments.get("ship_type") or None,
                    arguments.get("min_length_m"), valid_ship_types=VALID_VALUES["ship_type"])
                print(f"[port_calls] {arguments} -> {reply.get('vessels', reply.get('error'))}",
                      file=sys.stderr)
                # Logged as a data step (rows = the vessel count) so the evals can score it.
                result = ({"error": reply["error"]} if "error" in reply else
                          {"columns": ["vessels"], "rows": [[reply["vessels"]]], "row_count": 1,
                           "elapsed_seconds": reply["seconds"]})
                steps.append({"request": f"port_calls({json.dumps(arguments)})", "places": [],
                              "sql": "(port_calls)", "result": result})
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": json.dumps(reply, default=str)})
                continue
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
                step = ask_data(student, request, session, model=sql_model)
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


ROUTER_PROMPT = """\
You route questions about Danish AIS ship data to one of two ways of answering.

direct: a SQL specialist answers it with ONE query. It was trained on these question types,
each for a few named places at most and at most about a week:
- how many vessels (optionally of one ship type) on a day, in a time window or in a place
- vessels seen on every day of a short period
- counts per hour or per day (breakdowns) for one place or ship type
- rankings: the vessel, hour, grid cell, ship type or destination with the most of something
- average or highest speed of a ship type or in a place
- one named vessel: its position at a time, first/last seen, top speed, destination, type
  and length, distance travelled, number of reports
- vessels that came within a distance of a named vessel; the two moving vessels that came
  closest to each other in a place
- vessels underway, at anchor or moored at a moment
- share of vessels with a navigational status or of a ship type
- vessels of a type at least some length long; raw data quality (duplicates, base stations)
- comparisons in one query: two places, two ship types, or one place on two days (both
  counts and the difference); two or three named places ranked by their counts
- distinct vessels in any of two or three named ports (each vessel counted once); vessels
  seen in both of two ports; vessels in a place on one day that were not there on another
- the average daily number of vessels (of a type) in a place over a few days

Examples (the places and dates vary):
- "Which two moving vessels came closest to each other in <place> between 09:00 and 09:10
  UTC on <date>?" -> direct (one closest-pair query)
- "How many vessels were in <place> in each hour of <date>?" -> direct (one breakdown)
- "What percentage of the vessels in <place> on <date> were tugs?" -> direct
- "Were there more tankers in <place A> or in <place B> on <date>?" -> direct (comparison)
- "How did the number of vessels in <place> change from <date 1> to <date 2>?" -> direct
- "On average, how many cargo ships were in <place> per day from <date 1> to <date 3>?"
  -> direct (daily average)
- "How many cruise ships visited <country> in <month>?" -> plan (port calls in a country)
- "Did the busiest of <place A> and <place B> on <date> also have the most tankers the
  week after?" -> plan (a result feeds a second question)

plan: a planner splits it into several queries and combines the results. Needed when the
question asks about visits to a country; covers more than about a week or more than three
places; compares more than two periods, types or vessels; uses one result to ask a further
question; or otherwise needs arithmetic or reasoning across separate results.

Reply with only JSON: {"route": "direct" or "plan", "reason": "<a few words>"}"""

PHRASE_PROMPT = """\
Answer the user's question in one or two sentences from the SQL result below. Use the numbers
exactly as given (you may round decimals to two places), say what they count (e.g. distinct
vessels), and do not add anything that is not in the result.

Question: {question}
SQL: {sql}
Columns: {columns}
Rows: {rows}"""


def route(planner: OpenAI, question: str) -> tuple[str, str]:
    """("direct" or "plan", reason) from one short 8B call; "plan" if the reply is unusable."""
    response = planner.chat.completions.create(
        model=MODEL_PATH,
        messages=[{"role": "system", "content": ROUTER_PROMPT},
                  {"role": "user", "content": question}],
        temperature=0,
        max_tokens=60,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = strip_thinking(response.choices[0].message.content)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        decision = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        decision = {}
    choice = decision.get("route")
    return (choice if choice in ("direct", "plan") else "plan"), str(decision.get("reason", text))


def run_routed(question: str, max_turns: int = 10, enable_thinking: bool = False,
               return_trace: bool = False):
    """Simple questions go straight to the student (one query, answer worded by the 8B); the
    rest, and direct attempts that fail or return no rows, go through the planner. The
    comparison of 2026-10-14 showed the student alone is best and 4x faster on single-step
    questions, while the planner only pays off when a question needs several steps."""
    planner = OpenAI(base_url=get_base_url(), api_key="not-needed")
    choice, reason = route(planner, question)
    if choice == "direct":
        student = OpenAI(base_url=student_base_url(), api_key="not-needed")
        step = ask_data(student, question)
        result = step["result"]
        if "error" not in result and result.get("rows"):
            response = planner.chat.completions.create(
                model=MODEL_PATH,
                messages=[{"role": "user", "content": PHRASE_PROMPT.format(
                    question=question, sql=step["sql"], columns=result["columns"],
                    rows=json.dumps(result["rows"][:MAX_ROWS_FOR_PLANNER], default=str))}],
                temperature=0,
                max_tokens=300,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            answer = with_cruise_note(question, strip_thinking(response.choices[0].message.content),
                                      [step])
            trace = {"answer": answer, "plan": None, "tool_calls": [step], "place_calls": [],
                     "route": "direct", "route_reason": reason}
            return trace if return_trace else answer
        reason = f"direct attempt failed ({result.get('error', 'no rows')[:80]}); {reason}"
        choice = "fallback"
    trace = run(question, max_turns=max_turns, enable_thinking=enable_thinking, return_trace=True)
    if choice == "fallback":
        trace["tool_calls"] = [step] + trace["tool_calls"]
    trace.update(route="plan" if choice == "plan" else "direct->plan", route_reason=reason)
    return trace if return_trace else trace["answer"]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many tankers were in the Kattegat on 3 March 2026?"
    print(run_routed(question))
