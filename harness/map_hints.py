"""
What the map (August's AIS Explorer, src/ais/ui) can show for an answer: the vessels, places,
time and filters behind it, read from the run's trace - the SQL that ran and its rows - and
from the question. The map page uses them to set its time, zoom to the places, filter and
highlight the ships, or open a vessel's route.

    hints = map_hints(question, trace)
    {"vessels": [{"mmsi": 219000123, "name": "TYCHO BRAHE"}, ...],   # from result rows
     "vessel_query": "TYCHO BRAHE",       # a vessel named in the question (map search)
     "places": [{"name": "Skagen", "box": [lon0, lat0, lon1, lat1]}],
     "start": "2026-03-01T00:00:00", "end": "2026-03-08T00:00:00",    # period, UTC
     "at": "2026-03-05T12:00:00",         # a moment (ships at that time)
     "name_filter": "MAERSK", "ship_type": "Tanker",
     "action": "route" | "snapshot" | None}
"""

import re
from datetime import date, datetime, timedelta

import places

MAX_VESSELS = 200
MAX_ROUTE_DAYS = 31  # the map's route view covers at most 31 days

_DATE_RE = re.compile(r"DATE\s+'(\d{4}-\d{2}-\d{2})'", re.IGNORECASE)
_TS_RE = re.compile(r"TIMESTAMP\s+'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}(?::\d{2})?)'", re.IGNORECASE)
_TYPE_RE = re.compile(r"ship_type\s*=\s*'([^']+)'", re.IGNORECASE)
_NAME_LIKE_RE = re.compile(r"name\s+(?:I?LIKE)\s+'%?([^'%]+)%?'", re.IGNORECASE)
_NAME_EQ_RE = re.compile(r"\bname\s*=\s*'([^']+)'", re.IGNORECASE)
_MMSI_RE = re.compile(r"\b([2-7]\d{8})\b")
_QUOTED_RE = re.compile(r"[\"“”']([A-Za-z0-9ÆØÅæøå .\-]{3,40})[\"“”']")
_CAPS_RE = re.compile(r"\b([A-ZÆØÅ][A-ZÆØÅ0-9\-]+(?:\s+[A-ZÆØÅ0-9][A-ZÆØÅ0-9\-]*)*)\b")
_NOT_NAMES = {"AIS", "UTC", "MMSI", "IMO", "US", "USA", "EU", "DK", "SQL", "HSC", "SAR", "ETA"}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_TEXT_DATE_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b"
    r"|\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
    r"|\b(\d{4})-(\d{2})-(\d{2})\b", re.IGNORECASE)


# Ship types named in the question (when no query ran, e.g. a map request), as on the map.
_TYPE_WORDS = [("Tanker", r"tanker"), ("Cargo", r"cargo|container|freighter"),
               ("Passenger", r"passenger|ferr|cruise"), ("Fishing", r"fishing|trawler"),
               ("Sailing", r"sailing|sailboat"), ("Tug", r"\btugs?\b"),
               ("Pleasure", r"pleasure|yacht|leisure"), ("Pilot", r"\bpilot"),
               ("Military", r"military|naval|navy|warship")]
_UNSUPPORTED_RE = re.compile(r"\b(heat ?maps?|charts?|plots?|graphs?)\b", re.IGNORECASE)
_RANGE_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:-|–|to|and|until)\s*(\d{1,2})(?:st|nd|rd|th)?\s+("
    + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE)


def _text_dates(text: str) -> list[date]:
    out = []
    for m in _RANGE_RE.finditer(text):  # "from 1 to 3 March 2026": both days
        month = _MONTHS.index(m.group(3).lower()) + 1
        for d in (m.group(1), m.group(2)):
            try:
                out.append(date(int(m.group(4)), month, int(d)))
            except ValueError:
                continue
    for m in _TEXT_DATE_RE.finditer(text):
        try:
            if m.group(1):
                out.append(date(int(m.group(3)), _MONTHS.index(m.group(2).lower()) + 1,
                                int(m.group(1))))
            elif m.group(4):
                out.append(date(int(m.group(6)), _MONTHS.index(m.group(4).lower()) + 1,
                                int(m.group(5))))
            else:
                out.append(date(int(m.group(7)), int(m.group(8)), int(m.group(9))))
        except ValueError:
            continue
    return out


def _vessel_query(question: str) -> str | None:
    """A vessel the question names: an MMSI, a quoted name or a name in capitals."""
    m = _MMSI_RE.search(question)
    if m:
        return m.group(1)
    for m in _QUOTED_RE.finditer(question):
        if any(c.isalpha() for c in m.group(1)):
            return m.group(1).strip().upper()
    for m in _CAPS_RE.finditer(question):
        words = [w for w in m.group(1).split() if w not in _NOT_NAMES]
        if words and len(" ".join(words)) >= 3:
            return " ".join(words)
    return None


def describe(hints: dict) -> str:
    """What the map shows, for the answer to a map request."""
    def day(ts: str | None) -> str:
        return datetime.fromisoformat(ts).strftime("%-d %B %Y") if ts else ""
    if hints.get("action") == "route":
        last = (datetime.fromisoformat(hints["end"]) - timedelta(days=1)).isoformat()
        span = (day(hints["start"]) if day(hints["start"]) == day(last)
                else f"{day(hints['start'])} to {day(last)}")
        who = hints.get("vessel_query") or (hints["vessels"][0].get("name") or
                                            hints["vessels"][0]["mmsi"])
        return f"Showing the route of {who} from {span} on the map."
    if hints.get("action") == "snapshot":
        where = " and ".join(p["name"] for p in hints.get("places") or [])
        when = hints.get("at")
        what = f"{hints['ship_type'].lower()} ships" if hints.get("ship_type") else "ships"
        shown = (f"Showing {what}" + (f" in {where}" if where else "")
                 + (f" at {when.replace('T', ' ')[:16]} UTC" if when else "") + " on the map.")
        if hints.get("unsupported"):
            return "Heatmaps and charts are not supported yet. " + shown
        return shown
    return ("The map can show ships at a moment or one vessel's route over at most 31 days; "
            "name a vessel, place or time to show it. Heatmaps and charts are not supported yet.")


def map_hints(question: str, trace: dict) -> dict:
    read = trace.get("question_as_read") or question
    calls = trace.get("tool_calls") or []
    sqls = [c.get("sql") or "" for c in calls if "error" not in (c.get("result") or {})]

    vessels, seen = [], set()
    for c in calls:
        result = c.get("result") or {}
        columns = [str(x).lower() for x in result.get("columns") or []]
        if "mmsi" not in columns:
            continue
        i, j = columns.index("mmsi"), columns.index("name") if "name" in columns else None
        for row in result.get("rows") or []:
            mmsi = row[i]
            if isinstance(mmsi, int) and mmsi not in seen and len(vessels) < MAX_VESSELS:
                seen.add(mmsi)
                vessels.append({"mmsi": mmsi, "name": row[j] if j is not None else None})

    known = places.load()
    named = {p.name: p for p in places.mentioned(read, known)}
    for c in calls:
        for p in places.mentioned(c.get("request") or "", known):
            named.setdefault(p.name, p)

    days = sorted({date.fromisoformat(d) for s in sqls for d in _DATE_RE.findall(s)}
                  | set(_text_dates(read)))
    stamps = sorted(datetime.fromisoformat(f"{d}T{t if len(t) == 8 else t + ':00'}")
                    for s in sqls for d, t in _TS_RE.findall(s))
    start = end = at = None
    if days:
        start = datetime.combine(days[0], datetime.min.time())
        end = datetime.combine(days[-1], datetime.min.time()) + timedelta(days=1)
    if stamps:
        at = stamps[-1]  # the moment asked about is the end of the window
    elif start and end - start == timedelta(days=1):
        at = start + timedelta(hours=12)

    types = {t for s in sqls for t in _TYPE_RE.findall(s)}
    if not sqls:
        types = {t for t, pattern in _TYPE_WORDS if re.search(pattern, read, re.IGNORECASE)}
    names = [n for s in sqls for n in _NAME_LIKE_RE.findall(s)]
    exact = [n for s in sqls for n in _NAME_EQ_RE.findall(s)]
    query = _vessel_query(read) or (exact[0] if len(set(exact)) == 1 else None)

    hints = {
        "vessels": vessels,
        "vessel_query": query,
        "places": [{"name": p.name, "box": list(p.box)} for p in named.values()],
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "at": at.isoformat() if at else None,
        "name_filter": names[0].strip() if len(set(names)) == 1 else None,
        "ship_type": types.pop() if len(types) == 1 else None,
        "unsupported": bool(_UNSUPPORTED_RE.search(read)),
    }
    one_vessel = len(vessels) == 1 or (query is not None and not vessels)
    short = start is not None and end - start <= timedelta(days=MAX_ROUTE_DAYS)
    wants_route = bool(re.search(r"\b(route|track|path|went|sailed|travel)", read, re.I))
    if one_vessel and short and (wants_route or trace.get("route") == "reply"):
        hints["action"] = "route"
    elif hints["at"] or hints["places"] or vessels:
        hints["action"] = "snapshot"
    else:
        hints["action"] = None
    return hints
