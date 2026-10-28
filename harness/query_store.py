"""
The learned-question store: answered questions from the question log (logs/questions, written
by api.py and ask.py) become patterns that can answer the same kind of question again without
the LLMs. Modelled on EvolveR's offline self-distillation (arXiv 2510.16079): the harness only
logs while it answers; this job runs offline (nightly), distils patterns from the log, merges
them, scores them with the verdicts from the map and sets their status.

A pattern is the student's SQL with the question's values taken out as slots: places, dates,
times and ship types. "How many ships were docked in Skagen at 08:00 on 3 March 2026?" gives
the alias "how many ships were docked in {place1} at {time1} on {date1}" and its SQL with
{place1} (name or box), {date1} and {time1} in place of the values. Questions with the same SQL
skeleton are one pattern with several aliases. Anything not taken out as a slot stays literal
in the alias, so it only matches the same words. A run is refused unless every slot is found in
the SQL and no date, box or place name is left in it; the SQL must come back unchanged with the
original values and run with other values.

Score (EvolveR's): s = (correct + 1) / (correct + wrong + 2), from the latest verdict per run.
Status: approved at s >= 0.6 with at least one correct verdict, disabled below 0.3, otherwise
pending; a manual review (approve, reject, disable) overrides that until newer verdicts bring
the score below 0.3. A "wrong" verdict also gives a caution: the question's alias, the SQL that
was wrong and the note, for the planner to be warned with.

With P7_QUERY_STORE=1, planner.run_routed answers a question whose wording and slots match an
approved pattern from the pattern's SQL: worded from the alias's template (no LLM call) or, for
a wording without one (e.g. lists), by the 8B (one call instead of router, student and wording).
A wording the store has not seen can match a similar approved one (embedding model on the CPU,
guards on slots, numbers and meaning) if the 8B confirms both ask the same (one short call).

    python3 query_store.py distill               # rebuild data/query_store.json from the logs
    python3 query_store.py list [--status pending]
    python3 query_store.py show ID
    python3 query_store.py approve|reject|disable ID [note]
"""

import argparse
import getpass
import hashlib
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import places
from ais_tool import HAVE_VESSEL_DAYS, PROJECT_DIR, connect, query_ais_data
from ask import FEEDBACK_FILE, LOG_DIR
import planner
from planner import (_MONTH_NAMES, DATA_FIRST, DATA_LAST, FIRST_DAY, LAST_DAY, TYPE_SYNONYMS,
                     day_text, spell_out)

STORE_FILE = Path(os.environ.get("P7_QUERY_STORE_FILE",
                                PROJECT_DIR / "data" / "query_store.json"))
REVIEWS_FILE = PROJECT_DIR / "data" / "query_store_reviews.jsonl"
APPROVE_SCORE = 0.6
DISABLE_SCORE = 0.3
MAX_EXAMPLES = 5

# Ship type words that become a {type} slot, and their AIS value. Cruise ships are left out (they
# also need a length), and so are words that are verbs too (sailing, towing, diving).
_TYPE_WORDS = {w: v for v in ("Cargo", "Tanker", "Passenger", "Fishing", "Pleasure", "Tug",
                              "Pilot", "Military", "Dredging", "HSC", "SAR")
               for w in (v.lower(), v.lower() + "s")}
_TYPE_WORDS |= {w: v for w, v in TYPE_SYNONYMS.items() if "cruise" not in w}
_TYPE_WORDS |= {"container ships": "Cargo", "freighters": "Cargo", "yachts": "Pleasure",
                "trawlers": "Fishing", "warships": "Military", "dredgers": "Dredging"}

_M = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_ORD = r"(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+(\d{4}))?"
_DATE_RANGE_RE = re.compile(rf"\b(\d{{1,2}}){_ORD}(\s+(?:to|and|until|till|-|–)\s+)"
                            rf"(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_M}{_YEAR}", re.IGNORECASE)
# "December 30 to 1 January 2026": each date has its own month (and the year can turn).
_CROSS_RANGE_RE = re.compile(rf"\b{_M}\s+(\d{{1,2}}){_ORD}(\s+(?:to|and|until|till|-|–)\s+)"
                             rf"(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_M}{_YEAR}", re.IGNORECASE)
_MONTH_RANGE_RE = re.compile(rf"\b{_M}\s+(\d{{1,2}}){_ORD}(\s+(?:to|and|until|till|-|–)\s+)"
                             rf"(\d{{1,2}}){_ORD}\b(?!\s*:){_YEAR}", re.IGNORECASE)
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_M}{_YEAR}", re.IGNORECASE)
_MONTH_DAY_RE = re.compile(rf"\b{_M}\s+(\d{{1,2}}){_ORD}\b{_YEAR}", re.IGNORECASE)
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b")
# Hours as plain numbers: "between 12 and 13", "from 6 to 8" (not "between 12 and 13 knots"),
# "6 pm", "at 6 in the morning", "at 18 utc". They become time slots like "12:00".
_NOT_HOURS = (r"(?!\s*(?:knots?|kn|m|meters?|metres?|km|nm|%|percent|ships?|vessels?|boats?|"
              r"craft|hours?|minutes?|days?|times?|degrees?|tons?|tonnes?)\b)")
_HOUR_RANGE_RE = re.compile(rf"\b(between|from)\s+(\d{{1,2}})\s+(and|to|until|till)\s+"
                            rf"(\d{{1,2}})\b{_NOT_HOURS}", re.IGNORECASE)
_HOUR_AMPM_RE = re.compile(r"\b(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.)(?!\w)", re.IGNORECASE)
_HOUR_DAYPART_RE = re.compile(r"\b(\d{1,2})\s+in\s+the\s+(morning|afternoon|evening)\b",
                              re.IGNORECASE)
_HOUR_UTC_RE = re.compile(r"\bat\s+(\d{1,2})\s*(utc|z)\b", re.IGNORECASE)
# Limits with a unit: "longer than 270 m", "faster than 20 knots"; the number is the slot.
_LIMIT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?=\s*(?:m|meters?|metres?|knots?|kn|km|nm|"
                       r"nautical\s+miles?)\b)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"qq([a-z]+)(\d+)qq")
_SLOT_RE = re.compile(r"\{([a-z]+\d+)(?:\.([a-z_]+))?\}")
_QUOTED_RE = re.compile(r"'([^']*)'")
_BOX_KEYS = ("lon_min", "lat_min", "lon_max", "lat_max")


# --- slots in a question -----------------------------------------------------------------

def _year_for(month: int) -> int:
    """The year a month without one means: the one the data covers (as spell_out reads it)."""
    return next((y for y in range(DATA_LAST[0], DATA_FIRST[0] - 1, -1)
                 if DATA_FIRST <= (y, month) <= DATA_LAST), DATA_LAST[0])


def _date(day: str, month: str, year: str | None) -> date | None:
    m = [n[:3].lower() for n in _MONTH_NAMES].index(month[:3].lower()) + 1
    try:
        return date(int(year) if year else _year_for(m), m, int(day))
    except ValueError:
        return None


class _Slots:
    """Values found in a question, numbered per kind in order of appearance."""

    def __init__(self):
        self.values: dict[str, object] = {}  # "date1" -> value
        self.words: dict[str, str] = {}  # "type1" -> the word used, e.g. "tankers"

    def token(self, kind: str, value: object, word: str | None = None) -> str:
        for name, v in self.values.items():
            if name.rstrip("0123456789") == kind and v == value:
                return f" qq{kind}{name[len(kind):]}qq "
        n = sum(1 for name in self.values if name.rstrip("0123456789") == kind) + 1
        self.values[f"{kind}{n}"] = value
        if word:
            self.words[f"{kind}{n}"] = word
        return f" qq{kind}{n}qq "


def question_skeleton(question: str) -> tuple[str, dict[str, object], dict[str, str]]:
    """(alias, values, words): the question as planner.run_routed hands it on (after its
    rewrites and spell_out), normalised, with places, dates, times (also plain hours), ship
    types and limits with a unit as slots. Vessel names are found with the SQL (candidate)
    or the pattern's wording (match), not here."""
    slots = _Slots()
    text = spell_out(question)

    def date_range(m: re.Match) -> str:
        d2 = _date(m.group(3), m.group(4), m.group(5))
        d1 = _date(m.group(1), m.group(4), m.group(5))
        if not (d1 and d2):
            return m.group(0)
        return slots.token("date", d1) + m.group(2) + slots.token("date", d2)

    def cross_range(m: re.Match) -> str:
        y2 = m.group(6)
        d2 = _date(m.group(4), m.group(5), y2)
        if y2 and d2:  # the first date's year: the same, or the one before at a year's turn
            months = [n[:3].lower() for n in _MONTH_NAMES]
            y1 = int(y2) - (months.index(m.group(1)[:3].lower()) > d2.month - 1)
            d1 = _date(m.group(2), m.group(1), str(y1))
        else:
            d1 = _date(m.group(2), m.group(1), None)
        if not (d1 and d2):
            return m.group(0)
        return slots.token("date", d1) + m.group(3) + slots.token("date", d2)

    def month_range(m: re.Match) -> str:
        d1 = _date(m.group(2), m.group(1), m.group(5))
        d2 = _date(m.group(4), m.group(1), m.group(5))
        if not (d1 and d2):
            return m.group(0)
        return slots.token("date", d1) + m.group(3) + slots.token("date", d2)

    def iso(m: re.Match) -> str:
        try:
            return slots.token("date", date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            return m.group(0)

    def day_month(m: re.Match) -> str:
        d = _date(m.group(1), m.group(2), m.group(3))
        return slots.token("date", d) if d else m.group(0)

    def month_day(m: re.Match) -> str:
        d = _date(m.group(2), m.group(1), m.group(3))
        return slots.token("date", d) if d else m.group(0)

    text = _CROSS_RANGE_RE.sub(cross_range, text)
    text = _DATE_RANGE_RE.sub(date_range, text)
    text = _MONTH_RANGE_RE.sub(month_range, text)
    text = _ISO_RE.sub(iso, text)
    text = _DAY_MONTH_RE.sub(day_month, text)
    text = _MONTH_DAY_RE.sub(month_day, text)
    text = _TIME_RE.sub(lambda m: slots.token("time", f"{int(m.group(1)):02d}:{m.group(2)}"),
                        text)

    def hour(h: int) -> str:
        return slots.token("time", f"{h:02d}:00")

    def hour_range(m: re.Match) -> str:
        h1, h2 = int(m.group(2)), int(m.group(4))
        if h1 > 23 or h2 > 24 or h2 == 24 and h1 == 0:
            return m.group(0)
        return f"{m.group(1)} {hour(h1)} {m.group(3)} {hour(h2 % 24)}"

    def ampm(m: re.Match) -> str:
        h = int(m.group(1))
        if not 1 <= h <= 12:
            return m.group(0)
        return hour(h % 12 + (12 if m.group(2).lower().startswith("p") else 0))

    def daypart(m: re.Match) -> str:
        h = int(m.group(1))
        if not 1 <= h <= 12:
            return m.group(0)
        return hour(h % 12 + (0 if m.group(2).lower() == "morning" else 12))

    text = _HOUR_RANGE_RE.sub(hour_range, text)
    text = _HOUR_AMPM_RE.sub(ampm, text)
    text = _HOUR_DAYPART_RE.sub(daypart, text)
    text = _HOUR_UTC_RE.sub(lambda m: f"at {hour(int(m.group(1)))} {m.group(2)}"
                            if int(m.group(1)) <= 23 else m.group(0), text)
    text = _LIMIT_RE.sub(lambda m: slots.token("num", m.group(1)), text)
    norm = f" {places.normalise(text)} "

    # Places: longest names first, each span once, numbered in order of appearance.
    found = []
    for name, place in sorted(((n, p) for p in places.load() for n in places._names(p)),
                              key=lambda np: -len(np[0])):
        for m in re.finditer(rf"(?<= ){re.escape(name)}(?= )", norm):
            if not any(s < m.end() and m.start() < e for s, e, _ in found):
                found.append((m.start(), m.end(), place))
    parts, last = [], 0
    for start, end, place in sorted(found, key=lambda f: f[0]):
        parts += [norm[last:start], slots.token("place", place.name).strip()]
        last = end
    norm = "".join(parts) + norm[last:]

    for word in sorted(_TYPE_WORDS, key=len, reverse=True):
        norm = re.sub(rf"(?<= ){re.escape(word)}(?= )",
                      lambda m: slots.token("type", _TYPE_WORDS[m.group(0)], m.group(0)).strip(),
                      norm)
    alias = _TOKEN_RE.sub(r"{\1\2}", " ".join(norm.split()))
    return alias, slots.values, slots.words


# --- slots in the SQL --------------------------------------------------------------------

def _place(name: str) -> places.Place | None:
    return places.find(name)


def _in_literals(sql: str, change) -> str:
    """change(text) applied to the text of each quoted literal in the SQL."""
    return _QUOTED_RE.sub(lambda m: "'" + change(m.group(1)) + "'", sql)


def _box_slot(sql: str, name: str, place: places.Place) -> str:
    """The place's box numbers in the SQL as {place1.lon_min}, ... (either column order)."""
    for regex, order in ((places._LON_LAT, (0, 2, 1, 3)), (places._LAT_LON, (1, 3, 0, 2))):
        def box(m: re.Match, order=order) -> str:
            nums = [float(x) for x in m.groups()]
            if not all(abs(nums[i] - place.box[order[i]]) < 1e-3 for i in range(4)):
                return m.group(0)
            text, offset = m.group(0), m.start()
            for i in reversed(range(4)):
                s, e = m.span(i + 1)
                text = text[:s - offset] + f"{{{name}.{_BOX_KEYS[order[i]]}}}" + text[e - offset:]
            return text
        sql = regex.sub(box, sql)
    return sql


def _slot_in_sql(sql: str, name: str, value: object) -> str:
    if name.startswith("place"):
        place = _place(str(value))
        sql = _box_slot(sql, name, place)
        return _in_literals(sql, lambda s: f"{{{name}}}" if s.lower() == place.name.lower()
                            else s)
    if name.startswith("date"):
        return _in_literals(sql, lambda s: s.replace(value.isoformat(), f"{{{name}}}"))
    if name.startswith("time"):
        return _in_literals(sql, lambda s: re.sub(rf"(?<![\d:]){value}(?=:\d\d\b|$)",
                                                  f"{{{name}}}", s))
    if name.startswith("vessel"):  # a whole literal (or '%...%'), upper case or lower (ILIKE)
        def vessel(s: str) -> str:
            core = s.strip("%")
            pre, post = s[:len(s) - len(s.lstrip("%"))], s[len(s.rstrip("%")):]
            if core == str(value):
                return f"{pre}{{{name}}}{post}"
            if core == str(value).lower():
                return f"{pre}{{{name}.lower}}{post}"
            return s
        return _in_literals(sql, vessel)
    if name.startswith("num"):
        return _NUMBER_OUTSIDE_RE.sub(lambda m: f"{{{name}}}" if m.group(1) and
                                      float(m.group(1)) == float(value) else m.group(0), sql)
    return _in_literals(sql, lambda s: f"{{{name}}}" if s == value else s)  # ship type


# A number outside quoted literals and slots (group 1); quoted literals are matched whole.
_NUMBER_OUTSIDE_RE = re.compile(r"'[^']*'|(?<![\w.{])(\d+(?:\.\d+)?)(?![\w.}])")


def _numbers_equal(sql: str, value: object) -> int:
    return sum(1 for m in _NUMBER_OUTSIDE_RE.finditer(sql)
               if m.group(1) and float(m.group(1)) == float(value))


def sql_skeleton(sql: str, values: dict[str, object]) -> tuple[str | None, str]:
    """(skeleton, "") with the slot values replaced in the SQL, or (None, why not)."""
    out = sql
    for name, value in values.items():
        if name.startswith("place") and _place(str(value)) is None:
            return None, f"unknown place {value}"
        if name.startswith("num") and _numbers_equal(sql, value) != 1:
            return None, f"{name} not exactly once in the SQL"  # e.g. "1 km" as 1000, or twice
        out = _slot_in_sql(out, name, value)
    for name in values:
        if f"{{{name}" not in out:
            return None, f"{name} not found in the SQL"
    if re.search(r"\d{4}-\d{2}-\d{2}", out):
        return None, "a date in the SQL that is not in the question"
    if places.boxes_in_sql(out):
        return None, "a box in the SQL that is not a place in the question"
    known = {p.name.lower() for p in places.load()}
    if any(m.group(1).lower() in known for m in _QUOTED_RE.finditer(out)):
        return None, "a place name in the SQL that is not in the question"
    if _span_numbers(out, values):
        return None, "a number in the SQL that depends on the dates or times"
    return out, ""


def _span_numbers(skeleton: str, values: dict[str, object]) -> list[float]:
    """Numbers in the SQL that equal a span of the slot values: the student writes
    count(*) / 3.0 for "per day from 22 to 24 September", which would stay 3.0 for any other
    range. Days in a date range, hours or minutes between two times; 1 only after a /."""
    dates = sorted(v for k, v in values.items() if k.startswith("date"))
    times = sorted(int(v[:2]) * 60 + int(v[3:]) for k, v in values.items()
                   if k.startswith("time"))
    spans = set()
    if len(dates) > 1:
        days = (dates[-1] - dates[0]).days
        spans |= {days, days + 1, 24 * (days + 1)}
    if len(times) > 1:
        minutes = times[-1] - times[0]
        spans |= {minutes, minutes / 60}
    plain = _QUOTED_RE.sub("''", _SLOT_RE.sub(" ", skeleton))
    found = []
    for m in re.finditer(r"(/\s*)?(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", plain):
        x = float(m.group(2))
        if x in spans and (x >= 2 or m.group(1)):
            found.append(x)
    return found


def render_sql(skeleton: str, values: dict[str, object]) -> str:
    """The SQL for these slot values ({place1.lon_min} -> the place's box number, ...)."""
    def fill(m: re.Match) -> str:
        value = values[m.group(1)]
        if m.group(2) == "lower":
            return str(value).lower()
        if m.group(2):
            return places.coord(_place(str(value)).box[_BOX_KEYS.index(m.group(2))])
        return value.isoformat() if isinstance(value, date) else str(value)
    return _SLOT_RE.sub(fill, skeleton)


def canonical(sql: str) -> str:
    """Whitespace and decimal formatting (12.40 = 12.4) do not count when comparing SQL."""
    sql = re.sub(r"(?<![\w.])-?\d+\.\d+", lambda m: repr(float(m.group(0))), sql)
    return " ".join(sql.split()).rstrip(";").strip()


def pattern_id(skeleton: str) -> str:
    return hashlib.sha1(canonical(skeleton).encode()).hexdigest()[:10]


# Other values a pattern is tried with (it must run with them too): a port and an area that
# vessel_days lists, ship types, and every date a week earlier (or later near the start).
_OTHER = {"port": ["Aalborg", "Skagen", "Frederikshavn"], "area": ["Kattegat", "Øresund",
                                                                   "Fehmarn Belt"],
          "type": ["Cargo", "Tanker", "Passenger"]}
_OTHER_VESSELS = ["TYCHO BRAHE", "HAMLET", "AURORA AF HELSINGBORG", "PERNILLE", "JEPPE"]


def other_values(values: dict[str, object]) -> dict[str, object]:
    out, used = dict(values), set(values.values())
    shift = timedelta(days=-7)
    if any(isinstance(v, date) and v + shift < FIRST_DAY for v in values.values()):
        shift = timedelta(days=7)
    for name, value in values.items():
        if name.startswith("date"):
            out[name] = min(value + shift, LAST_DAY)
        elif name.startswith("vessel"):
            out[name] = next(v for v in _OTHER_VESSELS if v not in used
                             and places.normalise(v) in _vessel_names())
            used.add(out[name])
        elif name.startswith(("place", "type")):
            place = _place(str(value)) if name.startswith("place") else None
            kind = "type" if name.startswith("type") else (place.kind if place else "port")
            more = [] if kind == "type" else sorted(p.name for p in places.load(False)
                                                    if p.kind == kind)
            out[name] = next(v for v in [*_OTHER.get(kind, []), *more] if v not in used)
            used.add(out[name])
    return out


# --- answer template ---------------------------------------------------------------------

_NUMBER_WORDS_RE = re.compile(
    r"\b(no|none|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|single|"
    r"both|yes|several|a few|dozens?|hundreds?)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.{])\d+(?:\.\d+)?")
# Vessel names are upper case in AIS; one in an answer that the question does not name came
# from the result, and would be wrong for other values.
_NAME_RE = re.compile(r"\b[A-Z][A-Z0-9'\-]{2,}\b|\"[^\"]+\"")
_FIXED_WORDS = {"UTC", "AIS", "MMSI", "IMO", "SOG", "COG"}


def _date_forms(d: date) -> list[str]:
    month = _MONTH_NAMES[d.month - 1]
    return [f"{d.day} {month} {d.year}", f"{month} {d.day}, {d.year}", f"{d.day} {month}",
            f"{month} {d.day}", d.isoformat()]


def answer_template(answer: str, question: str, values: dict[str, object],
                    words: dict[str, str], rows: list | None,
                    row_count: int | None) -> str | None:
    """The answer with slot values as {date1}, ... and result numbers as {cell:R:C:D} (row,
    column, decimals) or {rows}; None if anything in it cannot be traced (names from the
    rows, number words, yes/no, a count of 1 that decides singular or plural)."""
    text = answer.split("\n\nNote:")[0].strip()
    if not rows or _NUMBER_WORDS_RE.search(text):
        return None
    if any(m.group(0) not in _FIXED_WORDS and m.group(0).strip('"').lower()
           not in question.lower() for m in _NAME_RE.finditer(text)):
        return None
    vessels = {str(v).lower() for k, v in values.items() if k.startswith("vessel")}
    for row in rows:
        for cell in row:
            if (isinstance(cell, str) and len(cell) > 2 and cell.lower() in text.lower()
                    and cell.lower() not in vessels):
                return None  # names or values from the rows: a list, not a template
    for name, value in values.items():
        if name.startswith("num"):  # only with its unit, as in the question
            text = re.sub(rf"(?<![\w.]){re.escape(str(value))}(?=\s*(?:m|meters?|metres?|"
                          rf"knots?|kn|km|nm|nautical\s+miles?)\b)", f"{{{name}}}", text,
                          flags=re.IGNORECASE)
            continue
        if name.startswith("date"):
            forms = _date_forms(value)
        elif name.startswith("time"):
            forms = [value]
        elif name.startswith("vessel"):
            forms = [str(value)]
        elif name.startswith("place"):
            place = _place(str(value))
            forms = sorted({place.name, *place.aliases}, key=len, reverse=True)
        else:
            forms = [words.get(name, "")]
        for form in filter(None, forms):
            text = re.sub(rf"(?<!\w){re.escape(form)}(?!\w)", f"{{{name}}}", text,
                          flags=re.IGNORECASE)
    hours = {v: k for k, v in values.items() if k.startswith("time")}

    def hour_range(m: re.Match) -> str:  # "between 12 and 13" for the time slots 12:00, 13:00
        a, b = f"{int(m.group(2)):02d}:00", f"{int(m.group(4)) % 24:02d}:00"
        if a in hours and b in hours:
            return f"{m.group(1)} {{{hours[a]}}} {m.group(3)} {{{hours[b]}}}"
        return m.group(0)

    text = _HOUR_RANGE_RE.sub(hour_range, text)
    cells = [(r, c, v) for r, row in enumerate(rows[:5]) for c, v in enumerate(row)
             if isinstance(v, (int, float)) and not isinstance(v, bool)]

    def number(m: re.Match) -> str:
        shown = m.group(0)
        x = float(shown.replace(",", ""))
        decimals = len(shown.split(".")[1]) if "." in shown else 0
        if row_count is not None and x == row_count and len(rows) > 1:
            return "{rows}"
        for r, c, v in cells:
            if round(float(v), decimals) == x:
                if x == 1:
                    raise ValueError("a count of 1 decides singular or plural")
                return f"{{cell:{r}:{c}:{decimals}{':,' if ',' in shown else ''}}}"
        raise ValueError(f"number {shown} not in the result")

    try:
        return _NUMBER_RE.sub(number, text)
    except ValueError:
        return None


# --- distilling the log ------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


_vessels: dict = {}


def _vessel_names() -> dict[str, list[str]]:
    """Normalised vessel name -> the names in the data (vessel_days), read once."""
    if not _vessels and HAVE_VESSEL_DAYS:
        names = connect().execute(
            "SELECT DISTINCT name FROM vessel_days WHERE name IS NOT NULL").fetchall()
        for (name,) in names:
            key = places.normalise(name)
            if len(key) >= 3:
                _vessels.setdefault(key, []).append(name)
    return _vessels


def _vessel_slots(alias: str, values: dict, sql: str) -> str:
    """Vessel names that the SQL compares with (in a quoted literal) and the question names,
    as {vessel1}, ... in the alias; their values are added to values."""
    names = _vessel_names()
    for m in _QUOTED_RE.finditer(sql):
        key = places.normalise(m.group(1).strip("%"))
        if key not in names or f" {key} " not in f" {alias} ":
            continue
        name = next((n for n in names[key] if n.upper() == m.group(1).strip("%").upper()),
                    names[key][0])
        if name in values.values():
            continue
        slot = f"vessel{sum(k.startswith('vessel') for k in values) + 1}"
        values[slot] = name
        alias = f" {alias} ".replace(f" {key} ", f" {{{slot}}} ", 1).strip()
    return alias


def candidate(run: dict) -> tuple[dict | None, str]:
    """(pattern part from one logged run, "") or (None, why the run gives no pattern)."""
    if run.get("error") or not run.get("answer"):
        return None, "no answer"
    steps = run.get("steps") or []
    if len(steps) != 1 or not steps[0].get("sql") or steps[0]["sql"] == "(port_calls)":
        return None, "not one SQL query"
    if steps[0].get("error"):
        return None, "the query failed"
    question = run.get("question_as_read") or run["question"]
    alias, values, words = question_skeleton(question)
    sql = steps[0]["sql"]
    alias = _vessel_slots(alias, values, sql)
    skeleton, why = sql_skeleton(sql, values)
    if skeleton is None:
        return None, why
    if canonical(render_sql(skeleton, values)) != canonical(sql):
        return None, "the SQL does not come back with the original values"
    return {"id": pattern_id(skeleton), "sql": skeleton, "alias": alias, "values": values,
            "words": words, "template": answer_template(
                run["answer"], question, values, words, steps[0].get("rows"),
                steps[0].get("row_count"))}, ""


def _score(correct: int, wrong: int) -> float:
    return round((correct + 1) / (correct + wrong + 2), 3)


def distill(log_dir: Path = LOG_DIR, feedback_file: Path = FEEDBACK_FILE,
            out: Path = STORE_FILE, check: bool = True) -> dict:
    runs = [r for f in sorted(log_dir.glob("20*.jsonl")) for r in _jsonl(f)]
    verdicts = {}  # run id -> latest feedback
    for v in _jsonl(feedback_file):
        verdicts[v["id"]] = v
    reviews = {}
    for r in _jsonl(REVIEWS_FILE):
        reviews[r["pattern"]] = r
    previous = {p["id"]: p for p in (json.loads(out.read_text())["patterns"]
                                     if out.exists() else [])}

    patterns, cautions, refused = {}, {}, {}
    for run in runs:
        part, why = candidate(run)
        verdict = verdicts.get(run["id"])
        if verdict and verdict["verdict"] == "wrong":
            alias = question_skeleton(run.get("question_as_read") or run["question"])[0]
            c = cautions.setdefault(alias, {"alias": alias, "notes": [], "sql": [], "runs": []})
            c["runs"].append(run["id"])
            if verdict.get("note"):
                c["notes"].append(verdict["note"])
            sql = next((s.get("sql") for s in run.get("steps") or [] if s.get("sql")), None)
            if sql and sql not in c["sql"]:
                c["sql"].append(sql)
        if part is None:
            refused[why] = refused.get(why, 0) + 1
            continue
        p = patterns.setdefault(part["id"], {
            "id": part["id"], "sql": part["sql"], "slots": sorted(part["values"]),
            "aliases": [], "templates": {}, "examples": [], "runs": [], "verdicts": []})
        if part["alias"] not in p["aliases"]:
            p["aliases"].append(part["alias"])
        if part["template"] and part["alias"] not in p["templates"]:
            p["templates"][part["alias"]] = part["template"]  # its own wording, per alias
        p["runs"].append(run["id"])
        if len(p["examples"]) < MAX_EXAMPLES:
            p["examples"].append({"run": run["id"], "question": run["question"],
                                  "values": {k: str(v) for k, v in part["values"].items()},
                                  "words": part["words"]})
        if verdict:
            p["verdicts"].append({"verdict": verdict["verdict"], "at": verdict["at"]})

    for p in patterns.values():
        p["correct"] = sum(v["verdict"] == "correct" for v in p["verdicts"])
        p["wrong"] = sum(v["verdict"] == "wrong" for v in p["verdicts"])
        p["score"] = _score(p["correct"], p["wrong"])
        old = previous.get(p["id"])
        if old and old.get("sql") == p["sql"] and "runs_with_other_values" in old:
            p["runs_with_other_values"] = old["runs_with_other_values"]
        elif check:
            values = {k: (date.fromisoformat(v) if k.startswith("date") else v)
                      for k, v in p["examples"][0]["values"].items()}
            result = query_ais_data(render_sql(p["sql"], other_values(values)))
            p["runs_with_other_values"] = "error" not in result
        p["status"], p["status_by"] = _status(p, reviews.get(p["id"]))
    store = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": len(runs),
             "refused": refused, "patterns": sorted(patterns.values(), key=lambda p: p["id"]),
             "cautions": list(cautions.values())}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, out)
    return store


def _status(p: dict, review: dict | None) -> tuple[str, str]:
    if p.get("runs_with_other_values") is False:
        return "broken", "auto"
    if review:
        newer = [v for v in p["verdicts"] if v["at"] > review["at"]]
        wrong = sum(v["verdict"] == "wrong" for v in newer)
        if wrong and _score(len(newer) - wrong, wrong) < DISABLE_SCORE:
            return "disabled", "auto"
        return {"approve": "approved", "reject": "rejected", "disable": "disabled"}[
            review["action"]], "manual"
    if p["score"] < DISABLE_SCORE:
        return "disabled", "auto"
    if p["correct"] and p["score"] >= APPROVE_SCORE:
        return "approved", "auto"
    return "pending", "auto"


# --- answering from the store (planner.run_routed) ---------------------------------------

_loaded: dict = {"mtime": None, "store": None, "vectors": None}

# Similar wordings (phase 4): a wording the store has not seen is compared with the approved
# wordings by a small embedding model (CPU, no LLM). Similarity alone matched 3-7% wrongly on
# the teacher rewrites ("docked" ~ "moving" 0.95), so it only picks a candidate: the slots, the
# literal numbers and the meaning key below must agree, the slot roles must be fixed by the
# wording, and the 8B confirms that both ask the same (planner._run_learned), as in EvolveR.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIR = PROJECT_DIR / "models" / "embed"
SIMILAR_MIN = 0.85
_CONCEPTS = {k: re.compile(v) for k, v in {
    "count": r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b",
    "list": r"\bwhich\b|\blist\b|\bname\b|^what (ships|vessels|boats)\b|\bidentify\b|\bshow\b",
    "average": r"\baverage\b|\bmean\b|\bper day\b|\btypical",
    "most": r"\bmost\b|\bhighest\b|\blargest\b|\blongest\b|\bbiggest\b|\bbusiest\b|\bfastest\b"
            r"|\bmax",
    "least": r"\bleast\b|\blowest\b|\bsmallest\b|\bshortest\b|\bslowest\b|\bfewest\b|\bmin",
    "percent": r"\bpercent|\bshare\b|\bproportion\b|\bfraction\b",
    "hour": r"\bhour",
    "change": r"\bchange|\bdifference\b|\bincrease|\bdecrease|\bcompare|\bmore\b|\bfewer\b"
              r"|\bthan\b",
    "both": r"\bboth\b|\bas well as\b|\band also\b",
    "or": r"\bor\b|\beither\b|\bany of\b",
    "docked": r"\bdock|\bharbou?r|\bin port\b|\bberth|\bport call|\bcalled at\b"
              r"|\bstopp?ed\b|\bstationary\b|\bstill\b|\bnot moving\b",
    "visit": r"\bvisit",  # present at a place (vessel_days.places), not docked (still_in)
    "moving": r"\bmov(e|ing|ed)\b|\bunderway\b|\bunder way\b|\bsail(s|ed)?\b|\btransit"
              r"|\bpass|\bcross|\btravers|\bin motion\b|\bunder\b|\bwent\b",
    "anchor": r"\banchor",
    "moored": r"\bmoor",
    "speed": r"\bspeed|\bknots?\b|\bfast\b|\bslow\b",
    "length": r"\blength|\blong(est)?\b|\bsize\b|\bbig(gest)?\b|\blarge(st)?\b"
              r"|\bsmallest\b|\bshortest\b|\bmet(er|re)s?\b|\bm\b",
    "draught": r"\bdraught|\bdraft",
    "distance": r"\bdistance|\bfar\b|\btravel|\bkm\b|\bnautical\b|\bvoyage|\btrip"
                r"|\bjourney",
    "destination": r"\bdestination",
    "reports": r"\breport|\bmessages?\b|\bposition\b",
    "new": r"\bnew\b|\bfirst time\b|\bnot seen\b",
    "neg": r"\bnot\b|\bno\b|\bwithout\b|\bexcept\b|\bnever\b",
    "where": r"\bwhere\b",
    "same": r"\bevery\b|\beach of\b|\ball (three|two|the) days\b",
    # ship types that are not slots (they are verbs too) must agree as words
    "sailing": r"\bsailing\b",
    "towing": r"\btowing\b",
}.items()}
_RANGE_RE = re.compile(r"(from|between) \{date1\} (to|and) \{date2\}")


def _similar_key(alias: str) -> tuple | None:
    """What two wordings must share to be compared: slots, literal numbers and meaning; None
    if the wording does not fix the slot roles (two places, or two dates not as a range)."""
    slots = re.findall(r"\{([a-z]+)(\d+)\}", alias)
    kinds: dict[str, int] = {}
    for kind, _ in slots:
        kinds[kind] = kinds.get(kind, 0) + 1
    if any(n > 1 for kind, n in kinds.items() if kind != "date"):
        return None
    if kinds.get("date", 0) > 1 and not (kinds["date"] == 2 and _RANGE_RE.search(alias)):
        return None
    plain = _SLOT_RE.sub(" ", alias)
    return (tuple(sorted(k + n for k, n in slots)),
            tuple(sorted(re.findall(r"\b\d+(?:\.\d+)?\b", plain))),
            frozenset(k for k, rx in _CONCEPTS.items() if rx.search(plain)))


def _embed(texts: list[str]):
    """Unit vectors for the texts, or None without fastembed (then only exact wordings match)."""
    try:
        import numpy as np
        from fastembed import TextEmbedding
    except ImportError:
        return None
    if "model" not in _loaded:
        _loaded["model"] = TextEmbedding(EMBED_MODEL, cache_dir=str(EMBED_DIR))
    vectors = np.array(list(_loaded["model"].embed(texts)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def load() -> dict | None:
    """The store, read again when the nightly distill has rewritten it; None if there is none."""
    try:
        mtime = STORE_FILE.stat().st_mtime
    except FileNotFoundError:
        return None
    if _loaded["mtime"] != mtime:
        _loaded.update(mtime=mtime, store=json.loads(STORE_FILE.read_text()), vectors=None)
    return _loaded["store"]


def snapshot() -> dict:
    """Whether the store answers questions and how big it is, for the question log."""
    store = load() or {"patterns": []}
    return {"on": planner.USE_QUERY_STORE, "built": store.get("built"),
            "patterns": len(store["patterns"]),
            "approved": sum(p["status"] == "approved" for p in store["patterns"])}


def _approved_wordings(store: dict):
    """(wording, pattern) for the approved patterns' wordings that may be matched by
    similarity, with their vectors (computed once per store)."""
    if _loaded["vectors"] is None:
        pairs = [(a, p) for p in store["patterns"] if p["status"] == "approved"
                 for a in p["aliases"] if _similar_key(a) is not None]
        vectors = _embed([a for a, _ in pairs]) if pairs else None
        _loaded["vectors"] = (pairs, vectors)
    return _loaded["vectors"]


def _vessel_match(store: dict, alias: str,
                  values: dict) -> tuple[list[dict], str, dict] | None:
    """An approved wording with {vessel1}, ... that this question's wording fits, with each
    vessel slot filled by exactly one vessel name in the data: ([pattern], wording, values)."""
    names = _vessel_names()
    for p in store["patterns"]:
        if p["status"] != "approved":
            continue
        for a in p["aliases"]:
            if "{vessel" not in a:
                continue
            rx = re.sub(r"\\\{(vessel\d+)\\\}", r"(?P<\1>.+?)", re.escape(a))
            m = re.fullmatch(rx, alias)
            if not m:
                continue
            found = {k: names.get(v, []) for k, v in m.groupdict().items()}
            if all(len(n) == 1 for n in found.values()):
                filled = {**values, **{k: n[0] for k, n in found.items()}}
                if p["slots"] == sorted(filled):
                    return [p], a, filled
    return None


def match(question: str, similar: bool = True) -> dict | None:
    """The best approved pattern for this question: one with its exact wording and slots, one
    whose wording it fits with a vessel name in a {vessel} slot, or (similar=True) the most
    similar approved wording that passes the guards, marked "similar" for the 8B to confirm.
    Its SQL with this question's values, and the template of the matched wording (None if it
    has none)."""
    store = load()
    if not store:
        return None
    alias, values, words = question_skeleton(question)
    hits = [p for p in store["patterns"] if p["status"] == "approved"
            and alias in p["aliases"] and p["slots"] == sorted(values)]
    matched, similarity, via = alias, 1.0, "exact"
    if not hits and "{vessel" in json.dumps(store["patterns"]):
        found = _vessel_match(store, alias, values)
        if found:
            hits, matched, values = found
            via = "vessel"
    if not hits and similar and _similar_key(alias) is not None:
        pairs, vectors = _approved_wordings(store)
        key = _similar_key(alias)
        allowed = [i for i, (a, _) in enumerate(pairs) if _similar_key(a) == key]
        query = _embed([alias]) if allowed else None
        if query is not None and vectors is not None:
            sims = vectors[allowed] @ query[0]
            best = int(sims.argmax())
            if sims[best] >= SIMILAR_MIN:
                matched, p = pairs[allowed[best]]
                hits, similarity, via = [p], round(float(sims[best]), 3), "similar"
    if not hits:
        return None
    p = max(hits, key=lambda p: (p["score"], len(p["runs"])))
    return {"pattern": p["id"], "alias": alias, "matched": matched, "similarity": similarity,
            "similar": via == "similar", "values": values, "words": words,
            "sql": render_sql(p["sql"], values), "template": p["templates"].get(matched),
            "places": [str(v) for k, v in values.items() if k.startswith("place")]}


def caution(question: str) -> dict | None:
    """The caution for this question's wording, if answers to it were marked wrong."""
    store = load()
    if not store:
        return None
    alias = question_skeleton(question)[0]
    return next((c for c in store["cautions"] if c["alias"] == alias), None)


_FILL_RE = re.compile(r"\{(rows|cell:[\d:,]+|[a-z]+\d+)\}")


def render_answer(hit: dict, result: dict) -> str | None:
    """The wording's template filled in from the slots and the result; None if there is no
    template, a cell is missing, or a count of 1 would need other grammar."""
    rows = result.get("rows") or []

    def fill(m: re.Match) -> str:
        key = m.group(1)
        if key == "rows":
            return str(result.get("total_rows", len(rows)))
        if key.startswith("cell:"):
            _, r, c, decimals, *comma = key.split(":")
            value = rows[int(r)][int(c)]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"cell {r}:{c} is not a number")
            text = f"{value:{',' if comma else ''}.{decimals}f}"
            if text == "1":
                raise ValueError("a count of 1 needs other grammar")
            return text
        value = hit["values"][key]
        if key.startswith("date"):
            return day_text(value)
        if key.startswith("type"):
            return hit["words"].get(key, str(value).lower())
        return str(value)

    if not hit["template"]:
        return None
    try:
        return _FILL_RE.sub(fill, hit["template"])
    except (IndexError, KeyError, ValueError):
        return None


# --- review CLI --------------------------------------------------------------------------

def _load_store() -> dict:
    if not STORE_FILE.exists():
        sys.exit(f"no store yet: run  python3 query_store.py distill  ({STORE_FILE})")
    return json.loads(STORE_FILE.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("distill", help="rebuild the store from the question log")
    d.add_argument("--log-dir", type=Path, default=LOG_DIR)
    d.add_argument("--feedback", type=Path, default=FEEDBACK_FILE)
    d.add_argument("--out", type=Path, default=STORE_FILE)
    d.add_argument("--no-check", action="store_true", help="skip running with other values")
    ls = sub.add_parser("list", help="list patterns")
    ls.add_argument("--status", choices=["approved", "pending", "disabled", "rejected", "broken"])
    show = sub.add_parser("show", help="one pattern in full")
    show.add_argument("id")
    for action in ("approve", "reject", "disable"):
        a = sub.add_parser(action, help=f"{action} a pattern (overrides the automatic status)")
        a.add_argument("id")
        a.add_argument("note", nargs="*")
    args = p.parse_args()

    if args.cmd == "distill":
        store = distill(args.log_dir, args.feedback, args.out, not args.no_check)
        counts = {}
        for pat in store["patterns"]:
            counts[pat["status"]] = counts.get(pat["status"], 0) + 1
        print(f"{store['runs']} runs -> {len(store['patterns'])} patterns {counts}, "
              f"{len(store['cautions'])} cautions; refused: {store['refused']}")
    elif args.cmd == "list":
        for pat in _load_store()["patterns"]:
            if args.status in (None, pat["status"]):
                print(f"{pat['id']}  {pat['status']:9} s={pat['score']:.2f} "
                      f"+{pat['correct']}/-{pat['wrong']} runs={len(pat['runs'])}  "
                      f"{pat['aliases'][0]}")
    elif args.cmd == "show":
        pat = next((x for x in _load_store()["patterns"] if x["id"] == args.id), None)
        if pat is None:
            sys.exit(f"no pattern {args.id}")
        print(json.dumps(pat, ensure_ascii=False, indent=1))
    else:
        if not any(x["id"] == args.id for x in _load_store()["patterns"]):
            sys.exit(f"no pattern {args.id}")
        REVIEWS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with REVIEWS_FILE.open("a") as f:
            f.write(json.dumps({"pattern": args.id, "action": args.cmd,
                                "note": " ".join(args.note), "by": getpass.getuser(),
                                "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        store = distill()
        pat = next(x for x in store["patterns"] if x["id"] == args.id)
        print(f"{args.id}: {pat['status']} ({pat['status_by']})")


if __name__ == "__main__":
    main()
