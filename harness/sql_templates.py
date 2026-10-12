"""
Question/SQL templates for text-to-SQL training data, filled with values sampled from the data.

Each template takes a Sampler and returns an Instance (canonical question, SQL, and the
patterns a paraphrase must keep) or None when the sampled values don't fit. The SQL follows
the conventions in schema_context.md. Dates in eval/questions.toml (plus a margin) are never
sampled, so no training pair repeats an eval instance.
"""

import json
import random
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from ais_tool import _DATES
import places as gazetteer

EXCLUDED_DATES = {date(2026, 9, d) for d in range(20, 27)}  # eval window 22-24 Sep + margin
EXCLUDED_DATE_RE = re.compile(r"2026-09-2[0-6]")

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ship_type value -> (plural phrase for questions, pattern a paraphrase must contain)
SHIP_TYPES = {
    "Cargo": ("cargo ships", r"cargo|freighter|container"),
    "Fishing": ("fishing vessels", r"fish"),
    "Passenger": ("passenger ships", r"passenger|ferr"),
    "Tanker": ("tankers", r"tanker"),
    "Sailing": ("sailing vessels", r"sail"),
    "Tug": ("tugs", r"tug"),
    "Pleasure": ("pleasure craft", r"pleasure|leisure|recreational|yacht"),
    "Pilot": ("pilot vessels", r"pilot"),
    "HSC": ("high-speed craft", r"high[- ]speed|hsc"),
    "SAR": ("search and rescue vessels", r"rescue|sar\b"),
    "Dredging": ("dredgers", r"dredg"),
    "Military": ("military vessels", r"militar|naval|navy|warship"),
    "Towing": ("towing vessels", r"tow"),
}
COMMON_TYPES = ["Cargo", "Fishing", "Passenger", "Tanker", "Sailing", "Tug", "Pleasure", "Pilot"]
# Places from places.json only (never the learned ones). Hour-level templates use sea areas
# and ports with at least 40 moored vessels in build_places' sample and a box of at least
# BIG_MIN_SIDE_DEG a side; smaller places are often empty for an hour.
_RAW = json.loads(gazetteer.PLACES_FILE.read_text())["places"]
_PLACES = {p.name: p for p in gazetteer.load(include_learned=False)}
_MOORED = {p["name"]: p.get("evidence", {}).get("moored_vessels", 0) for p in _RAW}
BIG_MIN_SIDE_DEG = 0.02
ALL_PLACES = sorted(_PLACES)
BIG_PLACES = sorted(
    n for n, p in _PLACES.items()
    if p.kind == "area"
    or (min(p.box[2] - p.box[0], p.box[3] - p.box[1]) >= BIG_MIN_SIDE_DEG
        and (p.source == "manual" or _MOORED.get(n, 0) >= 40))
)
# Qualifiers a paraphrase tends to drop, which would change the answer.
MOVING_MUST = r"underway|under way|moving|in motion|making way|\b1 (knot|kn)\b|at least 1"
EVERY_DAY_MUST = r"every|each|all|both|consecutive|throughout"
AT_LEAST_MUST = r"at least|≥|>=|or (more|longer)|minimum|longer than|over"
STATUSES = ["At anchor", "Moored", "Engaged in fishing", "Under way sailing",
            "Restricted maneuverability"]


@dataclass
class Instance:
    family: str
    question: str
    sql: str
    must: list[str] = field(default_factory=list)  # regexes a paraphrase must match

    def keeps_specifics(self, text: str) -> bool:
        return all(re.search(p, text, re.IGNORECASE) for p in self.must)


# --- formatting helpers ----------------------------------------------------------------


def sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def d_sql(d: date) -> str:
    return f"DATE '{d.isoformat()}'"


def ts_sql(d: date, hhmm: str) -> str:
    return f"TIMESTAMP '{d.isoformat()} {hhmm}:00'"


def d_text(d: date, rng: random.Random) -> str:
    month = MONTHS[d.month - 1]
    return rng.choice([f"{d.day} {month}", f"{month} {d.day}", f"{d.day} {month} {d.year}"])


def d_must(d: date) -> str:
    month = MONTHS[d.month - 1].lower()
    day = rf"0?{d.day}(st|nd|rd|th)?"
    mon = rf"({month}|{month[:3]}\.?)"
    return rf"{d.isoformat()}|\b{day}\s+(of\s+)?{mon}\b|\b{mon}\s+{day}\b"


def hhmm_must(hhmm: str) -> str:
    h, m = hhmm.split(":")
    h12 = int(h) % 12 or 12
    if m == "00":
        return rf"\b0?{int(h)}(:00|\.00)?\b|\b{h12}\s*(am|pm|a\.m\.|p\.m\.)|noon|midnight"
    return rf"\b0?{int(h)}[:.]{m}\b"


def place_text(name: str) -> tuple[str, str]:
    """How a question names the place, and a pattern any of its spellings matches."""
    place = _PLACES[name]
    shown = f"the {name}" if place.kind == "area" else name
    spellings = {name, *place.aliases, gazetteer.normalise(name)}
    alternatives = "|".join(re.escape(s) for s in sorted(spellings, key=len, reverse=True) if s)
    return shown, rf"(?<!\w)(?:{alternatives})(?!\w)"  # whole words: Fur, not "further"


def box_sql(name: str, prefix: str = "") -> str:
    """The box exactly as the student's place list prints it (places.coord)."""
    lon0, lat0, lon1, lat1 = (gazetteer.coord(x) for x in _PLACES[name].box)
    return (f"{prefix}lon BETWEEN {lon0} AND {lon1} "
            f"AND {prefix}lat BETWEEN {lat0} AND {lat1}")


def num_must(n: int | float) -> str:
    return rf"\b{n:g}\b"


def name_must(name: str) -> str:
    return re.escape(name.lower()).replace(r"\ ", r"\s+")


# --- sampling real values --------------------------------------------------------------


@dataclass
class Seed:
    """One sampled report of a vessel whose name is unique on that day."""
    mmsi: int
    name: str
    ship_type: str
    ts: datetime
    lat: float
    lon: float


class Sampler:
    def __init__(self, con, rng: random.Random, n_days: int):
        self.con = con
        self.rng = rng
        days = [date.fromisoformat(d) for d in _DATES]
        days = [d for d in days if d not in EXCLUDED_DATES]
        self.days = sorted(rng.sample(days, min(n_days, len(days))))
        self._seeds: dict[date, list[Seed]] = {}

    def day(self) -> date:
        return self.rng.choice(self.days)

    def day_range(self, min_len: int = 2, max_len: int = 4) -> tuple[date, date]:
        n = self.rng.randint(min_len, max_len)
        d0 = self.day()
        d1 = d0 + timedelta(days=n - 1)
        if any(d0 + timedelta(days=i) in EXCLUDED_DATES for i in range(n)) or d1.isoformat() > _DATES[-1]:
            return self.day_range(min_len, max_len)
        return d0, d1

    def seeds(self, d: date) -> list[Seed]:
        if d not in self._seeds:
            rows = self.con.execute(f"""
                WITH names AS (
                  SELECT name FROM vessel_positions WHERE date = {d_sql(d)} AND name IS NOT NULL
                  GROUP BY name HAVING count(DISTINCT mmsi) = 1
                )
                SELECT mmsi, name, ship_type, ts, lat, lon FROM (
                  SELECT * FROM vessel_positions
                  WHERE date = {d_sql(d)} AND name IN (SELECT name FROM names)
                    AND regexp_matches(name, '[A-Z]{{3}}') AND strlen(name) <= 20
                ) USING SAMPLE reservoir(300 ROWS) REPEATABLE ({self.rng.randrange(10**6)})
            """).fetchall()
            self._seeds[d] = [Seed(*r) for r in rows]
        return self._seeds[d]

    def seed(self, d: date, ship_type: str | None = None) -> Seed | None:
        pool = [s for s in self.seeds(d) if ship_type is None or s.ship_type == ship_type]
        return self.rng.choice(pool) if pool else None

    def hour(self, lo: int = 0, hi: int = 23) -> int:
        return self.rng.randint(lo, hi)

    def ship_type(self, common: bool = True) -> str:
        return self.rng.choice(COMMON_TYPES if common else list(SHIP_TYPES))

    def place(self, big: bool = True) -> str:
        return self.rng.choice(BIG_PLACES if big else ALL_PLACES)


# --- templates ---------------------------------------------------------------------------
# Each returns Instance | None. `r` is the Sampler's Random, used for wording choices.


def vessels_on_day(s: Sampler) -> Instance:
    d, r = s.day(), s.rng
    q = r.choice(["How many different vessels were seen on {d}?",
                  "How many distinct ships sent AIS reports on {d}?"]).format(d=d_text(d, r))
    sql = f"SELECT count(DISTINCT mmsi) AS vessels\nFROM vessel_positions WHERE date = {d_sql(d)}"
    return Instance("vessels_on_day", q, sql, [d_must(d)])


def vessels_type_day(s: Sampler) -> Instance:
    d, t, r = s.day(), s.ship_type(common=False), s.rng
    plural, t_must = SHIP_TYPES[t]
    q = f"How many {plural} were active on {d_text(d, r)}?"
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels\nFROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND ship_type = {sql_str(t)}")
    return Instance("vessels_type_day", q, sql, [d_must(d), t_must])


def vessels_window(s: Sampler) -> Instance:
    d, h, r = s.day(), s.hour(0, 21), s.rng
    k = r.randint(1, 2)
    t0, t1 = f"{h:02d}:00", f"{h + k:02d}:00"
    q = f"How many vessels sent AIS reports between {t0} and {t1} UTC on {d_text(d, r)}?"
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels\nFROM vessel_positions\n"
           f"WHERE date = {d_sql(d)}\n  AND ts >= {ts_sql(d, t0)} AND ts < {ts_sql(d, t1)}")
    return Instance("vessels_window", q, sql, [d_must(d), hhmm_must(t0), hhmm_must(t1)])


def vessels_place_window(s: Sampler) -> Instance:
    d, h, p, r = s.day(), s.hour(0, 22), s.place(big=False), s.rng
    t0, t1 = f"{h:02d}:00", f"{h + 1:02d}:00"
    pt, pm = place_text(p)
    q = f"How many vessels were in {pt} between {t0} and {t1} UTC on {d_text(d, r)}?"
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)}\n  AND ts >= {ts_sql(d, t0)} AND ts < {ts_sql(d, t1)}\n"
           f"  AND {box_sql(p)}")
    return Instance("vessels_place_window", q, sql, [d_must(d), hhmm_must(t0), pm])


def vessels_place_type_range(s: Sampler) -> Instance:
    (d0, d1), t, p, r = s.day_range(), s.ship_type(), s.place(big=False), s.rng
    plural, t_must = SHIP_TYPES[t]
    pt, pm = place_text(p)
    q = f"How many different {plural} visited {pt} between {d_text(d0, r)} and {d_text(d1, r)}?"
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_positions\n"
           f"WHERE date BETWEEN {d_sql(d0)} AND {d_sql(d1)}\n"
           f"  AND ship_type = {sql_str(t)} AND {box_sql(p)}")
    return Instance("vessels_place_type_range", q, sql, [d_must(d0), d_must(d1), t_must, pm])


def vessels_every_day(s: Sampler) -> Instance:
    (d0, d1), r = s.day_range(2, 5), s.rng
    n = (d1 - d0).days + 1
    q = f"How many vessels were seen on every day from {d_text(d0, r)} to {d_text(d1, r)}?"
    sql = (f"SELECT count(*) AS vessels FROM (\n  SELECT mmsi FROM vessel_positions\n"
           f"  WHERE date BETWEEN {d_sql(d0)} AND {d_sql(d1)}\n"
           f"  GROUP BY mmsi HAVING count(DISTINCT date) = {n}\n)")
    return Instance("vessels_every_day", q, sql, [d_must(d0), d_must(d1), EVERY_DAY_MUST])


def daily_counts(s: Sampler) -> Instance:
    (d0, d1), t, r = s.day_range(3, 7), s.ship_type(), s.rng
    plural, t_must = SHIP_TYPES[t]
    q = f"How many {plural} were seen on each day from {d_text(d0, r)} to {d_text(d1, r)}?"
    sql = (f"SELECT date, count(DISTINCT mmsi) AS vessels FROM vessel_positions\n"
           f"WHERE date BETWEEN {d_sql(d0)} AND {d_sql(d1)} AND ship_type = {sql_str(t)}\n"
           f"GROUP BY date ORDER BY date")
    return Instance("daily_counts", q, sql, [d_must(d0), d_must(d1), t_must])


def hourly_counts_place(s: Sampler) -> Instance:
    d, p, r = s.day(), s.place(), s.rng
    pt, pm = place_text(p)
    q = f"How many vessels were in {pt} in each hour of {d_text(d, r)}?"
    sql = (f"SELECT hour(ts) AS hour_utc, count(DISTINCT mmsi) AS vessels FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND {box_sql(p)}\nGROUP BY 1 ORDER BY 1")
    return Instance("hourly_counts_place", q, sql, [d_must(d), pm])


def reports_of_vessel(s: Sampler) -> Instance | None:
    d, r = s.day(), s.rng
    v = s.seed(d)
    if v is None:
        return None
    q = f"How many position reports did {v.name} send on {d_text(d, r)}?"
    sql = (f"SELECT count(DISTINCT ts) AS reports FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND name = {sql_str(v.name)}")
    return Instance("reports_of_vessel", q, sql, [d_must(d), name_must(v.name)])


def most_reports(s: Sampler) -> Instance:
    d, t, r = s.day(), s.ship_type(), s.rng
    plural, t_must = SHIP_TYPES[t]
    q = f"Which of the {plural} sent the most position reports on {d_text(d, r)}?"
    sql = (f"SELECT mmsi FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND ship_type = {sql_str(t)}\n"
           f"GROUP BY mmsi ORDER BY count(DISTINCT ts) DESC, mmsi LIMIT 1")
    return Instance("most_reports", q, sql, [d_must(d), t_must])


def top_types(s: Sampler) -> Instance:
    d, r = s.day(), s.rng
    n = r.randint(3, 5)
    q = f"Which {n} ship types had the most vessels on {d_text(d, r)}, not counting 'Undefined'?"
    sql = (f"SELECT ship_type, count(DISTINCT mmsi) AS vessels FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND ship_type <> 'Undefined'\n"
           f"GROUP BY ship_type ORDER BY vessels DESC, ship_type LIMIT {n}")
    return Instance("top_types", q, sql, [d_must(d), num_must(n), r"undefined|unknown|known"])


def busiest_hour_place(s: Sampler) -> Instance:
    d, p, r = s.day(), s.place(), s.rng
    pt, pm = place_text(p)
    q = f"Which hour on {d_text(d, r)} had the most vessels in {pt}?"
    sql = (f"SELECT hour(ts) AS hour_utc FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND {box_sql(p)}\n"
           f"GROUP BY 1 ORDER BY count(DISTINCT mmsi) DESC, hour_utc LIMIT 1")
    return Instance("busiest_hour_place", q, sql, [d_must(d), pm])


def avg_speed_type_place(s: Sampler) -> Instance:
    d, t, p, r = s.day(), s.ship_type(), s.place(), s.rng
    plural, t_must = SHIP_TYPES[t]
    pt, pm = place_text(p)
    q = f"What was the average speed of {plural} underway in {pt} on {d_text(d, r)}?"
    sql = (f"SELECT avg(sog) AS avg_sog_kn FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND ship_type = {sql_str(t)} AND sog >= 1\n"
           f"  AND {box_sql(p)}")
    return Instance("avg_speed_type_place", q, sql, [d_must(d), t_must, pm, MOVING_MUST])


def fastest_vessel_place(s: Sampler) -> Instance:
    d, p, r = s.day(), s.place(), s.rng
    pt, pm = place_text(p)
    q = f"Which vessel reported the highest speed in {pt} on {d_text(d, r)}?"
    sql = (f"SELECT mmsi, max(sog) AS max_sog_kn FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND sog IS NOT NULL AND {box_sql(p)}\n"
           f"GROUP BY mmsi ORDER BY max_sog_kn DESC, mmsi LIMIT 1")
    return Instance("fastest_vessel_place", q, sql, [d_must(d), pm])


def crowded_cell(s: Sampler) -> Instance:
    d, h, r = s.day(), s.hour(), s.rng
    size, inv = r.choice([(0.1, 10), (0.5, 2)])
    m = r.choice([10, 30])
    t0 = f"{h:02d}:00"
    t1 = f"{h:02d}:{m}"
    q = (f"Which {size}-degree grid cell had the most vessels between {t0} and {t1} UTC on "
         f"{d_text(d, r)}? Give the cell's south-west corner.")
    sql = (f"SELECT floor(lat * {inv}) / {inv} AS lat_min, floor(lon * {inv}) / {inv} AS lon_min\n"
           f"FROM vessel_positions\nWHERE date = {d_sql(d)}\n"
           f"  AND ts >= {ts_sql(d, t0)} AND ts < {ts_sql(d, t1)}\n"
           f"GROUP BY 1, 2 ORDER BY count(DISTINCT mmsi) DESC, lat_min, lon_min LIMIT 1")
    return Instance("crowded_cell", q, sql, [d_must(d), hhmm_must(t0), num_must(size)])


def position_at_time(s: Sampler) -> Instance | None:
    d, r = s.day(), s.rng
    v = s.seed(d)
    if v is None:
        return None
    t = (v.ts + timedelta(minutes=r.randint(0, 5))).strftime("%H:%M")
    q = f"Where was {v.name} at {t} UTC on {d_text(d, r)}?"
    sql = (f"SELECT lat, lon FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND name = {sql_str(v.name)}\n"
           f"  AND ts <= {ts_sql(d, t)} AND ts > {ts_sql(d, t)} - INTERVAL 10 MINUTE\n"
           f"ORDER BY ts DESC LIMIT 1")
    return Instance("position_at_time", q, sql, [d_must(d), name_must(v.name), hhmm_must(t)])


def first_last_seen(s: Sampler) -> Instance | None:
    d, r = s.day(), s.rng
    v = s.seed(d)
    if v is None:
        return None
    q = f"When was {v.name} first and last seen on {d_text(d, r)}?"
    sql = (f"SELECT min(ts) AS first_seen, max(ts) AS last_seen FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND name = {sql_str(v.name)}")
    return Instance("first_last_seen", q, sql, [d_must(d), name_must(v.name)])


def vessel_max_speed(s: Sampler) -> Instance | None:
    d, r = s.day(), s.rng
    v = s.seed(d)
    if v is None:
        return None
    q = f"What was the highest speed {v.name} reported on {d_text(d, r)}?"
    sql = (f"SELECT max(sog) AS max_sog_kn FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND name = {sql_str(v.name)}")
    return Instance("vessel_max_speed", q, sql, [d_must(d), name_must(v.name)])


def vessel_destination(s: Sampler) -> Instance | None:
    d, r = s.day(), s.rng
    v = s.seed(d)
    if v is None:
        return None
    q = f"Which destination did {v.name} report on {d_text(d, r)}?"
    sql = (f"SELECT DISTINCT destination FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND name = {sql_str(v.name)} AND destination IS NOT NULL\n"
           f"ORDER BY destination")
    return Instance("vessel_destination", q, sql, [d_must(d), name_must(v.name)])


def vessel_static(s: Sampler) -> Instance | None:
    d, r = s.day(), s.rng
    v = s.seed(d)
    if v is None:
        return None
    q = f"What type of ship is {v.name} and how long is it, according to its reports on {d_text(d, r)}?"
    sql = (f"SELECT DISTINCT ship_type, length FROM vessel_positions\n"
           f"WHERE date = {d_sql(d)} AND name = {sql_str(v.name)}")
    return Instance("vessel_static", q, sql, [d_must(d), name_must(v.name)])


def vessel_distance(s: Sampler) -> Instance | None:
    d, r = s.day(), s.rng
    v = s.seed(d)
    if v is None:
        return None
    q = f"How many kilometres did {v.name} travel on {d_text(d, r)}?"
    sql = (f"SELECT sum(haversine_m(prev_lat, prev_lon, lat, lon)) / 1000 AS km FROM (\n"
           f"  SELECT lat, lon, lag(lat) OVER (ORDER BY ts) AS prev_lat,\n"
           f"         lag(lon) OVER (ORDER BY ts) AS prev_lon\n"
           f"  FROM vessel_positions\n"
           f"  WHERE date = {d_sql(d)} AND name = {sql_str(v.name)}\n)")
    return Instance("vessel_distance", q, sql, [d_must(d), name_must(v.name)])


def underway_type_at_time(s: Sampler) -> Instance:
    d, t, r = s.day(), s.ship_type(), s.rng
    plural, t_must = SHIP_TYPES[t]
    hhmm = f"{s.hour(1, 23):02d}:00"
    q = f"How many {plural} were underway at {hhmm} UTC on {d_text(d, r)}?"
    sql = (f"SELECT count(*) AS vessels FROM (\n"
           f"  SELECT mmsi, arg_max(sog, ts) AS sog FROM vessel_positions\n"
           f"  WHERE date = {d_sql(d)} AND ship_type = {sql_str(t)}\n"
           f"    AND ts > {ts_sql(d, hhmm)} - INTERVAL 10 MINUTE AND ts <= {ts_sql(d, hhmm)}\n"
           f"  GROUP BY mmsi\n) WHERE sog >= 1")
    return Instance("underway_type_at_time", q, sql,
                    [d_must(d), t_must, hhmm_must(hhmm), MOVING_MUST])


def status_at_time_place(s: Sampler) -> Instance:
    d, p, r = s.day(), s.place(), s.rng
    status = r.choice(["At anchor", "Moored"])
    hhmm = f"{s.hour(1, 23):02d}:00"
    pt, pm = place_text(p)
    phrase = {"At anchor": "at anchor", "Moored": "moored"}[status]
    q = f"How many vessels in {pt} were {phrase} at {hhmm} UTC on {d_text(d, r)}?"
    sql = (f"SELECT count(*) AS vessels FROM (\n"
           f"  SELECT mmsi, arg_max(nav_status, ts) AS nav_status FROM vessel_positions\n"
           f"  WHERE date = {d_sql(d)} AND {box_sql(p)}\n"
           f"    AND ts > {ts_sql(d, hhmm)} - INTERVAL 10 MINUTE AND ts <= {ts_sql(d, hhmm)}\n"
           f"  GROUP BY mmsi\n) WHERE nav_status = {sql_str(status)}")
    return Instance("status_at_time_place", q, sql,
                    [d_must(d), pm, hhmm_must(hhmm), "anchor" if status == "At anchor" else "moor"])


def status_share(s: Sampler) -> Instance:
    d, r = s.day(), s.rng
    status = r.choice(STATUSES)
    q = (f"What percentage of vessels reported the navigational status '{status}' at least once "
         f"on {d_text(d, r)}?")
    sql = (f"SELECT 100.0 * count(DISTINCT mmsi) FILTER (WHERE nav_status = {sql_str(status)})\n"
           f"       / count(DISTINCT mmsi) AS pct\n"
           f"FROM vessel_positions WHERE date = {d_sql(d)}")
    return Instance("status_share", q, sql, [d_must(d), re.escape(status.lower())])


def near_vessel(s: Sampler) -> Instance | None:
    d, r = s.day(), s.rng
    v = s.seed(d)
    if v is None:
        return None
    dist = r.choice([500, 1000, 2000])
    t0 = v.ts.strftime("%H:00")
    t1 = (v.ts.replace(minute=0, second=0) + timedelta(hours=1)).strftime("%H:%M")
    if t1 == "00:00":
        return None
    q = f"Which vessels came within {dist} m of {v.name} between {t0} and {t1} UTC on {d_text(d, r)}?"
    sql = (f"WITH m AS (\n"
           f"  SELECT mmsi, date_trunc('minute', ts) AS minute,\n"
           f"         arg_max(lat, ts) AS lat, arg_max(lon, ts) AS lon\n"
           f"  FROM vessel_positions\n"
           f"  WHERE date = {d_sql(d)} AND ts >= {ts_sql(d, t0)} AND ts < {ts_sql(d, t1)}\n"
           f"  GROUP BY 1, 2\n"
           f"), target AS (\n"
           f"  SELECT * FROM m WHERE mmsi = (\n"
           f"    SELECT any_value(mmsi) FROM vessel_positions\n"
           f"    WHERE date = {d_sql(d)} AND name = {sql_str(v.name)})\n"
           f")\n"
           f"SELECT DISTINCT o.mmsi\n"
           f"FROM target t JOIN m o ON o.minute = t.minute AND o.mmsi <> t.mmsi\n"
           f"WHERE haversine_m(t.lat, t.lon, o.lat, o.lon) <= {dist}\nORDER BY o.mmsi")
    return Instance("near_vessel", q, sql, [d_must(d), name_must(v.name), num_must(dist)])


def closest_pair_place(s: Sampler) -> Instance:
    d, p, r = s.day(), s.place(), s.rng
    h = s.hour(0, 23)
    t0, t1 = f"{h:02d}:00", f"{h:02d}:10"
    pt, pm = place_text(p)
    q = (f"Which two moving vessels came closest to each other in {pt} between {t0} and {t1} "
         f"UTC on {d_text(d, r)}?")
    sql = (f"WITH m AS (\n"
           f"  SELECT mmsi, date_trunc('minute', ts) AS minute,\n"
           f"         arg_max(lat, ts) AS lat, arg_max(lon, ts) AS lon, arg_max(sog, ts) AS sog\n"
           f"  FROM vessel_positions\n"
           f"  WHERE date = {d_sql(d)} AND ts >= {ts_sql(d, t0)} AND ts < {ts_sql(d, t1)}\n"
           f"    AND {box_sql(p)}\n"
           f"  GROUP BY 1, 2\n)\n"
           f"SELECT a.mmsi AS mmsi_a, b.mmsi AS mmsi_b\n"
           f"FROM m a JOIN m b ON a.minute = b.minute AND a.mmsi < b.mmsi\n"
           f"WHERE a.sog >= 1 AND b.sog >= 1\nGROUP BY 1, 2\n"
           f"ORDER BY min(haversine_m(a.lat, a.lon, b.lat, b.lon)), mmsi_a, mmsi_b LIMIT 1")
    return Instance("closest_pair_place", q, sql, [d_must(d), pm, hhmm_must(t0), MOVING_MUST])


def long_ships(s: Sampler) -> Instance:
    (d0, d1), r = s.day_range(), s.rng
    t, lo, hi = r.choice([("Cargo", 250, 400), ("Tanker", 200, 330), ("Passenger", 150, 300)])
    plural, t_must = SHIP_TYPES[t]
    length = r.randrange(lo, hi, 10)
    q = f"Which {plural} at least {length} m long were seen from {d_text(d0, r)} to {d_text(d1, r)}?"
    sql = (f"SELECT DISTINCT mmsi FROM vessel_positions\n"
           f"WHERE date BETWEEN {d_sql(d0)} AND {d_sql(d1)}\n"
           f"  AND ship_type = {sql_str(t)} AND length >= {length}\nORDER BY mmsi")
    return Instance("long_ships", q, sql,
                    [d_must(d0), d_must(d1), t_must, num_must(length), AT_LEAST_MUST])


def top_destinations(s: Sampler) -> Instance:
    (d0, d1), r = s.day_range(), s.rng
    t = r.choice(["Cargo", "Tanker", "Passenger", "Fishing"])
    plural, t_must = SHIP_TYPES[t]
    n = r.randint(3, 5)
    q = (f"What were the {n} most common destinations declared by {plural} from "
         f"{d_text(d0, r)} to {d_text(d1, r)}?")
    sql = (f"SELECT upper(destination) AS destination FROM vessel_positions\n"
           f"WHERE date BETWEEN {d_sql(d0)} AND {d_sql(d1)} AND ship_type = {sql_str(t)}\n"
           f"GROUP BY 1 ORDER BY count(DISTINCT mmsi) DESC, destination LIMIT {n}")
    return Instance("top_destinations", q, sql, [d_must(d0), d_must(d1), t_must, num_must(n)])


def duplicate_share(s: Sampler) -> Instance:
    d, r = s.day(), s.rng
    q = f"What percentage of the raw AIS rows on {d_text(d, r)} were duplicate receptions of the same report?"
    sql = (f"SELECT 100.0 * (count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM ais\n"
           f"                                                   WHERE date = {d_sql(d)})))\n"
           f"       / count(*) AS pct\nFROM ais WHERE date = {d_sql(d)}")
    return Instance("duplicate_share", q, sql, [d_must(d), r"duplicat|repeat|same report"])


def non_vessel_rows(s: Sampler) -> Instance:
    d, r = s.day(), s.rng
    kind, phrase, pm = r.choice([("Base Station", "base stations", r"base station"),
                                 ("AtoN", "aids to navigation (AtoN)", r"aton|aids? to navigation")])
    q = f"How many raw AIS rows on {d_text(d, r)} came from {phrase}?"
    sql = (f"SELECT count(*) AS n_rows FROM ais\n"
           f"WHERE date = {d_sql(d)} AND mobile_type = {sql_str(kind)}")
    return Instance("non_vessel_rows", q, sql, [d_must(d), pm])


TEMPLATES = [
    vessels_on_day, vessels_type_day, vessels_window, vessels_place_window,
    vessels_place_type_range, vessels_every_day, daily_counts, hourly_counts_place,
    reports_of_vessel, most_reports, top_types, busiest_hour_place, avg_speed_type_place,
    fastest_vessel_place, crowded_cell, position_at_time, first_last_seen, vessel_max_speed,
    vessel_destination, vessel_static, vessel_distance, underway_type_at_time,
    status_at_time_place, status_share, near_vessel, closest_pair_place, long_ships,
    top_destinations, duplicate_share, non_vessel_rows,
]
