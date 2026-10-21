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


def sql_ident(name: str) -> str:
    """A column alias from a place or type name: 'Great Belt' -> great_belt."""
    return re.sub(r"\W+", "_", gazetteer.normalise(name)).strip("_").lower() or "n"


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


# --- composite templates (sql-v4) --------------------------------------------------------
# Questions the router used to send to the planner because they combine several counts:
# two days, two places, two ship types, several places at once, averages of daily counts.
# Each is still ONE query (CTEs, UNION ALL, INTERSECT, EXCEPT), so the student can answer it
# directly. Places are BIG_PLACES (or ports) so the single counts are rarely zero.
BOTH_MUST = r"both|as well as|and also|each of|and then"
AVERAGE_MUST = r"average|mean|per day|daily|a day"
NOT_BEFORE_MUST = r"\bnot\b|n't|\bnew\b|absent|without|never|first time"
PORTS = sorted(n for n, p in _PLACES.items() if p.kind == "port")


def _centre(name: str) -> tuple[float, float]:
    lon0, lat0, lon1, lat1 = _PLACES[name].box
    return (lon0 + lon1) / 2, (lat0 + lat1) / 2


def _near(a: str, b: str) -> float:
    (x0, y0), (x1, y1) = _centre(a), _centre(b)
    return ((x0 - x1) * 0.55) ** 2 + (y0 - y1) ** 2  # ~cos(57 deg) for longitude


# The 5 nearest other ports of each port: random pairs (Landskrona, Brunsbuettel) almost
# never share a vessel within a few days.
NEAR_PORTS = {p: sorted((q for q in PORTS if q != p), key=lambda q: _near(p, q))[:5]
              for p in PORTS}


def what(t: str | None) -> tuple[str, list[str]]:
    """Plural phrase for 'vessels' or one ship type, and the patterns it adds."""
    if t is None:
        return "vessels", []
    plural, t_must = SHIP_TYPES[t]
    return plural, [t_must]


def date_where(d0: date, d1: date) -> str:
    return f"date = {d_sql(d0)}" if d0 == d1 else f"date BETWEEN {d_sql(d0)} AND {d_sql(d1)}"


def period_text(d0: date, d1: date, r: random.Random) -> tuple[str, list[str]]:
    """'on <day>' or 'between <day> and <day>', and the patterns a paraphrase must keep."""
    if d0 == d1:
        return f"on {d_text(d0, r)}", [d_must(d0)]
    return (r.choice(["between {a} and {b}", "from {a} to {b}"]).format(
        a=d_text(d0, r), b=d_text(d1, r)), [d_must(d0), d_must(d1)])


def period(s: Sampler) -> tuple[date, date]:
    """One day (60 %) or a 2-3 day range."""
    if s.rng.random() < 0.6:
        d = s.day()
        return d, d
    return s.day_range(2, 3)


def type_clause(t: str | None) -> str:
    return f" AND ship_type = {sql_str(t)}" if t else ""


def count_part(where: str, p: str | None, t: str | None) -> str:
    box = f" AND {box_sql(p)}" if p else ""
    return (f"SELECT count(DISTINCT mmsi) AS n FROM vessel_positions\n"
            f"  WHERE {where}{box}{type_clause(t)}")


def two_days(s: Sampler) -> tuple[date, date]:
    """Two different days: half the time at most a week apart, otherwise any two."""
    if s.rng.random() < 0.5:
        d1, d2 = s.day_range(2, 8)
        return d1, d2
    d1, d2 = sorted(s.rng.sample(s.days, 2))
    return d1, d2


def day_change(s: Sampler) -> Instance:
    r = s.rng
    (d1, d2) = two_days(s)
    p = s.place() if r.random() < 0.75 else None
    t = s.ship_type() if r.random() < 0.5 else None
    noun, musts = what(t)
    where_text, pm = (f" in {place_text(p)[0]}", [place_text(p)[1]]) if p else ("", [])
    q = r.choice([
        "How did the number of {w}{p} change from {a} to {b}?",
        "Compare the number of {w}{p} on {a} and on {b}. What was the difference?",
        "Were there more or fewer {w}{p} on {b} than on {a}, and by how many?",
        "How many {w} were seen{p} on {a} and on {b}, and what is the change?",
    ]).format(w=noun, p=where_text, a=d_text(d1, r), b=d_text(d2, r))
    sql = (f"WITH a AS ({count_part(f'date = {d_sql(d1)}', p, t)}),\n"
           f"     b AS ({count_part(f'date = {d_sql(d2)}', p, t)})\n"
           f"SELECT a.n AS first_day, b.n AS second_day, b.n - a.n AS change FROM a, b")
    return Instance("day_change", q, sql, [d_must(d1), d_must(d2), *pm, *musts])


def compare_places(s: Sampler) -> Instance:
    r = s.rng
    p1, p2 = r.sample(BIG_PLACES, 2)
    d0, d1 = period(s)
    t = s.ship_type() if r.random() < 0.7 else None
    noun, musts = what(t)
    (a, am), (b, bm) = place_text(p1), place_text(p2)
    when, dm = period_text(d0, d1, r)
    q = r.choice([
        "Were there more {w} in {a} or in {b} {when}?",
        "Which had more {w} {when}, {a} or {b}? Give both numbers.",
        "Compare the number of {w} in {a} and in {b} {when}.",
    ]).format(w=noun, a=a, b=b, when=when)
    where = date_where(d0, d1)
    sql = (f"WITH a AS ({count_part(where, p1, t)}),\n"
           f"     b AS ({count_part(where, p2, t)})\n"
           f"SELECT CASE WHEN a.n > b.n THEN {sql_str(p1)} WHEN b.n > a.n THEN {sql_str(p2)} "
           f"END AS more_in,\n       a.n AS {sql_ident(p1)}, b.n AS {sql_ident(p2)} FROM a, b")
    return Instance("compare_places", q, sql, [*dm, am, bm, *musts])


def compare_types(s: Sampler) -> Instance:
    r = s.rng
    t1, t2 = r.sample(COMMON_TYPES, 2)
    (n1, m1), (n2, m2) = SHIP_TYPES[t1], SHIP_TYPES[t2]
    p = s.place() if r.random() < 0.7 else None
    d0, d1 = period(s)
    when, dm = period_text(d0, d1, r)
    where_text, pm = (f" in {place_text(p)[0]}", [place_text(p)[1]]) if p else ("", [])
    q = r.choice([
        "Were there more {a} or {b}{p} {when}?",
        "How many {a} and how many {b} were seen{p} {when}?",
        "Compare the number of {a} with the number of {b}{p} {when}.",
    ]).format(a=n1, b=n2, p=where_text, when=when)
    box = f" AND {box_sql(p)}" if p else ""
    sql = (f"SELECT count(DISTINCT mmsi) FILTER (WHERE ship_type = {sql_str(t1)}) "
           f"AS {sql_ident(t1)},\n"
           f"       count(DISTINCT mmsi) FILTER (WHERE ship_type = {sql_str(t2)}) "
           f"AS {sql_ident(t2)}\n"
           f"FROM vessel_positions WHERE {date_where(d0, d1)}{box}")
    return Instance("compare_types", q, sql, [*dm, *pm, m1, m2])


def union_places(s: Sampler) -> Instance:
    r = s.rng
    chosen = r.sample(PORTS, r.choice([2, 3]))
    d0, d1 = period(s)
    when, dm = period_text(d0, d1, r)
    names = [place_text(p)[0] for p in chosen]
    listed = ", ".join(names[:-1]) + " or " + names[-1]
    q = r.choice([
        "How many different vessels were in {l} {when}?",
        "How many distinct vessels visited at least one of {l2} {when}?",
        "Counting each vessel once, how many ships were seen in {l} {when}?",
    ]).format(l=listed, l2=", ".join(names[:-1]) + " and " + names[-1], when=when)
    boxes = "\n    OR ".join(f"({box_sql(p)})" for p in chosen)
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_positions\n"
           f"WHERE {date_where(d0, d1)}\n  AND ({boxes})")
    return Instance("union_places", q, sql, [*dm, *(place_text(p)[1] for p in chosen)])


def both_places(s: Sampler) -> Instance:
    r = s.rng
    p1 = r.choice(PORTS)
    p2 = r.choice(NEAR_PORTS[p1])
    d0, d1 = s.day_range(3, 7)
    when, dm = period_text(d0, d1, r)
    (a, am), (b, bm) = place_text(p1), place_text(p2)
    q = r.choice([
        "How many vessels were in both {a} and {b} {when}?",
        "How many ships visited {a} and also {b} {when}?",
        "Which vessels were seen in both {a} and {b} {when}?",
    ]).format(a=a, b=b, when=when)
    part = "SELECT mmsi FROM vessel_positions WHERE {w} AND {box}"
    inner = (part.format(w=date_where(d0, d1), box=box_sql(p1)) + "\n  INTERSECT\n  "
             + part.format(w=date_where(d0, d1), box=box_sql(p2)))
    if q.startswith("Which"):
        sql = f"SELECT mmsi FROM (\n  {inner}\n) ORDER BY mmsi"
    else:
        sql = f"SELECT count(*) AS vessels FROM (\n  {inner}\n)"
    return Instance("both_places", q, sql, [*dm, am, bm, BOTH_MUST])


def daily_average(s: Sampler) -> Instance:
    r = s.rng
    d0, d1 = s.day_range(3, 7)
    n = (d1 - d0).days + 1
    p = s.place()
    t = s.ship_type() if r.random() < 0.7 else None
    noun, musts = what(t)
    pt, pm = place_text(p)
    q = r.choice([
        "On average, how many {w} were in {p} per day from {a} to {b}?",
        "What was the average daily number of {w} in {p} between {a} and {b}?",
        "How many {w} were in {p} on an average day from {a} to {b}?",
    ]).format(w=noun, p=pt, a=d_text(d0, r), b=d_text(d1, r))
    # Distinct (day, vessel) pairs over the number of days: days without vessels count as 0.
    sql = (f"SELECT count(*) / {n}.0 AS per_day FROM (\n"
           f"  SELECT DISTINCT date, mmsi FROM vessel_positions\n"
           f"  WHERE {date_where(d0, d1)} AND {box_sql(p)}{type_clause(t)}\n)")
    return Instance("daily_average", q, sql, [d_must(d0), d_must(d1), pm, AVERAGE_MUST, *musts])


def busiest_of_places(s: Sampler) -> Instance:
    r = s.rng
    chosen = r.sample(BIG_PLACES, 3)
    d0, d1 = period(s)
    when, dm = period_text(d0, d1, r)
    t = s.ship_type() if r.random() < 0.5 else None
    noun, musts = what(t)
    names = [place_text(p)[0] for p in chosen]
    q = r.choice([
        "Which of {a}, {b} and {c} had the most {w} {when}? Give the number for each.",
        "Rank {a}, {b} and {c} by the number of {w} {when}.",
        "How many {w} were in each of {a}, {b} and {c} {when}?",
    ]).format(a=names[0], b=names[1], c=names[2], w=noun, when=when)
    parts = "\n  UNION ALL\n  ".join(
        f"SELECT {sql_str(p)} AS place, mmsi FROM vessel_positions\n"
        f"  WHERE {date_where(d0, d1)} AND {box_sql(p)}{type_clause(t)}" for p in chosen)
    sql = (f"SELECT place, count(DISTINCT mmsi) AS vessels FROM (\n  {parts}\n)\n"
           f"GROUP BY place ORDER BY vessels DESC, place")
    return Instance("busiest_of_places", q, sql,
                    [*dm, *(place_text(p)[1] for p in chosen), *musts])


def new_vessels(s: Sampler) -> Instance:
    r = s.rng
    d1, d2 = two_days(s)
    p = s.place()
    pt, pm = place_text(p)
    q = r.choice([
        "How many vessels were in {p} on {b} that had not been there on {a}?",
        "How many ships seen in {p} on {b} were not seen there on {a}?",
        "Of the vessels in {p} on {b}, how many were new compared with {a}?",
    ]).format(p=pt, a=d_text(d1, r), b=d_text(d2, r))
    part = "SELECT mmsi FROM vessel_positions WHERE date = {d} AND {box}"
    sql = (f"SELECT count(*) AS vessels FROM (\n"
           f"  {part.format(d=d_sql(d2), box=box_sql(p))}\n  EXCEPT\n"
           f"  {part.format(d=d_sql(d1), box=box_sql(p))}\n)")
    return Instance("new_vessels", q, sql, [d_must(d1), d_must(d2), pm, NOT_BEFORE_MUST])


COMPOSITE_TEMPLATES = [
    day_change, compare_places, compare_types, union_places, both_places, daily_average,
    busiest_of_places, new_vessels,
]


# --- sql-v5 templates -------------------------------------------------------------------
# Long periods, whole-data totals and per-vessel averages from vessel_days (one row per vessel
# per day, build_vessel_days.py), and "harbored at a moment" alone and in comparisons (sql-v4
# compared whole days and dropped the harbour condition). Months are never September 2026,
# which holds the eval days.
HARBOR_MUST = r"harbou?r|dock|in port|berth|quay|tied up"
PASS_MUST = r"pass|under|through|transit|cross|went|go"
LARGEST_MUST = r"larg|long|big|max"
DISTANCE_MUST = r"distance|far|travel|longest|most (?:km|kilomet)"
_DATA_MONTHS = sorted({(int(d[:4]), int(d[5:7])) for d in _DATES} - {(2026, 9)})
BRIDGES = ["Great Belt Bridge", "Little Belt bridges"]
MAX_REAL_LENGTH_M = 400  # the longest real ships are ~400 m; longer AIS lengths are errors


def singular(plural: str) -> str:
    return plural if plural.endswith("craft") else plural[:-1]


def month_span(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return first, last


def long_period(s: Sampler) -> tuple[date, date, str, list[str]]:
    """A month (60 %) or a 8-40 day range, its wording and the patterns to keep."""
    r = s.rng
    if r.random() < 0.6:
        year, month = r.choice(_DATA_MONTHS)
        d0, d1 = month_span(year, month)
        name = MONTHS[month - 1]
        with_year = r.random() < 0.7
        text = r.choice(["in {m}", "during {m}", "in the month of {m}"]).format(
            m=f"{name} {year}" if with_year else name)
        musts = [name.lower()[:3]] + ([str(year)] if with_year else [])
        return d0, d1, text, musts
    while True:
        d0 = s.day()
        d1 = d0 + timedelta(days=r.randint(8, 40))
        days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
        if d1.isoformat() <= _DATES[-1] and not any(d in EXCLUDED_DATES for d in days):
            break
    text = r.choice(["between {a} and {b}", "from {a} to {b}"]).format(
        a=d_text(d0, r), b=d_text(d1, r))
    return d0, d1, text, [d_must(d0), d_must(d1)]


def vd_where(d0: date, d1: date, extra: str = "") -> str:
    when = f"date = {d_sql(d0)}" if d0 == d1 else f"date BETWEEN {d_sql(d0)} AND {d_sql(d1)}"
    return f"{when}{extra}"


def vd_type(t: str | None) -> str:
    return f" AND ship_type = {sql_str(t)}" if t else ""


def vd_count_place_period(s: Sampler) -> Instance:
    r = s.rng
    d0, d1, when, dm = long_period(s)
    p = s.place(big=False)
    t = s.ship_type() if r.random() < 0.5 else None
    noun, tm = what(t)
    pt, pm = place_text(p)
    q = r.choice(["How many different {w} visited {p} {when}?",
                  "How many distinct {w} were seen in {p} {when}?",
                  "Count the {w} that were in {p} {when}."]).format(w=noun, p=pt, when=when)
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_days\n"
           f"WHERE {vd_where(d0, d1)} AND list_contains(places, {sql_str(p)}){vd_type(t)}")
    return Instance("vd_count_place_period", q, sql, [*dm, pm, *tm])


def vd_docked_period(s: Sampler) -> Instance:
    r = s.rng
    d0, d1, when, dm = long_period(s)
    p = r.choice(PORTS)
    t = s.ship_type() if r.random() < 0.4 else None
    noun, tm = what(t)
    pt, pm = place_text(p)
    q = r.choice(["How many {w} docked in {p} {when}?",
                  "How many different {w} were harbored in {p} {when}?",
                  "How many {w} lay in port at {p} {when}?"]).format(w=noun, p=pt, when=when)
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_days\n"
           f"WHERE {vd_where(d0, d1)} AND list_contains(still_in, {sql_str(p)}){vd_type(t)}")
    return Instance("vd_docked_period", q, sql, [*dm, pm, HARBOR_MUST, *tm])


def vd_moving_through(s: Sampler) -> Instance:
    r = s.rng
    d0, d1, when, dm = long_period(s)
    p = r.choice(BRIDGES) if r.random() < 0.6 else s.place()
    t = s.ship_type() if r.random() < 0.3 else None
    noun, tm = what(t)
    pt, pm = place_text(p)
    if p in BRIDGES:
        q = r.choice(["How many {w} passed under {p} {when}?",
                      "How many different {w} went under {p} {when}?"])
    else:
        q = r.choice(["How many {w} moved through {p} {when}?",
                      "How many different {w} passed through {p} {when}?"])
    q = q.format(w=noun, p=pt, when=when)
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_days\n"
           f"WHERE {vd_where(d0, d1)} AND list_contains(moving_in, {sql_str(p)}){vd_type(t)}")
    return Instance("vd_moving_through", q, sql, [*dm, pm, PASS_MUST, *tm])


def vd_compare_period(s: Sampler) -> Instance:
    r = s.rng
    d0, d1, when, dm = long_period(s)
    bridges = r.random() < 0.4
    p1, p2 = (BRIDGES if r.random() < 0.5 else BRIDGES[::-1]) if bridges else r.sample(BIG_PLACES, 2)
    column = "moving_in" if bridges else "places"
    t = s.ship_type() if r.random() < 0.4 else None
    noun, tm = what(t)
    (a, am), (b, bm) = place_text(p1), place_text(p2)
    if bridges:
        q = r.choice(["Did more {w} pass under {a} or under {b} {when}?",
                      "Which had more {w} passing under it {when}, {a} or {b}?"])
    else:
        q = r.choice(["Were there more {w} in {a} or in {b} {when}?",
                      "Which saw more different {w} {when}, {a} or {b}? Give both numbers."])
    q = q.format(w=noun, a=a, b=b, when=when)
    part = ("SELECT count(DISTINCT mmsi) AS n FROM vessel_days WHERE {w} AND "
            "list_contains({c}, {p}){t}")
    sql = (f"WITH a AS ({part.format(w=vd_where(d0, d1), c=column, p=sql_str(p1), t=vd_type(t))}),\n"
           f"     b AS ({part.format(w=vd_where(d0, d1), c=column, p=sql_str(p2), t=vd_type(t))})\n"
           f"SELECT CASE WHEN a.n > b.n THEN {sql_str(p1)} WHEN b.n > a.n THEN {sql_str(p2)} "
           f"END AS more_in,\n       a.n AS {sql_ident(p1)}, b.n AS {sql_ident(p2)} FROM a, b")
    return Instance("vd_compare_period", q, sql,
                    [*dm, am, bm, *tm] + ([PASS_MUST] if bridges else []))


def vd_daily_average(s: Sampler) -> Instance | None:
    r = s.rng
    whole = r.random() < 0.2
    t = s.ship_type() if r.random() < 0.5 else None
    noun, tm = what(t)
    if whole:
        p = None
        q = r.choice(["On average, how many {w} are seen per day in the data?",
                      "Across the whole data, what is the mean daily count of {w}?",
                      "How many {w} are there on an average day?"]).format(w=noun)
        where, dm = (f"ship_type = {sql_str(t)}" if t else ""), []
    else:
        d0, d1, when, dm = long_period(s)
        p = s.place() if r.random() < 0.5 else None
        where_text = f" in {place_text(p)[0]}" if p else ""
        q = r.choice(["On average, how many {w} were seen per day{p} {when}?",
                      "What was the average daily number of {w}{p} {when}?"]).format(
            w=noun, p=where_text, when=when)
        where = vd_where(d0, d1, (f" AND list_contains(places, {sql_str(p)})" if p else "")
                         + vd_type(t))
    sql = (f"SELECT round(avg(n), 1) AS per_day FROM (\n  SELECT date, count(*) AS n "
           f"FROM vessel_days{f' WHERE {where}' if where else ''} GROUP BY date\n)")
    musts = [*dm, AVERAGE_MUST, *tm] + ([place_text(p)[1]] if p else [])
    return Instance("vd_daily_average", q, sql, musts)


def vd_total_vessels(s: Sampler) -> Instance:
    r = s.rng
    t = s.ship_type(common=False) if r.random() < 0.5 else None
    noun, tm = what(t)
    if r.random() < 0.3:
        q = r.choice(["How many different {w} are in the data in total?",
                      "Over the whole period covered, how many distinct {w} were recorded?",
                      "How many distinct {w} appear in the whole data?"]).format(w=noun)
        sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_days"
               + (f" WHERE ship_type = {sql_str(t)}" if t else ""))
        return Instance("vd_total_vessels", q, sql, [r"total|whole|all|in the data", *tm])
    d0, d1, when, dm = long_period(s)
    q = r.choice(["How many different {w} were seen {when}?",
                  "What was the total number of distinct {w} {when}?"]).format(w=noun, when=when)
    sql = (f"SELECT count(DISTINCT mmsi) AS vessels FROM vessel_days WHERE "
           f"{vd_where(d0, d1)}{vd_type(t)}")
    return Instance("vd_total_vessels", q, sql, [*dm, *tm])


def vd_largest_ship(s: Sampler) -> Instance:
    r = s.rng
    d0, d1, when, dm = long_period(s)
    p = s.place() if r.random() < 0.4 else None
    t = s.ship_type() if r.random() < 0.4 else None
    noun = singular(SHIP_TYPES[t][0]) if t else "ship"
    tm = [SHIP_TYPES[t][1]] if t else []
    where_text = f" in {place_text(p)[0]}" if p else ""
    q = r.choice(["What was the largest {w} seen{p} {when}?",
                  "Which was the longest {w}{p} {when}?",
                  "What is the biggest {w} that sailed{p} {when}?"]).format(
        w=noun, p=where_text, when=when)
    where = vd_where(d0, d1, (f" AND list_contains(places, {sql_str(p)})" if p else "")
                     + vd_type(t) + f" AND length <= {MAX_REAL_LENGTH_M}")
    sql = (f"SELECT DISTINCT name, length FROM vessel_days\nWHERE {where}\n"
           f"  AND length = (SELECT max(length) FROM vessel_days WHERE {where})\n"
           f"ORDER BY name LIMIT 10")
    return Instance("vd_largest_ship", q, sql,
                    [*dm, LARGEST_MUST, *tm] + ([place_text(p)[1]] if p else []))


def vd_avg_length(s: Sampler) -> Instance:
    r = s.rng
    if r.random() < 0.5:
        d0 = d1 = s.day()
        when, dm = f"on {d_text(d0, r)}", [d_must(d0)]
    else:
        d0, d1, when, dm = long_period(s)
    p = s.place()
    t = s.ship_type() if r.random() < 0.4 else None
    noun, tm = what(t)
    pt, pm = place_text(p)
    q = r.choice(["How big were the {w} in {p} {when}, on average?",
                  "What was the average length of the {w} in {p} {when}?",
                  "How long were the {w} in {p} {when} on average?"]).format(
        w=noun, p=pt, when=when)
    sql = (f"SELECT round(avg(length), 1) AS avg_length_m FROM (\n"
           f"  SELECT mmsi, max(length) AS length FROM vessel_days\n"
           f"  WHERE {vd_where(d0, d1)} AND list_contains(places, {sql_str(p)}){vd_type(t)}\n"
           f"  GROUP BY mmsi\n)")
    return Instance("vd_avg_length", q, sql, [*dm, pm, r"size|length|long|big", *tm])


def vd_busiest_day(s: Sampler) -> Instance:
    r = s.rng
    d0, d1, when, dm = long_period(s)
    p = s.place() if r.random() < 0.6 else None
    t = s.ship_type() if r.random() < 0.4 else None
    noun, tm = what(t)
    where_text = f" in {place_text(p)[0]}" if p else ""
    q = r.choice(["Which day {when} had the most {w}{p}?",
                  "On which day {when} were the most {w} seen{p}?"]).format(
        w=noun, p=where_text, when=when)
    where = vd_where(d0, d1, (f" AND list_contains(places, {sql_str(p)})" if p else "")
                     + vd_type(t))
    sql = (f"SELECT date, count(*) AS vessels FROM vessel_days WHERE {where}\n"
           f"GROUP BY date ORDER BY vessels DESC, date LIMIT 1")
    return Instance("vd_busiest_day", q, sql,
                    [*dm, r"most|busiest", *tm] + ([place_text(p)[1]] if p else []))


def vd_farthest(s: Sampler) -> Instance:
    r = s.rng
    if r.random() < 0.5:
        d0 = d1 = s.day()
        when, dm = f"on {d_text(d0, r)}", [d_must(d0)]
    else:
        d0, d1, when, dm = long_period(s)
    t = s.ship_type() if r.random() < 0.5 else None
    noun = singular(SHIP_TYPES[t][0]) if t else "vessel"
    tm = [SHIP_TYPES[t][1]] if t else []
    q = r.choice(["Which {w} travelled the longest distance {when}?",
                  "Which {w} covered the most kilometres {when}?"]).format(w=noun, when=when)
    sql = (f"SELECT mmsi, any_value(name) AS name, round(sum(distance_km), 1) AS km\n"
           f"FROM vessel_days WHERE {vd_where(d0, d1)}{vd_type(t)}\n"
           f"GROUP BY mmsi ORDER BY km DESC, mmsi LIMIT 1")
    return Instance("vd_farthest", q, sql, [*dm, DISTANCE_MUST, *tm])


def harbored_at_part(p: str, d: date, hhmm: str) -> str:
    return (f"SELECT count(*) AS n FROM (\n"
            f"  SELECT mmsi, arg_max(sog, ts) AS sog FROM vessel_positions\n"
            f"  WHERE date = {d_sql(d)} AND {box_sql(p)}\n"
            f"    AND ts > {ts_sql(d, hhmm)} - INTERVAL 10 MINUTE AND ts <= {ts_sql(d, hhmm)}\n"
            f"  GROUP BY mmsi\n) WHERE sog < 0.5")


def _odd_time(r: random.Random) -> str:
    return f"{r.randint(0, 23):02d}:{r.choice([0, 0, 15, 30, 45, r.randint(1, 59)]):02d}"


def harbored_at_time(s: Sampler) -> Instance:
    r = s.rng
    d, p, hhmm = s.day(), r.choice(PORTS), _odd_time(r)
    pt, pm = place_text(p)
    q = r.choice(["How many vessels were docked in {p} at {t} UTC on {d}?",
                  "How many ships lay in port at {p} at {t} on {d}?",
                  "At {t} on {d}, how many boats were in harbour at {p}?"]).format(
        p=pt, d=d_text(d, r), t=hhmm)
    return Instance("harbored_at_time", q, harbored_at_part(p, d, hhmm),
                    [d_must(d), pm, hhmm_must(hhmm), HARBOR_MUST])


def harbored_compare_at_time(s: Sampler) -> Instance:
    r = s.rng
    d, hhmm = s.day(), _odd_time(r)
    p1 = r.choice(PORTS)
    p2 = r.choice(NEAR_PORTS[p1] + r.sample(PORTS, 3))
    if p2 == p1:
        p2 = r.choice([x for x in PORTS if x != p1])
    (a, am), (b, bm) = place_text(p1), place_text(p2)
    q = r.choice(["At {t} UTC on {d}, were more ships docked in {a} or in {b}?",
                  "Were more ships docked in {a} or in {b} at {t} UTC on {d}?",
                  "Compare the number of vessels in port at {a} and at {b} at {t} on {d}."]).format(
        a=a, b=b, t=hhmm, d=d_text(d, r))
    sql = (f"WITH a AS ({harbored_at_part(p1, d, hhmm)}),\n"
           f"     b AS ({harbored_at_part(p2, d, hhmm)})\n"
           f"SELECT CASE WHEN a.n > b.n THEN {sql_str(p1)} WHEN b.n > a.n THEN {sql_str(p2)} "
           f"END AS more_in,\n       a.n AS {sql_ident(p1)}, b.n AS {sql_ident(p2)} FROM a, b")
    return Instance("harbored_compare_at_time", q, sql,
                    [d_must(d), am, bm, hhmm_must(hhmm), HARBOR_MUST])


V5_TEMPLATES = [
    vd_count_place_period, vd_docked_period, vd_moving_through, vd_compare_period,
    vd_daily_average, vd_total_vessels, vd_largest_ship, vd_avg_length, vd_busiest_day,
    vd_farthest, harbored_at_time, harbored_compare_at_time,
]

TEMPLATES = [
    vessels_on_day, vessels_type_day, vessels_window, vessels_place_window,
    vessels_place_type_range, vessels_every_day, daily_counts, hourly_counts_place,
    reports_of_vessel, most_reports, top_types, busiest_hour_place, avg_speed_type_place,
    fastest_vessel_place, crowded_cell, position_at_time, first_last_seen, vessel_max_speed,
    vessel_destination, vessel_static, vessel_distance, underway_type_at_time,
    status_at_time_place, status_share, near_vessel, closest_pair_place, long_ships,
    top_destinations, duplicate_share, non_vessel_rows,
] + COMPOSITE_TEMPLATES + V5_TEMPLATES
