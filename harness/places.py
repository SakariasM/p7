"""
Named places for the SQL student: lookup, finding the places a request mentions, the
student's place list, checking SQL boxes, and places learned at runtime.

  places.json                  in the repo, built by build_places.py (manual areas and
                               AIS-located ports)
  data/places_learned.jsonl    in the project folder, appended by planner.py when the
                               planner proposes a place that passes validate(); status
                               "unverified" until someone moves it to places_manual.json
"""

import fcntl
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ais_tool import _DATES, PROJECT_DIR, connect

PLACES_FILE = Path(__file__).with_name("places.json")
LEARNED_FILE = PROJECT_DIR / "data" / "places_learned.jsonl"
PLACES_HEADER = "Places (lon_min, lat_min, lon_max, lat_max):"

COVERAGE = (-5.0, 50.0, 32.0, 64.0)  # vessel_positions keeps lon -5..32, lat 50..64
MAX_SIDE_DEG = {"port": 0.3, "area": 3.0}
MIN_SIDE_DEG = 0.002
MIN_VESSELS = 3  # distinct vessels in the box on the validation days
VALIDATION_DAYS = 3


@dataclass
class Place:
    name: str
    box: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    kind: str = "port"
    aliases: list[str] = field(default_factory=list)
    source: str = "manual"
    status: str = "verified"

    def as_dict(self) -> dict:
        return {"name": self.name, "box": list(self.box), "kind": self.kind,
                "status": self.status, "source": self.source}


def normalise(text: str) -> str:
    """Lower case, Danish/Swedish/German letters folded (Å->aa, Ø->o, Ö->o, ...), accents
    dropped, anything else that is not a letter or digit becomes one space."""
    t = text.lower().replace("å", "aa").replace("æ", "ae").replace("ø", "o")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def load() -> list[Place]:
    places = [
        Place(p["name"], tuple(p["box"]), p["kind"], p.get("aliases", []), p["source"])
        for p in json.loads(PLACES_FILE.read_text())["places"]
    ]
    if LEARNED_FILE.exists():
        for line in LEARNED_FILE.read_text().splitlines():
            p = json.loads(line)
            places.append(Place(p["name"], tuple(p["box"]), p["kind"], [], "learned",
                                p.get("status", "unverified")))
    return places


def _names(place: Place) -> list[str]:
    return [n for n in {normalise(x) for x in [place.name, *place.aliases]} if n]


def find(name: str, places: list[Place] | None = None) -> Place | None:
    key = normalise(name)
    for place in places if places is not None else load():
        if key in _names(place):
            return place
    return None


def mentioned(text: str, places: list[Place] | None = None) -> list[Place]:
    """Places named in the text (whole words), longest names first, each place once."""
    norm = f" {normalise(text)} "
    found: list[Place] = []
    pairs = sorted(((n, p) for p in (places if places is not None else load())
                    for n in _names(p)), key=lambda np: -len(np[0]))
    for name, place in pairs:
        if f" {name} " in norm and place not in found:
            found.append(place)
            norm = norm.replace(f" {name} ", " ")
    return found


def format_place(place: Place) -> str:
    return f"  {place.name}: " + ", ".join(f"{x:.2f}" if abs(x * 100 - round(x * 100)) < 1e-9
                                           else f"{x:.3f}" for x in place.box)


def with_places(context: str, places: list[Place]) -> str:
    """The student's prompt with its place list replaced by `places`."""
    head, sep, _ = context.partition(PLACES_HEADER)
    if not sep:
        raise ValueError("context has no place list")
    lines = [format_place(p) for p in places] or ["  (no named place in this request)"]
    return head + PLACES_HEADER + "\n" + "\n".join(lines) + "\n"


_NUM = r"(-?\d+(?:\.\d+)?)"
_LON_LAT = re.compile(rf"(?:\w+\.)?lon\s+BETWEEN\s+{_NUM}\s+AND\s+{_NUM}\s+AND\s+"
                      rf"(?:\w+\.)?lat\s+BETWEEN\s+{_NUM}\s+AND\s+{_NUM}", re.IGNORECASE)
_LAT_LON = re.compile(rf"(?:\w+\.)?lat\s+BETWEEN\s+{_NUM}\s+AND\s+{_NUM}\s+AND\s+"
                      rf"(?:\w+\.)?lon\s+BETWEEN\s+{_NUM}\s+AND\s+{_NUM}", re.IGNORECASE)


def boxes_in_sql(sql: str) -> list[tuple[float, ...]]:
    boxes = [(float(a), float(c), float(b), float(d)) for a, b, c, d in _LON_LAT.findall(sql)]
    boxes += [(float(c), float(a), float(d), float(b)) for a, b, c, d in _LAT_LON.findall(sql)]
    return boxes


def unexplained_boxes(sql: str, places: list[Place], request: str) -> list[tuple[float, ...]]:
    """Boxes in the SQL that belong neither to a given place nor to numbers in the request."""
    known = [p.box for p in places]
    numbers = {round(float(x), 3) for x in re.findall(r"-?\d+\.\d+", request)}
    out = []
    for box in boxes_in_sql(sql):
        if any(all(abs(a - b) < 1e-3 for a, b in zip(box, k, strict=True)) for k in known):
            continue
        if all(round(x, 3) in numbers for x in box):
            continue
        out.append(box)
    return out


def validate(name: str, kind: str, box: tuple[float, float, float, float]) -> tuple[bool, str]:
    """Is a proposed place plausible? Inside coverage, sensible size, vessels present."""
    if kind not in MAX_SIDE_DEG:
        return False, f"kind must be one of {sorted(MAX_SIDE_DEG)}"
    lon0, lat0, lon1, lat1 = box
    if not (lon0 < lon1 and lat0 < lat1):
        return False, "box must be lon_min < lon_max and lat_min < lat_max"
    c = COVERAGE
    if lon0 < c[0] or lat0 < c[1] or lon1 > c[2] or lat1 > c[3]:
        return False, "box is outside the data's coverage (lon -5..32, lat 50..64)"
    if max(lon1 - lon0, lat1 - lat0) > MAX_SIDE_DEG[kind]:
        return False, f"box is too large for a {kind} (max {MAX_SIDE_DEG[kind]} degrees a side)"
    if min(lon1 - lon0, lat1 - lat0) < MIN_SIDE_DEG:
        return False, "box is too small"
    if find(name):
        return False, f"{name} is already known"
    days = ", ".join(f"DATE '{d}'" for d in _DATES[-VALIDATION_DAYS:])
    vessels, moored = connect().execute(f"""
        SELECT count(DISTINCT mmsi), count(DISTINCT mmsi) FILTER (WHERE nav_status = 'Moored')
        FROM vessel_positions
        WHERE date IN ({days}) AND lon BETWEEN {lon0} AND {lon1} AND lat BETWEEN {lat0} AND {lat1}
    """).fetchone()
    if vessels < MIN_VESSELS:
        return False, f"only {vessels} vessels in that box on the last {VALIDATION_DAYS} days"
    if kind == "port" and moored == 0:
        return False, "no moored vessels in that box, so it does not look like a port"
    return True, f"{vessels} vessels ({moored} moored) on the last {VALIDATION_DAYS} days"


def learn(name: str, kind: str, box: tuple[float, float, float, float], evidence: str,
          context: str) -> Place:
    """Append a validated place to LEARNED_FILE (append-only, locked) and return it."""
    record = {"name": name, "kind": kind, "box": list(box), "status": "unverified",
              "evidence": evidence, "context": context,
              "learned_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    LEARNED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LEARNED_FILE.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    return Place(name, box, kind, [], "learned", "unverified")
