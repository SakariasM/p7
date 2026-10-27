"""
Deterministic tools for the planner (the "function calls" step): fixed, tested SQL for
questions the SQL student cannot answer from one request, run with the usual guardrails.

port_calls: which vessels called at a country's ports (or the ports inside a named area, e.g.
Bornholm) in a period. A port call is at least one report with speed under PORT_CALL_SOG_KN
inside the box of one of those ports in places.json. The period is queried a week at a time (to stay under the query timeout) and the
vessels are combined as a set, so a vessel calling in two weeks counts once.
"""

import re
from datetime import date, timedelta

import places
from ais_tool import _DATES, query_ais_data

PORT_CALL_SOG_KN = 0.5
MAX_DAYS = 92
CHUNK_DAYS = 7
COUNTRIES = {"DK": ["denmark", "danmark", "dk", "danish"],
             "SE": ["sweden", "sverige", "se", "swedish"],
             "NO": ["norway", "norge", "no", "norwegian"],
             "DE": ["germany", "deutschland", "de", "german"],
             "PL": ["poland", "polska", "pl", "polish"]}
CRUISE_MIN_LENGTH_M = 250
CRUISE_ASSUMPTION = (
    "Assumption: passenger ships of 250 m or longer are counted as cruise ships. AIS has no "
    "cruise ship category, so this cannot be determined with certainty: large ferries may "
    "be included and smaller cruise ships missed. Tell the user this.")

PORT_CALLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "port_calls",
        "description": (
            "Vessels that called at (stopped in) any port of a country, or any port inside a "
            "named area such as an island or sea area, in a period: the number of distinct "
            "vessels, the busiest ports and the MMSI list. Use it for questions like 'how many "
            "ships visited Denmark in August' (country) or 'how many ships docked at Bornholm' "
            "(area). Give country or area, not both. Up to 3 months. Not for a single port "
            "or for listing vessels by name: use ask_data for those."),
        "parameters": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "Denmark, Sweden, Norway, Germany "
                                                             "or Poland (or DK, SE, NO, DE, PL)"},
                "area": {"type": "string", "description": "instead of country: a place name "
                                                          "from find_place, e.g. Bornholm; "
                                                          "the ports inside it count"},
                "date_from": {"type": "string", "description": "first day, YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "last day, YYYY-MM-DD"},
                "ship_type": {"type": "string", "description": "optional AIS ship type, e.g. "
                                                               "Passenger, Cargo, Tanker"},
                "min_length_m": {"type": "number", "description": "optional minimum length"},
            },
            "required": ["date_from", "date_to"],
        },
    },
}


def country_code(name: str) -> str | None:
    key = places.normalise(name)
    return next((code for code, names in COUNTRIES.items() if key in names), None)


def _box(p: places.Place) -> str:
    lon0, lat0, lon1, lat1 = (places.coord(x) for x in p.box)
    return f"(lon BETWEEN {lon0} AND {lon1} AND lat BETWEEN {lat0} AND {lat1})"


def _inside(port: places.Place, area: places.Place) -> bool:
    lon = (port.box[0] + port.box[2]) / 2
    lat = (port.box[1] + port.box[3]) / 2
    return area.box[0] <= lon <= area.box[2] and area.box[1] <= lat <= area.box[3]


def port_calls(country: str | None, date_from: str, date_to: str, ship_type: str | None = None,
               min_length_m: float | None = None, valid_ship_types: set[str] | None = None,
               area: str | None = None) -> dict:
    known = places.load(include_learned=False)
    if area:
        region = places.find(area, known)
        if region is None:
            return {"error": f"unknown area {area!r}; check it with find_place"}
        ports = [p for p in known if p.kind == "port" and _inside(p, region)]
        code, scope = None, f"the ports inside {region.name}"
        if not ports:
            return {"error": f"no known ports inside {region.name}"}
    else:
        code = country_code(country or "")
        if code is None:
            return {"error": f"unknown country {country!r}; use one of {', '.join(COUNTRIES)}"}
        ports = [p for p in known if p.kind == "port" and p.country == code]
        scope = f"the {code} ports"
        if not ports:
            return {"error": f"no ports known for {code}"}
    try:
        d0, d1 = date.fromisoformat(date_from), date.fromisoformat(date_to)
    except ValueError:
        return {"error": "dates must be YYYY-MM-DD"}
    first, last = date.fromisoformat(_DATES[0]), date.fromisoformat(_DATES[-1])
    if d0 > d1 or d0 < first or d1 > last:
        return {"error": f"period must lie within {first}..{last} with date_from <= date_to"}
    if (d1 - d0).days + 1 > MAX_DAYS:
        return {"error": f"at most {MAX_DAYS} days; split the period"}
    if ship_type is not None and valid_ship_types and ship_type not in valid_ship_types:
        return {"error": f"'{ship_type}' is not a ship_type value; valid: "
                         f"{', '.join(sorted(valid_ship_types))}"}
    if ship_type is not None and not re.fullmatch(r"[A-Za-z /-]+", ship_type):
        return {"error": "invalid ship_type"}

    case = "\n".join(f"WHEN {_box(p)} THEN '{p.name.replace(chr(39), chr(39) * 2)}'"
                     for p in ports)
    filters = [f"sog < {PORT_CALL_SOG_KN}"]
    if ship_type is not None:
        filters.append(f"ship_type = '{ship_type}'")
    if min_length_m is not None:
        filters.append(f"length >= {float(min_length_m):g}")
    any_port = " OR ".join(_box(p) for p in ports)

    vessels_by_port: dict[str, set[int]] = {}
    seconds = 0.0
    day = d0
    while day <= d1:
        end = min(day + timedelta(days=CHUNK_DAYS - 1), d1)
        sql = (f"SELECT DISTINCT mmsi, CASE {case} END AS port FROM vessel_positions "
               f"WHERE date BETWEEN DATE '{day}' AND DATE '{end}' AND {' AND '.join(filters)} "
               f"AND ({any_port}) LIMIT 20000")
        result = query_ais_data(sql)
        if "error" in result:
            return {"error": f"{day}..{end}: {result['error']}"}
        seconds += result["elapsed_seconds"]
        for mmsi, port in result["rows"]:
            vessels_by_port.setdefault(port, set()).add(mmsi)
        day = end + timedelta(days=1)

    vessels = sorted(set().union(*vessels_by_port.values())) if vessels_by_port else []
    by_port = sorted(((p, len(v)) for p, v in vessels_by_port.items()), key=lambda x: -x[1])
    cruise = (ship_type == "Passenger" and min_length_m is not None
              and float(min_length_m) >= CRUISE_MIN_LENGTH_M)
    return {
        **({"assumption": CRUISE_ASSUMPTION} if cruise else {}),
        "country": code, **({"area": region.name, "ports": [p.name for p in ports]} if area else {}),
        "date_from": str(d0), "date_to": str(d1),
        "ship_type": ship_type, "min_length_m": min_length_m,
        "vessels": len(vessels),
        "ports_with_calls": dict(by_port[:15]),
        "mmsi": vessels[:100],
        "definition": f"a port call is a report with speed < {PORT_CALL_SOG_KN} kn inside one of "
                      f"the boxes of {scope} in places.json ({len(ports)} ports)",
        "seconds": round(seconds, 1),
    }
