"""
Write places.json: named places with lon/lat boxes for the SQL student. Only the places a
request mentions go into the student's prompt (see places.py), so the list can be long.

  places_manual.json  hand-made: the ten Phase 3 boxes (unchanged, the eval depends on them)
                      and sea areas; a manual entry wins over a generated one with its name
  UN/LOCODE           names and codes of ports in DK, SE, NO, DE and PL
                      (data/external/un-locode/code-list.csv); most Danish ports have no
                      coordinates there, so ports are located from AIS instead:
  AIS                 a port is where moored vessels that declare it as their destination
                      ("DKAAL", "AALBORG") lie on the sample days; a port can have several
                      sites (e.g. Aalborg city and east harbour). Its box covers the moored
                      positions of all vessels around its sites, each moored cell going to
                      the nearest site. LOCODE coordinates, when present, are only a check.
  places_reference_ports.json  port points from a list of unverified origin: a fallback
                      start point for ports the destination method misses (still needs
                      moored vessels there) and a cross-check (> 3 km off is reported).

Run as a CPU job, it reads the sample days (from harness/):
    python3 build_places.py /ceph/project/P7-Agent-Harness/data/external/un-locode/code-list.csv
"""

import argparse
import csv
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

from ais_tool import _DATES, connect
from sql_templates import EXCLUDED_DATES

HERE = Path(__file__).parent
MANUAL_FILE = HERE / "places_manual.json"
REFERENCE_FILE = HERE / "places_reference_ports.json"
OUT = HERE / "places.json"

COUNTRIES = ("DK", "SE", "NO", "DE", "PL")
SAMPLE_DAYS = 24
MIN_VESSELS = 3  # distinct moored vessels declaring the port, near its centre
CLUSTER_KM = 3.0  # declared positions this close to a site's cell form that site
SITE_RADIUS_KM = 15.0  # further sites of one port lie within this distance of the busiest
SITE_SHARE = 0.3  # a further site needs this share of the busiest site's vessels
BOX_KM = 2.5  # moored cells this close to the centre can belong to its box
PAD_DEG = 0.003
MIN_SIDE_DEG, MAX_SIDE_DEG, MAX_SIDE_MULTI_DEG = 0.01, 0.15, 0.3


def km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def name_keys(text: str) -> set[str]:
    """Spellings of a name as crews type it: upper case, letters and digits only,
    with Danish letters both as Å->AA/Ø->OE and as Å->A/Ø->O."""
    t = text.upper()
    variants = {t, t.replace("Å", "AA").replace("Æ", "AE").replace("Ø", "OE"),
                t.replace("Å", "A").replace("Æ", "AE").replace("Ø", "O")}
    out = set()
    for v in variants:
        ascii_ = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode()
        out.add(re.sub(r"[^A-Z0-9]", "", ascii_))
    return {k for k in out if len(k) >= 4}


def parse_coord(text: str) -> tuple[float, float] | None:
    parts = text.split()
    if len(parts) != 2:
        return None
    a, b = parts
    lat = int(a[:2]) + int(a[2:4]) / 60
    lon = int(b[:3]) + int(b[3:5]) / 60
    return (lat if a.endswith("N") else -lat, lon if b.endswith("E") else -lon)


def locode_ports(path: Path) -> dict[str, dict]:
    ports = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Country"] in COUNTRIES and r["Function"].startswith("1"):
                code = r["Country"] + r["Location"]
                ports[code] = {
                    "code": code,
                    "name": r["Name"],
                    "names": {r["Name"], r["NameWoDiacritics"]},
                    "coords": parse_coord(r["Coordinates"]),
                }
    return ports


def destination_index(ports: dict[str, dict]) -> dict[str, str]:
    """Normalised destination -> LOCODE; names shared by several ports are left out."""
    by_key: dict[str, set[str]] = defaultdict(set)
    for code, p in ports.items():
        by_key[code].add(code)
        for name in p["names"]:
            for k in name_keys(name):
                by_key[k].add(code)
    return {k: next(iter(codes)) for k, codes in by_key.items() if len(codes) == 1}


def sample_days() -> list[str]:
    days = [d for d in _DATES if date.fromisoformat(d) not in EXCLUDED_DATES]
    step = max(1, len(days) // SAMPLE_DAYS)
    return days[step // 2 :: step][:SAMPLE_DAYS]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("locode_csv", type=Path)
    args = p.parse_args()

    manual = json.loads(MANUAL_FILE.read_text())["places"]
    manual_keys = {k for m in manual for n in [m["name"], *m["aliases"]] for k in name_keys(n)}
    ports = locode_ports(args.locode_csv)
    index = destination_index(ports)
    days = sample_days()
    in_days = ", ".join(f"DATE '{d}'" for d in days)
    print(f"{len(ports)} LOCODE ports in {', '.join(COUNTRIES)}; sample days {days[0]}..{days[-1]} "
          f"({len(days)})", flush=True)

    con = connect()
    moored = f"date IN ({in_days}) AND nav_status = 'Moored' AND sog < 0.5"
    declared = con.execute(f"""
        SELECT mmsi, round(lat, 3), round(lon, 3), upper(destination) FROM vessel_positions
        WHERE {moored} AND destination IS NOT NULL GROUP BY ALL
    """).fetchall()
    cells = con.execute(f"""
        SELECT round(lat, 3), round(lon, 3), count(DISTINCT mmsi) FROM vessel_positions
        WHERE {moored} GROUP BY ALL
    """).fetchall()
    print(f"{len(declared):,} moored (vessel, cell, destination) rows, {len(cells):,} moored cells",
          flush=True)

    # Where vessels declaring each port lie moored.
    points: dict[str, list] = defaultdict(list)
    for mmsi, lat, lon, dest in declared:
        for k in name_keys(dest) | {re.sub(r"[^A-Z0-9]", "", dest)}:
            if k in index:
                points[index[k]].append((mmsi, lat, lon))
                break

    # A port can have several sites (Aalborg city and east harbour, Odense canal and Lindo):
    # the busiest 0.01-degree cell, plus cells within SITE_RADIUS_KM with at least
    # SITE_SHARE of its vessels, each more than CLUSTER_KM from the sites found so far.
    sites: dict[str, list[tuple[float, float]]] = {}
    evidence: dict[str, int] = {}
    for code, pts in points.items():
        by_cell: dict[tuple, set] = defaultdict(set)
        for mmsi, lat, lon in pts:
            by_cell[(round(lat, 2), round(lon, 2))].add(mmsi)
        ranked = sorted(by_cell.items(), key=lambda kv: -len(kv[1]))
        (lat0, lon0), top = ranked[0]
        found: list[tuple[float, float]] = []
        vessels: set = set()
        for (clat, clon), ms in ranked:
            if len(ms) < max(MIN_VESSELS, SITE_SHARE * len(top)):
                break
            if km(clat, clon, lat0, lon0) > SITE_RADIUS_KM:
                continue
            if any(km(clat, clon, la, lo) <= CLUSTER_KM for la, lo in found):
                continue
            near = [(m, la, lo) for m, la, lo in pts if km(la, lo, clat, clon) <= CLUSTER_KM]
            found.append((statistics.median(la for _, la, _ in near),
                          statistics.median(lo for _, _, lo in near)))
            vessels |= {m for m, _, _ in near}
        if found:
            sites[code], evidence[code] = found, len(vessels)

    # Fallback: reference points (places_reference_ports.json) for ports the destination
    # method missed, if vessels lie moored around the point.
    reference = json.loads(REFERENCE_FILE.read_text())["ports"]
    names_by_key = {k: code for code, pt in ports.items()
                    for n in pt["names"] for k in name_keys(n)}
    located_keys = manual_keys | {k for code in sites for n in ports[code]["names"]
                                  for k in name_keys(n)}
    from_reference = []
    for ref in reference:
        keys = name_keys(ref["name"])
        if keys & located_keys:
            continue
        moored_near = sum(n for la, lo, n in cells
                          if km(la, lo, ref["latitude"], ref["longitude"]) <= BOX_KM)
        if moored_near < MIN_VESSELS:
            continue
        code = next((names_by_key[k] for k in keys if k in names_by_key), None)
        if code is None:
            code = "REF:" + ref["name"]
            ports[code] = {"code": None, "name": ref["name"], "names": {ref["name"]},
                           "coords": None}
        sites[code], evidence[code] = [(ref["latitude"], ref["longitude"])], moored_near
        from_reference.append(ref["name"])

    # Each moored cell goes to the nearest site within BOX_KM.
    site_list = [(code, la, lo) for code, ss in sites.items() for la, lo in ss]
    owned: dict[str, list] = defaultdict(list)
    for lat, lon, _ in cells:
        best, best_d = None, BOX_KM
        for code, slat, slon in site_list:
            if abs(lat - slat) < 0.05 and abs(lon - slon) < 0.08:
                d = km(lat, lon, slat, slon)
                if d <= best_d:
                    best, best_d = code, d
        if best:
            owned[best].append((lat, lon))

    generated, skipped_manual = [], []
    for code, port_sites in sorted(sites.items()):
        port = ports[code]
        if any(k in manual_keys for nm in port["names"] for k in name_keys(nm)):
            skipped_manual.append(port["name"])
            continue
        clat, clon = port_sites[0]
        lats = sorted([la for la, _ in owned[code]] + [la for la, _ in port_sites])
        lons = sorted([lo for _, lo in owned[code]] + [lo for _, lo in port_sites])
        cut = len(lats) // 50
        max_side = MAX_SIDE_DEG if len(port_sites) == 1 else MAX_SIDE_MULTI_DEG

        def side(lo: float, hi: float, c: float, max_side: float = max_side):
            lo, hi = lo - PAD_DEG, hi + PAD_DEG
            half = min(max((hi - lo) / 2, MIN_SIDE_DEG / 2), max_side / 2)
            mid = (lo + hi) / 2 if hi - lo <= max_side else c
            return round(mid - half, 3), round(mid + half, 3)

        lat0, lat1 = side(lats[cut], lats[-1 - cut], clat)
        lon0, lon1 = side(lons[cut], lons[-1 - cut], clon)
        offset = round(km(clat, clon, *port["coords"]), 1) if port["coords"] else None
        locode = port["code"]
        aliases = sorted({nm for nm in port["names"] if nm != port["name"]}
                         | ({locode} if locode else set()))
        generated.append({
            "name": port["name"], "aliases": aliases, "kind": "port",
            "box": [lon0, lat0, lon1, lat1],
            "source": "reference+ais" if code.startswith("REF:") or port["name"] in
                      from_reference else "locode+ais",
            "locode": locode,
            "evidence": {"moored_vessels": evidence[code], "sites": len(port_sites),
                         "locode_offset_km": offset},
        })

    places = [{**m, "source": "manual"} for m in manual] + generated
    OUT.write_text(json.dumps({"_note": "Generated by build_places.py; edit places_manual.json "
                                        "instead.", "places": places},
                              ensure_ascii=False, indent=1) + "\n")
    far = [g for g in generated if (g["evidence"]["locode_offset_km"] or 0) > 10]
    print(f"\n{len(manual)} manual + {len(generated)} AIS-located ports -> {OUT}")
    print(f"skipped (manual entry exists): {', '.join(sorted(skipped_manual))}")
    print(f"located from reference points: {', '.join(from_reference) or 'none'}")
    print(f"LOCODE ports without enough AIS evidence: {len(ports) - len(sites)}")
    far_text = ", ".join(f"{g['name']} ({g['evidence']['locode_offset_km']} km)" for g in far)
    print(f"AIS centre > 10 km from LOCODE coordinates: {far_text or 'none'}")
    multi = [g for g in generated if g["evidence"]["sites"] > 1]
    print("multi-site ports: " + ", ".join(f"{g['name']} ({g['evidence']['sites']})"
                                           for g in multi))
    print("\nreference points vs places (> 3 km from the box, or missing):")
    by_key = {k: p for p in places for n in [p["name"], *p["aliases"]] for k in name_keys(n)}
    for ref in reference:
        place = next((by_key[k] for k in name_keys(ref["name"]) if k in by_key), None)
        if place is None:
            print(f"  {ref['name']:16} missing")
            continue
        lon0, lat0, lon1, lat1 = place["box"]
        d = km(ref["latitude"], ref["longitude"], min(max(ref["latitude"], lat0), lat1),
               min(max(ref["longitude"], lon0), lon1))
        if d > 3:
            print(f"  {ref['name']:16} {d:5.1f} km from {place['name']} {place['box']}")
    for g in sorted(generated, key=lambda g: -g["evidence"]["moored_vessels"])[:25]:
        print(f"  {g['name']:22} {g['locode']}  box {g['box']}  "
              f"vessels {g['evidence']['moored_vessels']}  sites {g['evidence']['sites']}")


if __name__ == "__main__":
    main()
