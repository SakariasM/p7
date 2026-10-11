Write one DuckDB SQL query that answers the question. Reply with the SQL only.

Table vessel_positions: AIS position reports from ships in Danish waters, 2025-10-01 through 2026-09-30,
about 18 million rows per day.
  date DATE            partition column, always filter on it: date = DATE '2026-09-23'
  ts TIMESTAMP         report time, UTC
  mmsi BIGINT          vessel id
  lat, lon DOUBLE      degrees
  sog FLOAT            speed over ground in knots, NULL when unknown
  cog FLOAT            course over ground in degrees; heading SMALLINT; rot FLOAT
  nav_status VARCHAR   'Under way using engine', 'Unknown value', 'Engaged in fishing', 'Restricted maneuverability', 'Moored', 'Under way sailing', 'Constrained by her draught', 'Reserved for future amendment [HSC]', 'At anchor', 'Power-driven vessel pushing ahead or towing alongside', 'Power-driven vessel towing astern', 'Reserved for future use', 'Not under command', 'Reserved for future amendment [WIG]', 'Aground'
  ship_type VARCHAR    'Cargo', 'Fishing', 'Passenger', 'Tanker', 'Sailing', 'Other', 'Tug', 'Pleasure', 'Pilot', 'HSC', 'SAR', 'Dredging', 'Undefined', 'Military', 'Towing', 'Law enforcement', 'Reserved', 'Port tender', 'Diving', 'Not party to conflict', 'Towing long/wide', 'Medical', 'Anti-pollution', 'Spare 1', 'WIG', 'Spare 2'
  name, callsign, destination VARCHAR   typed by the crew, spelling varies
  length, width SMALLINT (m); draught FLOAT (m); imo BIGINT; eta TIMESTAMP
  mobile_type VARCHAR  'Class A' or 'Class B'
Table ais: the same columns over every raw row, incl. base stations and bad positions.
Use it only for questions about raw data quality.

Rules:
- One report is often received several times: count vessels with count(DISTINCT mmsi) and
  a vessel's reports with count(DISTINCT ts).
- Moving or underway means sog >= 1. Use nav_status only when the question asks about it.
- ts is a full timestamp: ts >= TIMESTAMP '2026-09-23 12:00:00' AND
  ts < TIMESTAMP '2026-09-23 13:00:00', never TIME literals.
- Names are upper case: name = 'TYCHO BRAHE'.
- A vessel's position or speed at a moment is its latest report in the 10 minutes before
  (arg_max(lat, ts)).
- Areas are boxes: lon BETWEEN lon_min AND lon_max AND lat BETWEEN lat_min AND lat_max.
- haversine_m(lat1, lon1, lat2, lon2) gives the distance in metres.
- Missing text is 'Undefined' or 'Unknown value', not NULL.
- Dates without a year are in the data range above. Queries over more than a few days must
  also filter on mmsi, an area or ship_type.

Places (lon_min, lat_min, lon_max, lat_max):
  Øresund (the Sound): 12.40, 55.50, 12.95, 56.15
  Kattegat: 10.50, 56.00, 12.50, 57.60
  Copenhagen: 12.53, 55.64, 12.66, 55.73
  Aarhus: 10.20, 56.13, 10.27, 56.17
  Esbjerg: 8.40, 55.44, 8.47, 55.49
  Frederikshavn: 10.52, 57.42, 10.60, 57.45
  Skagen: 10.55, 57.70, 10.65, 57.74
  Gothenburg: 11.60, 57.63, 11.98, 57.73
  Helsingør: 12.60, 56.03, 12.63, 56.05
  Helsingborg: 12.68, 56.03, 12.71, 56.06
