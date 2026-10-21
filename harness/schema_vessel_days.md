Table vessel_days: one row per vessel per day, made from vessel_positions and much faster.
Use it for periods longer than a few days, whole-data totals and averages per vessel; use
vessel_positions for times within a day, positions and speeds at a moment.
  date DATE; mmsi BIGINT; name, ship_type, destination VARCHAR (most common that day)
  length, width (m, plausible values only); draught
  reports BIGINT; first_ts, last_ts TIMESTAMP; distance_km DOUBLE; max_sog, avg_sog (knots)
  places VARCHAR[]     named places (list below) where it had a report that day
  still_in VARCHAR[]   places where it lay still (sog < 0.5): docked or harbored there
  moving_in VARCHAR[]  places where it moved (sog >= 1), e.g. passing under a bridge
  e.g. list_contains(still_in, 'Skagen'), list_has_any(places, ['Rønne', 'Nexø'])
