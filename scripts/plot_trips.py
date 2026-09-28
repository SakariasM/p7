"""Visual check of trip segmentation: `uv run python scripts/plot_trips.py [n_trips] [seed]`.

Writes data/derived/plots/trips.png (all positions as a density background, a random
sample of simplified trips, and stops) and zoomed per-trip panels in trips_detail.png.
"""

import json
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ais.paths import DataPaths

paths = DataPaths.from_env()
n_trips = int(sys.argv[1]) if len(sys.argv) > 1 else 40
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
out = paths.derived / "plots"
out.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")

# Density background: count positions per ~0.02 degree cell.
cells = con.execute(f"""
    SELECT round(lon / 0.02)::INT AS x, round(lat / 0.02)::INT AS y, count(*) AS n
    FROM read_parquet('{paths.glob("positions")}') GROUP BY ALL
""").fetchnumpy()
x0, y0 = cells["x"].min(), cells["y"].min()
grid = np.zeros((cells["y"].max() - y0 + 1, cells["x"].max() - x0 + 1))
grid[cells["y"] - y0, cells["x"] - x0] = np.log1p(cells["n"])
extent = (x0 * 0.02, (cells["x"].max() + 1) * 0.02, y0 * 0.02, (cells["y"].max() + 1) * 0.02)

trips = con.execute(f"""
    SELECT t.trip_id, t.point_count, t.distance_m, t.duration_s, g.geojson
    FROM '{paths.derived}/trips/*.parquet' t
    JOIN '{paths.derived}/trip_geometry/*.parquet' g USING (trip_id)
    USING SAMPLE {n_trips} ROWS (reservoir, {seed})
""").fetchall()
stops = con.execute(
    f"SELECT lon, lat, duration_s FROM '{paths.derived}/stops/*.parquet'"
).fetchnumpy()


def coords(geojson: str) -> np.ndarray:
    g = json.loads(geojson)
    return np.array(g["coordinates"] if g["type"] == "LineString" else [g["coordinates"]])


fig, ax = plt.subplots(figsize=(14, 12))
ax.imshow(
    grid, origin="lower", extent=extent, cmap="Greys", alpha=0.6, aspect=1 / np.cos(np.radians(56))
)
ax.scatter(stops["lon"], stops["lat"], s=2, c="tab:red", alpha=0.3, label="stops")
for i, (_tid, _n, _dist, _dur, gj) in enumerate(trips):
    c = coords(gj)
    ax.plot(c[:, 0], c[:, 1], lw=1.2, color=plt.cm.tab20(i % 20))
    ax.plot(*c[0], "o", ms=3, color="green")
    ax.plot(*c[-1], "s", ms=3, color="black")
ax.set_xlim(3, 17)
ax.set_ylim(53.5, 59.5)
ax.set_title(f"{len(trips)} random trips (green = start, black = end), red = stops")
ax.legend(loc="upper right")
fig.savefig(out / "trips.png", dpi=110, bbox_inches="tight")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for ax, (tid, n, dist, dur, gj) in zip(axes.flat, trips, strict=False):
    c = coords(gj)
    ax.plot(c[:, 0], c[:, 1], "-", lw=1)
    ax.plot(*c[0], "o", color="green")
    ax.plot(*c[-1], "s", color="black")
    ax.set_aspect(1 / np.cos(np.radians(c[0, 1])))
    ax.set_title(f"{tid}\n{n} pts, {dist / 1000:.1f} km, {dur / 60:.0f} min", fontsize=8)
    ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(out / "trips_detail.png", dpi=90)
print(f"wrote {out / 'trips.png'} and {out / 'trips_detail.png'}")
