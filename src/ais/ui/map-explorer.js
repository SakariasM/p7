import { getJson } from "./api-client.js";
import {
  $, escapeHtml, formatTimestamp, OTHER_SHIP_COLOR, setStatus, SHIP_COLORS, SHIP_TYPES,
  shipColor, toIsoSeconds,
} from "./shared.js";

const DAY_MS = 86_400_000;
let highlightedVessels = new Set();
let currentVessel = null;
let searchTimer;
let player = null;

export const map = L.map("map", { preferCanvas: true, zoomControl: false }).setView([56.0, 11.0], 7);
L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

const shipsLayer = L.layerGroup().addTo(map);
const pathLayer = L.layerGroup().addTo(map);

$("legend").innerHTML = [...Object.entries(SHIP_COLORS), ["Other", OTHER_SHIP_COLOR]]
  .map(([label, color]) => `<span><i class="dot" style="background:${color}"></i>${label}</span>`)
  .join("");
$("type").innerHTML += SHIP_TYPES.map(type => `<option>${type}</option>`).join("");

export function setHighlightedVessels(vessels) {
  highlightedVessels = new Set(vessels.map(vessel => vessel.mmsi));
}

export function showActiveFilters() {
  $("clear-filter").hidden = !($("name").value.trim() || $("type").value);
}

function selectedTimestamp() {
  const value = $("at").value;
  return value.length === 16 ? `${value}:00` : value;
}

export async function loadSnapshot() {
  const bounds = map.getBounds();
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const bbox = [
    clamp(bounds.getWest(), -180, 180), clamp(bounds.getSouth(), -90, 90),
    clamp(bounds.getEast(), -180, 180), clamp(bounds.getNorth(), -90, 90),
  ].map(value => value.toFixed(4)).join(",");
  $("snap").disabled = true;
  setStatus("snap-status", "Loading ships…");
  const started = performance.now();
  try {
    const params = { at: selectedTimestamp(), lookback_min: $("lookback").value, bbox };
    const name = $("name").value.trim();
    const type = $("type").value;
    if (name) params.name = name;
    if (type) params.ship_type = type;
    const snapshot = await getJson("/snapshot", params);
    const filtered = Boolean(name || type);
    shipsLayer.clearLayers();
    for (const vessel of snapshot.items) {
      const highlighted = highlightedVessels.has(vessel.mmsi);
      L.circleMarker([vessel.lat, vessel.lon], {
        radius: highlighted ? 8 : filtered ? 6 : 4,
        weight: highlighted ? 3 : 1,
        color: highlighted ? "#14201f" : "#ffffff",
        fillColor: shipColor(vessel.ship_type),
        fillOpacity: 0.92,
      }).bindPopup(() => `
        <b>${escapeHtml(vessel.name || "Unknown name")}</b><br>
        ${escapeHtml(vessel.ship_type || "Unknown type")} · MMSI ${vessel.mmsi}<br>
        ${vessel.sog != null ? `${vessel.sog.toFixed(1)} kn` : "speed –"} · seen ${formatTimestamp(vessel.ts)}<br>
        <button onclick="selectVessel(${vessel.mmsi})">Details &amp; trips</button>`)
        .addTo(shipsLayer);
    }
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    const filterLabel = [name && `named “${name}”`, type].filter(Boolean).join(", ");
    const capped = snapshot.truncated ? " · capped at 5,000; zoom in" : "";
    setStatus(
      "snap-status",
      `${snapshot.items.length.toLocaleString()} ships${filterLabel ? ` · ${filterLabel}` : ""}${capped} · ${seconds}s`,
    );
  } catch (error) {
    setStatus("snap-status", error.message, true);
  } finally {
    $("snap").disabled = false;
  }
}

$("snap").addEventListener("click", loadSnapshot);
$("presets").addEventListener("click", event => {
  const button = event.target.closest("button");
  if (!button) return;
  $("name").value = button.dataset.name || "";
  $("type").value = button.dataset.type || "";
  showActiveFilters();
  loadSnapshot();
});
$("name").addEventListener("input", showActiveFilters);
$("name").addEventListener("keydown", event => { if (event.key === "Enter") loadSnapshot(); });
$("type").addEventListener("change", () => { showActiveFilters(); loadSnapshot(); });

$("q").addEventListener("input", () => {
  clearTimeout(searchTimer);
  const query = $("q").value.trim();
  if (!query) {
    $("results").innerHTML = "";
    setStatus("search-status", "");
    return;
  }
  searchTimer = setTimeout(async () => {
    try {
      const vessels = await getJson("/vessels", { q: query, limit: 15 });
      setStatus("search-status", vessels.length ? "" : "No matches");
      $("results").innerHTML = vessels.map(vessel => `
        <li data-mmsi="${vessel.mmsi}"><b>${escapeHtml(vessel.name || "Unknown name")}</b>
        <div class="meta">${escapeHtml(vessel.ship_type || "–")} · ${vessel.mmsi}${vessel.callsign ? ` · ${escapeHtml(vessel.callsign)}` : ""}</div></li>`).join("");
    } catch (error) {
      setStatus("search-status", error.message, true);
    }
  }, 250);
});
$("results").addEventListener("click", event => {
  const item = event.target.closest("li");
  if (item) selectVessel(Number(item.dataset.mmsi));
});

export async function selectVessel(mmsi) {
  currentVessel = mmsi;
  map.closePopup();
  const selectedDay = new Date(`${selectedTimestamp()}Z`);
  $("route-from").value = toIsoSeconds(new Date(selectedDay - 3 * DAY_MS)).slice(0, 10);
  $("route-to").value = toIsoSeconds(new Date(+selectedDay + 3 * DAY_MS)).slice(0, 10);
  $("vessel-panel").hidden = false;
  $("vessel").innerHTML = "";
  $("trips").innerHTML = "";
  setStatus("vessel-status", "Loading vessel…");
  try {
    const vessel = await getJson(`/vessels/${mmsi}`);
    const dimensions = vessel.length ? `${vessel.length} × ${vessel.width ?? "?"} m` : "–";
    $("vessel").innerHTML = `
      <dt>Name</dt><dd><b>${escapeHtml(vessel.name || "Unknown")}</b></dd>
      <dt>MMSI</dt><dd>${vessel.mmsi}${vessel.imo ? ` · IMO ${vessel.imo}` : ""}</dd>
      <dt>Type</dt><dd>${escapeHtml(vessel.ship_type || "–")}</dd>
      <dt>Size</dt><dd>${dimensions}</dd>
      <dt>Destination</dt><dd>${escapeHtml(vessel.destination || "–")}</dd>
      <dt>Seen</dt><dd>${formatTimestamp(vessel.first_seen)} → ${formatTimestamp(vessel.last_seen)}</dd>`;
    const at = new Date(`${selectedTimestamp()}Z`);
    const trips = await getJson("/trips", {
      mmsi,
      start: toIsoSeconds(new Date(at - 15 * DAY_MS)),
      end: toIsoSeconds(new Date(+at + 15 * DAY_MS)),
    });
    setStatus("vessel-status", `${trips.length} trips in the 30 days around the selected time`);
    $("trips").innerHTML = trips.map(trip => `
      <li data-trip="${escapeHtml(trip.trip_id)}">${formatTimestamp(trip.start_ts)}
      <div class="meta">${(trip.distance_m / 1852).toFixed(1)} nm · ${(trip.duration_s / 3600).toFixed(1)} h · ${trip.point_count} points</div></li>`).join("");
  } catch (error) {
    setStatus("vessel-status", error.message, true);
  }
}
window.selectVessel = selectVessel;

function drawTrack(latlngs, label) {
  pathLayer.clearLayers();
  stopPlayer();
  $("route-legend").hidden = true;
  if (latlngs.length < 2) {
    setStatus("vessel-status", "Not enough points to draw", true);
    return;
  }
  const line = L.polyline(latlngs, { color: "#e11d48", weight: 3 }).addTo(pathLayer);
  L.circleMarker(latlngs[0], { radius: 5, color: "#15803d", fillOpacity: 1 }).bindTooltip("Start").addTo(pathLayer);
  L.circleMarker(latlngs.at(-1), { radius: 5, color: "#b42318", fillOpacity: 1 }).bindTooltip("End").addTo(pathLayer);
  map.fitBounds(line.getBounds(), { padding: [40, 40], maxZoom: 13 });
  setStatus("vessel-status", label);
}

$("trips").addEventListener("click", async event => {
  const item = event.target.closest("li");
  if (!item) return;
  document.querySelectorAll("#trips li.active").forEach(element => element.classList.remove("active"));
  item.classList.add("active");
  try {
    const geometry = await getJson(`/trips/${encodeURIComponent(item.dataset.trip)}/geometry`);
    drawTrack(
      geometry.geometry.coordinates.map(([longitude, latitude]) => [latitude, longitude]),
      `Trip drawn · ${geometry.geometry.coordinates.length} points`,
    );
  } catch (error) {
    setStatus("vessel-status", error.message, true);
  }
});

$("track-btn").addEventListener("click", async () => {
  if (!currentVessel) return;
  const at = new Date(`${selectedTimestamp()}Z`);
  const halfDay = 12 * 3_600_000;
  try {
    const track = await getJson(`/vessels/${currentVessel}/track`, {
      start: toIsoSeconds(new Date(at - halfDay)),
      end: toIsoSeconds(new Date(+at + halfDay)),
    });
    const downsampled = track.downsampled ? ` · showing ${track.points.length}` : "";
    drawTrack(track.points.map(point => [point.lat, point.lon]), `24 h track · ${track.total_points.toLocaleString()} positions${downsampled}`);
  } catch (error) {
    setStatus("vessel-status", error.message, true);
  }
});

$("clear-btn").addEventListener("click", () => {
  pathLayer.clearLayers();
  stopPlayer();
  $("route-legend").hidden = true;
  setStatus("vessel-status", "");
});

const timeColor = fraction => `hsl(${200 + 145 * fraction} ${80 - 10 * Math.sin(Math.PI * fraction)}% ${55 - 10 * fraction}%)`;
const formatDuration = seconds => seconds >= 86_400 ? `${(seconds / 86_400).toFixed(1)} days` : `${(seconds / 3600).toFixed(1)} h`;
const timestampMs = timestamp => Date.parse(`${timestamp}Z`);

function setupPlayer(legs) {
  const prepared = legs.map(leg => {
    const points = leg.geometry.coordinates.map(([longitude, latitude]) => L.latLng(latitude, longitude));
    const distances = [0];
    for (let index = 1; index < points.length; index += 1) {
      distances.push(distances[index - 1] + points[index - 1].distanceTo(points[index]));
    }
    const total = distances.at(-1) || 1;
    const start = timestampMs(leg.start_ts);
    const end = timestampMs(leg.end_ts);
    return { start, end, points, times: distances.map(distance => start + (end - start) * distance / total) };
  });
  const marker = L.circleMarker(prepared[0].points[0], {
    radius: 8, color: "#ffffff", weight: 3, fillColor: "#14201f", fillOpacity: 1,
  }).addTo(pathLayer);
  player = {
    legs: prepared,
    start: prepared[0].start,
    end: prepared.at(-1).end,
    marker,
    time: prepared[0].start,
    animationFrame: 0,
    lastFrame: 0,
  };
  $("player").hidden = false;
  $("play-btn").textContent = "Play";
  showPlaybackTime(player.start);
}

function positionAt(time) {
  let index = player.legs.length - 1;
  while (index > 0 && player.legs[index].start > time) index -= 1;
  const leg = player.legs[index];
  if (time >= leg.end) return leg.points.at(-1);
  if (time <= leg.start) return leg.points[0];
  let pointIndex = 1;
  while (pointIndex < leg.times.length - 1 && leg.times[pointIndex] < time) pointIndex += 1;
  const before = leg.times[pointIndex - 1];
  const after = leg.times[pointIndex];
  const fraction = after > before ? (time - before) / (after - before) : 1;
  const start = leg.points[pointIndex - 1];
  const end = leg.points[pointIndex];
  return L.latLng(start.lat + (end.lat - start.lat) * fraction, start.lng + (end.lng - start.lng) * fraction);
}

function showPlaybackTime(time) {
  player.time = time;
  player.marker.setLatLng(positionAt(time));
  $("scrub").value = Math.round(1000 * (time - player.start) / (player.end - player.start || 1));
  $("play-time").textContent = formatTimestamp(new Date(time).toISOString());
}

function pausePlayer() {
  if (!player) return;
  cancelAnimationFrame(player.animationFrame);
  player.animationFrame = 0;
  $("play-btn").textContent = player.time >= player.end ? "Replay" : "Play";
}

function stopPlayer() {
  pausePlayer();
  player = null;
  $("player").hidden = true;
}

function animatePlayer(now) {
  const step = (now - player.lastFrame) * (player.end - player.start) / Number($("speed").value);
  player.lastFrame = now;
  const time = Math.min(player.end, player.time + step);
  showPlaybackTime(time);
  if (time < player.end) player.animationFrame = requestAnimationFrame(animatePlayer);
  else pausePlayer();
}

$("play-btn").addEventListener("click", () => {
  if (!player) return;
  if (player.animationFrame) { pausePlayer(); return; }
  if (player.time >= player.end) showPlaybackTime(player.start);
  player.lastFrame = performance.now();
  $("play-btn").textContent = "Pause";
  player.animationFrame = requestAnimationFrame(animatePlayer);
});
$("scrub").addEventListener("input", () => {
  if (!player) return;
  pausePlayer();
  showPlaybackTime(player.start + (player.end - player.start) * $("scrub").value / 1000);
  $("play-btn").textContent = "Play";
});

$("route-btn").addEventListener("click", async () => {
  if (!currentVessel) return;
  const from = $("route-from").value;
  const to = $("route-to").value;
  if (!from || !to || to < from) {
    setStatus("vessel-status", "Pick a From date on or before To", true);
    return;
  }
  const start = new Date(`${from}T00:00:00Z`);
  const end = new Date(+new Date(`${to}T00:00:00Z`) + DAY_MS);
  if (end - start > 31 * DAY_MS) {
    setStatus("vessel-status", "A route can cover at most 31 days", true);
    return;
  }
  $("route-btn").disabled = true;
  setStatus("vessel-status", "Loading route…");
  try {
    const route = await getJson(`/vessels/${currentVessel}/route`, { start: toIsoSeconds(start), end: toIsoSeconds(end) });
    pathLayer.clearLayers();
    stopPlayer();
    if (!route.legs.length) {
      setStatus("vessel-status", "No trips in this range");
      $("route-legend").hidden = true;
      return;
    }
    const span = end - start;
    const bounds = L.latLngBounds([]);
    for (const leg of route.legs) {
      const fraction = Math.min(1, Math.max(0, (new Date(`${leg.start_ts}Z`) - start) / span));
      const points = leg.geometry.coordinates.map(([longitude, latitude]) => [latitude, longitude]);
      L.polyline(points, { color: timeColor(fraction), weight: 3, opacity: 0.86 })
        .bindTooltip(`${formatTimestamp(leg.start_ts)} → ${formatTimestamp(leg.end_ts)}<br>${(leg.distance_m / 1852).toFixed(1)} nm · ${formatDuration(leg.duration_s)}`, { sticky: true })
        .addTo(pathLayer);
      points.forEach(point => bounds.extend(point));
    }
    for (const stop of route.stops) {
      L.circleMarker([stop.lat, stop.lon], { radius: 5, color: "#14201f", weight: 2, fillColor: "#ffffff", fillOpacity: 1 })
        .bindTooltip(`Stopped ${formatDuration(stop.duration_s)}<br>${formatTimestamp(stop.start_ts)} → ${formatTimestamp(stop.end_ts)}`)
        .addTo(pathLayer);
    }
    const first = route.legs[0].geometry.coordinates[0];
    const last = route.legs.at(-1).geometry.coordinates.at(-1);
    L.marker([first[1], first[0]], { title: "Start" }).bindTooltip(`Start ${formatTimestamp(route.legs[0].start_ts)}`).addTo(pathLayer);
    L.marker([last[1], last[0]], { title: "End" }).bindTooltip(`End ${formatTimestamp(route.legs.at(-1).end_ts)}`).addTo(pathLayer);
    map.flyToBounds(bounds, { padding: [40, 40], maxZoom: 13, duration: 1.6 });
    $("route-legend").hidden = false;
    setupPlayer(route.legs);
    const nauticalMiles = route.legs.reduce((sum, leg) => sum + leg.distance_m, 0) / 1852;
    const capped = route.truncated ? " · capped at 2,000 trips; shorten the range" : "";
    setStatus("vessel-status", `${route.legs.length} trips · ${Math.round(nauticalMiles).toLocaleString()} nm · ${route.stops.length} stops${capped}`);
  } catch (error) {
    setStatus("vessel-status", error.message, true);
  } finally {
    $("route-btn").disabled = false;
  }
});

export function flyTo(bounds) {
  return new Promise(resolve => {
    map.once("moveend", resolve);
    map.flyToBounds(bounds, { padding: [30, 30], maxZoom: 12, duration: 1.6 });
    setTimeout(resolve, 2500);
  });
}
