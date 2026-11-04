import { getJson, postJson } from "./api-client.js";
import { clearSelectedArea, getSelectedArea } from "./area-selection.js";
import {
  flyTo, loadSnapshot, map, selectVessel, setHighlightedVessels, showActiveFilters,
} from "./map-explorer.js";
import { $, escapeHtml, setStatus, SHIP_TYPES, toIsoSeconds } from "./shared.js";

const DAY_MS = 86_400_000;
let runId = null;

async function showAnswerOnMap(hints) {
  setHighlightedVessels(hints.vessels || []);
  if (hints.places?.length) {
    const bounds = L.latLngBounds([]);
    for (const place of hints.places) {
      bounds.extend([place.box[1], place.box[0]]);
      bounds.extend([place.box[3], place.box[2]]);
    }
    await flyTo(bounds);
  }
  const when = hints.at || hints.start;
  if (when) $("at").value = when.slice(0, 16);
  $("name").value = hints.name_filter || "";
  $("type").value = SHIP_TYPES.includes(hints.ship_type) ? hints.ship_type : "";
  showActiveFilters();
  $("ask-vessels").innerHTML = (hints.vessels || []).slice(0, 50).map(vessel => `
    <li data-mmsi="${vessel.mmsi}"><b>${escapeHtml(vessel.name || `MMSI ${vessel.mmsi}`)}</b>
    <div class="meta">${vessel.mmsi}</div></li>`).join("");

  if (hints.action === "route" && hints.start && hints.end) {
    let mmsi = hints.vessels?.length === 1 ? hints.vessels[0].mmsi : null;
    if (!mmsi && hints.vessel_query) {
      const vessels = await getJson("/vessels", { q: hints.vessel_query, limit: 1 });
      mmsi = vessels[0]?.mmsi;
    }
    if (mmsi) {
      await selectVessel(mmsi);
      $("route-from").value = hints.start.slice(0, 10);
      $("route-to").value = toIsoSeconds(new Date(new Date(`${hints.end}Z`) - DAY_MS)).slice(0, 10);
      $("route-btn").click();
      return;
    }
  }
  if (hints.action) await loadSnapshot();
}

function showFeedback(id) {
  runId = id || null;
  $("ask-feedback").hidden = !runId;
  $("ask-fb-text").textContent = "Was this answer right?";
  $("ask-fb-buttons").hidden = false;
  $("ask-fb-note").hidden = true;
  $("ask-note").value = "";
}

async function sendFeedback(verdict) {
  const note = verdict === "wrong" ? $("ask-note").value.trim() : "";
  $("ask-fb-send").disabled = true;
  try {
    await postJson("/feedback", { id: runId, verdict, note });
    $("ask-fb-text").textContent = "Thanks, saved.";
    $("ask-fb-buttons").hidden = true;
    $("ask-fb-note").hidden = true;
  } catch (error) {
    setStatus("ask-status", `Feedback not saved: ${error.message}`, true);
  } finally {
    $("ask-fb-send").disabled = false;
  }
}

async function askAgent() {
  const question = $("ask-q").value.trim();
  if (!question) {
    setStatus("ask-status", "Write a question first", true);
    return;
  }
  $("ask-btn").disabled = true;
  $("ask-answer").hidden = true;
  showFeedback(null);
  $("ask-vessels").innerHTML = "";
  const selectedArea = getSelectedArea();
  setStatus(
    "ask-status",
    `Thinking${selectedArea ? " inside the selected rectangle" : ""}… simple questions take about 10 seconds`,
  );
  const started = performance.now();
  try {
    const request = { question };
    if (selectedArea) request.selected_area = selectedArea;
    const response = await postJson("/ask", request);
    $("ask-answer").textContent = response.answer;
    $("ask-answer").hidden = false;
    showFeedback(response.id);
    const seconds = ((performance.now() - started) / 1000).toFixed(0);
    setStatus("ask-status", `Answered in ${seconds} seconds${selectedArea ? " · exact area used" : ""}`);
    await showAnswerOnMap(response.map || {});
  } catch (error) {
    setStatus("ask-status", error.message, true);
  } finally {
    $("ask-btn").disabled = false;
  }
}

$("ask-btn").addEventListener("click", askAgent);
$("ask-fb-yes").addEventListener("click", () => sendFeedback("correct"));
$("ask-fb-no").addEventListener("click", () => {
  $("ask-fb-buttons").hidden = true;
  $("ask-fb-note").hidden = false;
  $("ask-note").focus();
});
$("ask-fb-send").addEventListener("click", () => sendFeedback("wrong"));
$("ask-note").addEventListener("keydown", event => { if (event.key === "Enter") sendFeedback("wrong"); });
$("ask-q").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askAgent();
  }
});
$("ask-examples").addEventListener("click", event => {
  const button = event.target.closest("button");
  if (button) {
    $("ask-q").value = button.textContent;
    askAgent();
  }
});
$("ask-vessels").addEventListener("click", event => {
  const item = event.target.closest("li");
  if (item) selectVessel(Number(item.dataset.mmsi));
});
$("ask-clear").addEventListener("click", () => {
  setHighlightedVessels([]);
  clearSelectedArea();
  $("ask-q").value = "";
  $("ask-answer").hidden = true;
  showFeedback(null);
  $("ask-vessels").innerHTML = "";
  setStatus("ask-status", "");
  loadSnapshot();
});
