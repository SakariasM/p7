import { getJson, postJson } from "./api-client.js";
import { getSelectedArea } from "./area-selection.js";
import {
  clearRoute, flyTo, focusChatVessel, highlightChatVessels, loadSnapshot, loadVesselRoute, map, selectVessel,
  setHighlightedVessels, showActiveFilters,
  setSnapshotTime,
} from "./map-explorer.js";
import { $, setStatus, SHIP_TYPES, toIsoSeconds } from "./shared.js";

const DAY_MS = 86_400_000;
const thread = $("chat-thread");
const chatContent = $("chat-content");
const questionInput = $("ask-q");
const sendButton = $("ask-btn");
const clearButton = $("ask-clear");
let answerMapVersion = 0;
let questionVersion = 0;
let pendingQuestion = null;
const HISTORY_PREFIX = "ais-explorer.chat.v1.";
const savedTurns = new Map();

function historyUnavailable() {
  $("history-status").hidden = false;
  $("history-status").textContent = "Chat history could not be saved in this browser.";
}

function historyKey(record) {
  return HISTORY_PREFIX + (record.response?.id || record.localId);
}

function readSavedTurn(key) {
  const record = JSON.parse(localStorage.getItem(key));
  if (!record || typeof record.localId !== "string" || typeof record.question !== "string"
      || !Number.isFinite(record.createdAt)) return null;
  if (record.response && typeof record.response.answer !== "string") return null;
  return record;
}

function saveTurn(record, changes) {
  if (savedTurns.get(record.localId)?.record !== record) return;
  const previousKey = historyKey(record);
  let latest = record;
  try {
    latest = readSavedTurn(previousKey) || record;
  } catch {
    historyUnavailable();
  }
  Object.assign(record, latest, changes, {
    feedback: { ...latest.feedback, ...changes.feedback },
  });
  try {
    localStorage.setItem(historyKey(record), JSON.stringify(record));
    if (previousKey !== historyKey(record)) localStorage.removeItem(previousKey);
  } catch {
    historyUnavailable();
  }
}

function createTurn(record) {
  const turn = $("chat-turn-template").content.firstElementChild.cloneNode(true);
  turn.querySelector(".chat-question").textContent = record.question;
  if (record.selectedArea) {
    const areaNote = document.createElement("span");
    areaNote.className = "chat-area-note";
    areaNote.textContent = "Selected area attached";
    turn.querySelector(".chat-user").append(areaNote);
  }
  const status = turn.querySelector(".chat-turn-status");
  status.textContent = "Thinking…";
  thread.append(turn);
  savedTurns.set(record.localId, { record, turn });
  $("ask-examples").hidden = true;
  return turn;
}

function restoreHistory() {
  const records = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key?.startsWith(HISTORY_PREFIX)) continue;
      try {
        const record = readSavedTurn(key);
        if (record) records.push(record);
      } catch {
        // A malformed entry must not prevent other answers from being restored.
      }
    }
  } catch {
    historyUnavailable();
  }
  const unique = new Map();
  for (const record of records) {
    if (record.response || !unique.has(record.localId)) unique.set(record.localId, record);
  }
  for (const record of [...unique.values()].sort((a, b) => a.createdAt - b.createdAt)) {
    if (savedTurns.has(record.localId)) continue;
    const turn = createTurn(record);
    const status = turn.querySelector(".chat-turn-status");
    if (record.response) {
      renderAnswer(turn, record, false, () => savedTurns.has(record.localId)).catch(() => {
        status.hidden = false;
        status.textContent = "Saved map references could not be restored.";
      });
    } else {
      status.textContent = record.error || "This request was interrupted. Please ask it again.";
      status.classList.add("error");
    }
  }
  scrollToLatest();
}

function clearHistory() {
  savedTurns.clear();
  try {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(HISTORY_PREFIX)) keys.push(key);
    }
    for (const key of keys) localStorage.removeItem(key);
  } catch {
    historyUnavailable();
  }
}

async function showAnswerOnMap(hints, selectedArea) {
  const version = ++answerMapVersion;
  clearRoute();
  const area = selectedArea && [selectedArea.west, selectedArea.south, selectedArea.east, selectedArea.north];
  const boxes = area ? [area] : (hints.places || []).map(place => place.box);
  const highlightContext = { start: hints.start, end: hints.end, boxes };
  if (boxes.length) {
    const bounds = L.latLngBounds([]);
    for (const [west, south, east, north] of boxes) {
      bounds.extend([south, west]);
      bounds.extend([north, east]);
    }
    await flyTo(bounds);
    if (version !== answerMapVersion) return null;
  }
  const when = hints.at || hints.start;
  const timeWasReset = when ? await setSnapshotTime(when) : false;
  if (version !== answerMapVersion) return null;
  $("name").value = hints.name_filter || "";
  $("type").value = SHIP_TYPES.includes(hints.ship_type) ? hints.ship_type : "";
  showActiveFilters();

  if (hints.action === "route" && hints.start && hints.end) {
    if (hints.vessels?.length !== 1) throw new Error("Choose a ship to show its route");
    const mmsi = hints.vessels[0].mmsi;
    await selectVessel(mmsi);
    if (version !== answerMapVersion) return null;
    $("route-from").value = hints.start.slice(0, 10);
    $("route-to").value = toIsoSeconds(new Date(new Date(`${hints.end}Z`) - DAY_MS)).slice(0, 10);
    const route = await loadVesselRoute(mmsi, new Date(`${hints.start}Z`), new Date(`${hints.end}Z`));
    if (!route || version !== answerMapVersion) return null;
    if (!route.legs.length) {
      setHighlightedVessels([]);
      return { located: 0, total: 0, route };
    }
    const highlighted = await highlightChatVessels(hints.vessels, highlightContext);
    if (version !== answerMapVersion) return null;
    return { ...highlighted, route };
  }
  if (timeWasReset || (hints.action && (!area || hints.at))) await loadSnapshot({ area });
  if (version !== answerMapVersion) return null;
  return highlightChatVessels(hints.vessels || [], highlightContext);
}

function renderVessels(list, vessels, onFocus) {
  for (const vessel of vessels.slice(0, 50)) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "link chat-vessel-button";
    button.title = "Highlight and focus this ship on the map";
    button.textContent = [vessel.name, `MMSI ${vessel.mmsi}`, vessel.ship_type, vessel.callsign]
      .filter(Boolean).join(" · ");
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await onFocus(vessel.mmsi);
      } finally {
        button.disabled = false;
      }
    });
    item.append(button);
    list.append(item);
  }
}

function bindFeedback(feedback, record) {
  const text = feedback.querySelector(".feedback-text");
  const buttons = feedback.querySelector(".feedback-buttons");
  const form = feedback.querySelector(".feedback-form");
  const note = form.querySelector("input");
  const send = form.querySelector("button");
  const yes = buttons.querySelector(".feedback-yes");
  const no = buttons.querySelector(".feedback-no");
  let verdict = record.feedback?.verdict || null;
  note.value = record.feedback?.note || "";
  if (record.feedback?.saved) {
    text.textContent = "Thanks, saved.";
    buttons.hidden = form.hidden = true;
  } else if (verdict) {
    buttons.hidden = true;
    form.hidden = false;
    text.textContent = record.feedback.error || "Add an optional comment, then send your feedback.";
  }

  async function sendFeedback() {
    if (!verdict || send.disabled) return;
    yes.disabled = no.disabled = send.disabled = true;
    try {
      const submittedNote = note.value.trim();
      await postJson("/feedback", { id: record.response.id, verdict, note: submittedNote });
      saveTurn(record, { feedback: { verdict, note: submittedNote, saved: true, error: null } });
      text.textContent = "Thanks, saved.";
      buttons.hidden = form.hidden = true;
    } catch (error) {
      text.textContent = `Feedback not saved: ${error.message}`;
      saveTurn(record, { feedback: { error: text.textContent } });
    } finally {
      yes.disabled = no.disabled = send.disabled = false;
    }
  }

  function chooseVerdict(value) {
    verdict = value;
    saveTurn(record, { feedback: { verdict, saved: false, error: null } });
    text.textContent = value === "correct" ? "Marked correct. Add an optional comment." : "Marked incorrect. Add an optional comment.";
    buttons.hidden = true;
    form.hidden = false;
    note.focus();
  }
  yes.addEventListener("click", () => chooseVerdict("correct"));
  no.addEventListener("click", () => chooseVerdict("wrong"));
  send.addEventListener("click", sendFeedback);
  note.addEventListener("input", () => saveTurn(record, { feedback: { note: note.value } }));
  note.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.isComposing) sendFeedback();
  });
  feedback.hidden = false;
}

function scrollToLatest() {
  chatContent.scrollTop = chatContent.scrollHeight;
}

function unresolvedVesselNames(response, hints) {
  const names = new Set(hints.vessel_names || []);
  for (const vessel of hints.vessels || []) {
    if (!vessel.mmsi && vessel.name) names.add(vessel.name);
  }
  // Older harness responses expose name-only references in the query results.
  for (const call of response.tool_calls || []) {
    const result = call.result || {};
    if (result.error) continue;
    const columns = (result.columns || []).map(column => String(column).toLowerCase());
    const nameIndex = columns.findIndex(column => ["name", "vessel_name", "ship_name"].includes(column));
    const mmsiIndex = columns.indexOf("mmsi");
    for (const row of result.rows || []) {
      if (nameIndex >= 0 && (mmsiIndex < 0 || !row[mmsiIndex]) && typeof row[nameIndex] === "string") {
        names.add(row[nameIndex].trim());
      }
    }
  }
  if (!hints.vessels?.length && hints.vessel_query) names.add(String(hints.vessel_query));
  return [...names].filter(Boolean).slice(0, 50);
}

async function renderAnswer(turn, record, activateMap, isCurrent) {
  const status = turn.querySelector(".chat-turn-status");
  const answer = turn.querySelector(".answer");
  const wasAtBottom = chatContent.scrollHeight - chatContent.scrollTop - chatContent.clientHeight < 32;
  const response = record.response;
  const selectedArea = record.selectedArea;
  const hints = response.map || {};
  const routeRequest = hints.action === "route" && hints.start && hints.end;
  status.hidden = true;
  answer.textContent = record.displayAnswer || response.answer;
  answer.hidden = false;
  if (response.id) bindFeedback(turn.querySelector(".chat-feedback"), record);
  const resolved = (hints.vessels || []).filter(vessel => vessel.mmsi);
  const ambiguities = activateMap ? [] : (record.ambiguities || []);
  for (const name of activateMap ? unresolvedVesselNames(response, hints) : []) {
    const matches = await getJson("/vessels", { q: name, limit: 100 });
    if (!isCurrent()) return;
    const query = name.trim().toUpperCase();
    const exact = matches.filter(vessel =>
      String(vessel.mmsi) === query || vessel.name?.toUpperCase() === query
      || vessel.callsign?.toUpperCase() === query);
    const candidates = exact.length || !routeRequest ? exact : matches;
    if (routeRequest || candidates.length === 1) resolved.push(...candidates);
    else if (candidates.length > 1) ambiguities.push({ name, candidates });
  }
  const vessels = [...new Map(resolved.map(vessel => [Number(vessel.mmsi), vessel])).values()].slice(0, 50);
  hints.vessels = vessels;
  if (routeRequest && activateMap) {
    answer.textContent = vessels.length
      ? "Choose a ship below to show its route. Each MMSI identifies a different vessel."
      : `No matching ship was found for ${hints.vessel_query || "this route request"}. Try its MMSI.`;
  }
  saveTurn(record, { response, ambiguities, displayAnswer: answer.textContent });
  async function showShips(mmsi) {
    if (!isCurrent()) return;
    if (routeRequest && !mmsi && vessels.length !== 1) return;
    const chosen = vessels.find(vessel => Number(vessel.mmsi) === Number(mmsi)) || vessels[0];
    const mapHints = routeRequest ? { ...hints, vessels: [chosen] } : hints;
    const choices = turn.querySelectorAll(".chat-vessel-button");
    choices.forEach(button => { button.disabled = true; });
    status.hidden = !vessels.length;
    status.classList.remove("error");
    status.textContent = "Locating ships mentioned in this answer…";
    if (routeRequest) answer.textContent = `Loading the route of ${chosen.name || chosen.mmsi}…`;
    try {
      const result = await showAnswerOnMap(mapHints, selectedArea);
      if (!result || !isCurrent()) return;
      const { located, total, route } = result;
      if (routeRequest) {
        const identity = `${chosen.name || "Ship"} (MMSI ${chosen.mmsi})`;
        const lastDay = toIsoSeconds(new Date(new Date(`${hints.end}Z`) - DAY_MS)).slice(0, 10);
        answer.textContent = route.legs.length
          ? `Showing the route of ${identity} from ${hints.start.slice(0, 10)} through ${lastDay} UTC.`
          : `No recorded trips were found for ${identity} from ${hints.start.slice(0, 10)} through ${lastDay} UTC.`;
        status.hidden = !route.legs.length;
      }
      status.hidden = !total;
      status.textContent = !total ? "" : located
        ? `${located} of ${total} ships highlighted in yellow`
        : "Ship identified, but no matching position is available for this period.";
      if (mmsi && !routeRequest && !focusChatVessel(mmsi)) {
        status.textContent += " · No matching position found for this ship";
      }
    } catch (error) {
      status.hidden = false;
      status.classList.add("error");
      status.textContent = `Map update failed: ${error.message}`;
      if (routeRequest) answer.textContent = `Could not load the route of ${chosen.name || chosen.mmsi}.`;
    } finally {
      saveTurn(record, { displayAnswer: answer.textContent });
      choices.forEach(button => { button.disabled = false; });
    }
  }
  const vesselList = turn.querySelector(".chat-vessels");
  renderVessels(vesselList, vessels, showShips);
  for (const { name, candidates } of ambiguities) {
    const choice = document.createElement("li");
    const label = document.createElement("p");
    label.className = "helper";
    label.textContent = `Several ships are named ${name}. Choose one to highlight:`;
    const options = document.createElement("ul");
    options.className = "list";
    renderVessels(options, candidates, async mmsi => {
      const vessel = candidates.find(candidate => Number(candidate.mmsi) === Number(mmsi));
      if (!vessels.some(existing => Number(existing.mmsi) === Number(mmsi))) {
        vessels.push(vessel);
        renderVessels(vesselList, [vessel], showShips);
      }
      saveTurn(record, { response, ambiguities: record.ambiguities.filter(group => group.name !== name) });
      choice.remove();
      await showShips(mmsi);
    });
    choice.append(label, options);
    vesselList.append(choice);
  }
  turn.querySelector(".chat-reply").append(status);
  if (record.error) {
    status.hidden = false;
    status.textContent = record.error;
    status.classList.add("error");
  }
  if (activateMap && wasAtBottom) {
    chatContent.scrollTop += answer.getBoundingClientRect().top - chatContent.getBoundingClientRect().top;
  }
  if (activateMap) await showShips();
}

async function askAgent() {
  if (sendButton.disabled) return;
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("ask-status", "Write a question first", true);
    return;
  }

  const selectedArea = getSelectedArea();
  const request = { question };
  if (selectedArea) request.selected_area = selectedArea;
  const version = ++questionVersion;
  answerMapVersion++;
  clearRoute();
  setHighlightedVessels([]);
  pendingQuestion = new AbortController();
  const record = {
    localId: crypto.randomUUID(), createdAt: Date.now(), question,
    selectedArea: request.selected_area || null,
    response: null, feedback: {}, error: null,
  };
  const turn = createTurn(record);
  saveTurn(record, {});
  const status = turn.querySelector(".chat-turn-status");
  const answer = turn.querySelector(".answer");
  questionInput.value = "";
  questionInput.focus({ preventScroll: true });
  setStatus("ask-status", "");
  sendButton.disabled = true;
  scrollToLatest();

  try {
    const response = await postJson("/ask", request, pendingQuestion.signal);
    if (version !== questionVersion) return;
    const hints = response.map || {};
    hints.vessel_names = unresolvedVesselNames(response, hints);
    const storedResponse = { id: response.id, answer: response.answer, map: hints };
    const isRoute = hints.action === "route" && hints.start && hints.end;
    saveTurn(record, {
      response: storedResponse,
      displayAnswer: isRoute ? "Select a ship below to load its route." : response.answer,
      error: null,
    });
    await renderAnswer(turn, record, true, () => savedTurns.has(record.localId));

  } catch (error) {
    if (version !== questionVersion) return;
    status.textContent = answer.hidden ? error.message : `Map update failed: ${error.message}`;
    status.hidden = false;
    status.classList.add("error");
    saveTurn(record, { error: status.textContent });
    if (answer.hidden && !questionInput.value) questionInput.value = question;
  } finally {
    if (version === questionVersion) {
      pendingQuestion = null;
      sendButton.disabled = false;
    }
  }
}

sendButton.addEventListener("click", askAgent);
questionInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    askAgent();
  }
});
$("ask-examples").addEventListener("click", event => {
  const button = event.target.closest("button");
  if (button) {
    questionInput.value = button.textContent;
    askAgent();
  }
});
clearButton.addEventListener("click", () => {
  questionVersion++;
  answerMapVersion++;
  pendingQuestion?.abort();
  pendingQuestion = null;
  sendButton.disabled = false;
  clearRoute();
  setHighlightedVessels([]);
  clearHistory();
  thread.replaceChildren();
  $("ask-examples").hidden = false;
  questionInput.value = "";
  questionInput.focus();
  setStatus("ask-status", "");
});

restoreHistory();
