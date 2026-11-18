import { getJson, postJson } from "./api-client.js";
import { getSelectedArea } from "./area-selection.js";
import {
  clearRoute, flyTo, focusChatVessel, highlightChatVessels, loadSnapshot, loadVesselRoute, map, selectVessel,
  setHighlightedVessels, showActiveFilters,
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
  if (when) $("at").value = when.slice(0, 16);
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
  if (hints.action && (!area || hints.at)) await loadSnapshot({ area });
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

function bindFeedback(feedback, id) {
  const text = feedback.querySelector(".feedback-text");
  const buttons = feedback.querySelector(".feedback-buttons");
  const form = feedback.querySelector(".feedback-form");
  const note = form.querySelector("input");
  const send = form.querySelector("button");
  const yes = buttons.querySelector(".feedback-yes");
  const no = buttons.querySelector(".feedback-no");

  async function sendFeedback(verdict) {
    yes.disabled = no.disabled = send.disabled = true;
    try {
      await postJson("/feedback", { id, verdict, note: verdict === "wrong" ? note.value.trim() : "" });
      text.textContent = "Thanks, saved.";
      buttons.hidden = form.hidden = true;
    } catch (error) {
      text.textContent = `Feedback not saved: ${error.message}`;
    } finally {
      yes.disabled = no.disabled = send.disabled = false;
    }
  }

  yes.addEventListener("click", () => sendFeedback("correct"));
  no.addEventListener("click", () => {
    buttons.hidden = true;
    form.hidden = false;
    note.focus();
  });
  send.addEventListener("click", () => sendFeedback("wrong"));
  note.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.isComposing && !send.disabled) sendFeedback("wrong");
  });
  feedback.hidden = false;
}

function scrollToLatest() {
  chatContent.scrollTop = chatContent.scrollHeight;
}

async function askAgent() {
  if (sendButton.disabled) return;
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("ask-status", "Write a question first", true);
    return;
  }

  const selectedArea = getSelectedArea();
  const version = ++questionVersion;
  answerMapVersion++;
  clearRoute();
  setHighlightedVessels([]);
  pendingQuestion = new AbortController();
  const turn = $("chat-turn-template").content.firstElementChild.cloneNode(true);
  const status = turn.querySelector(".chat-turn-status");
  const answer = turn.querySelector(".answer");
  turn.querySelector(".chat-question").textContent = question;
  turn.querySelector(".chat-area-note").hidden = !selectedArea;
  status.textContent = "Thinking…";
  $("ask-examples").hidden = true;
  thread.append(turn);
  questionInput.value = "";
  questionInput.focus({ preventScroll: true });
  setStatus("ask-status", "");
  sendButton.disabled = true;
  scrollToLatest();

  try {
    const request = { question };
    if (selectedArea) request.selected_area = selectedArea;
    const response = await postJson("/ask", request, pendingQuestion.signal);
    if (version !== questionVersion) return;
    const wasAtBottom = chatContent.scrollHeight - chatContent.scrollTop - chatContent.clientHeight < 32;
    const hints = response.map || {};
    const routeRequest = hints.action === "route" && hints.start && hints.end;
    status.hidden = true;
    answer.textContent = routeRequest ? "Finding matching ships…" : response.answer;
    answer.hidden = false;
    if (!hints.vessels?.length && hints.vessel_query) {
      const matches = await getJson("/vessels", { q: hints.vessel_query, limit: 100 });
      if (version !== questionVersion) return;
      const query = String(hints.vessel_query).trim().toUpperCase();
      const exact = matches.filter(vessel =>
        String(vessel.mmsi) === query || vessel.name?.toUpperCase() === query
        || vessel.callsign?.toUpperCase() === query);
      hints.vessels = exact.length || !routeRequest ? exact : matches;
    }
    const vessels = (hints.vessels || []).slice(0, 50);
    hints.vessels = vessels;
    if (routeRequest) {
      answer.textContent = vessels.length
        ? "Choose a ship below to show its route. Each MMSI identifies a different vessel."
        : `No matching ship was found for ${hints.vessel_query || "this route request"}. Try its MMSI.`;
    }
    async function showShips(mmsi) {
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
        if (!result) return;
        const { located, total, route } = result;
        if (routeRequest) {
          const identity = `${chosen.name || "Ship"} (MMSI ${chosen.mmsi})`;
          const lastDay = toIsoSeconds(new Date(new Date(`${hints.end}Z`) - DAY_MS)).slice(0, 10);
          answer.textContent = route.legs.length
            ? `Showing the route of ${identity} from ${hints.start.slice(0, 10)} through ${lastDay} UTC.`
            : `No recorded trips were found for ${identity} from ${hints.start.slice(0, 10)} through ${lastDay} UTC.`;
          status.hidden = !route.legs.length;
        }
        status.textContent = `${located} of ${total} ships highlighted in yellow`;
        if (mmsi && !routeRequest && !focusChatVessel(mmsi)) {
          status.textContent += " · No matching position found for this ship";
        }
      } catch (error) {
        status.hidden = false;
        status.classList.add("error");
        status.textContent = `Map update failed: ${error.message}`;
        if (routeRequest) answer.textContent = `Could not load the route of ${chosen.name || chosen.mmsi}.`;
      } finally {
        choices.forEach(button => { button.disabled = false; });
      }
    }
    renderVessels(turn.querySelector(".chat-vessels"), vessels, showShips);
    turn.querySelector(".chat-reply").append(status);
    if (response.id) bindFeedback(turn.querySelector(".chat-feedback"), response.id);
    if (wasAtBottom) {
      chatContent.scrollTop += answer.getBoundingClientRect().top - chatContent.getBoundingClientRect().top;
    }
    await showShips();
  } catch (error) {
    if (version !== questionVersion) return;
    status.textContent = answer.hidden ? error.message : `Map update failed: ${error.message}`;
    status.hidden = false;
    status.classList.add("error");
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
  thread.replaceChildren();
  $("ask-examples").hidden = false;
  questionInput.value = "";
  questionInput.focus();
  setStatus("ask-status", "");
});
