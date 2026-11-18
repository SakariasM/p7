import { $ } from "./shared.js";

let selectedArea = null;
let drawnRectangle = null;
let drawRectangle = null;
let mapInstance = null;

function rounded(value) {
  return Number(value.toFixed(6));
}

function areaFromBounds(bounds) {
  return {
    west: rounded(bounds.getWest()),
    south: rounded(bounds.getSouth()),
    east: rounded(bounds.getEast()),
    north: rounded(bounds.getNorth()),
  };
}

function renderArea() {
  const chip = $("selected-area-chip");
  const coordinates = $("selected-area-coordinates");
  chip.hidden = !selectedArea;
  if (!selectedArea) {
    chip.querySelector("details").open = false;
    $("ask-q").placeholder = "Ask about a vessel, route, or area…";
    return;
  }
  const { west, south, east, north } = selectedArea;
  $("ask-q").placeholder = "How many ships passed through here from 3 to 5 May 2026?";
  coordinates.textContent = [
    `SW ${south.toFixed(6)}, ${west.toFixed(6)}`,
    `NW ${north.toFixed(6)}, ${west.toFixed(6)}`,
    `NE ${north.toFixed(6)}, ${east.toFixed(6)}`,
    `SE ${south.toFixed(6)}, ${east.toFixed(6)}`,
  ].join("  ·  ");
}

function useRectangle(layer) {
  if (drawnRectangle) mapInstance.removeLayer(drawnRectangle);
  drawnRectangle = layer;
  drawnRectangle.setStyle({ color: "#0f766e", weight: 2, fillColor: "#14b8a6", fillOpacity: 0.14 });
  drawnRectangle.addTo(mapInstance);
  selectedArea = areaFromBounds(drawnRectangle.getBounds());
  renderArea();
}

export function clearSelectedArea() {
  if (drawnRectangle) mapInstance.removeLayer(drawnRectangle);
  drawnRectangle = null;
  selectedArea = null;
  renderArea();
}

export function getSelectedArea() {
  return selectedArea ? { ...selectedArea } : null;
}

export function initializeAreaSelection(map) {
  mapInstance = map;
  drawRectangle = new L.Draw.Rectangle(map, {
    shapeOptions: { color: "#0f766e", weight: 2, fillColor: "#14b8a6", fillOpacity: 0.14 },
    showArea: true,
    metric: true,
  });
  map.on(L.Draw.Event.CREATED, event => useRectangle(event.layer));
  $("draw-area-btn").addEventListener("click", () => {
    drawRectangle.enable();
    if (window.matchMedia("(max-width: 760px)").matches) {
      map.getContainer().scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
  $("clear-area-btn").addEventListener("click", () => {
    clearSelectedArea();
    $("draw-area-btn").focus();
  });
  renderArea();
}
