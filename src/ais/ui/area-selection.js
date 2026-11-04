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
  const summary = $("selected-area-summary");
  const coordinates = $("selected-area-coordinates");
  const clearButton = $("clear-area-btn");
  if (!selectedArea) {
    summary.textContent = "No exact area selected";
    coordinates.textContent = "Draw a rectangle to limit the agent to every position inside it.";
    clearButton.hidden = true;
    return;
  }
  const { west, south, east, north } = selectedArea;
  summary.textContent = "Exact area attached to your question";
  coordinates.textContent = [
    `SW ${south.toFixed(5)}, ${west.toFixed(5)}`,
    `NW ${north.toFixed(5)}, ${west.toFixed(5)}`,
    `NE ${north.toFixed(5)}, ${east.toFixed(5)}`,
    `SE ${south.toFixed(5)}, ${east.toFixed(5)}`,
  ].join("  ·  ");
  clearButton.hidden = false;
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
    if (window.matchMedia("(max-width: 820px)").matches) {
      map.getContainer().scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
  $("clear-area-btn").addEventListener("click", clearSelectedArea);
  renderArea();
}
