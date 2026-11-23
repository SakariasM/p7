import "./agent-assistant.js";
import { initializeAreaSelection } from "./area-selection.js";
import { loadSnapshot, map } from "./map-explorer.js";
import { initializeSidebars } from "./sidebars.js";

initializeSidebars();
new ResizeObserver(() => map.invalidateSize({ pan: false })).observe(document.getElementById("map"));
initializeAreaSelection(map);
loadSnapshot();
