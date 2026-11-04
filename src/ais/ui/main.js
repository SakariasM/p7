import "./agent-assistant.js";
import { initializeAreaSelection } from "./area-selection.js";
import { loadSnapshot, map } from "./map-explorer.js";

initializeAreaSelection(map);
loadSnapshot();
