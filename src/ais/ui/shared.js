export const SHIP_COLORS = {
  Cargo: "#2563eb",
  Tanker: "#ea580c",
  Passenger: "#7c3aed",
  Fishing: "#16a34a",
  Pleasure: "#db2777",
  Sailing: "#0891b2",
  Tug: "#ca8a04",
};

export const OTHER_SHIP_COLOR = "#64748b";
export const SHIP_TYPES = [
  "Cargo", "Tanker", "Passenger", "Fishing", "Pleasure", "Sailing", "Tug", "HSC",
  "Pilot", "SAR", "Dredging", "Military", "Law enforcement", "Towing", "Port tender", "Other",
];

export const $ = id => document.getElementById(id);
export const escapeHtml = value => String(value ?? "").replace(
  /[&<>"']/g,
  character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character],
);
export const formatTimestamp = timestamp => timestamp ? timestamp.replace("T", " ").slice(0, 16) : "–";
export const toIsoSeconds = date => date.toISOString().slice(0, 19);
export const shipColor = type => SHIP_COLORS[type] || OTHER_SHIP_COLOR;

export function setStatus(id, text, error = false) {
  $(id).textContent = text;
  $(id).classList.toggle("error", error);
}
