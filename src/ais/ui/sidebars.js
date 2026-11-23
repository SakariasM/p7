const shell = document.querySelector(".app-shell");
const compactLayout = window.matchMedia("(max-width: 1100px)");
const names = ["controls", "chat"];

function setSidebarOpen(name, open) {
  const content = document.getElementById(name === "controls" ? "controls-content" : "chat-panel-content");
  const button = document.getElementById(`${name}-toggle`);
  shell.classList.toggle(`${name}-collapsed`, !open);
  content.hidden = !open;
  button.setAttribute("aria-expanded", String(open));
  button.setAttribute("aria-label", `${open ? "Collapse" : "Expand"} ${name === "controls" ? "map controls" : "chat"}`);
  button.textContent = (name === "controls") === open ? "‹" : "›";
}

export function showSidebar(name) {
  if (compactLayout.matches) {
    for (const other of names) setSidebarOpen(other, other === name);
  } else {
    setSidebarOpen(name, true);
  }
}

export function showMap() {
  if (compactLayout.matches) {
    for (const name of names) setSidebarOpen(name, false);
  }
}

export function initializeSidebars() {
  function resetLayout() {
    for (const name of names) setSidebarOpen(name, !compactLayout.matches);
  }
  for (const name of names) {
    const button = document.getElementById(`${name}-toggle`);
    button.addEventListener("click", () => {
      if (button.getAttribute("aria-expanded") === "false") showSidebar(name);
      else setSidebarOpen(name, false);
    });
  }
  compactLayout.addEventListener("change", resetLayout);
  resetLayout();
}
