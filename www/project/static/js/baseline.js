/* Real captures of one fixed scene under five camera sampling policies. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const base = "static/media/baseline/";
  const panel = $("baseline-explorer");
  const state = { room: 0, level: 2, mode: "rgb" };
  const labels = ["Narrow", "Close", "Balanced", "Wide", "Very wide"];
  const cache = new Map();
  let gallery, revision = 0;

  function preload(name) {
    if (!cache.has(name)) {
      const pending = new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => image.decode().then(resolve, reject);
        image.onerror = () => reject(new Error(`Baseline image unavailable: ${name}`));
        image.src = base + name;
      });
      // A temporary failed request may be retried on the next selection.
      cache.set(name, pending.catch((error) => { cache.delete(name); throw error; }));
    }
    return cache.get(name);
  }

  async function update() {
    const version = ++revision;
    const selected = { ...state };
    const room = gallery.rooms[selected.room];
    const level = room.levels[selected.level];
    panel.setAttribute("aria-busy", "true");
    $("baseline-status").textContent = "Loading matched views…";
    try {
      // Decode every plane first; rapid slider/room/mode changes cannot mix sets.
      await Promise.all([preload(level.plan), ...level.views.map((v) => preload(v[selected.mode]))]);
      if (version !== revision) return;
      const value = level.baseline.toFixed(2);
      $("baseline-slider").value = level.baseline;
      $("baseline-slider").setAttribute("aria-valuetext", `${value}, ${labels[selected.level].toLowerCase()} baseline`);
      $("baseline-room").value = selected.room;
      $("baseline-value").textContent = value;
      $("baseline-name").textContent = labels[selected.level];
      panel.querySelectorAll("[data-baseline]").forEach((button) => {
        button.setAttribute("aria-pressed", String(Number(button.dataset.baseline) === level.baseline));
      });
      panel.querySelectorAll("[data-baseline-mode]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.baselineMode === selected.mode));
      });
      $("baseline-plan").src = base + level.plan;
      $("baseline-plan").alt = `Top-down camera layout, seed ${room.seed}, baseline ${value}; common room scale`;
      $("baseline-distance").textContent = `${level.mean_reference_m.toFixed(2)} m`;
      $("baseline-shared").textContent = `${(100 * level.shared_any).toFixed(1)}%`;
      $("baseline-views").querySelectorAll("figure").forEach((tile, c) => {
        const view = level.views[c];
        const image = tile.querySelector("img");
        image.src = base + view[selected.mode];
        image.alt = `Seed ${room.seed}, baseline ${value}, camera ${c}, t=0: ${selected.mode === "rgb" ? "RGB" : "surfaces shared with any peer highlighted in teal"}`;
        tile.querySelector("a").href = image.src;
        tile.querySelector("a").setAttribute("aria-label", `${image.alt}. Open full resolution in a new tab.`);
        tile.querySelector(".baseline-fov").textContent = `${view.camera.fovy_degrees.toFixed(1)}° FOV`;
        tile.querySelector(".baseline-fov").title = "Vertical field of view";
      });
      $("baseline-contact-link").href = base + room.contact;
      $("baseline-contact-image").src = base + room.contact;
      $("baseline-contact-image").alt = `Seed ${room.seed}: all four cameras and room plans at baseline 0, 0.5 and 1`;
      $("baseline-display-help").textContent = selected.mode === "rgb"
        ? "Four synchronized RGB views at t = 0. Select an image to open it at full resolution."
        : "Teal highlights surfaces also seen by at least one other camera. Unshared pixels and background are dimmed. Exact masks are available below.";
      $("baseline-status").textContent = `Seed ${room.seed} · baseline ${value} · ${labels[selected.level].toLowerCase()} · same geometry & lighting`;
      panel.dataset.selection = `${room.seed}:${value}:${selected.mode}`;
      panel.dataset.ready = "true";
      panel.setAttribute("aria-busy", "false");
    } catch (error) {
      if (version !== revision) return;
      panel.setAttribute("aria-busy", "false");
      $("baseline-status").textContent = "Capture unavailable. The previous views remain; choose a setting to retry.";
      console.error(error);
    }
  }

  fetch(base + "gallery.json").then((response) => {
    if (!response.ok) throw new Error(`Baseline gallery request: ${response.status}`);
    return response.json();
  }).then((data) => {
    gallery = data;
    panel.querySelectorAll("button,input,select").forEach((control) => { control.disabled = false; });
    $("baseline-room").addEventListener("change", (e) => { state.room = Number(e.target.value); update(); });
    $("baseline-slider").addEventListener("input", (e) => {
      state.level = gallery.levels.indexOf(Number(e.target.value)); update();
    });
    panel.querySelectorAll("[data-baseline]").forEach((button) => button.addEventListener("click", () => {
      state.level = gallery.levels.indexOf(Number(button.dataset.baseline)); update();
    }));
    panel.querySelectorAll("[data-baseline-mode]").forEach((button) => button.addEventListener("click", () => {
      state.mode = button.dataset.baselineMode; update();
    }));
    update();
  }).catch((error) => {
    panel.setAttribute("aria-busy", "false");
    $("baseline-status").textContent = "Interactive controls unavailable. Default captures and the side-by-side figure remain available.";
    console.error(error);
  });
})();
