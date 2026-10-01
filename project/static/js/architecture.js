/* All current architectural captures. Plans and pixels share room/time identity. */
"use strict";
(() => {
  const root = document.getElementById("architecture-explorer");
  if (!root) return;
  const $ = (name) => document.getElementById(`architecture-${name}`);
  const base = "static/media/architecture/";
  const labels = { color: "RGB", depth: "Depth", normal: "Normals", semantic: "Semantic", position: "Position", co_visibility: "Co-visibility" };
  const state = { seed: 7, step: 0, mode: location.hash === "#explore" ? "co_visibility" : "color", peer: null };
  const cache = new Map();
  let gallery, revision = 0;
  function preload(file) {
    if (!cache.has(file)) {
      cache.set(file, new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = resolve;
        image.onerror = () => { cache.delete(file); reject(new Error(`Missing architecture image: ${file}`)); };
        image.src = base + file;
      }));
    }
    return cache.get(file);
  }
  async function update() {
    const token = ++revision;
    const { seed, step, mode, peer } = state;
    const scene = gallery.scenes.find((s) => s.seed === seed);
    const frame = scene.frames[step];
    const annotation = (view, camera) => mode === "co_visibility" && peer !== null
      ? (camera === peer ? view.images.color : view.images.peers[peer]) : view.images[mode];
    root.setAttribute("aria-busy", "true");
    $("status").textContent = `Loading seed ${seed}, t=${frame.time}…`;
    try {
      await Promise.all([preload(frame.plan), ...frame.views.flatMap((v,c) => [preload(v.images.color), preload(annotation(v,c))])]);
      if (token !== revision) return;
      document.querySelectorAll("[data-architecture-camera]").forEach((tile, c) => {
        const view = frame.views[c];
        const identity = `seed ${seed}, camera ${c}, t=${frame.time}`;
        tile.querySelector(".architecture-rgb").src = base + view.images.color;
        tile.querySelector(".architecture-rgb").alt = `Native RGB: ${identity}`;
        const description = mode === "co_visibility" && peer !== null
          ? (c === peer ? "RGB reference" : `Shared with camera ${peer}`) : labels[mode];
        tile.querySelector(".architecture-annotation").src = base + annotation(view,c);
        tile.querySelector(".architecture-annotation").alt = `${description}: ${identity}`;
        tile.querySelector("a").href = base + annotation(view,c);
        tile.querySelector("a").setAttribute("aria-label", `Open full ${description} image: ${identity}`);
        tile.querySelector(".architecture-fov").textContent = `${(view.camera.fov_y * 180/Math.PI).toFixed(1)}° FOV`;
        const shared = tile.querySelector(".architecture-shared");
        shared.hidden = mode !== "co_visibility";
        shared.textContent = peer === null ? `${(view.visibility.shared_fraction_valid*100).toFixed(1)}% shared with a peer`
          : c === peer ? "RGB reference · own bit excluded" : `${(view.visibility.peer_fraction_valid[peer]*100).toFixed(1)}% shared with camera ${peer}`;
      });
      $("plan").src = base + frame.plan;
      $("plan").alt = `Manifest plan and envelope section for seed ${seed}, cameras at t=${frame.time}`;
      $("plan-link").href = base + frame.plan;
      $("metadata").href = base + scene.metadata;
      $("masks").href = base + scene.masks;
      $("room").value = String(seed);
      $("title").textContent = `Seed ${seed} · ${scene.activity}`;
      $("dimensions").textContent = `${scene.area_m2.toFixed(1)} m² footprint · ${scene.roof_pitch_degrees.toFixed(1)}° roof pitch`;
      $("features").replaceChildren(...scene.features.map((feature) => {
        const item = document.createElement("li");
        item.textContent = feature;
        return item;
      }));
      $("description").textContent = mode === "co_visibility" && peer !== null
        ? `Green highlights mark surfaces also visible to camera ${peer}. Other pixels are dimmed; camera ${peer} remains the RGB reference. Exact masks use the same camera-bit order.` : gallery.display[mode];
      $("mode-label").textContent = mode === "co_visibility" && peer !== null ? `Shared with camera ${peer}` : labels[mode];
      $("visibility").hidden = mode !== "co_visibility";
      $("legend").hidden = peer !== null;
      $("status").textContent = `Seed ${seed} · t=${frame.time} · four matched 640 × 400 views · ${labels[mode]}`;
      document.querySelectorAll("[data-architecture-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.architectureMode === mode)));
      document.querySelectorAll("[data-architecture-step]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.architectureStep) === step)));
      document.querySelectorAll("[data-architecture-seed]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.architectureSeed) === seed)));
      root.querySelectorAll("[data-architecture-peer]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.architecturePeer === (peer === null ? "all" : String(peer)))));
      $("reveal").disabled = mode === "color";
      root.dataset.mode = mode;
      root.dataset.selection = `${seed}:${step}:${mode}`;
      root.dataset.peer = peer === null ? "all" : String(peer);
      root.dataset.ready = "true";
    } catch (error) {
      if (token === revision) $("status").textContent = "Capture unavailable. The previous matched view is retained; retry or use the downloads below.";
      console.error(error);
    } finally {
      if (token === revision) root.setAttribute("aria-busy", "false");
    }
  }
  function roomOptions(feature = "") {
    const scenes = gallery.scenes.filter((s) => !feature || s.features.includes(feature));
    $("room").replaceChildren(...scenes.map((scene) => new Option(`${scene.seed} · ${scene.activity}`, String(scene.seed))));
    if (!scenes.some((s) => s.seed === state.seed)) state.seed = scenes[0].seed;
    $("room").value = String(state.seed);
    $("count").textContent = `${scenes.length} of ${gallery.rendered_rooms} captured rooms`;
  }
  fetch(base + "gallery.json").then((r) => {
    if (!r.ok) throw new Error(`Architecture gallery: ${r.status}`);
    return r.json();
  }).then((data) => {
    gallery = data;
    roomOptions();
    Object.keys(data.feature_room_counts).forEach((feature) => {
      const count = data.scenes.filter((s) => s.features.includes(feature)).length;
      if (count) $("filter").add(new Option(`${feature} · ${count} rooms`, feature));
    });
    $("filter").addEventListener("change", (event) => { roomOptions(event.target.value); update(); });
    $("room").addEventListener("change", (event) => { state.seed = Number(event.target.value); update(); });
    document.querySelectorAll("[data-architecture-seed]").forEach((button) => button.addEventListener("click", () => {
      state.seed = Number(button.dataset.architectureSeed);
      $("filter").value = "";
      roomOptions(); update();
    }));
    document.querySelectorAll("[data-architecture-mode]").forEach((button) => button.addEventListener("click", () => {
      state.mode = button.dataset.architectureMode; update();
    }));
    document.querySelectorAll("[data-architecture-step]").forEach((button) => button.addEventListener("click", () => {
      state.step = Number(button.dataset.architectureStep); update();
    }));
    root.querySelectorAll("[data-architecture-peer]").forEach((button) => button.addEventListener("click", () => {
      state.peer = button.dataset.architecturePeer === "all" ? null : Number(button.dataset.architecturePeer); update();
    }));
    document.querySelectorAll("[data-architecture-jump]").forEach((link) => link.addEventListener("click", () => {
      state.mode = link.dataset.architectureJump; update();
    }));
    $("reveal").addEventListener("input", (event) => {
      root.style.setProperty("--architecture-split", `${event.target.value}%`);
      event.target.setAttribute("aria-valuetext", `${event.target.value} percent RGB`);
      $("reveal-value").textContent = `${event.target.value}% RGB`;
    });
    root.querySelectorAll("button, select").forEach((control) => { control.disabled = false; });
    update();
  }).catch((error) => {
    $("status").textContent = "Interactive controls unavailable. Full-resolution captures, plans and paper figures remain linked below.";
    console.error(error);
  });
})();
