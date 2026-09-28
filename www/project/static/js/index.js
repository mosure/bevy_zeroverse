/* Matched capture explorer. All images and values come from native exports. */
"use strict";
(() => {
  const base = "static/media/";
  const $ = (id) => document.getElementById(id);
  const labels = {
    color: "RGB", depth: "Depth", normal: "Normals", semantic: "Semantic",
    position: "Position", optical_flow: "Optical flow", motion_vectors: "Motion vectors",
    co_visibility: "Co-visibility",
  };
  const state = { scene: 0, time: 0, camera: 0, mode: "normal", peer: null };
  let gallery;
  let request = 0;
  const imageCache = new Map();
  function preload(path) {
    if (!imageCache.has(path)) {
      imageCache.set(path, new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve();
        image.onerror = () => reject(new Error(`Image unavailable: ${path}`));
        image.src = base + path;
      }));
    }
    return imageCache.get(path);
  }
  function swatch(label, rgb) {
    const span = document.createElement("span");
    span.className = "legend-swatch";
    const color = document.createElement("i");
    color.style.backgroundColor = `rgb(${rgb.join(",")})`;
    span.append(color, document.createTextNode(label));
    return span;
  }
  function describe(scene, frame, view) {
    const legend = $("mode-legend");
    legend.replaceChildren();
    const next = scene.frames[state.time + 1];
    const descriptions = {
      color: "Renderer-tonemapped sRGB. This is the same source image used on the left of every annotation comparison.",
      depth: `Linear camera-space Z in metres. One fixed 0–${scene.depth_max} m display scale is used for all views and times in this room; farther values are clipped for display.`,
      normal: "View-space unit normals encoded as RGB = (n + 1) / 2. Colors describe surface direction, not material.",
      semantic: "Shared semantic classes, aligned with geometric depth. The palette is identical across cameras. Ceiling light fixtures use the Lamp class.",
      position: "World XYZ affine-normalized by the exported primary-room AABB. RGB shows X/Y/Z; out-of-box values are clipped in this visualization only.",
      optical_flow: next ? `Forward surface displacement from t=${frame.time.toFixed(2)} to t=${next.time.toFixed(2)}, in pixels. Hue = direction; saturation reaches its maximum at 32 pixels. White = zero; black = invalid correspondence.` : "Terminal capture: there is no successor frame. All forward vectors and correspondence-valid masks are zero.",
      motion_vectors: next ? `The same forward displacement from t=${frame.time.toFixed(2)} to t=${next.time.toFixed(2)}, divided by image width and height. Saturation reaches its maximum at 0.05 normalized displacement. Black = invalid.` : "Terminal capture: normalized motion vectors and validity masks are zero.",
      co_visibility: state.peer === null ? "Additive camera membership at this same timestep. Each legend color contributes one camera bit; combined colors mean shared visibility. Black can be an unshared surface or background—use the exact mask + validity export to distinguish them." : `Surfaces also seen by camera ${state.peer} are highlighted over RGB (${(view.shared_fraction[state.peer] * 100).toFixed(1)}% of this source image). The other pixels are dimmed.`,
    };
    $("mode-description").textContent = descriptions[state.mode];
    $("mode-label").textContent = state.mode === "co_visibility" && state.peer !== null ? `Shared with camera ${state.peer}` : labels[state.mode];
    if (state.mode === "depth") {
      const scale = document.createElement("i");
      scale.className = "legend-gradient";
      legend.append("0 m", scale, `${scene.depth_max} m`);
    } else if (state.mode === "normal" || state.mode === "position") {
      legend.append(swatch("+X", [255, 128, 128]), swatch("+Y", [128, 255, 128]), swatch("+Z", [128, 128, 255]));
    } else if (state.mode === "co_visibility") {
      scene.legend.forEach((entry) => legend.append(swatch(`Camera ${entry.camera_index}${entry.camera_index === state.camera ? " (source; excluded)" : ""}`, entry.rgb8)));
    } else if (state.mode === "semantic") {
      // All labels are available without hiding less common classes.
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Semantic palette (40 classes)";
      const content = document.createElement("div");
      content.className = "mode-legend";
      gallery.semantic_palette.forEach((entry) => content.append(swatch(entry.label.replace(/([a-z])([A-Z])/g, "$1 $2"), entry.rgb8)));
      details.append(summary, content); legend.append(details);
    } else if (state.mode === "optical_flow" || state.mode === "motion_vectors") {
      legend.append(swatch("Right", [255, 0, 0]), swatch("Down", [128, 255, 0]), swatch("Left", [0, 255, 255]), swatch("Up", [128, 0, 255]));
    }
  }
  function peers() {
    $("peer-controls").hidden = state.mode !== "co_visibility";
    const buttons = $("peer-buttons");
    buttons.replaceChildren();
    [null, 0, 1, 2, 3].forEach((peer) => {
      const button = document.createElement("button");
      button.textContent = peer === null ? "All cameras" : `Camera ${peer}`;
      button.setAttribute("aria-pressed", String(state.peer === peer));
      button.disabled = peer === state.camera;
      button.addEventListener("click", () => { state.peer = peer; update(); });
      buttons.append(button);
    });
  }
  async function update() {
    const revision = ++request;
    const scene = gallery.scenes[state.scene];
    const frame = scene.frames[state.time];
    const view = frame.views[state.camera];
    const rgb = view.images.color;
    const annotation = state.mode === "co_visibility" && state.peer !== null ? view.images.peers[state.peer] : view.images[state.mode];
    try {
      // Commit both planes together, so rapid switching cannot mix captures.
      await Promise.all([preload(rgb), preload(annotation)]);
      if (revision !== request) return;
      $("rgb-image").src = base + rgb;
      $("annotation-image").src = base + annotation;
      const identity = `seed ${scene.seed}, camera ${state.camera}, t=${frame.time.toFixed(2)}`;
      $("rgb-image").alt = `RGB capture: ${identity}`;
      $("annotation-image").alt = `${labels[state.mode]} annotation: ${identity}`;
      document.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode)));
      document.querySelectorAll("[data-camera]").forEach((button) => {
        const index = Number(button.dataset.camera);
        button.setAttribute("aria-pressed", String(index === state.camera));
        button.querySelector("img").src = base + frame.views[index].images.color;
        button.querySelector("img").alt = `Seed ${scene.seed}, camera ${index}, time ${frame.time.toFixed(2)}`;
      });
      describe(scene, frame, view); peers();
      $("camera-info").textContent = `Camera ${state.camera} · vertical FOV ${view.camera.fovy_degrees.toFixed(1)}° · t=${frame.time.toFixed(2)}`;
      $("capture-status").textContent = `Seed ${scene.seed} · ${scene.width} × ${scene.height} · four capture cameras · same-time annotations`;
      $("calibration-link").href = base + scene.calibration;
      $("mask-link").href = base + scene.masks;
      $("reveal").disabled = state.mode === "color";
      $("comparison").querySelector(".divider").hidden = state.mode === "color";
      $("comparison").dataset.ready = "true";
    } catch (error) {
      $("capture-status").textContent = "This image could not be loaded. Try another view or download the capture metadata.";
      console.error(error);
    }
  }
  $("reveal").addEventListener("input", (event) => {
    const amount = Number(event.target.value);
    $("comparison").style.setProperty("--split", `${amount}%`);
    event.target.setAttribute("aria-valuetext", `${amount} percent RGB`);
    $("reveal-value").textContent = `${amount} / ${100 - amount}`;
  });
  fetch(base + "gallery.json").then((response) => {
    if (!response.ok) throw new Error(`Gallery request: ${response.status}`);
    return response.json();
  }).then((data) => {
    gallery = data;
    $("scene-select").addEventListener("change", (event) => { state.scene = Number(event.target.value); update(); });
    $("time-select").addEventListener("change", (event) => { state.time = Number(event.target.value); update(); });
    document.querySelectorAll("[data-camera]").forEach((b) => b.addEventListener("click", () => {
      state.camera = Number(b.dataset.camera);
      if (state.peer === state.camera) state.peer = null;
      update();
    }));
    document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode; update(); }));
    update();
  }).catch((error) => {
    $("capture-status").textContent = "Interactive gallery unavailable. Static figures, videos and the whitepaper are still available.";
    console.error(error);
  });

  // Start the teaser once when visible. Reduced-motion users opt in with Play.
  const teaser = $("room-video");
  let started = false;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.target === teaser && entry.isIntersecting && !started && !reduced.matches) {
          started = true; teaser.play().catch(() => {});
        } else if (!entry.isIntersecting) {
          entry.target.pause();
        }
      }
    }, { threshold: 0.2 });
    observer.observe(teaser);
    observer.observe($("motion-video"));
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) document.querySelectorAll("video").forEach((v) => v.pause());
  });
})();
