/* Current construction evidence; data strings enter the DOM as text. */
(() => {
  const root = document.getElementById("object-atlas");
  if (!root) return;
  const select = document.getElementById("object-category");
  const mode = document.getElementById("object-mode");
  const study = document.getElementById("object-study");
  const studyControl = document.getElementById("object-study-control");
  const protocol = document.getElementById("object-protocol");
  const download = document.getElementById("object-download");
  const content = document.getElementById("object-content");
  const inventory = document.getElementById("object-summary");
  const base = root.dataset.source.replace(/[^/]+$/, "");
  const assemblyBase = root.dataset.assemblies.replace(/[^/]+$/, "");
  const cache = new Map();
  let atlas, assemblies, generation = 0;
  const node = (tag, text, cls) => {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  };
  const label = text => text.replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2");
  const number = value => value === undefined || value === null ? "—" : typeof value === "number" ? Number(value.toPrecision(4)).toString() : typeof value === "object" ? JSON.stringify(value) : String(value);
  function distribution(p) {
    const el = node("div");
    if (Number.isInteger(p.present) && p.present < p.total) {
      el.append(node("p", "Active in " + p.present + " of " + p.total + " samples. Statistics below use only those active samples."));
    }
    if (p.type === "numeric") {
      const s = p.observed;
      el.append(node("p", number(s.min) + " – " + number(s.max) + "; median " + number(s.median) + ". Eight bins across this observed range."));
      const bars = node("div", undefined, "object-bars");
      const peak = Math.max(1, ...p.bins);
      for (const count of p.bins) {
        const bar = node("span"); bar.style.height = (100 * count / peak) + "%"; bar.title = count + " samples"; bars.append(bar);
      }
      bars.setAttribute("aria-label", "Bin counts: " + p.bins.join(", "));
      el.append(bars);
    } else {
      el.append(node("p", Object.entries(p.counts).map(([v,n]) => v + ": " + n).join("; ")));
    }
    return el;
  }
  async function renderGeometry(c, serial) {
    const file = c.samples_file;
    if (!cache.has(file)) {
      const response = await fetch(base + file);
      if (!response.ok) throw new Error("Missing " + file);
      cache.set(file, await response.json());
    }
    if (serial !== generation) return;
    const samples = cache.get(file);
    const rows = c.selected_seeds.map(seed => samples.find(s => s.object.seed === seed));
    const metrics = node("div", undefined, "object-metrics");
    const counts = c.coverage.map(v => v.representatives.length);
    for (const [title, value] of [
      ["Geometry coverage · .005 / .015 / .030", counts.join(" / ") + " of " + atlas.sample_count_per_category],
      ["Triangles · median / p95", number(c.mesh_summary.triangles.median) + " / " + number(c.mesh_summary.triangles.p95)],
      ["Mesh build · median / p95", number(c.build_ms_summary.median) + " / " + number(c.build_ms_summary.p95) + " ms"],
      ["Mesh validation", atlas.sample_count_per_category + " / " + atlas.sample_count_per_category + " passed"],
    ]) { const metric=node("div"); metric.append(node("strong",value), node("span",title)); metrics.append(metric); }
    const size = rows[0].object.size.map(number).join(" × ");
    const near = c.nearest_shape_distances.filter(d => d <= 0.005).length;
    const note = counts[1] === atlas.sample_count_per_category ? "Coverage saturates this sample size." : counts[1] < 8 ? "Low coarse-shape variation at this envelope; inspect fine geometry and conditional controls below." : "Coverage describes realized surfaces at this fixed envelope.";
    const status = node("p", "Envelope: " + size + " m (width × height × depth). " + near + " / " + atlas.sample_count_per_category + " samples have a neighbor within distance .005. " + note + " A dash marks a parameter that is inactive for that sample.", "object-status");
    const table = node("table", undefined, "object-grid");
    const head = node("thead"), images = node("tr");
    images.append(node("th", label(c.kind) + " · " + (mode.value === "color" ? "PBR color" : "View-space normals")));
    rows.forEach((s,i) => {
      const td=node("td"), link=node("a"), img=node("img");
      link.href=base+c.kind+"-"+i+"-"+mode.value+".webp"; link.target="_blank"; link.rel="noopener";
      img.src=link.href; img.width=320; img.height=320;
      img.alt=label(c.kind)+", seed "+s.object.seed+", "+mode.value+"; matched studio view";
      link.append(img); td.append(link, node("span","seed "+s.object.seed)); images.append(td);
    });
    head.append(images); table.append(head);
    const body = node("tbody");
    const variants = node("tr");
    variants.append(node("th","Construction variant"));
    for (const sample of rows) variants.append(node("td",sample.object.variant));
    body.append(variants);
    for (const [key,p] of Object.entries(c.parameters)) {
      const tr=node("tr"), th=node("th"), detail=node("details");
      th.scope="row"; detail.append(node("summary",label(key)), distribution(p)); th.append(detail); tr.append(th);
      for (const sample of rows) tr.append(node("td",number(sample.program[key])));
      body.append(tr);
    }
    table.append(body);
    const scroller=node("div",undefined,"object-scroll");
    scroller.tabIndex=0; scroller.setAttribute("role","region"); scroller.setAttribute("aria-label",label(c.kind)+" parameter grid; scroll horizontally for all six examples");
    scroller.append(table);
    content.replaceChildren(metrics,status,scroller);
  }
  function renderAssembly(c) {
    const rows = c.selected;
    const metrics = node("div", undefined, "object-metrics");
    for (const [title, value] of [
      ["Construction validation", assemblies.samples_per_category + " / " + assemblies.samples_per_category + " passed"],
      ["Illustrative examples", rows.length],
      ["Aligned channels", "Color · normals · semantics"],
    ]) { const metric=node("div"); metric.append(node("strong",value),node("span",title)); metrics.append(metric); }
    const isDoor = c.kind.endsWith("Door");
    const status = node("p", isDoor
      ? "Leaves, frames and hardware share a construction instance with distinct semantic labels. Revolving examples show static rotor phases; other doors include shut, ajar and open states."
      : "Cabinet fronts and glazing retain the furniture's semantic identity; pulls and hinges have their own hardware label. Envelopes include front travel.", "object-status");
    const legend = node("div", undefined, "object-legend");
    if (mode.value === "semantic") {
      const visible = new Set(c.semantic_pixel_counts.flatMap(counts => Object.keys(counts).filter(key => counts[key] > 0)));
      for (const entry of assemblies.semantic_palette.filter(e => visible.has(e.label))) {
        const swatch = node("span", label(entry.label));
        swatch.style.borderLeft = "10px solid rgb(" + entry.srgb8.join(",") + ")";
        legend.append(swatch);
      }
    }
    const table = node("table", undefined, "object-grid"), head=node("thead"), images=node("tr");
    images.append(node("th",label(c.kind)+" · "+mode.selectedOptions[0].textContent));
    rows.forEach((sample,i) => {
      const td=node("td"), link=node("a"), img=node("img");
      link.href=assemblyBase+c.kind+"-"+i+"-"+mode.value+".png"; link.target="_blank"; link.rel="noopener";
      img.src=link.href; img.width=320; img.height=320;
      img.alt=label(c.kind)+", seed "+sample.seed+", "+mode.value+"; opening and hardware study";
      link.append(img); td.append(link,node("span","seed "+sample.seed)); images.append(td);
    });
    head.append(images); table.append(head);
    const programs = rows.map(s => {
      const {program, ...envelope} = s.program;
      return program ? {...envelope, ...program} : envelope;
    });
    const body=node("tbody");
    for (const key of new Set(programs.flatMap(p => Object.keys(p)))) {
      const tr=node("tr"), th=node("th",label(key)); th.scope="row"; tr.append(th);
      for (const program of programs) tr.append(node("td",number(program[key])));
      body.append(tr);
    }
    table.append(body);
    const scroller=node("div",undefined,"object-scroll");
    scroller.tabIndex=0; scroller.setAttribute("role","region");
    scroller.setAttribute("aria-label",label(c.kind)+" opening and hardware parameter grid; scroll horizontally for all six examples");
    scroller.append(table); content.replaceChildren(metrics,status,legend,scroller);
  }
  async function render() {
    const serial = ++generation;
    const geometry = atlas.categories.find(c => c.kind === select.value);
    const assembly = assemblies.categories.find(c => c.kind === select.value);
    studyControl.hidden = !(geometry && assembly);
    const useAssembly = !!assembly && (!geometry || study.value === "assembly");
    mode.querySelector('[value="semantic"]').disabled = !useAssembly;
    if (!useAssembly && mode.value === "semantic") mode.value = "color";
    download.href = useAssembly ? root.dataset.assemblies : root.dataset.source;
    protocol.textContent = useAssembly
      ? assemblies.protocol + " Opening and hardware studies do not report the object's 256-seed geometry coverage."
      : atlas.sample_count_per_category + " construction seeds per category. Six examples are selected for depth-map spread, with camera, finish seed and studio lighting held constant. Parameter samples are correlated; open a row to inspect its observed distribution.";
    try {
      if (useAssembly) renderAssembly(assembly); else await renderGeometry(geometry,serial);
      if (serial !== generation) return;
      const url=new URL(location.href);
      url.searchParams.set("object",select.value); url.searchParams.set("object_mode",mode.value);
      if (useAssembly) url.searchParams.set("object_study","assembly"); else url.searchParams.delete("object_study");
      history.replaceState(null,"",url);
    } catch(error) { if(serial === generation) failure(error); }
  }
  function failure(error) { content.replaceChildren(node("p","Object atlas unavailable: "+error.message)); }
  Promise.all([root.dataset.source,root.dataset.assemblies].map(async url => {
    const response=await fetch(url); if(!response.ok) throw new Error("Missing "+url); return response.json();
  })).then(([objects,doors]) => {
    atlas=objects; assemblies=doors;
    const furniture=node("optgroup"); furniture.label="Furniture + everyday objects";
    const openings=node("optgroup"); openings.label="Doors + openings";
    for (const c of atlas.categories) { const option=node("option",label(c.kind)); option.value=c.kind; furniture.append(option); }
    for (const c of assemblies.categories.filter(c=>!atlas.categories.some(o=>o.kind===c.kind))) {
      const option=node("option",label(c.kind)); option.value=c.kind; openings.append(option);
    }
    select.append(furniture,openings);
    const query=new URLSearchParams(location.search);
    const kinds=[...atlas.categories,...assemblies.categories].map(c=>c.kind);
    select.value=kinds.includes(query.get("object"))?query.get("object"):location.hash==="#doors"?"HingedDoor":"FloorLamp";
    study.value=query.get("object_study")==="assembly"?"assembly":"geometry";
    mode.value=["normal","semantic"].includes(query.get("object_mode"))?query.get("object_mode"):"color";
    const table=node("table"), head=node("thead"), titles=node("tr");
    for(const title of ["Category","Coverage .015 / 256","Close-neighbor samples ≤ .005","Median triangles"]) titles.append(node("th",title));
    head.append(titles); table.append(head);
    const body=node("tbody");
    for(const c of atlas.categories) {
      const tr=node("tr"), td=node("td"), button=node("button",label(c.kind));
      button.type="button"; button.addEventListener("click",()=>{select.value=c.kind; study.value="geometry"; render().catch(failure); select.focus();});
      td.append(button);
      if(assemblies.categories.some(a=>a.kind===c.kind)) {
        const opening=node("button","Opening states");
        opening.type="button"; opening.addEventListener("click",()=>{select.value=c.kind; study.value="assembly"; render(); select.focus();});
        td.append(node("span"," · "),opening);
      }
      tr.append(td); tr.dataset.low=String(c.coverage[1].representatives.length<8);
      tr.append(node("td",c.coverage[1].representatives.length+" / 256"),node("td",c.nearest_shape_distances.filter(d=>d<=.005).length),node("td",number(c.mesh_summary.triangles.median))); body.append(tr);
    }
    for(const c of assemblies.categories.filter(c=>!atlas.categories.some(o=>o.kind===c.kind))) {
      const tr=node("tr"), td=node("td"), button=node("button",label(c.kind)+" · opening states");
      button.type="button"; button.addEventListener("click",()=>{select.value=c.kind; study.value="assembly"; render(); select.focus();});
      td.append(button); tr.append(td,node("td","Not measured"),node("td","Not measured"),node("td","See state parameters")); body.append(tr);
    }
    table.append(body); inventory.append(table);
    select.addEventListener("change",()=>render()); mode.addEventListener("change",()=>render()); study.addEventListener("change",()=>render());
    return render().then(()=>{if(location.hash==="#doors") document.getElementById("objects").scrollIntoView();});
  }).catch(failure);
})();
