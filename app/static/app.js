const $ = (s) => document.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const WORLD = 7; // svg spans +-WORLD world units

const state = {
  preset: "dolly_in",
  dsl: "",          // custom DSL overrides the preset when set
  shareCode: "",    // a pasted code overrides both
  duration: 5,
  strength: 1,
  aspect: "16:9",
  resolution: "448p",
  scene: "set",
  seed: 42,
  file: null,
  imageId: "",
  preview: null,
};
let presets = [];
const presetPreviewCache = new Map();

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `${r.status} ${r.statusText}`);
  return body;
}

// ------------------------------------------------------------ presets

function presetCard(p) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "preset";
  b.dataset.id = p.id;
  b.innerHTML = `<video muted loop playsinline></video><b></b><span></span>`;
  b.querySelector("b").textContent = p.label;
  b.querySelector("span").textContent = p.description;
  b.addEventListener("click", () => selectPreset(p.id));
  const v = b.querySelector("video");
  b.addEventListener("mouseenter", async () => {
    try {
      let url = presetPreviewCache.get(p.id);
      if (!url) {
        const res = await api("/api/preview", jsonBody({ preset: p.id, duration_s: 5, strength: 1 }));
        url = res.video_url;
        presetPreviewCache.set(p.id, url);
      }
      if (!v.src.endsWith(url)) { v.src = url; v.oncanplay = () => v.classList.add("ready"); }
      v.play().catch(() => {});
    } catch { /* hover previews are best effort */ }
  });
  b.addEventListener("mouseleave", () => v.pause());
  return b;
}

function selectPreset(id) {
  state.preset = id;
  state.dsl = "";
  state.shareCode = "";
  document.querySelectorAll(".preset").forEach((el) => el.classList.toggle("on", el.dataset.id === id));
  refreshPreview();
}

// ------------------------------------------------------------ preview

const jsonBody = (o) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(o) });

let previewSeq = 0;
let previewTimer;
function refreshPreview(delay = 0) {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(doPreview, delay);
}

async function doPreview() {
  const seq = ++previewSeq;
  $("#guideBusy").hidden = false;
  $("#dslErr").hidden = true;
  try {
    const res = await api("/api/preview", jsonBody({
      preset: state.preset, dsl: state.dsl, share_code: state.shareCode,
      duration_s: state.duration, strength: state.strength,
    }));
    if (seq !== previewSeq) return;
    state.preview = res;
    if (state.shareCode) {
      // A share code carries its own duration/strength; reflect them in the controls.
      setRange("duration", res.duration_s);
      setRange("strength", res.strength);
    }
    $("#dsl").value = res.dsl;
    $("#code").value = res.share_code;
    $("#custom").hidden = !(state.dsl || state.shareCode);
    const g = $("#guide");
    g.src = res.video_url;
    g.play().catch(() => {});
    drawPlan(res.path);
  } catch (e) {
    if (seq === previewSeq) { $("#dslErr").textContent = e.message; $("#dslErr").hidden = false; }
  } finally {
    if (seq === previewSeq) $("#guideBusy").hidden = true;
  }
}

function setRange(id, v) {
  const el = $("#" + id);
  el.value = v;
  state[id] = Number(v);
  updateOutputs();
}

function updateOutputs() {
  $("#durOut").textContent = `${Number(state.duration).toFixed(1)}s`;
  $("#strOut").textContent = Number(state.strength).toFixed(2);
}

// ------------------------------------------------------------ top-down plan

function el(name, attrs, parent) {
  const n = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  parent?.appendChild(n);
  return n;
}

let planPath = [];
let camMarker, fovWedge;
function drawPlan(path) {
  planPath = path;
  const svg = $("#plan");
  svg.innerHTML = "";
  const s = 1 / WORLD;
  for (const r of [2, 4, 6]) el("circle", { cx: 0, cy: 0, r: r * s, class: "ring" }, svg);
  el("text", { x: 4 * s + 0.02, y: -0.02 }, svg).textContent = "1×";
  el("circle", { cx: 0, cy: 0, r: 0.035, class: "subject" }, svg);
  const pts = path.map(([x, z]) => `${(x * s).toFixed(4)},${(z * s).toFixed(4)}`).join(" ");
  el("polyline", { points: pts, class: "trail" }, svg);
  fovWedge = el("path", { class: "fov" }, svg);
  camMarker = el("circle", { r: 0.03, class: "cam" }, svg);
  placeCam(0);
}

function placeCam(u) {
  if (!planPath.length || !camMarker) return;
  const i = Math.min(planPath.length - 1, Math.max(0, Math.round(u * (planPath.length - 1))));
  const [x, z, hx, hz, fov] = planPath[i];
  const s = 1 / WORLD;
  const cx = x * s, cz = z * s;
  camMarker.setAttribute("cx", cx);
  camMarker.setAttribute("cy", cz);
  const aspect = state.aspect === "16:9" ? 16 / 9 : state.aspect === "9:16" ? 9 / 16 : 1;
  const half = Math.atan(Math.tan((fov * Math.PI) / 360) * aspect);
  const len = 0.32;
  const norm = Math.hypot(hx, hz) || 1;
  const a = Math.atan2(hz / norm, hx / norm);
  const p1 = [cx + len * Math.cos(a - half), cz + len * Math.sin(a - half)];
  const p2 = [cx + len * Math.cos(a + half), cz + len * Math.sin(a + half)];
  fovWedge.setAttribute("d", `M${cx},${cz} L${p1[0]},${p1[1]} L${p2[0]},${p2[1]} Z`);
}

function animatePlan() {
  const g = $("#guide");
  if (g.duration) placeCam(g.currentTime / g.duration);
  requestAnimationFrame(animatePlan);
}

// ------------------------------------------------------------ inputs

function bindInputs() {
  $("#duration").addEventListener("input", (e) => { state.duration = Number(e.target.value); state.shareCode = ""; updateOutputs(); refreshPreview(250); });
  $("#strength").addEventListener("input", (e) => { state.strength = Number(e.target.value); state.shareCode = ""; updateOutputs(); refreshPreview(250); });
  $("#seed").addEventListener("change", (e) => { state.seed = Math.max(0, parseInt(e.target.value || "0", 10)); });
  $("#dice").addEventListener("click", () => { state.seed = Math.floor(Math.random() * 2 ** 31); $("#seed").value = state.seed; });

  document.querySelectorAll(".seg").forEach((seg) => {
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      state[seg.dataset.name] = b.dataset.v;
    });
  });

  const applyDsl = () => {
    const v = $("#dsl").value.trim();
    state.shareCode = "";
    state.dsl = v;
    document.querySelectorAll(".preset").forEach((x) => x.classList.remove("on"));
    refreshPreview();
  };
  $("#apply").addEventListener("click", applyDsl);
  $("#dsl").addEventListener("keydown", (e) => { if (e.key === "Enter") applyDsl(); });

  $("#copy").addEventListener("click", async () => {
    const code = $("#code").value;
    const link = `${location.origin}/?move=${encodeURIComponent(code)}`;
    try { await navigator.clipboard.writeText(link); flash($("#copy"), "Copied link"); }
    catch { $("#code").select(); }
  });
  const loadCode = () => {
    const v = $("#paste").value.trim();
    if (!v) return;
    state.shareCode = v;
    state.dsl = "";
    document.querySelectorAll(".preset").forEach((x) => x.classList.remove("on"));
    refreshPreview();
  };
  $("#load").addEventListener("click", loadCode);
  $("#paste").addEventListener("keydown", (e) => { if (e.key === "Enter") loadCode(); });

  $("#more").addEventListener("click", () => {
    const m = $("#morePresets");
    m.hidden = !m.hidden;
    $("#more").textContent = m.hidden ? "More moves" : "Fewer moves";
  });

  // image upload
  const drop = $("#drop");
  const setFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    state.file = f;
    state.imageId = "";
    const img = $("#thumb");
    img.src = URL.createObjectURL(f);
    img.hidden = false;
    $("#dropText").hidden = true;
  };
  $("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); setFile(e.dataTransfer.files[0]); });

  $("#generate").addEventListener("click", generate);
  $("#remix").addEventListener("click", () => {
    // Keep the exact move: pin the current share code, then ask for a new image.
    state.shareCode = $("#code").value;
    $("#paste").value = state.shareCode;
    $("#file").click();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function flash(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = old), 1200);
}

// ------------------------------------------------------------ jobs

async function generate() {
  $("#jobErr").hidden = true;
  const fd = new FormData();
  if (state.file && !state.imageId) fd.append("image", state.file);
  if (state.imageId) fd.append("image_id", state.imageId);
  fd.append("prompt", $("#prompt").value);
  fd.append("preset", state.preset);
  fd.append("dsl", state.dsl);
  fd.append("share_code", state.shareCode);
  fd.append("duration_s", state.duration);
  fd.append("strength", state.strength);
  fd.append("resolution", state.resolution);
  fd.append("aspect", state.aspect);
  fd.append("scene", state.scene);
  fd.append("seed", state.seed);

  const btn = $("#generate");
  btn.disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    state.imageId = job.params.image_id || state.imageId;
    follow(job);
  } catch (e) {
    $("#jobErr").textContent = e.message;
    $("#jobErr").hidden = false;
    btn.disabled = false;
  }
}

function follow(job) {
  const prog = $("#progress");
  prog.hidden = false;
  const es = new EventSource(`/api/jobs/${job.id}/events`);
  es.onmessage = (ev) => {
    const j = JSON.parse(ev.data);
    prog.querySelector("i").style.width = `${Math.round(j.progress * 100)}%`;
    prog.querySelector(".msg").textContent = `${j.message} · ${Math.round(j.progress * 100)}%`;
    if (j.status === "succeeded" || j.status === "failed") {
      es.close();
      $("#generate").disabled = false;
      if (j.status === "succeeded") showResult(j);
      else { $("#jobErr").textContent = j.error; $("#jobErr").hidden = false; }
      addTake(j);
    }
  };
  es.onerror = () => {
    es.close();
    // Fall back to one poll so a dropped stream doesn't strand the UI.
    setTimeout(async () => {
      const j = await api(`/api/jobs/${job.id}`).catch(() => null);
      if (j && (j.status === "queued" || j.status === "running")) follow(j);
      else { $("#generate").disabled = false; if (j?.status === "succeeded") showResult(j); }
    }, 1000);
  };
}

function showResult(j) {
  $("#result").hidden = false;
  const v = $("#out");
  v.src = j.video_url;
  v.play().catch(() => {});
  $("#download").href = j.video_url;
  $("#download").setAttribute("download", `dollygrip-${j.id}.mp4`);
}

function addTake(j) {
  $("#takesPanel").hidden = false;
  const b = document.createElement("button");
  b.type = "button";
  b.className = "take" + (j.status === "failed" ? " failed" : "");
  const label = j.params.preset || j.params.dsl;
  b.innerHTML = j.video_url ? `<video muted loop playsinline preload="metadata" src="${j.video_url}#t=0.5"></video><span></span>` : `<span></span>`;
  b.querySelector("span").textContent = `${label} · ${j.params.duration_s}s · seed ${j.params.seed}`;
  const v = b.querySelector("video");
  if (v) { b.onmouseenter = () => v.play().catch(() => {}); b.onmouseleave = () => v.pause(); }
  b.onclick = () => { if (j.video_url) showResult(j); $("#paste").value = j.params.share_code; };
  $("#takes").prepend(b);
}

// ------------------------------------------------------------ samples & examples

const EXAMPLES = [
  { clip: "dolly_zoom", label: "Dolly Zoom (Vertigo)", note: "subject holds size, kitchen stretches behind it", verdict: "works" },
  { clip: "crash_zoom_in", label: "Crash Zoom In", note: "holds, then snaps in on the beat of the guide", verdict: "works" },
  { clip: "dolly_in", label: "Dolly In", note: "real push with background parallax", verdict: "works" },
  { clip: "dolly_in_lighthouse", guide: null, label: "Dolly In · landscape", note: "made through this app", verdict: "works" },
  { clip: "orbit_360", label: "360 Orbit", note: "timing transfers, but LTX turns the subject instead of circling the camera", verdict: "partial" },
];

function renderExamples() {
  const box = $("#examples");
  for (const ex of EXAMPLES) {
    const d = document.createElement("div");
    d.className = "ex";
    const guide = ex.guide === null ? "" : `<video class="g" src="/static/examples/${ex.clip}.guide.mp4" muted loop playsinline autoplay></video>`;
    d.innerHTML = `<div class="pair"><video src="/static/examples/${ex.clip}.mp4" muted loop playsinline autoplay preload="metadata"></video>${guide}</div>
      <div class="cap"><b></b><span></span> <span class="verdict"></span></div>`;
    d.querySelector("b").textContent = ex.label;
    d.querySelector(".cap span").textContent = ex.note;
    d.querySelector(".verdict").textContent = ex.verdict === "works" ? "" : "· " + ex.verdict;
    box.appendChild(d);
  }
}

function bindSamples() {
  document.querySelectorAll(".sample").forEach((b) => {
    b.addEventListener("click", async () => {
      const r = await fetch(`/static/samples/${b.dataset.name}.jpg`);
      const blob = await r.blob();
      const f = new File([blob], `${b.dataset.name}.jpg`, { type: "image/jpeg" });
      state.file = f;
      state.imageId = "";
      const img = $("#thumb");
      img.src = URL.createObjectURL(f);
      img.hidden = false;
      $("#dropText").hidden = true;
      if (!$("#prompt").value.trim()) $("#prompt").value = b.dataset.prompt;
      document.querySelectorAll(".sample").forEach((x) => x.classList.toggle("on", x === b));
    });
  });
}

// ------------------------------------------------------------ boot

async function boot() {
  bindInputs();
  bindSamples();
  renderExamples();
  updateOutputs();
  const info = await api("/api/presets");
  presets = info.presets;
  const badge = $("#backend");
  badge.textContent = info.backend === "mock" ? "mock backend" : "LTX-2.5 · live";
  badge.classList.toggle("live", info.backend !== "mock");
  for (const p of presets) (p.mvp ? $("#presets") : $("#morePresets")).appendChild(presetCard(p));

  const move = new URLSearchParams(location.search).get("move");
  if (move) {
    state.shareCode = move;
    $("#paste").value = move;
    refreshPreview();
  } else {
    selectPreset("dolly_zoom");
  }
  const past = await api("/api/jobs").catch(() => ({ jobs: [] }));
  for (const j of past.jobs.reverse()) if (j.status === "succeeded") addTake(j);
  requestAnimationFrame(animatePlan);
}

boot();
