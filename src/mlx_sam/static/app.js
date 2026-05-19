const els = {
  videoFile: document.getElementById("videoFile"),
  uploadBtn: document.getElementById("uploadBtn"),
  loadFrameBtn: document.getElementById("loadFrameBtn"),
  runBtn: document.getElementById("runBtn"),
  downloadOverlayBtn: document.getElementById("downloadOverlayBtn"),
  clearBtn: document.getElementById("clearBtn"),
  modelId: document.getElementById("modelId"),
  frameIdx: document.getElementById("frameIdx"),
  sourcePlayBtn: document.getElementById("sourcePlayBtn"),
  sourcePlayhead: document.getElementById("sourcePlayhead"),
  sourcePlayheadLabel: document.getElementById("sourcePlayheadLabel"),
  overlayEnabled: document.getElementById("overlayEnabled"),
  yieldEvery: document.getElementById("yieldEvery"),
  imageSize: document.getElementById("imageSize"),
  maskColor: document.getElementById("maskColor"),
  frameStep: document.getElementById("frameStep"),
  precomputeVideo: document.getElementById("precomputeVideo"),
  featureBatchSize: document.getElementById("featureBatchSize"),
  direction: document.getElementById("direction"),
  memoryDtype: document.getElementById("memoryDtype"),
  attentionDtype: document.getElementById("attentionDtype"),
  objectId: document.getElementById("objectId"),
  objectSelect: document.getElementById("objectSelect"),
  posBtn: document.getElementById("posBtn"),
  negBtn: document.getElementById("negBtn"),
  canvas: document.getElementById("canvas"),
  videoPlayer: document.getElementById("videoPlayer"),
  previewOverlay: document.getElementById("previewOverlay"),
  promptWrap: document.getElementById("promptWrap"),
  points: document.getElementById("points"),
  meta: document.getElementById("meta"),
  status: document.getElementById("status"),
  toolInfo: document.getElementById("toolInfo"),
  promptInfo: document.getElementById("promptInfo"),
  streamInfo: document.getElementById("streamInfo"),
  progress: document.getElementById("progress"),
};
const ctx = els.canvas.getContext("2d");
let session = null;
let overlayVideoUrl = null;
let finalMaskReady = false;
let finalOverlayCache = new Map();
let finalOverlayInFlight = null;
let lastOverlayFrame = null;
let points = [];
let mode = null;
let sourceLoadTimer = null;

function setStatus(text) { els.status.textContent = text; }
function setPromptFrame(idx) {
  els.frameIdx.value = String(idx);
  els.sourcePlayhead.value = String(idx);
  els.sourcePlayheadLabel.textContent = String(idx);
}
function resetOutputFrames() {
  overlayVideoUrl = null;
  finalMaskReady = false;
  finalOverlayCache = new Map();
  finalOverlayInFlight = null;
  lastOverlayFrame = null;
  els.downloadOverlayBtn.href = "#";
  els.downloadOverlayBtn.classList.add("disabled");
  els.downloadOverlayBtn.setAttribute("aria-disabled", "true");
  els.previewOverlay.removeAttribute("src");
  applyOverlayVisibility();
}
function currentFrame() {
  if (!session) return 0;
  return Math.max(0, Math.min(session.frames - 1, Math.floor(els.videoPlayer.currentTime * session.fps + 0.0001)));
}
function syncFrameFromVideo() {
  if (!session) return;
  setPromptFrame(currentFrame());
}
function resizeCanvas() {
  const rect = els.videoPlayer.getBoundingClientRect();
  els.canvas.width = Math.max(1, Math.round(rect.width));
  els.canvas.height = Math.max(1, Math.round(rect.height));
  draw();
}
function applyOverlayVisibility() {
  const show = els.overlayEnabled.checked;
  els.previewOverlay.style.display = show && els.previewOverlay.src ? "block" : "none";
  if (show && finalMaskReady) updateFinalOverlayFrame(currentFrame());
  if (!show) els.previewOverlay.removeAttribute("src");
}
async function updateFinalOverlayFrame(frameIdx) {
  if (!session || !finalMaskReady || !els.overlayEnabled.checked) return;
  const idx = Math.max(0, Math.min(session.frames - 1, Number(frameIdx || 0)));
  if (idx === lastOverlayFrame && els.previewOverlay.src) return;
  lastOverlayFrame = idx;
  if (finalOverlayCache.has(idx)) {
    els.previewOverlay.src = finalOverlayCache.get(idx);
    applyOverlayVisibility();
    return;
  }
  if (finalOverlayInFlight === idx) return;
  finalOverlayInFlight = idx;
  const url = `/api/overlay_frame/${session.session_id}/${idx}`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    finalOverlayCache.set(idx, objectUrl);
    if (lastOverlayFrame === idx && els.overlayEnabled.checked) {
      els.previewOverlay.src = objectUrl;
      applyOverlayVisibility();
    }
  } catch (err) {
    setStatus(String(err));
  } finally {
    if (finalOverlayInFlight === idx) finalOverlayInFlight = null;
  }
}
function setMode(next) {
  mode = mode === next ? null : next;
  els.posBtn.classList.toggle("active", mode === "pos");
  els.negBtn.classList.toggle("active", mode === "neg");
  els.canvas.classList.toggle("armed", mode !== null);
  els.toolInfo.textContent = mode === null ? "playback" : mode === "pos" ? "+ point armed" : "- point armed";
  setStatus(mode === null ? "Playback mode. Click + Point or - Point to annotate." : `${mode === "pos" ? "Positive" : "Negative"} point tool armed.`);
}
function currentObjectId() {
  return Math.max(1, Math.floor(Number(els.objectId.value || 1)));
}
function objectColor(objId, label = 1) {
  if (label === 0) return "#ff4d6d";
  const colors = ["#ff1ac2", "#19c37d", "#2f80ff", "#ffd166", "#8b5cf6", "#00d4ff"];
  return colors[(Math.max(1, objId) - 1) % colors.length];
}
function syncObjectOptions() {
  const ids = [...new Set([1, currentObjectId(), ...points.map((p) => p.obj_id)])].sort((a, b) => a - b);
  const selected = String(currentObjectId());
  els.objectSelect.innerHTML = "";
  for (const id of ids) {
    const option = document.createElement("option");
    option.value = String(id);
    option.textContent = `Object ${id}`;
    if (option.value === selected) option.selected = true;
    els.objectSelect.appendChild(option);
  }
}
function renderPoints() {
  syncObjectOptions();
  els.points.innerHTML = "";
  els.promptInfo.textContent = `${points.length} point${points.length === 1 ? "" : "s"}`;
  for (const [i, p] of points.entries()) {
    const row = document.createElement("div");
    row.className = "point";
    row.innerHTML = `<span class="pill ${p.label === 1 ? "pos" : "neg"}">O${p.obj_id} ${p.label === 1 ? "+" : "-"}</span><span>${Math.round(p.x)}, ${Math.round(p.y)}</span><button class="x">x</button>`;
    row.querySelector(".pill").style.background = objectColor(p.obj_id, p.label);
    row.querySelector("button").onclick = () => {
      points.splice(i, 1);
      draw();
      renderPoints();
      previewPrompt();
    };
    els.points.appendChild(row);
  }
}
function draw() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!session) return;
  const scaleX = els.canvas.width / session.width;
  const scaleY = els.canvas.height / session.height;
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x * scaleX, p.y * scaleY, 8, 0, Math.PI * 2);
    ctx.fillStyle = objectColor(p.obj_id, p.label);
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#11110f";
    ctx.stroke();
  }
}
async function loadSourceFrame(idx, clearPrompt = true, runPreview = true) {
  if (!session) return;
  const frameIdx = Math.max(0, Math.min(Number(idx || 0), session.frames - 1));
  setPromptFrame(frameIdx);
  els.videoPlayer.pause();
  els.videoPlayer.currentTime = frameIdx / session.fps;
  if (clearPrompt) points = [];
  renderPoints();
  if (runPreview) previewPrompt();
  setStatus(`Frame ${frameIdx} loaded.`);
}
async function loadPreview(dataUrl) {
  els.previewOverlay.src = dataUrl;
  applyOverlayVisibility();
}
let previewSeq = 0;
async function previewPrompt() {
  const seq = ++previewSeq;
  if (!session) return;
  if (points.length === 0) {
    els.previewOverlay.removeAttribute("src");
    applyOverlayVisibility();
    draw();
    els.streamInfo.textContent = "idle";
    setStatus("Points cleared.");
    return;
  }
  if (!points.some((p) => p.label === 1)) {
    els.streamInfo.textContent = "needs +";
    setStatus("Add at least one positive point for a mask preview.");
    return;
  }
  els.streamInfo.textContent = "preview";
  els.promptWrap.classList.add("busy");
  setStatus("Previewing selected mask...");
  els.previewOverlay.removeAttribute("src");
  applyOverlayVisibility();
  draw();
  const body = {
    session_id: session.session_id,
    model_id: els.modelId.value.trim(),
    frame_idx: Number(els.frameIdx.value || 0),
    points,
    image_size: Number(els.imageSize.value || 768),
    color: els.maskColor.value,
    memory_dtype: els.memoryDtype.value || null,
    memory_attention_dtype: els.attentionDtype.value || null,
  };
  try {
    const res = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (seq !== previewSeq) return;
    if (!res.ok) throw new Error(data.error || JSON.stringify(data));
    await loadPreview(data.overlay);
    els.streamInfo.textContent = `frame ${data.frame_idx}`;
    setStatus(`Preview ready in ${data.ms.toFixed(1)} ms.`);
  } catch (err) {
    if (seq !== previewSeq) return;
    els.streamInfo.textContent = "preview failed";
    setStatus(String(err));
  } finally {
    if (seq === previewSeq) els.promptWrap.classList.remove("busy");
  }
}
els.posBtn.onclick = () => setMode("pos");
els.negBtn.onclick = () => setMode("neg");
els.objectId.onchange = () => {
  els.objectId.value = String(currentObjectId());
  syncObjectOptions();
};
els.objectSelect.onchange = () => {
  els.objectId.value = els.objectSelect.value;
  syncObjectOptions();
};
els.clearBtn.onclick = () => {
  points = [];
  mode = null;
  els.posBtn.classList.remove("active");
  els.negBtn.classList.remove("active");
  els.canvas.classList.remove("armed");
  els.toolInfo.textContent = "playback";
  draw();
  renderPoints();
  previewPrompt();
};
els.canvas.onclick = (event) => {
  if (!session || mode === null) return;
  els.videoPlayer.pause();
  syncFrameFromVideo();
  const rect = els.canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (session.width / rect.width);
  const y = (event.clientY - rect.top) * (session.height / rect.height);
  points.push({ x, y, label: mode === "pos" ? 1 : 0, obj_id: currentObjectId() });
  mode = null;
  els.posBtn.classList.remove("active");
  els.negBtn.classList.remove("active");
  els.canvas.classList.remove("armed");
  els.toolInfo.textContent = "playback";
  draw();
  renderPoints();
  previewPrompt();
};
els.uploadBtn.onclick = async () => {
  const file = els.videoFile.files[0];
  if (!file) return setStatus("Choose a video first.");
  const form = new FormData();
  form.append("video", file);
  setStatus("Uploading video...");
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  session = await res.json();
  overlayVideoUrl = null;
  els.videoPlayer.src = session.video_url;
  els.frameIdx.max = Math.max(0, session.frames - 1);
  els.sourcePlayhead.max = Math.max(0, session.frames - 1);
  els.sourcePlayhead.disabled = false;
  els.sourcePlayBtn.disabled = false;
  els.meta.textContent = `${session.width}x${session.height}, ${session.frames} frames, ${session.fps.toFixed(2)} FPS`;
  points = [];
  resetOutputFrames();
  setPromptFrame(0);
  els.videoPlayer.currentTime = 0;
  resizeCanvas();
  renderPoints();
  previewPrompt();
  els.runBtn.disabled = false;
  setStatus("Video loaded. Click points on the prompt frame.");
};
els.loadFrameBtn.onclick = async () => {
  if (!session) return;
  await loadSourceFrame(Number(els.frameIdx.value || 0), true);
};
els.frameIdx.onchange = () => {
  if (!session) return;
  loadSourceFrame(Number(els.frameIdx.value || 0), true);
};
els.sourcePlayhead.oninput = () => {
  const idx = Number(els.sourcePlayhead.value || 0);
  setPromptFrame(idx);
  if (sourceLoadTimer) clearTimeout(sourceLoadTimer);
  sourceLoadTimer = setTimeout(() => loadSourceFrame(idx, true), 120);
};
els.sourcePlayBtn.onclick = () => {
  if (!session) return;
  if (els.videoPlayer.paused) {
    els.videoPlayer.play().catch(() => {});
  } else {
    els.videoPlayer.pause();
  }
};
els.downloadOverlayBtn.onclick = (event) => {
  if (!overlayVideoUrl) {
    event.preventDefault();
    setStatus("Run segmentation first to generate an overlay video.");
  }
};
els.overlayEnabled.onchange = applyOverlayVisibility;
els.videoPlayer.onloadedmetadata = resizeCanvas;
els.videoPlayer.onresize = resizeCanvas;
els.videoPlayer.onplay = () => {
  els.sourcePlayBtn.textContent = "Stop";
  applyOverlayVisibility();
  draw();
};
els.videoPlayer.onpause = () => {
  els.sourcePlayBtn.textContent = "Play";
  syncFrameFromVideo();
  applyOverlayVisibility();
  draw();
};
els.videoPlayer.ontimeupdate = () => {
  syncFrameFromVideo();
  if (finalMaskReady) updateFinalOverlayFrame(currentFrame());
};
window.addEventListener("resize", resizeCanvas);
els.runBtn.onclick = async () => {
  if (!session || points.length === 0) return setStatus("Add at least one point.");
  els.videoPlayer.pause();
  els.runBtn.disabled = true;
  els.progress.value = 0;
  els.streamInfo.textContent = "starting";
  resetOutputFrames();
  setStatus("Starting segmentation...");
  const body = {
    session_id: session.session_id,
    model_id: els.modelId.value.trim(),
    frame_idx: Number(els.frameIdx.value || 0),
    points,
    image_size: Number(els.imageSize.value || 768),
    direction: els.direction.value,
    yield_every: Number(els.yieldEvery.value || 30),
    frame_step: Number(els.frameStep.value || 2),
    precompute_image_features: els.precomputeVideo.value === "true",
    feature_batch_size: Number(els.featureBatchSize.value || 4),
    color: els.maskColor.value,
    memory_dtype: els.memoryDtype.value || null,
    memory_attention_dtype: els.attentionDtype.value || null,
  };
  const res = await fetch("/api/segment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    els.runBtn.disabled = false;
    throw new Error(await res.text());
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "frame") {
        if (els.overlayEnabled.checked) await loadPreview(event.overlay);
        els.progress.value = event.progress;
        els.streamInfo.textContent = `frame ${event.frame_idx}`;
        setStatus(`Streaming ${event.phase}: frame ${event.frame_idx}, ${event.ms.toFixed(1)} ms`);
      } else if (event.type === "final") {
        els.progress.value = 1;
        els.streamInfo.textContent = "done";
        overlayVideoUrl = event.overlay_url;
        if (overlayVideoUrl) {
          els.downloadOverlayBtn.href = overlayVideoUrl;
          els.downloadOverlayBtn.classList.remove("disabled");
          els.downloadOverlayBtn.setAttribute("aria-disabled", "false");
        }
        finalMaskReady = true;
        finalOverlayCache = new Map();
        lastOverlayFrame = null;
        applyOverlayVisibility();
        const overlay = event.overlay_url ? ` Overlay: ${event.overlay_url}` : "";
        setStatus(`Done. Masks: ${event.mask_url}.${overlay}`);
      } else if (event.type === "status") {
        els.progress.value = event.progress || 0;
        els.streamInfo.textContent = "precompute";
        setStatus(event.message);
      } else if (event.type === "error") {
        setStatus(event.message);
      }
    }
  }
  els.runBtn.disabled = false;
};
