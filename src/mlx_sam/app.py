from __future__ import annotations

import argparse
import base64
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from mlx_sam.overlay import write_mask_overlay_video
from mlx_sam.video_predictor import SAM2VideoPredictor


APP_ROOT = Path("outputs/manual_app")
SESSIONS: dict[str, dict] = {}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>mlx-sam lab</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #11110f;
      --panel: #1a1a17;
      --panel-2: #24241f;
      --line: #38382f;
      --text: #f2efe3;
      --muted: #a7a18e;
      --accent: #ff9f1c;
      --bad: #ff4d6d;
      --good: #19c37d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 360px 1fr;
    }
    aside {
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, #181814, #11110f);
      padding: 18px;
      overflow: auto;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr;
      min-width: 0;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 16px 20px;
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
      max-width: 520px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 14px 0 6px;
    }
    input, select, button {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
    }
    input[type="file"] { padding: 8px; }
    input[type="color"] {
      height: 40px;
      padding: 4px;
      cursor: pointer;
    }
    input[type="range"] {
      padding: 0;
      accent-color: var(--accent);
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: #171008;
      border-color: #ffb247;
      font-weight: 700;
    }
    button.secondary {
      background: var(--panel-2);
      color: var(--text);
      border-color: var(--line);
      font-weight: 500;
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .segmented {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .segmented button.active[data-mode="pos"] {
      background: var(--good);
      border-color: var(--good);
      color: #06170f;
    }
    .segmented button.active[data-mode="neg"] {
      background: var(--bad);
      border-color: var(--bad);
      color: #21060b;
    }
    .stage {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      overflow: hidden;
    }
    .viewer {
      min-width: 0;
      background: #0b0b0a;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .viewer-title {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
    }
    .playhead {
      border-top: 1px solid var(--line);
      padding: 10px 12px 12px;
      display: grid;
      grid-template-columns: 42px 1fr auto;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .playhead button {
      width: 42px;
      height: 32px;
      padding: 0;
      border-radius: 6px;
    }
    .canvas-wrap {
      position: relative;
      display: grid;
      place-items: center;
      min-height: 0;
      padding: 12px;
    }
    .canvas-wrap.busy::after {
      content: "previewing mask";
      position: absolute;
      left: 22px;
      top: 22px;
      border: 1px solid var(--line);
      background: rgba(17, 17, 15, 0.82);
      color: var(--accent);
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 700;
    }
    .media-layer {
      position: relative;
      display: inline-block;
      max-width: 100%;
      max-height: calc(100vh - 150px);
    }
    #videoPlayer, #previewOverlay, canvas {
      max-width: 100%;
      max-height: calc(100vh - 130px);
      background: #050505;
      border-radius: 4px;
    }
    #videoPlayer {
      display: block;
      width: auto;
      height: auto;
    }
    #previewOverlay, #canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }
    #previewOverlay {
      display: block;
      object-fit: contain;
      pointer-events: none;
    }
    #canvas {
      background: transparent;
      pointer-events: none;
      cursor: default;
    }
    #canvas.armed {
      pointer-events: auto;
      cursor: crosshair;
    }
    .points {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      max-height: 200px;
      overflow-y: auto;
    }
    .point {
      display: grid;
      grid-template-columns: 46px 1fr 32px;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
    }
    .point:last-child { border-bottom: 0; }
    .pill {
      border-radius: 999px;
      padding: 3px 7px;
      text-align: center;
      color: #080806;
      font-weight: 700;
    }
    .pill.pos { background: var(--good); }
    .pill.neg { background: var(--bad); }
    .x {
      border: 0;
      background: transparent;
      color: var(--muted);
      padding: 0;
      width: auto;
      font-size: 16px;
    }
    progress {
      width: 100%;
      height: 8px;
      accent-color: var(--accent);
    }
    .meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .stage { grid-template-columns: 1fr; }
      .status { text-align: left; white-space: normal; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>mlx-sam lab</h1>
      <label>Video</label>
      <input id="videoFile" type="file" accept="video/*" />
      <button id="uploadBtn" style="margin-top:10px">Load video</button>

      <label>Model</label>
      <input id="modelId" value="avbiswas/sam2.1-hiera-small-mlx-4bit" />

      <div class="row">
        <div>
          <label>Frame</label>
          <input id="frameIdx" type="number" value="0" min="0" />
        </div>
        <div>
          <label>Yield every</label>
          <input id="yieldEvery" type="number" value="10" min="1" />
        </div>
      </div>

      <div class="row">
        <div>
          <label>Image size</label>
          <input id="imageSize" type="number" value="768" min="256" step="16" />
        </div>
        <div>
          <label>Direction</label>
          <select id="direction">
            <option value="forward">Forward</option>
            <option value="backward">Backward</option>
            <option value="both">Both</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label>Mask color</label>
          <input id="maskColor" type="color" value="#ff1ac2" />
        </div>
        <div>
          <label>Temporal jump</label>
          <input id="frameStep" type="number" value="2" min="1" step="1" />
        </div>
      </div>

      <div class="row">
        <div>
          <label>Precompute video</label>
          <select id="precomputeVideo">
            <option value="false" selected>Off</option>
            <option value="true">On</option>
          </select>
        </div>
        <div>
          <label>Feature batch</label>
          <input id="featureBatchSize" type="number" value="4" min="1" step="1" />
        </div>
      </div>

      <div class="row">
        <div>
          <label>Memory dtype</label>
          <select id="memoryDtype">
            <option value="">fp32</option>
            <option value="bfloat16" selected>bfloat16</option>
            <option value="float16">float16</option>
          </select>
        </div>
        <div>
          <label>Attention dtype</label>
          <select id="attentionDtype">
            <option value="">fp32</option>
            <option value="bfloat16" selected>bfloat16</option>
            <option value="float16">float16</option>
          </select>
        </div>
      </div>

      <label>Point tool</label>
      <div class="segmented">
        <button id="posBtn" class="secondary" data-mode="pos">+ Point</button>
        <button id="negBtn" class="secondary" data-mode="neg">- Point</button>
      </div>

      <div class="row" style="margin-top:12px">
        <button id="loadFrameBtn" class="secondary">Load frame</button>
        <button id="clearBtn" class="secondary">Clear points</button>
      </div>

      <label>
        <input id="overlayEnabled" type="checkbox" checked style="width:auto;margin-right:8px" />
        Overlay
      </label>

      <button id="runBtn" style="margin-top:14px" disabled>Run segmentation</button>
      <progress id="progress" value="0" max="1" style="margin-top:14px"></progress>
      <div id="meta" class="meta">No video loaded.</div>
      <div id="points" class="points"></div>
    </aside>

    <main>
      <header>
        <h1>Manual video segmentation</h1>
        <div id="status" class="status">Load a video, click positive or negative points, then run.</div>
      </header>
      <section class="stage">
        <div class="viewer">
          <div class="viewer-title">
            <span>Video</span>
            <span><span id="toolInfo">playback</span> / <span id="promptInfo">0 points</span> / <span id="streamInfo">idle</span></span>
          </div>
          <div id="promptWrap" class="canvas-wrap">
            <div class="media-layer">
              <video id="videoPlayer" playsinline></video>
              <img id="previewOverlay" alt="" />
              <canvas id="canvas"></canvas>
            </div>
          </div>
          <div class="playhead">
            <button id="sourcePlayBtn" class="secondary" disabled>Play</button>
            <input id="sourcePlayhead" type="range" min="0" max="0" value="0" disabled />
            <span id="sourcePlayheadLabel">0</span>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const els = {
      videoFile: document.getElementById("videoFile"),
      uploadBtn: document.getElementById("uploadBtn"),
      loadFrameBtn: document.getElementById("loadFrameBtn"),
      runBtn: document.getElementById("runBtn"),
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
    function renderPoints() {
      els.points.innerHTML = "";
      els.promptInfo.textContent = `${points.length} point${points.length === 1 ? "" : "s"}`;
      for (const [i, p] of points.entries()) {
        const row = document.createElement("div");
        row.className = "point";
        row.innerHTML = `<span class="pill ${p.label === 1 ? "pos" : "neg"}">${p.label === 1 ? "+" : "-"}</span><span>${Math.round(p.x)}, ${Math.round(p.y)}</span><button class="x">x</button>`;
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
        ctx.fillStyle = p.label === 1 ? "#19c37d" : "#ff4d6d";
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
      points.push({ x, y, label: mode === "pos" ? 1 : 0 });
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
  </script>
</body>
</html>
"""


def _frame_data_url(frame: np.ndarray, quality: int = 86) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _jpeg_bytes(frame: np.ndarray, quality: int = 86) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode frame")
    return encoded.tobytes()


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_idx}")
        return frame
    finally:
        cap.release()


def _video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1,
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        }
    finally:
        cap.release()


def _parse_color(value) -> tuple[int, int, int]:
    if value is None:
        return (194, 26, 255)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 6:
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
            return (b, g, r)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        r, g, b = [max(0, min(255, int(v))) for v in value]
        return (b, g, r)
    raise ValueError(f"Expected color as #rrggbb or [r,g,b], got {value!r}")


def _overlay_frame(frame: np.ndarray, masks: np.ndarray, color: tuple[int, int, int] = (194, 26, 255)) -> np.ndarray:
    mask = masks[0, 0] > 0
    overlay = frame.copy()
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask] = color_arr
    blended = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, 2)
    return blended


def _write_sparse_overlay_video(
    video_path: Path,
    frame_indices: list[int],
    masks: np.ndarray,
    output_path: Path,
    color: tuple[int, int, int],
) -> None:
    if not frame_indices:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = _video_info(video_path)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"],
        (info["width"], info["height"]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    try:
        for idx, mask in zip(frame_indices, masks):
            frame = _read_frame(video_path, idx)
            writer.write(_overlay_frame(frame, mask[None], color=color))
    finally:
        writer.release()


def _load_saved_mask_for_frame(session: dict, frame_idx: int) -> np.ndarray:
    path = session["session_dir"] / "masks.npz"
    if not path.exists():
        raise FileNotFoundError("No final masks are available yet")
    data = np.load(path)
    masks = data["masks"]
    frame_indices = data["frame_indices"].astype(np.int32)
    if frame_indices.size == 0:
        raise ValueError("Final mask artifact contains no frames")
    frame_idx = int(frame_idx)
    right = int(np.searchsorted(frame_indices, frame_idx, side="left"))
    if right < frame_indices.size and int(frame_indices[right]) == frame_idx:
        mask = masks[right]
    elif right <= 0:
        mask = masks[0]
    elif right >= frame_indices.size:
        mask = masks[-1]
    else:
        left = right - 1
        left_idx = int(frame_indices[left])
        right_idx = int(frame_indices[right])
        alpha = (frame_idx - left_idx) / max(right_idx - left_idx, 1)
        mask = (1.0 - alpha) * masks[left] + alpha * masks[right]
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4:
        raise ValueError(f"Unexpected saved mask shape: {mask.shape}")
    return mask.astype(np.float32)


def _json_line(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def _model_cache_key(request: dict) -> tuple:
    return (
        request.get("model_id") or "avbiswas/sam2.1-hiera-small-mlx-4bit",
        int(request.get("image_size") or 768),
        request.get("memory_dtype"),
        request.get("memory_attention_dtype"),
    )


def _get_preview_context(session: dict, request: dict) -> tuple[SAM2VideoPredictor, dict]:
    cache = session.setdefault("preview_cache", {})
    key = _model_cache_key(request)
    cached = cache.get(key)
    if cached is None:
        predictor = SAM2VideoPredictor.from_pretrained(
            key[0],
            image_size=key[1],
            memory_dtype=key[2],
            memory_attention_dtype=key[3],
        )
        state = predictor.init_state(session["video_path"], precompute_image_features=False)
        cached = {"predictor": predictor, "state": state}
        cache[key] = cached
    else:
        predictor = cached["predictor"]
        state = cached["state"]
        predictor.reset_state(state)
    return predictor, state


def _run_one_direction(
    predictor: SAM2VideoPredictor,
    state: dict,
    video_path: Path,
    start_frame_idx: int,
    reverse: bool,
    yield_every: int,
    frame_step: int,
    color: tuple[int, int, int],
    phase: str,
) -> Iterable[tuple[dict, np.ndarray]]:
    total = state["num_frames"]
    max_frames = start_frame_idx + 1 if reverse else total - start_frame_idx
    for event in predictor.stream_in_video(
        state,
        start_frame_idx=start_frame_idx,
        max_frame_num_to_track=max_frames,
        reverse=reverse,
        yield_every=yield_every,
        return_full=True,
        frame_step=frame_step,
    ):
        if event["type"] == "final":
            if event["masks"] is None:
                continue
            yield {
                "type": "direction_final",
                "phase": phase,
                "frame_indices": event["frame_indices"],
            }, event["masks"]
            continue
        if event["type"] != "frame":
            continue
        frame_idx = int(event["frame_idx"])
        frame = _read_frame(video_path, frame_idx)
        overlay = _overlay_frame(frame, event["masks"], color=color)
        yield {
            "type": "frame",
            "phase": phase,
            "frame_idx": frame_idx,
            "obj_ids": event["obj_ids"],
            "overlay": _frame_data_url(overlay),
            "progress": 0.0,
        }, event["masks"]


def create_app():
    app = FastAPI(title="mlx-sam lab")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.post("/api/upload")
    async def upload(video: UploadFile = File(...)):
        APP_ROOT.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        session_dir = APP_ROOT / session_id
        session_dir.mkdir(parents=True)
        suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
        video_path = session_dir / f"input{suffix}"
        with video_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)
        info = _video_info(video_path)
        frame = _read_frame(video_path, 0)
        SESSIONS[session_id] = {"video_path": video_path, "session_dir": session_dir, "preview_cache": {}, **info}
        return JSONResponse({"session_id": session_id, "video_url": f"/api/video/{session_id}", "frame": _frame_data_url(frame), **info})

    @app.get("/api/frame/{session_id}/{frame_idx}")
    def frame(session_id: str, frame_idx: int):
        session = SESSIONS.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        frame_idx = max(0, min(int(frame_idx), int(session["frames"]) - 1))
        frame_bgr = _read_frame(session["video_path"], frame_idx)
        return JSONResponse({"frame_idx": frame_idx, "frame": _frame_data_url(frame_bgr)})

    @app.get("/api/video/{session_id}")
    def video(session_id: str):
        session = SESSIONS.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        return FileResponse(session["video_path"], media_type="video/mp4", filename="input.mp4")

    @app.get("/api/artifact/{session_id}/{name}")
    def artifact(session_id: str, name: str):
        session = SESSIONS.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        if name not in {"masks.npy", "masks.npz", "overlay.mp4"}:
            return JSONResponse({"error": "unknown artifact"}, status_code=404)
        path = session["session_dir"] / name
        if not path.exists():
            return JSONResponse({"error": "artifact not ready"}, status_code=404)
        media_type = "video/mp4" if name.endswith(".mp4") else "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=name)

    @app.get("/api/overlay_frame/{session_id}/{frame_idx}")
    def overlay_frame(session_id: str, frame_idx: int):
        session = SESSIONS.get(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        try:
            frame_idx = max(0, min(int(frame_idx), int(session["frames"]) - 1))
            frame_bgr = _read_frame(session["video_path"], frame_idx)
            mask = _load_saved_mask_for_frame(session, frame_idx)
            color = session.get("last_overlay_color", (194, 26, 255))
            overlay = _overlay_frame(frame_bgr, mask, color=color)
            return Response(
                content=_jpeg_bytes(overlay),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=404)

    @app.post("/api/prompt")
    async def prompt(request: dict):
        started = time.perf_counter()
        try:
            session = SESSIONS[request["session_id"]]
            points = request.get("points") or []
            if not points:
                return JSONResponse({"error": "at least one point is required"}, status_code=400)
            coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
            labels = np.array([p["label"] for p in points], dtype=np.int32)
            if not np.any(labels == 1):
                return JSONResponse({"error": "at least one positive point is required"}, status_code=400)
            color = _parse_color(request.get("color"))
            frame_idx = max(0, min(int(request.get("frame_idx", 0)), int(session["frames"]) - 1))
            predictor, state = _get_preview_context(session, request)
            out_frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                state,
                frame_idx=frame_idx,
                obj_id=1,
                points=coords,
                labels=labels,
                normalize_coords=True,
            )
            frame_bgr = _read_frame(session["video_path"], out_frame_idx)
            overlay = _overlay_frame(frame_bgr, masks, color=color)
            return JSONResponse(
                {
                    "type": "prompt",
                    "frame_idx": int(out_frame_idx),
                    "obj_ids": list(obj_ids),
                    "overlay": _frame_data_url(overlay),
                    "ms": (time.perf_counter() - started) * 1000.0,
                }
            )
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.post("/api/segment")
    async def segment(request: dict):
        def generate():
            started = time.perf_counter()
            try:
                session = SESSIONS[request["session_id"]]
                video_path = session["video_path"]
                session_dir = session["session_dir"]
                frame_idx = int(request.get("frame_idx", 0))
                points = request.get("points") or []
                if not points:
                    raise ValueError("At least one positive or negative point is required")
                coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
                labels = np.array([p["label"] for p in points], dtype=np.int32)
                predictor = SAM2VideoPredictor.from_pretrained(
                    request.get("model_id") or "avbiswas/sam2.1-hiera-small-mlx-4bit",
                    image_size=int(request.get("image_size") or 768),
                    memory_dtype=request.get("memory_dtype"),
                    memory_attention_dtype=request.get("memory_attention_dtype"),
                )
                precompute = bool(request.get("precompute_image_features"))
                feature_batch_size = max(1, int(request.get("feature_batch_size") or 4))
                if precompute:
                    yield _json_line(
                        {
                            "type": "status",
                            "message": f"Precomputing video features with batch size {feature_batch_size}...",
                            "progress": 0.0,
                            "ms": (time.perf_counter() - started) * 1000.0,
                        }
                    )
                state = predictor.init_state(
                    video_path,
                    precompute_image_features=precompute,
                    feature_batch_size=feature_batch_size,
                )
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=coords,
                    labels=labels,
                    normalize_coords=True,
                )
                direction = request.get("direction") or "forward"
                yield_every = max(1, int(request.get("yield_every") or 30))
                frame_step = max(1, int(request.get("frame_step") or 1))
                color = _parse_color(request.get("color"))
                collected: dict[int, np.ndarray] = {}
                phases = []
                if direction in {"backward", "both"}:
                    phases.append(("backward", True))
                if direction in {"forward", "both"}:
                    phases.append(("forward", False))
                if not phases:
                    raise ValueError(f"Unknown direction: {direction}")
                total_steps = sum(frame_idx + 1 if reverse else state["num_frames"] - frame_idx for _, reverse in phases)
                seen = 0
                for phase, reverse in phases:
                    for payload, masks in _run_one_direction(
                        predictor,
                        state,
                        video_path,
                        frame_idx,
                        reverse=reverse,
                        yield_every=yield_every,
                        frame_step=frame_step,
                        color=color,
                        phase=phase,
                    ):
                        if payload["type"] == "direction_final":
                            for out_frame_idx, mask in zip(payload["frame_indices"].tolist(), masks):
                                collected[int(out_frame_idx)] = mask
                            continue
                        collected[int(payload["frame_idx"])] = masks
                        seen += yield_every
                        payload["progress"] = min(0.98, seen / max(total_steps, 1))
                        payload["ms"] = (time.perf_counter() - started) * 1000.0
                        yield _json_line(payload)
                if collected:
                    ordered = sorted(collected)
                    masks = np.stack([collected[idx][:, 0] for idx in ordered], axis=0)
                    mask_path = session_dir / "masks.npz"
                    overlay_path = session_dir / "overlay.mp4"
                    session["last_overlay_color"] = color
                    np.savez(mask_path, masks=masks, frame_indices=np.asarray(ordered, dtype=np.int32))
                    if frame_step == 1 and direction == "forward":
                        write_mask_overlay_video(video_path, masks[:, 0], overlay_path, color=color, limit=len(ordered))
                    else:
                        _write_sparse_overlay_video(video_path, ordered, masks, overlay_path, color=color)
                else:
                    ordered = []
                    mask_path = None
                    overlay_path = None
                yield _json_line(
                    {
                        "type": "final",
                        "is_final": True,
                        "frame_indices": ordered,
                        "mask_path": None if mask_path is None else str(mask_path),
                        "overlay_path": None if overlay_path is None else str(overlay_path),
                        "mask_url": f"/api/artifact/{request['session_id']}/masks.npz" if mask_path else None,
                        "overlay_url": f"/api/artifact/{request['session_id']}/overlay.mp4" if overlay_path else None,
                        "ms": (time.perf_counter() - started) * 1000.0,
                    }
                )
            except Exception as exc:
                yield _json_line({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install app dependencies with: uv sync --extra app") from exc
    uvicorn.run("mlx_sam.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
