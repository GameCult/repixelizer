"use strict";
function byId(id) {
    const element = document.getElementById(id);
    if (!element) {
        throw new Error(`Missing element: ${id}`);
    }
    return element;
}
const fileInput = byId("fileInput");
const dropzone = byId("dropzone");
const dropzoneLabel = byId("dropzoneLabel");
const inputPreviewCanvas = byId("inputPreviewCanvas");
const pickFileButton = byId("pickFileButton");
const runButton = byId("runButton");
const statusBadge = byId("statusBadge");
const statusText = byId("statusText");
const inferenceSummary = byId("inferenceSummary");
const candidateList = byId("candidateList");
const eventLog = byId("eventLog");
const stepSlider = byId("stepSlider");
const stepValue = byId("stepValue");
const lossCanvas = byId("lossCanvas");
const leftCanvas = byId("leftCanvas");
const rightCanvas = byId("rightCanvas");
const leftVizLabel = byId("leftVizLabel");
const rightVizLabel = byId("rightVizLabel");
const metricsPanel = byId("metricsPanel");
const paintSwatch = byId("paintSwatch");
const zoomInput = byId("zoomInput");
const zoomValue = byId("zoomValue");
const gridToggle = byId("gridToggle");
const editorCanvas = byId("editorCanvas");
const editorGridCanvas = byId("editorGridCanvas");
const editorSurface = byId("editorSurface");
const editorMeta = byId("editorMeta");
const summaryPanel = byId("summaryPanel");
const resetEditorButton = byId("resetEditorButton");
const downloadButton = byId("downloadButton");
const targetSizeInput = byId("targetSizeInput");
const targetWidthInput = byId("targetWidthInput");
const targetHeightInput = byId("targetHeightInput");
const phaseXInput = byId("phaseXInput");
const phaseYInput = byId("phaseYInput");
const stepsInput = byId("stepsInput");
const seedInput = byId("seedInput");
const deviceInput = byId("deviceInput");
const stripBackgroundInput = byId("stripBackgroundInput");
const skipRerankInput = byId("skipRerankInput");
const paintInputs = {
    r: byId("paintR"),
    g: byId("paintG"),
    b: byId("paintB"),
    a: byId("paintA"),
};
const imageCache = new Map();
const eventTypes = [
    "job_state",
    "job_failed",
    "source_loaded",
    "preprocess_completed",
    "inference_candidates_ready",
    "phase_rerank_started",
    "phase_selection_completed",
    "analysis_completed",
    "phase_field_prepared",
    "phase_field_initial",
    "phase_field_step",
    "phase_field_final",
    "cleanup_completed",
    "palette_completed",
    "pipeline_completed",
];
const state = {
    file: null,
    jobId: null,
    status: "idle",
    statusText: "Waiting for input.",
    inputPreviewImage: null,
    sourceImage: null,
    preprocessedImage: null,
    latticeImage: null,
    guidanceImage: null,
    inference: null,
    frames: [],
    cleanupImage: null,
    heatmapImage: null,
    finalOutputImage: null,
    runSummary: null,
    selectedFrameIndex: 0,
    autoFollow: true,
    eventLog: [],
    paintColor: [255, 255, 255, 255],
    editorBaseAsset: null,
    editorDirty: false,
    altHeld: false,
    zoom: Number(zoomInput.value),
    showGrid: gridToggle.checked,
};
let currentEventSource = null;
let painting = false;
let panning = false;
let activePanPointerId = null;
let panStartClientX = 0;
let panStartClientY = 0;
let panStartScrollLeft = 0;
let panStartScrollTop = 0;
let offscreenCanvas = null;
let offscreenContext = null;
let singlePixel = new ImageData(1, 1);
function clampByte(value) {
    return Math.max(0, Math.min(255, Math.round(value)));
}
function clampZoom(value) {
    const minZoom = Number(zoomInput.min) || 1;
    const maxZoom = Number(zoomInput.max) || 64;
    return Math.max(minZoom, Math.min(maxZoom, Math.round(value)));
}
function syncEditorCursorState() {
    editorSurface.classList.toggle("is-panning", panning);
    editorSurface.classList.toggle("is-eyedropper", state.altHeld);
}
function setPaintColor(color) {
    state.paintColor = color;
    paintInputs.r.value = String(color[0]);
    paintInputs.g.value = String(color[1]);
    paintInputs.b.value = String(color[2]);
    paintInputs.a.value = String(color[3]);
    paintSwatch.style.background = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3] / 255})`;
}
function addLog(label, detail) {
    state.eventLog.unshift({ label, detail });
    state.eventLog = state.eventLog.slice(0, 8);
    renderEventLog();
}
function renderEventLog() {
    eventLog.innerHTML = "";
    if (state.eventLog.length === 0) {
        eventLog.innerHTML = `<p class="muted">Nothing to report yet.</p>`;
        return;
    }
    for (const entry of state.eventLog) {
        const node = document.createElement("div");
        node.className = "event-item";
        node.innerHTML = `<strong>${entry.label}</strong><span>${entry.detail}</span>`;
        eventLog.appendChild(node);
    }
}
function setStatus(status, detail) {
    state.status = status;
    state.statusText = detail;
    statusBadge.textContent = status;
    statusText.textContent = detail;
}
function resetRunArtifacts() {
    state.jobId = null;
    state.sourceImage = null;
    state.preprocessedImage = null;
    state.latticeImage = null;
    state.guidanceImage = null;
    state.inference = null;
    state.frames = [];
    state.cleanupImage = null;
    state.heatmapImage = null;
    state.finalOutputImage = null;
    state.runSummary = null;
    state.selectedFrameIndex = 0;
    state.autoFollow = true;
    state.eventLog = [];
    state.editorBaseAsset = null;
    state.editorDirty = false;
}
function parseOptionalInteger(input) {
    if (input.value.trim() === "") {
        return null;
    }
    const value = Number(input.value);
    return Number.isFinite(value) ? Math.round(value) : null;
}
function parseOptionalFloat(input) {
    if (input.value.trim() === "") {
        return null;
    }
    const value = Number(input.value);
    return Number.isFinite(value) ? value : null;
}
async function loadImage(asset) {
    const cached = imageCache.get(asset.dataUrl);
    if (cached) {
        return cached;
    }
    const promise = new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("Failed to load image asset"));
        image.src = asset.dataUrl;
    });
    imageCache.set(asset.dataUrl, promise);
    return promise;
}
async function drawAsset(canvas, asset) {
    const context = canvas.getContext("2d");
    if (!context) {
        return;
    }
    if (!asset) {
        canvas.width = 640;
        canvas.height = 360;
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = "#081012";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = "rgba(255,255,255,0.2)";
        context.font = "16px sans-serif";
        context.fillText("Waiting for image data", 24, 32);
        return;
    }
    const image = await loadImage(asset);
    canvas.width = image.naturalWidth || asset.width;
    canvas.height = image.naturalHeight || asset.height;
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
}
async function fileToImageAsset(file) {
    const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            if (typeof reader.result !== "string") {
                reject(new Error("Failed to read image preview."));
                return;
            }
            resolve(reader.result);
        };
        reader.onerror = () => reject(reader.error ?? new Error("Failed to read image preview."));
        reader.readAsDataURL(file);
    });
    const image = await loadImage({ dataUrl, width: 1, height: 1 });
    return {
        dataUrl,
        width: image.naturalWidth || 1,
        height: image.naturalHeight || 1,
    };
}
async function renderInputPreview() {
    const context = inputPreviewCanvas.getContext("2d");
    if (!context) {
        return;
    }
    const asset = state.inputPreviewImage;
    if (!asset) {
        inputPreviewCanvas.width = 320;
        inputPreviewCanvas.height = 180;
        context.clearRect(0, 0, inputPreviewCanvas.width, inputPreviewCanvas.height);
        context.fillStyle = "#081012";
        context.fillRect(0, 0, inputPreviewCanvas.width, inputPreviewCanvas.height);
        context.fillStyle = "rgba(255,255,255,0.2)";
        context.font = "16px sans-serif";
        context.fillText("No source loaded yet", 20, 32);
        return;
    }
    const image = await loadImage(asset);
    inputPreviewCanvas.width = image.naturalWidth || asset.width;
    inputPreviewCanvas.height = image.naturalHeight || asset.height;
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, inputPreviewCanvas.width, inputPreviewCanvas.height);
    context.drawImage(image, 0, 0, inputPreviewCanvas.width, inputPreviewCanvas.height);
}
function readNestedNumber(root, path) {
    let current = root;
    for (const key of path) {
        if (!current || typeof current !== "object" || !(key in current)) {
            return null;
        }
        current = current[key];
    }
    return typeof current === "number" ? current : null;
}
function formatNumber(value, digits = 3) {
    if (value === null || !Number.isFinite(value)) {
        return "n/a";
    }
    return value.toFixed(digits);
}
function getSelectedFrame() {
    if (state.frames.length === 0) {
        return null;
    }
    return state.frames[Math.max(0, Math.min(state.selectedFrameIndex, state.frames.length - 1))];
}
function renderInference() {
    if (!state.inference) {
        inferenceSummary.innerHTML = `<p class="muted">No lattice picked yet.</p>`;
        candidateList.innerHTML = "";
        return;
    }
    inferenceSummary.innerHTML = `
    <div class="summary-card">
      <strong>Chosen Grid</strong>
      <span>${state.inference.target_width} x ${state.inference.target_height}</span>
    </div>
    <div class="summary-card">
      <strong>Phase</strong>
      <span>${formatNumber(state.inference.phase_x, 2)}, ${formatNumber(state.inference.phase_y, 2)}</span>
    </div>
    <div class="summary-card">
      <strong>Confidence</strong>
      <span>${formatNumber(state.inference.confidence, 3)}</span>
    </div>
  `;
    candidateList.innerHTML = "";
    const topCandidates = state.inference.top_candidates.slice(0, 6);
    for (const candidate of topCandidates) {
        const node = document.createElement("div");
        node.className = "candidate-card";
        const rerank = candidate.breakdown["phase_rerank_rank"];
        node.innerHTML = `
      <strong>${candidate.target_width} x ${candidate.target_height}</strong>
      <span>phase ${formatNumber(candidate.phase_x, 2)}, ${formatNumber(candidate.phase_y, 2)}</span><br />
      <span>score ${formatNumber(candidate.score, 3)}${typeof rerank === "number" ? ` • rerank #${Math.round(rerank)}` : ""}</span>
    `;
        candidateList.appendChild(node);
    }
}
function renderMetrics() {
    metricsPanel.querySelectorAll("[data-dynamic='true']").forEach((node) => {
        node.remove();
    });
    const frame = getSelectedFrame();
    const items = [];
    if (frame) {
        items.push(["Loss", frame.loss === null ? "n/a" : formatNumber(frame.loss, 4)]);
        for (const [key, value] of Object.entries(frame.terms)) {
            items.push([key.replaceAll("_", " "), formatNumber(value, 4)]);
        }
    }
    else if (state.runSummary) {
        items.push(["Structure score", formatNumber(readNestedNumber(state.runSummary, ["source_structure", "score"]))]);
        items.push([
            "Final fidelity",
            formatNumber(readNestedNumber(state.runSummary, ["source_fidelity", "final_output", "score"])),
        ]);
        items.push(["Source color ratio", formatNumber(readNestedNumber(state.runSummary, ["output_colors_from_source_ratio"]))]);
    }
    for (const [label, value] of items) {
        const node = document.createElement("div");
        node.className = "metric-chip";
        node.dataset.dynamic = "true";
        node.innerHTML = `<strong>${label}</strong><span>${value}</span>`;
        metricsPanel.appendChild(node);
    }
}
function renderSummary() {
    summaryPanel.innerHTML = "";
    if (!state.runSummary) {
        summaryPanel.innerHTML = `<p class="muted">Final metrics show up here after the run finishes.</p>`;
        return;
    }
    const entries = [
        ["Structure score", readNestedNumber(state.runSummary, ["source_structure", "score"])],
        ["Edge F1", readNestedNumber(state.runSummary, ["source_structure", "edge_f1"])],
        ["Exact match", readNestedNumber(state.runSummary, ["source_structure", "exact_match_ratio"])],
        ["Final fidelity", readNestedNumber(state.runSummary, ["source_fidelity", "final_output", "score"])],
        ["Mean displacement", readNestedNumber(state.runSummary, ["optimizer_displacement", "final_output", "mean_magnitude_px"])],
    ];
    for (const [label, value] of entries) {
        const card = document.createElement("div");
        card.className = "summary-card";
        card.innerHTML = `<strong>${label}</strong><span>${formatNumber(value, 3)}</span>`;
        summaryPanel.appendChild(card);
    }
}
async function renderViewer() {
    const frame = getSelectedFrame();
    let leftAsset = state.preprocessedImage ?? state.sourceImage;
    let rightAsset = state.finalOutputImage ?? state.cleanupImage ?? state.sourceImage;
    let leftLabel = "Source";
    let rightLabel = "Output";
    if (state.latticeImage) {
        leftAsset = state.latticeImage;
        leftLabel = "Lattice Prep";
    }
    if (state.guidanceImage) {
        leftAsset = state.guidanceImage;
        leftLabel = "Guidance";
    }
    if (frame) {
        leftAsset = frame.samplingOverlayImage;
        rightAsset = frame.outputImage;
        leftLabel = "Sampling Overlay";
        rightLabel = "Current Output";
    }
    else if (state.cleanupImage) {
        if (state.heatmapImage) {
            leftAsset = state.heatmapImage;
            leftLabel = "Cleanup Heatmap";
        }
        rightAsset = state.cleanupImage;
        rightLabel = "Cleaned Output";
    }
    leftVizLabel.textContent = leftLabel;
    rightVizLabel.textContent = rightLabel;
    await Promise.all([drawAsset(leftCanvas, leftAsset), drawAsset(rightCanvas, rightAsset)]);
    renderMetrics();
}
function renderSlider() {
    if (state.frames.length === 0) {
        stepSlider.disabled = true;
        stepSlider.max = "0";
        stepSlider.value = "0";
        stepValue.textContent = "No frames yet";
        return;
    }
    stepSlider.disabled = false;
    stepSlider.max = String(state.frames.length - 1);
    stepSlider.value = String(state.selectedFrameIndex);
    const frame = getSelectedFrame();
    if (!frame) {
        stepValue.textContent = "No frames yet";
        return;
    }
    stepValue.textContent = `Frame ${frame.step} / ${frame.totalSteps}${state.autoFollow ? " • following live" : ""}`;
}
function renderLossChart() {
    const context = lossCanvas.getContext("2d");
    if (!context) {
        return;
    }
    const width = lossCanvas.width;
    const height = lossCanvas.height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "rgba(7, 15, 17, 0.96)";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "rgba(255,255,255,0.08)";
    context.lineWidth = 1;
    for (let index = 1; index <= 4; index += 1) {
        const y = (height / 5) * index;
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
    }
    const losses = state.frames
        .filter((frame) => typeof frame.loss === "number")
        .map((frame) => frame.loss);
    if (losses.length === 0) {
        context.fillStyle = "rgba(255,255,255,0.28)";
        context.font = "14px sans-serif";
        context.fillText("Loss curve wakes up once the solver starts moving.", 18, 28);
        return;
    }
    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const span = Math.max(1e-6, maxLoss - minLoss);
    context.strokeStyle = "#8fe2c8";
    context.lineWidth = 3;
    context.beginPath();
    losses.forEach((loss, index) => {
        const x = losses.length === 1 ? width / 2 : (index / (losses.length - 1)) * (width - 24) + 12;
        const y = height - 18 - ((loss - minLoss) / span) * (height - 36);
        if (index === 0) {
            context.moveTo(x, y);
        }
        else {
            context.lineTo(x, y);
        }
    });
    context.stroke();
    const selected = getSelectedFrame();
    if (selected && selected.loss !== null) {
        const matchingIndex = losses.findIndex((value) => Math.abs(value - selected.loss) < 1e-6);
        if (matchingIndex >= 0) {
            const x = losses.length === 1 ? width / 2 : (matchingIndex / (losses.length - 1)) * (width - 24) + 12;
            const y = height - 18 - ((selected.loss - minLoss) / span) * (height - 36);
            context.fillStyle = "#f5b76b";
            context.beginPath();
            context.arc(x, y, 5, 0, Math.PI * 2);
            context.fill();
        }
    }
}
async function loadEditorAsset(asset) {
    state.editorBaseAsset = asset;
    state.editorDirty = false;
    if (!asset) {
        offscreenCanvas = null;
        offscreenContext = null;
        renderEditor();
        return;
    }
    const image = await loadImage(asset);
    offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = image.naturalWidth || asset.width;
    offscreenCanvas.height = image.naturalHeight || asset.height;
    offscreenContext = offscreenCanvas.getContext("2d");
    if (!offscreenContext) {
        return;
    }
    offscreenContext.imageSmoothingEnabled = false;
    offscreenContext.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
    offscreenContext.drawImage(image, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    renderEditor();
}
function renderEditor() {
    const displayContext = editorCanvas.getContext("2d");
    const gridContext = editorGridCanvas.getContext("2d");
    if (!displayContext || !gridContext) {
        return;
    }
    if (!offscreenCanvas) {
        editorCanvas.width = 1;
        editorCanvas.height = 1;
        editorGridCanvas.width = 1;
        editorGridCanvas.height = 1;
        editorMeta.innerHTML = `<p class="muted">Final output lands here once the solver finishes its little pilgrimage.</p>`;
        return;
    }
    const width = offscreenCanvas.width * state.zoom;
    const height = offscreenCanvas.height * state.zoom;
    for (const canvas of [editorCanvas, editorGridCanvas]) {
        canvas.width = width;
        canvas.height = height;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
    }
    displayContext.imageSmoothingEnabled = false;
    displayContext.clearRect(0, 0, width, height);
    displayContext.drawImage(offscreenCanvas, 0, 0, width, height);
    gridContext.clearRect(0, 0, width, height);
    if (state.showGrid && state.zoom >= 8) {
        gridContext.strokeStyle = "rgba(255,255,255,0.12)";
        gridContext.lineWidth = 1;
        for (let x = 0; x <= offscreenCanvas.width; x += 1) {
            const xpos = x * state.zoom + 0.5;
            gridContext.beginPath();
            gridContext.moveTo(xpos, 0);
            gridContext.lineTo(xpos, height);
            gridContext.stroke();
        }
        for (let y = 0; y <= offscreenCanvas.height; y += 1) {
            const ypos = y * state.zoom + 0.5;
            gridContext.beginPath();
            gridContext.moveTo(0, ypos);
            gridContext.lineTo(width, ypos);
            gridContext.stroke();
        }
    }
    editorMeta.innerHTML = `
    <div class="summary-card">
      <strong>Canvas</strong>
      <span>${offscreenCanvas.width} x ${offscreenCanvas.height}</span>
    </div>
    <div class="summary-card">
      <strong>State</strong>
      <span>${state.editorDirty ? "Edited" : "Matches solver output"}</span>
    </div>
  `;
}
function setZoom(nextZoom, anchor = null) {
    const clampedZoom = clampZoom(nextZoom);
    const previousZoom = state.zoom;
    if (clampedZoom === previousZoom) {
        zoomInput.value = String(clampedZoom);
        zoomValue.textContent = `${clampedZoom}x`;
        return;
    }
    let viewportX = 0;
    let viewportY = 0;
    let imageX = null;
    let imageY = null;
    if (offscreenCanvas && anchor) {
        const surfaceRect = editorSurface.getBoundingClientRect();
        viewportX = anchor.clientX - surfaceRect.left;
        viewportY = anchor.clientY - surfaceRect.top;
        imageX = (editorSurface.scrollLeft + viewportX - editorCanvas.offsetLeft) / previousZoom;
        imageY = (editorSurface.scrollTop + viewportY - editorCanvas.offsetTop) / previousZoom;
    }
    state.zoom = clampedZoom;
    zoomInput.value = String(clampedZoom);
    zoomValue.textContent = `${clampedZoom}x`;
    renderEditor();
    if (offscreenCanvas && imageX !== null && imageY !== null) {
        const nextScrollLeft = imageX * clampedZoom + editorCanvas.offsetLeft - viewportX;
        const nextScrollTop = imageY * clampedZoom + editorCanvas.offsetTop - viewportY;
        const maxScrollLeft = Math.max(0, editorSurface.scrollWidth - editorSurface.clientWidth);
        const maxScrollTop = Math.max(0, editorSurface.scrollHeight - editorSurface.clientHeight);
        editorSurface.scrollLeft = Math.max(0, Math.min(maxScrollLeft, nextScrollLeft));
        editorSurface.scrollTop = Math.max(0, Math.min(maxScrollTop, nextScrollTop));
    }
}
function editorPixelFromPointer(event) {
    if (!offscreenCanvas) {
        return null;
    }
    const rect = editorCanvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
        return null;
    }
    const x = Math.floor((event.clientX - rect.left) / state.zoom);
    const y = Math.floor((event.clientY - rect.top) / state.zoom);
    if (x < 0 || y < 0 || x >= offscreenCanvas.width || y >= offscreenCanvas.height) {
        return null;
    }
    return { x, y };
}
function samplePaintColor(x, y) {
    if (!offscreenContext) {
        return;
    }
    const pixel = offscreenContext.getImageData(x, y, 1, 1).data;
    setPaintColor([pixel[0], pixel[1], pixel[2], pixel[3]]);
}
function paintPixel(x, y) {
    if (!offscreenContext) {
        return;
    }
    singlePixel.data[0] = state.paintColor[0];
    singlePixel.data[1] = state.paintColor[1];
    singlePixel.data[2] = state.paintColor[2];
    singlePixel.data[3] = state.paintColor[3];
    offscreenContext.putImageData(singlePixel, x, y);
    state.editorDirty = true;
    renderEditor();
}
function exportEditorPng() {
    if (!offscreenCanvas) {
        return;
    }
    const link = document.createElement("a");
    link.href = offscreenCanvas.toDataURL("image/png");
    link.download = (state.file?.name?.replace(/\.[^.]+$/, "") || "repixelized") + "-edited.png";
    link.click();
}
function beginPan(event) {
    panning = true;
    activePanPointerId = event.pointerId;
    panStartClientX = event.clientX;
    panStartClientY = event.clientY;
    panStartScrollLeft = editorSurface.scrollLeft;
    panStartScrollTop = editorSurface.scrollTop;
    editorSurface.setPointerCapture(event.pointerId);
    syncEditorCursorState();
}
function endPan(pointerId = null) {
    if (pointerId !== null && activePanPointerId !== pointerId) {
        return;
    }
    if (activePanPointerId !== null && editorSurface.hasPointerCapture(activePanPointerId)) {
        editorSurface.releasePointerCapture(activePanPointerId);
    }
    panning = false;
    activePanPointerId = null;
    syncEditorCursorState();
}
function buildFormData() {
    if (!state.file) {
        throw new Error("No file selected");
    }
    const data = new FormData();
    data.set("image", state.file);
    const values = [
        ["target_size", parseOptionalInteger(targetSizeInput)?.toString() ?? null],
        ["target_width", parseOptionalInteger(targetWidthInput)?.toString() ?? null],
        ["target_height", parseOptionalInteger(targetHeightInput)?.toString() ?? null],
        ["phase_x", parseOptionalFloat(phaseXInput)?.toString() ?? null],
        ["phase_y", parseOptionalFloat(phaseYInput)?.toString() ?? null],
        ["steps", String(parseOptionalInteger(stepsInput) ?? 48)],
        ["seed", String(parseOptionalInteger(seedInput) ?? 7)],
        ["device", deviceInput.value],
        ["strip_background", stripBackgroundInput.checked ? "true" : "false"],
        ["skip_phase_rerank", skipRerankInput.checked ? "true" : "false"],
    ];
    for (const [key, value] of values) {
        if (value !== null) {
            data.set(key, value);
        }
    }
    return data;
}
async function renderEverything() {
    renderInference();
    renderSlider();
    renderLossChart();
    renderSummary();
    await renderViewer();
}
async function handleEvent(eventName, payload) {
    switch (eventName) {
        case "job_state":
            setStatus(String(payload.status ?? "working"), payload.status === "completed" ? "Run complete." : "Running.");
            break;
        case "job_failed":
            setStatus("failed", String(payload.message ?? "The GUI run fell over."));
            addLog("Failure", String(payload.message ?? "The run failed."));
            runButton.disabled = false;
            currentEventSource?.close();
            break;
        case "source_loaded":
            state.sourceImage = payload.sourceImage;
            setStatus("running", "Source loaded. Survey crew is measuring the mess.");
            addLog("Source", "Loaded the input image.");
            break;
        case "preprocess_completed":
            state.preprocessedImage = payload.sourceImage;
            addLog("Preprocess", "Stripped edge-connected background noise.");
            break;
        case "inference_candidates_ready":
            state.inference = payload.inference;
            addLog("Inference", "Scored candidate grids and phase offsets.");
            break;
        case "phase_rerank_started":
            addLog("Rerank", `Running ${String(payload.previewSteps)} preview steps across ${String(payload.candidateCount)} low-confidence candidates.`);
            break;
        case "phase_selection_completed":
            state.inference = payload.inference;
            addLog("Selection", "Committed to a ruler and phase.");
            break;
        case "analysis_completed":
            addLog("Scout", "Built the edge map that tells the solver where the floorboards creak.");
            break;
        case "phase_field_prepared":
            state.latticeImage = payload.latticeImage;
            state.guidanceImage = payload.guidanceImage;
            addLog("Prep", `Locked ${String(payload.targetWidth)} x ${String(payload.targetHeight)} lattice centers.`);
            break;
        case "phase_field_initial":
        case "phase_field_step":
        case "phase_field_final": {
            const frame = payload;
            const existingIndex = state.frames.findIndex((candidate) => candidate.step === frame.step);
            if (existingIndex >= 0) {
                state.frames[existingIndex] = frame;
            }
            else {
                state.frames.push(frame);
                state.frames.sort((a, b) => a.step - b.step);
            }
            if (state.autoFollow) {
                state.selectedFrameIndex = Math.max(0, state.frames.findIndex((candidate) => candidate.step === frame.step));
            }
            if (eventName === "phase_field_final") {
                addLog("Solver", "Final nearest-source sample committed.");
            }
            break;
        }
        case "cleanup_completed":
            state.cleanupImage = payload.cleanedImage;
            state.heatmapImage = payload.heatmapImage;
            addLog("Cleanup", "Ran the local cleanup sweep.");
            break;
        case "palette_completed":
            state.finalOutputImage = payload.outputImage;
            addLog("Output", "Final output image is ready.");
            break;
        case "pipeline_completed":
            state.finalOutputImage = payload.outputImage;
            state.runSummary = payload.runSummary ?? null;
            setStatus("completed", "Run finished. If the machine still did something stupid, fix it pixel by pixel.");
            addLog("Done", "Full pipeline completed.");
            runButton.disabled = false;
            currentEventSource?.close();
            await loadEditorAsset(state.finalOutputImage);
            break;
        default:
            break;
    }
    await renderEverything();
}
function connectEventStream(jobId) {
    currentEventSource?.close();
    const stream = new EventSource(`/api/jobs/${jobId}/events`);
    currentEventSource = stream;
    for (const eventType of eventTypes) {
        stream.addEventListener(eventType, (raw) => {
            const message = raw;
            const payload = JSON.parse(message.data);
            void handleEvent(eventType, payload);
        });
    }
    stream.onerror = () => {
        if (state.status === "running") {
            setStatus("waiting", "Event stream hiccup. If it stays this way, rerun it.");
        }
    };
}
async function startRun() {
    if (!state.file) {
        setStatus("idle", "Pick a file first. The machine is not clairvoyant.");
        return;
    }
    resetRunArtifacts();
    renderEventLog();
    runButton.disabled = true;
    setStatus("queued", "Queued the run. Small miracle pending.");
    addLog("Queued", `Starting ${state.file.name}.`);
    await renderEverything();
    try {
        const response = await fetch("/api/jobs", {
            method: "POST",
            body: buildFormData(),
        });
        if (!response.ok) {
            throw new Error(await response.text());
        }
        const payload = (await response.json());
        state.jobId = payload.jobId;
        connectEventStream(payload.jobId);
    }
    catch (error) {
        runButton.disabled = false;
        setStatus("failed", error instanceof Error ? error.message : "Failed to start GUI job.");
        addLog("Failure", error instanceof Error ? error.message : "Failed to start GUI job.");
    }
}
async function acceptFile(file) {
    state.file = file;
    dropzoneLabel.textContent = file.name;
    setStatus("idle", `${file.name} is loaded. Hit run when you want the machine to start sweating.`);
    try {
        const previewImage = await fileToImageAsset(file);
        if (state.file !== file) {
            return;
        }
        state.inputPreviewImage = previewImage;
    }
    catch (error) {
        if (state.file !== file) {
            return;
        }
        state.inputPreviewImage = null;
        setStatus("failed", error instanceof Error ? error.message : "Failed to load image preview.");
    }
    await renderInputPreview();
}
dropzone.addEventListener("click", () => fileInput.click());
pickFileButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) {
        void acceptFile(file);
    }
});
dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
});
dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) {
        void acceptFile(file);
    }
});
runButton.addEventListener("click", () => {
    void startRun();
});
stepSlider.addEventListener("input", () => {
    state.selectedFrameIndex = Number(stepSlider.value);
    state.autoFollow = state.selectedFrameIndex >= state.frames.length - 1;
    void renderEverything();
});
zoomInput.addEventListener("input", () => {
    setZoom(Number(zoomInput.value));
});
gridToggle.addEventListener("change", () => {
    state.showGrid = gridToggle.checked;
    renderEditor();
});
for (const input of Object.values(paintInputs)) {
    input.addEventListener("change", () => {
        setPaintColor([
            clampByte(Number(paintInputs.r.value)),
            clampByte(Number(paintInputs.g.value)),
            clampByte(Number(paintInputs.b.value)),
            clampByte(Number(paintInputs.a.value)),
        ]);
    });
}
window.addEventListener("keydown", (event) => {
    if (event.key === "Alt") {
        state.altHeld = true;
        syncEditorCursorState();
    }
});
window.addEventListener("keyup", (event) => {
    if (event.key === "Alt") {
        state.altHeld = false;
        syncEditorCursorState();
    }
});
window.addEventListener("blur", () => {
    state.altHeld = false;
    painting = false;
    endPan();
    syncEditorCursorState();
});
editorSurface.addEventListener("mousedown", (event) => {
    if (event.button === 1) {
        event.preventDefault();
    }
});
editorSurface.addEventListener("auxclick", (event) => {
    if (event.button === 1) {
        event.preventDefault();
    }
});
editorSurface.addEventListener("wheel", (event) => {
    if (!offscreenCanvas) {
        return;
    }
    event.preventDefault();
    const delta = event.deltaY < 0 ? 1 : -1;
    setZoom(state.zoom + delta, { clientX: event.clientX, clientY: event.clientY });
}, { passive: false });
editorSurface.addEventListener("pointerdown", (event) => {
    if (event.button !== 1 || !offscreenCanvas) {
        return;
    }
    event.preventDefault();
    beginPan(event);
});
editorSurface.addEventListener("pointermove", (event) => {
    if (!panning || activePanPointerId !== event.pointerId) {
        return;
    }
    event.preventDefault();
    editorSurface.scrollLeft = panStartScrollLeft - (event.clientX - panStartClientX);
    editorSurface.scrollTop = panStartScrollTop - (event.clientY - panStartClientY);
});
editorSurface.addEventListener("pointerup", (event) => {
    endPan(event.pointerId);
});
editorSurface.addEventListener("pointercancel", (event) => {
    endPan(event.pointerId);
});
editorCanvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) {
        return;
    }
    const pixel = editorPixelFromPointer(event);
    if (!pixel) {
        return;
    }
    event.preventDefault();
    editorCanvas.setPointerCapture(event.pointerId);
    if (event.altKey || state.altHeld) {
        samplePaintColor(pixel.x, pixel.y);
        return;
    }
    painting = true;
    paintPixel(pixel.x, pixel.y);
});
editorCanvas.addEventListener("pointermove", (event) => {
    if (panning) {
        return;
    }
    const pixel = editorPixelFromPointer(event);
    if (!pixel) {
        return;
    }
    if (event.altKey || state.altHeld) {
        if (event.buttons === 1) {
            samplePaintColor(pixel.x, pixel.y);
        }
        return;
    }
    if (painting && event.buttons === 1) {
        paintPixel(pixel.x, pixel.y);
    }
});
editorCanvas.addEventListener("pointerup", () => {
    painting = false;
});
editorCanvas.addEventListener("pointerleave", () => {
    painting = false;
});
resetEditorButton.addEventListener("click", () => {
    void loadEditorAsset(state.editorBaseAsset);
});
downloadButton.addEventListener("click", () => {
    exportEditorPng();
});
setPaintColor(state.paintColor);
syncEditorCursorState();
zoomValue.textContent = `${state.zoom}x`;
renderEventLog();
void renderInputPreview();
void renderEverything();
