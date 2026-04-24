type JsonRecord = Record<string, unknown>;

type ImageAsset = {
  dataUrl: string;
  width: number;
  height: number;
};

type InferenceCandidate = {
  target_width: number;
  target_height: number;
  phase_x: number;
  phase_y: number;
  score: number;
  breakdown: Record<string, number | null | undefined>;
};

type InferencePayload = {
  target_width: number;
  target_height: number;
  phase_x: number;
  phase_y: number;
  confidence: number;
  top_candidates: InferenceCandidate[];
};

type FramePayload = {
  step: number;
  totalSteps: number;
  loss: number | null;
  terms: Record<string, number>;
  phaseMetrics: Record<string, number>;
  outputImage: ImageAsset;
  samplingOverlayImage: ImageAsset;
  displacementImage: ImageAsset;
  lossHistory: number[];
};

type LogItem = {
  label: string;
  detail: string;
};

type AppState = {
  file: File | null;
  jobId: string | null;
  status: string;
  statusText: string;
  inputPreviewImage: ImageAsset | null;
  sourceImage: ImageAsset | null;
  preprocessedImage: ImageAsset | null;
  edgeMapImage: ImageAsset | null;
  latticeImage: ImageAsset | null;
  guidanceImage: ImageAsset | null;
  inference: InferencePayload | null;
  frames: FramePayload[];
  cleanupImage: ImageAsset | null;
  heatmapImage: ImageAsset | null;
  finalOutputImage: ImageAsset | null;
  runSummary: JsonRecord | null;
  selectedFrameIndex: number;
  autoFollow: boolean;
  eventLog: LogItem[];
  paintColor: [number, number, number, number];
  editorBaseAsset: ImageAsset | null;
  editorDirty: boolean;
  altHeld: boolean;
  zoom: number;
  showGrid: boolean;
  currentStage: string;
};

type ViewportAnchor = {
  clientX: number;
  clientY: number;
};

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element: ${id}`);
  }
  return element as T;
}

const fileInput = byId<HTMLInputElement>("fileInput");
const dropzone = byId<HTMLButtonElement>("dropzone");
const dropzoneLabel = byId<HTMLSpanElement>("dropzoneLabel");
const inputPreviewCanvas = byId<HTMLCanvasElement>("inputPreviewCanvas");
const pickFileButton = byId<HTMLButtonElement>("pickFileButton");
const runButton = byId<HTMLButtonElement>("runButton");
const statusBadge = byId<HTMLSpanElement>("statusBadge");
const statusText = byId<HTMLParagraphElement>("statusText");
const inferenceSummary = byId<HTMLDivElement>("inferenceSummary");
const candidateList = byId<HTMLDivElement>("candidateList");
const eventLog = byId<HTMLDivElement>("eventLog");
const stepSlider = byId<HTMLInputElement>("stepSlider");
const stepValue = byId<HTMLSpanElement>("stepValue");
const lossCanvas = byId<HTMLCanvasElement>("lossCanvas");
const leftCanvas = byId<HTMLCanvasElement>("leftCanvas");
const middleCanvas = byId<HTMLCanvasElement>("middleCanvas");
const rightCanvas = byId<HTMLCanvasElement>("rightCanvas");
const leftVizLabel = byId<HTMLDivElement>("leftVizLabel");
const middleVizLabel = byId<HTMLDivElement>("middleVizLabel");
const rightVizLabel = byId<HTMLDivElement>("rightVizLabel");
const metricsPanel = byId<HTMLDivElement>("metricsPanel");
const paintSwatch = byId<HTMLDivElement>("paintSwatch");
const zoomInput = byId<HTMLInputElement>("zoomInput");
const zoomValue = byId<HTMLSpanElement>("zoomValue");
const gridToggle = byId<HTMLInputElement>("gridToggle");
const editorCanvas = byId<HTMLCanvasElement>("editorCanvas");
const editorGridCanvas = byId<HTMLCanvasElement>("editorGridCanvas");
const editorSurface = byId<HTMLDivElement>("editorSurface");
const editorMeta = byId<HTMLDivElement>("editorMeta");
const summaryPanel = byId<HTMLDivElement>("summaryPanel");
const resetEditorButton = byId<HTMLButtonElement>("resetEditorButton");
const downloadButton = byId<HTMLButtonElement>("downloadButton");

const targetSizeInput = byId<HTMLInputElement>("targetSizeInput");
const targetWidthInput = byId<HTMLInputElement>("targetWidthInput");
const targetHeightInput = byId<HTMLInputElement>("targetHeightInput");
const phaseXInput = byId<HTMLInputElement>("phaseXInput");
const phaseYInput = byId<HTMLInputElement>("phaseYInput");
const stepsInput = byId<HTMLInputElement>("stepsInput");
const seedInput = byId<HTMLInputElement>("seedInput");
const deviceInput = byId<HTMLSelectElement>("deviceInput");
const stripBackgroundInput = byId<HTMLInputElement>("stripBackgroundInput");
const skipRerankInput = byId<HTMLInputElement>("skipRerankInput");

const paintInputs = {
  r: byId<HTMLInputElement>("paintR"),
  g: byId<HTMLInputElement>("paintG"),
  b: byId<HTMLInputElement>("paintB"),
  a: byId<HTMLInputElement>("paintA"),
};

const imageCache = new Map<string, Promise<HTMLImageElement>>();
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
] as const;

const state: AppState = {
  file: null,
  jobId: null,
  status: "idle",
  statusText: "Waiting for input.",
  inputPreviewImage: null,
  sourceImage: null,
  preprocessedImage: null,
  edgeMapImage: null,
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
  currentStage: "Idle",
};

let currentEventSource: EventSource | null = null;
let painting = false;
let panning = false;
let activePanPointerId: number | null = null;
let panStartClientX = 0;
let panStartClientY = 0;
let panStartScrollLeft = 0;
let panStartScrollTop = 0;
let offscreenCanvas: HTMLCanvasElement | null = null;
let offscreenContext: CanvasRenderingContext2D | null = null;
let singlePixel = new ImageData(1, 1);

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function clampZoom(value: number): number {
  const minZoom = Number(zoomInput.min) || 1;
  const maxZoom = Number(zoomInput.max) || 64;
  return Math.max(minZoom, Math.min(maxZoom, Math.round(value)));
}

function syncEditorCursorState(): void {
  editorSurface.classList.toggle("is-panning", panning);
  editorSurface.classList.toggle("is-eyedropper", state.altHeld);
}

function setPaintColor(color: [number, number, number, number]): void {
  state.paintColor = color;
  paintInputs.r.value = String(color[0]);
  paintInputs.g.value = String(color[1]);
  paintInputs.b.value = String(color[2]);
  paintInputs.a.value = String(color[3]);
  paintSwatch.style.background = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3] / 255})`;
}

function addLog(label: string, detail: string): void {
  state.eventLog.unshift({ label, detail });
  state.eventLog = state.eventLog.slice(0, 8);
  renderEventLog();
}

function renderEventLog(): void {
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

function setStatus(status: string, detail: string): void {
  state.status = status;
  state.statusText = detail;
  statusBadge.textContent = status;
  statusText.textContent = detail;
}

function resetRunArtifacts(): void {
  state.jobId = null;
  state.sourceImage = null;
  state.preprocessedImage = null;
  state.edgeMapImage = null;
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
  state.currentStage = "Waiting";
}

function parseOptionalInteger(input: HTMLInputElement): number | null {
  if (input.value.trim() === "") {
    return null;
  }
  const value = Number(input.value);
  return Number.isFinite(value) ? Math.round(value) : null;
}

function parseOptionalFloat(input: HTMLInputElement): number | null {
  if (input.value.trim() === "") {
    return null;
  }
  const value = Number(input.value);
  return Number.isFinite(value) ? value : null;
}

async function loadImage(asset: ImageAsset): Promise<HTMLImageElement> {
  const cached = imageCache.get(asset.dataUrl);
  if (cached) {
    return cached;
  }
  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load image asset"));
    image.src = asset.dataUrl;
  });
  imageCache.set(asset.dataUrl, promise);
  return promise;
}

async function drawAsset(canvas: HTMLCanvasElement, asset: ImageAsset | null): Promise<void> {
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

async function fileToImageAsset(file: File): Promise<ImageAsset> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
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

async function renderInputPreview(): Promise<void> {
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

function readNestedNumber(root: JsonRecord | null, path: string[]): number | null {
  let current: unknown = root;
  for (const key of path) {
    if (!current || typeof current !== "object" || !(key in current)) {
      return null;
    }
    current = (current as JsonRecord)[key];
  }
  return typeof current === "number" ? current : null;
}

function formatNumber(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function getSelectedFrame(): FramePayload | null {
  if (state.frames.length === 0) {
    return null;
  }
  return state.frames[Math.max(0, Math.min(state.selectedFrameIndex, state.frames.length - 1))];
}

function renderInference(): void {
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

function renderMetrics(): void {
  metricsPanel.innerHTML = "";
  const frame = getSelectedFrame();
  const items: Array<[string, string]> = [["Stage", state.currentStage]];
  if (frame) {
    items.push(["Loss", frame.loss === null ? "n/a" : formatNumber(frame.loss, 4)]);
    for (const [key, value] of Object.entries(frame.terms)) {
      items.push([key.replaceAll("_", " "), formatNumber(value, 4)]);
    }
  } else if (state.runSummary) {
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
    node.innerHTML = `<strong>${label}</strong><span>${value}</span>`;
    metricsPanel.appendChild(node);
  }
}

function renderSummary(): void {
  summaryPanel.innerHTML = "";
  if (!state.runSummary) {
    summaryPanel.innerHTML = `<p class="muted">Final metrics show up here after the run finishes.</p>`;
    return;
  }
  const entries: Array<[string, number | null]> = [
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

async function renderViewer(): Promise<void> {
  const frame = getSelectedFrame();
  let leftAsset = state.preprocessedImage ?? state.sourceImage;
  let middleAsset = state.edgeMapImage;
  let rightAsset = state.finalOutputImage ?? state.cleanupImage ?? state.sourceImage;
  let leftLabel = "Source";
  let middleLabel = "Edge Scout";
  let rightLabel = "Output";

  if (state.latticeImage) {
    leftAsset = state.latticeImage;
    leftLabel = "Lattice Prep";
  }
  if (state.guidanceImage) {
    middleAsset = state.guidanceImage;
    middleLabel = "Guidance";
  }
  if (frame) {
    leftAsset = frame.samplingOverlayImage;
    middleAsset = frame.displacementImage;
    rightAsset = frame.outputImage;
    leftLabel = "Sampling Overlay";
    middleLabel = "Displacement";
    rightLabel = "Current Output";
  } else if (state.cleanupImage) {
    middleAsset = state.heatmapImage ?? middleAsset;
    middleLabel = "Cleanup Heatmap";
    rightAsset = state.cleanupImage;
    rightLabel = "Cleaned Output";
  }

  leftVizLabel.textContent = leftLabel;
  middleVizLabel.textContent = middleLabel;
  rightVizLabel.textContent = rightLabel;
  await Promise.all([drawAsset(leftCanvas, leftAsset), drawAsset(middleCanvas, middleAsset), drawAsset(rightCanvas, rightAsset)]);
  renderMetrics();
}

function renderSlider(): void {
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

function renderLossChart(): void {
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
    .map((frame) => frame.loss as number);
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
    } else {
      context.lineTo(x, y);
    }
  });
  context.stroke();

  const selected = getSelectedFrame();
  if (selected && selected.loss !== null) {
    const matchingIndex = losses.findIndex((value) => Math.abs(value - selected.loss!) < 1e-6);
    if (matchingIndex >= 0) {
      const x = losses.length === 1 ? width / 2 : (matchingIndex / (losses.length - 1)) * (width - 24) + 12;
      const y = height - 18 - (((selected.loss as number) - minLoss) / span) * (height - 36);
      context.fillStyle = "#f5b76b";
      context.beginPath();
      context.arc(x, y, 5, 0, Math.PI * 2);
      context.fill();
    }
  }
}

async function loadEditorAsset(asset: ImageAsset | null): Promise<void> {
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

function renderEditor(): void {
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

function setZoom(nextZoom: number, anchor: ViewportAnchor | null = null): void {
  const clampedZoom = clampZoom(nextZoom);
  const previousZoom = state.zoom;
  if (clampedZoom === previousZoom) {
    zoomInput.value = String(clampedZoom);
    zoomValue.textContent = `${clampedZoom}x`;
    return;
  }

  let viewportX = 0;
  let viewportY = 0;
  let imageX: number | null = null;
  let imageY: number | null = null;
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

function editorPixelFromPointer(event: PointerEvent): { x: number; y: number } | null {
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

function samplePaintColor(x: number, y: number): void {
  if (!offscreenContext) {
    return;
  }
  const pixel = offscreenContext.getImageData(x, y, 1, 1).data;
  setPaintColor([pixel[0], pixel[1], pixel[2], pixel[3]]);
}

function paintPixel(x: number, y: number): void {
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

function exportEditorPng(): void {
  if (!offscreenCanvas) {
    return;
  }
  const link = document.createElement("a");
  link.href = offscreenCanvas.toDataURL("image/png");
  link.download = (state.file?.name?.replace(/\.[^.]+$/, "") || "repixelized") + "-edited.png";
  link.click();
}

function beginPan(event: PointerEvent): void {
  panning = true;
  activePanPointerId = event.pointerId;
  panStartClientX = event.clientX;
  panStartClientY = event.clientY;
  panStartScrollLeft = editorSurface.scrollLeft;
  panStartScrollTop = editorSurface.scrollTop;
  editorSurface.setPointerCapture(event.pointerId);
  syncEditorCursorState();
}

function endPan(pointerId: number | null = null): void {
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

function buildFormData(): FormData {
  if (!state.file) {
    throw new Error("No file selected");
  }
  const data = new FormData();
  data.set("image", state.file);
  const values: Array<[string, string | null]> = [
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

async function renderEverything(): Promise<void> {
  renderInference();
  renderSlider();
  renderLossChart();
  renderSummary();
  await renderViewer();
}

async function handleEvent(eventName: string, payload: JsonRecord): Promise<void> {
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
      state.sourceImage = payload.sourceImage as ImageAsset;
      state.currentStage = "Loaded source";
      setStatus("running", "Source loaded. Survey crew is measuring the mess.");
      addLog("Source", "Loaded the input image.");
      break;
    case "preprocess_completed":
      state.preprocessedImage = payload.sourceImage as ImageAsset;
      state.currentStage = "Preprocessed source";
      addLog("Preprocess", "Stripped edge-connected background noise.");
      break;
    case "inference_candidates_ready":
      state.inference = payload.inference as InferencePayload;
      state.currentStage = "Scoring lattice candidates";
      addLog("Inference", "Scored candidate grids and phase offsets.");
      break;
    case "phase_rerank_started":
      state.currentStage = "Phase rerank preview";
      addLog(
        "Rerank",
        `Running ${String(payload.previewSteps)} preview steps across ${String(payload.candidateCount)} low-confidence candidates.`,
      );
      break;
    case "phase_selection_completed":
      state.inference = payload.inference as InferencePayload;
      state.currentStage = "Selected lattice";
      addLog("Selection", "Committed to a ruler and phase.");
      break;
    case "analysis_completed":
      state.edgeMapImage = payload.edgeMapImage as ImageAsset;
      state.currentStage = "Edge scout";
      addLog("Scout", "Built the edge map that tells the solver where the floorboards creak.");
      break;
    case "phase_field_prepared":
      state.latticeImage = payload.latticeImage as ImageAsset;
      state.guidanceImage = payload.guidanceImage as ImageAsset;
      state.currentStage = "Prepared phase field";
      addLog("Prep", `Locked ${String(payload.targetWidth)} x ${String(payload.targetHeight)} lattice centers.`);
      break;
    case "phase_field_initial":
    case "phase_field_step":
    case "phase_field_final": {
      const frame = payload as unknown as FramePayload;
      const existingIndex = state.frames.findIndex((candidate) => candidate.step === frame.step);
      if (existingIndex >= 0) {
        state.frames[existingIndex] = frame;
      } else {
        state.frames.push(frame);
        state.frames.sort((a, b) => a.step - b.step);
      }
      if (state.autoFollow) {
        state.selectedFrameIndex = Math.max(0, state.frames.findIndex((candidate) => candidate.step === frame.step));
      }
      state.currentStage = frame.step === 0 ? "Initial placement" : `Solver step ${frame.step}/${frame.totalSteps}`;
      if (eventName === "phase_field_final") {
        addLog("Solver", "Final nearest-source sample committed.");
      }
      break;
    }
    case "cleanup_completed":
      state.cleanupImage = payload.cleanedImage as ImageAsset;
      state.heatmapImage = payload.heatmapImage as ImageAsset;
      state.currentStage = "Cleanup";
      addLog("Cleanup", "Ran the local cleanup sweep.");
      break;
    case "palette_completed":
      state.finalOutputImage = payload.outputImage as ImageAsset;
      state.currentStage = "Final output";
      addLog("Output", "Final output image is ready.");
      break;
    case "pipeline_completed":
      state.finalOutputImage = payload.outputImage as ImageAsset;
      state.runSummary = (payload.runSummary as JsonRecord | null) ?? null;
      state.currentStage = "Completed";
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

function connectEventStream(jobId: string): void {
  currentEventSource?.close();
  const stream = new EventSource(`/api/jobs/${jobId}/events`);
  currentEventSource = stream;
  for (const eventType of eventTypes) {
    stream.addEventListener(eventType, (raw) => {
      const message = raw as MessageEvent<string>;
      const payload = JSON.parse(message.data) as JsonRecord;
      void handleEvent(eventType, payload);
    });
  }
  stream.onerror = () => {
    if (state.status === "running") {
      setStatus("waiting", "Event stream hiccup. If it stays this way, rerun it.");
    }
  };
}

async function startRun(): Promise<void> {
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
    const payload = (await response.json()) as { jobId: string };
    state.jobId = payload.jobId;
    connectEventStream(payload.jobId);
  } catch (error) {
    runButton.disabled = false;
    setStatus("failed", error instanceof Error ? error.message : "Failed to start GUI job.");
    addLog("Failure", error instanceof Error ? error.message : "Failed to start GUI job.");
  }
}

async function acceptFile(file: File): Promise<void> {
  state.file = file;
  dropzoneLabel.textContent = file.name;
  setStatus("idle", `${file.name} is loaded. Hit run when you want the machine to start sweating.`);
  try {
    const previewImage = await fileToImageAsset(file);
    if (state.file !== file) {
      return;
    }
    state.inputPreviewImage = previewImage;
  } catch (error) {
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

editorSurface.addEventListener(
  "wheel",
  (event) => {
    if (!offscreenCanvas) {
      return;
    }
    event.preventDefault();
    const delta = event.deltaY < 0 ? 1 : -1;
    setZoom(state.zoom + delta, { clientX: event.clientX, clientY: event.clientY });
  },
  { passive: false },
);

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
