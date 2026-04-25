import base64
import contextlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .diagnostics import _displacement_preview_rgba
from .inference import inference_to_json
from .observe import PipelineCancelled
from .pipeline import _resolve_requested_target_dims, run_pipeline_rgba
from .types import InferenceResult, PaletteResult


def _require_gui_dependencies():
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised only when GUI deps missing
        raise RuntimeError(
            "The GUI requires FastAPI, uvicorn, and python-multipart. Install the project dependencies first."
        ) from exc
    return {
        "FastAPI": FastAPI,
        "File": File,
        "Form": Form,
        "HTTPException": HTTPException,
        "UploadFile": UploadFile,
        "HTMLResponse": HTMLResponse,
        "JSONResponse": JSONResponse,
        "RedirectResponse": RedirectResponse,
        "StreamingResponse": StreamingResponse,
        "StaticFiles": StaticFiles,
    }


def _to_uint8(rgba: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)


def _rgba_data_url(rgba: np.ndarray) -> str:
    image = Image.fromarray(_to_uint8(rgba), mode="RGBA")
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_asset(rgba: np.ndarray) -> dict[str, Any]:
    return {
        "dataUrl": _rgba_data_url(rgba),
        "width": int(rgba.shape[1]),
        "height": int(rgba.shape[0]),
    }


def _scalar_to_rgba(values: np.ndarray, *, tint: tuple[float, float, float] = (0.15, 0.85, 0.75)) -> np.ndarray:
    scaled = values.astype(np.float32)
    max_value = float(np.max(scaled)) if scaled.size else 0.0
    if max_value > 0.0:
        scaled = scaled / max_value
    rgba = np.zeros((*scaled.shape, 4), dtype=np.float32)
    rgba[..., 0] = scaled * float(tint[0]) + np.sqrt(scaled) * 0.08
    rgba[..., 1] = scaled * float(tint[1])
    rgba[..., 2] = scaled * float(tint[2]) + np.sqrt(scaled) * 0.18
    rgba[..., 3] = np.clip(0.12 + scaled * 0.88, 0.0, 1.0)
    return np.clip(rgba, 0.0, 1.0)


def _render_sample_overlay(source_rgba: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    overlay = source_rgba.copy()
    accent = np.asarray([0.10, 0.95, 0.82, 1.0], dtype=np.float32)
    shadow = np.asarray([0.02, 0.08, 0.10, 1.0], dtype=np.float32)
    height, width = overlay.shape[:2]
    for dy, dx, color in (
        (0, 0, accent),
        (-1, 0, shadow),
        (1, 0, shadow),
        (0, -1, shadow),
        (0, 1, shadow),
    ):
        yy = np.clip(sample_y + dy, 0, height - 1)
        xx = np.clip(sample_x + dx, 0, width - 1)
        overlay[yy, xx] = overlay[yy, xx] * 0.22 + color[None, :] * 0.78
    return np.clip(overlay, 0.0, 1.0)


def _render_lattice_overlay(source_rgba: np.ndarray, uv0_px: np.ndarray) -> np.ndarray:
    image = Image.fromarray(_to_uint8(source_rgba), mode="RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    color = (54, 219, 198, 180)
    shadow = (8, 18, 24, 170)
    height, width = uv0_px.shape[:2]
    for y in range(height):
        row = uv0_px[y]
        for x in range(width - 1):
            left = row[x]
            right = row[x + 1]
            draw.line((float(left[0]), float(left[1]), float(right[0]), float(right[1])), fill=shadow, width=3)
            draw.line((float(left[0]), float(left[1]), float(right[0]), float(right[1])), fill=color, width=1)
    for x in range(width):
        column = uv0_px[:, x]
        for y in range(height - 1):
            top = column[y]
            bottom = column[y + 1]
            draw.line((float(top[0]), float(top[1]), float(bottom[0]), float(bottom[1])), fill=shadow, width=3)
            draw.line((float(top[0]), float(top[1]), float(bottom[0]), float(bottom[1])), fill=color, width=1)
    for point in uv0_px.reshape(-1, 2):
        px = float(point[0])
        py = float(point[1])
        draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=(255, 244, 168, 230))
    return np.asarray(image, dtype=np.float32) / 255.0


def _decode_rgba(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"), dtype=np.float32) / 255.0


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class HostedDemoConfig:
    hosted_demo: bool
    max_upload_bytes: int
    max_input_dimension: int
    max_output_dimension: int
    default_steps: int
    max_steps: int
    queue_capacity: int
    heartbeat_interval_seconds: int
    stale_after_seconds: int
    phase_field_preview_stride: int
    spool_dir: Path

    @classmethod
    def from_env(cls) -> "HostedDemoConfig":
        hosted_demo = _env_flag("REPIXELIZER_HOSTED_DEMO", False)
        defaults = {
            "max_upload_bytes": 1_048_576 if hosted_demo else 16 * 1_048_576,
            "max_input_dimension": 2048 if hosted_demo else 4096,
            "max_output_dimension": 256 if hosted_demo else 1024,
            "default_steps": 32 if hosted_demo else 48,
            "max_steps": 48 if hosted_demo else 256,
            "queue_capacity": 10 if hosted_demo else 32,
            "heartbeat_interval_seconds": 10,
            "stale_after_seconds": 30,
            "phase_field_preview_stride": 4,
        }
        spool_dir_raw = os.getenv("REPIXELIZER_SPOOL_DIR")
        if spool_dir_raw:
            spool_dir = Path(spool_dir_raw).expanduser()
        else:
            spool_dir = Path(tempfile.gettempdir()) / "repixelizer-gui-spool"
        return cls(
            hosted_demo=hosted_demo,
            max_upload_bytes=max(1, _env_int("REPIXELIZER_MAX_UPLOAD_BYTES", defaults["max_upload_bytes"])),
            max_input_dimension=max(1, _env_int("REPIXELIZER_MAX_INPUT_DIMENSION", defaults["max_input_dimension"])),
            max_output_dimension=max(1, _env_int("REPIXELIZER_MAX_OUTPUT_DIMENSION", defaults["max_output_dimension"])),
            default_steps=max(0, _env_int("REPIXELIZER_DEFAULT_STEPS", defaults["default_steps"])),
            max_steps=max(0, _env_int("REPIXELIZER_MAX_STEPS", defaults["max_steps"])),
            queue_capacity=max(1, _env_int("REPIXELIZER_QUEUE_CAPACITY", defaults["queue_capacity"])),
            heartbeat_interval_seconds=max(
                1, _env_int("REPIXELIZER_HEARTBEAT_INTERVAL_SECONDS", defaults["heartbeat_interval_seconds"])
            ),
            stale_after_seconds=max(2, _env_int("REPIXELIZER_STALE_AFTER_SECONDS", defaults["stale_after_seconds"])),
            phase_field_preview_stride=max(
                1, _env_int("REPIXELIZER_PHASE_FIELD_PREVIEW_STRIDE", defaults["phase_field_preview_stride"])
            ),
            spool_dir=spool_dir,
        )

    def ui_flags(self) -> dict[str, bool]:
        return {
            "showDeviceControl": not self.hosted_demo,
            "showStripBackgroundControl": not self.hosted_demo,
        }

    def public_payload(self) -> dict[str, Any]:
        return {
            "hostedDemo": self.hosted_demo,
            "limits": {
                "maxUploadBytes": self.max_upload_bytes,
                "maxInputDimension": self.max_input_dimension,
                "maxOutputDimension": self.max_output_dimension,
                "defaultSteps": self.default_steps,
                "maxSteps": self.max_steps,
                "queueCapacity": self.queue_capacity,
                "heartbeatIntervalSeconds": self.heartbeat_interval_seconds,
                "staleAfterSeconds": self.stale_after_seconds,
            },
            "ui": self.ui_flags(),
        }


def _palette_to_json(palette_result: PaletteResult | None) -> dict[str, Any] | None:
    if palette_result is None:
        return None
    return {
        "palette": [list(color) for color in palette_result.palette],
        "hasIndexedOutput": palette_result.indexed_rgba is not None,
    }


@dataclass(slots=True)
class GuiJob:
    job_id: str
    filename: str
    options: dict[str, Any]
    spool_path: Path
    created_at: float = field(default_factory=time.time)
    last_heartbeat_at: float = field(default_factory=time.time)
    phase_field_preview_stride: int = 4
    phase_field_include_snapshot: bool = True
    status: str = "queued"
    error: str | None = None
    cancel_reason: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    current_source_rgba: np.ndarray | None = None
    current_inference: InferenceResult | None = None
    run_summary: dict[str, Any] | None = None
    _condition: threading.Condition = field(default_factory=threading.Condition, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def observe(self, event: str, payload: dict[str, Any]) -> None:
        serialized = self._serialize_event(event, payload)
        if serialized is None:
            return
        self.publish(event, serialized)

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "id": len(self.events) + 1,
            "event": event,
            "payload": payload,
            "timestamp": time.time(),
        }
        with self._condition:
            self.events.append(record)
            self._condition.notify_all()

    def mark_running(self) -> None:
        self.status = "running"
        self.touch_heartbeat()
        self.publish("job_state", {"status": self.status})

    def mark_completed(self) -> None:
        self.status = "completed"
        self.publish("job_state", {"status": self.status})

    def mark_failed(self, message: str) -> None:
        self.status = "failed"
        self.error = message
        self.publish("job_failed", {"status": self.status, "message": message})

    def mark_canceled(self, message: str) -> None:
        self.status = "canceled"
        self.error = message
        self.cancel_reason = message
        self.publish("job_canceled", {"status": self.status, "message": message})
        self.publish("job_state", {"status": self.status, "message": message})

    def touch_heartbeat(self) -> None:
        self.last_heartbeat_at = time.time()

    def request_cancel(self, reason: str) -> None:
        self.cancel_reason = reason
        self._cancel_event.set()

    @property
    def cancellation_message(self) -> str:
        return self.cancel_reason or "Pipeline canceled."

    def check_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def is_stale(self, *, now: float, stale_after_seconds: int) -> bool:
        if self.status not in {"queued", "running"}:
            return False
        return (now - self.last_heartbeat_at) >= float(stale_after_seconds)

    def wait_for_events(self, index: int, timeout: float = 0.5) -> list[dict[str, Any]]:
        with self._condition:
            if index >= len(self.events) and self.status not in {"completed", "failed", "canceled"}:
                self._condition.wait(timeout=timeout)
            return self.events[index:]

    def _serialize_event(self, event: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if event == "source_loaded":
            source = payload["source_rgba"]
            self.current_source_rgba = source.copy()
            return {
                "filename": self.filename,
                "sourceImage": _image_asset(source),
            }
        if event == "preprocess_completed":
            source = payload["source_rgba"]
            self.current_source_rgba = source.copy()
            return {
                "operation": payload.get("operation"),
                "sourceImage": _image_asset(source),
            }
        if event in {"inference_candidates_ready", "phase_selection_completed"}:
            inference = payload["inference"]
            self.current_inference = inference
            return {
                "inference": inference_to_json(inference),
                "inferenceMode": payload.get("inference_mode"),
            }
        if event == "phase_rerank_started":
            return {
                "previewSteps": int(payload["preview_steps"]),
                "candidateCount": int(payload["candidate_count"]),
                "confidence": float(payload["confidence"]),
            }
        if event == "lattice_search_started":
            return {
                "candidateCount": int(payload["candidate_count"]),
                "phaseSampleCount": int(payload["phase_sample_count"]),
                "device": str(payload["device"]),
            }
        if event == "lattice_search_progress":
            return {
                "completedCandidates": int(payload["completed_candidates"]),
                "totalCandidates": int(payload["total_candidates"]),
                "targetWidth": int(payload["target_width"]),
                "targetHeight": int(payload["target_height"]),
                "phaseSampleCount": int(payload["phase_sample_count"]),
                "bestScore": None if payload.get("best_score") is None else float(payload["best_score"]),
            }
        if event == "phase_rerank_candidate_started":
            return {
                "candidateIndex": int(payload["candidate_index"]),
                "totalCandidates": int(payload["total_candidates"]),
                "targetWidth": int(payload["target_width"]),
                "targetHeight": int(payload["target_height"]),
                "phaseX": float(payload["phase_x"]),
                "phaseY": float(payload["phase_y"]),
                "previewSteps": int(payload["preview_steps"]),
            }
        if event == "phase_rerank_candidate_step":
            return {
                "candidateIndex": int(payload["candidate_index"]),
                "totalCandidates": int(payload["total_candidates"]),
                "targetWidth": int(payload["target_width"]),
                "targetHeight": int(payload["target_height"]),
                "phaseX": float(payload["phase_x"]),
                "phaseY": float(payload["phase_y"]),
                "step": int(payload["step"]),
                "totalSteps": int(payload["total_steps"]),
                "loss": None if payload.get("loss") is None else float(payload["loss"]),
            }
        if event == "phase_rerank_candidate_completed":
            return {
                "candidateIndex": int(payload["candidate_index"]),
                "totalCandidates": int(payload["total_candidates"]),
                "completedCandidates": int(payload["completed_candidates"]),
                "targetWidth": int(payload["target_width"]),
                "targetHeight": int(payload["target_height"]),
                "phaseX": float(payload["phase_x"]),
                "phaseY": float(payload["phase_y"]),
                "totalSteps": int(payload["total_steps"]),
                "finalLoss": None if payload.get("final_loss") is None else float(payload["final_loss"]),
            }
        if event == "stage_started":
            return {
                "stage": str(payload["stage"]),
                "label": str(payload["label"]),
                "detail": str(payload["detail"]),
            }
        if event == "analysis_completed":
            edge_map = payload["edge_map"]
            return {
                "edgeMapImage": _image_asset(_scalar_to_rgba(edge_map, tint=(0.18, 0.84, 0.82))),
            }
        if event == "phase_field_prepared":
            if self.current_source_rgba is None:
                return None
            return {
                "targetWidth": int(payload["target_width"]),
                "targetHeight": int(payload["target_height"]),
                "cellX": float(payload["cell_x"]),
                "cellY": float(payload["cell_y"]),
                "latticeImage": _image_asset(_render_lattice_overlay(self.current_source_rgba, payload["uv0_px"])),
                "guidanceImage": _image_asset(_scalar_to_rgba(payload["guidance"], tint=(0.92, 0.58, 0.18))),
            }
        if event in {"phase_field_initial", "phase_field_step", "phase_field_final"}:
            if self.current_source_rgba is None:
                return None
            return {
                "step": int(payload["step"]),
                "totalSteps": int(payload["total_steps"]),
                "loss": None if payload.get("loss") is None else float(payload["loss"]),
                "terms": {key: float(value) for key, value in payload.get("terms", {}).items()},
                "phaseMetrics": {key: float(value) for key, value in payload.get("phase_metrics", {}).items()},
                "outputImage": _image_asset(payload["target_rgba"]),
                "samplingOverlayImage": _image_asset(
                    _render_sample_overlay(self.current_source_rgba, payload["sample_x"], payload["sample_y"])
                ),
                "displacementImage": _image_asset(
                    _displacement_preview_rgba(payload["displacement_x"], payload["displacement_y"])
                ),
                "lossHistory": [float(value) for value in payload.get("loss_history", [])],
            }
        if event == "cleanup_completed":
            return {
                "cleanedImage": _image_asset(payload["cleaned_rgba"]),
                "heatmapImage": _image_asset(_scalar_to_rgba(payload["isolated_heatmap"], tint=(0.95, 0.38, 0.24))),
            }
        if event == "palette_completed":
            return {
                "outputImage": _image_asset(payload["output_rgba"]),
                "paletteMode": payload.get("palette_mode"),
                "palette": _palette_to_json(payload.get("palette_result")),
            }
        if event == "pipeline_completed":
            self.run_summary = payload.get("run_summary")
            return {
                "outputImage": _image_asset(payload["output_rgba"]),
                "diagnostics": payload["diagnostics"],
                "runSummary": self.run_summary,
            }
        return None


def _job_state_payload(job: GuiJob) -> dict[str, Any]:
    return {
        "jobId": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "error": job.error,
        "createdAt": job.created_at,
        "eventCount": len(job.events),
        "runSummary": job.run_summary,
    }


def _inspect_upload_image(raw: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
    return int(width), int(height)


def _normalize_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return resolved


def _normalize_job_options(
    config: HostedDemoConfig,
    *,
    source_width: int,
    source_height: int,
    target_size: int | None,
    target_width: int | None,
    target_height: int | None,
    phase_x: float | None,
    phase_y: float | None,
    steps: int | None,
    seed: int,
    device: str,
    strip_background: bool,
    skip_phase_rerank: bool,
) -> dict[str, Any]:
    normalized_target_size = _normalize_optional_positive_int("target_size", target_size)
    normalized_target_width = _normalize_optional_positive_int("target_width", target_width)
    normalized_target_height = _normalize_optional_positive_int("target_height", target_height)
    try:
        fixed_dims = _resolve_requested_target_dims(
            source_width=source_width,
            source_height=source_height,
            target_size=normalized_target_size,
            target_width=normalized_target_width,
            target_height=normalized_target_height,
            phase_x=phase_x,
            phase_y=phase_y,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if fixed_dims is not None and max(fixed_dims) > config.max_output_dimension:
        raise ValueError(
            f"Requested output {fixed_dims[0]}x{fixed_dims[1]} exceeds the maximum hosted output dimension of {config.max_output_dimension}."
        )
    requested_steps = config.default_steps if steps is None else max(0, int(steps))
    normalized_steps = min(requested_steps, config.max_steps)
    normalized_device = "cpu" if config.hosted_demo else (device.strip() if device.strip() else "auto")
    normalized_strip_background = False if config.hosted_demo else bool(strip_background)
    effective_target_size = normalized_target_size
    if fixed_dims is None:
        effective_target_size = (
            config.max_output_dimension
            if effective_target_size is None
            else min(effective_target_size, config.max_output_dimension)
        )
    return {
        "target_size": effective_target_size,
        "target_width": normalized_target_width,
        "target_height": normalized_target_height,
        "phase_x": phase_x,
        "phase_y": phase_y,
        "steps": normalized_steps,
        "seed": int(seed),
        "device": normalized_device,
        "strip_background": normalized_strip_background,
        "skip_phase_rerank": bool(skip_phase_rerank),
    }


def _validate_upload_request(
    config: HostedDemoConfig,
    *,
    raw: bytes,
    filename: str,
    target_size: int | None,
    target_width: int | None,
    target_height: int | None,
    phase_x: float | None,
    phase_y: float | None,
    steps: int | None,
    seed: int,
    device: str,
    strip_background: bool,
    skip_phase_rerank: bool,
) -> tuple[dict[str, Any], int, int]:
    if len(raw) > config.max_upload_bytes:
        raise ValueError(f"Upload is too large. Limit is {config.max_upload_bytes // 1024} KiB for the hosted demo.")
    try:
        source_width, source_height = _inspect_upload_image(raw)
    except Exception as exc:
        raise ValueError(f"{filename or 'Upload'} is not a readable PNG or image file.") from exc
    if source_width > config.max_input_dimension or source_height > config.max_input_dimension:
        raise ValueError(
            f"Input image is {source_width}x{source_height}. Limit is {config.max_input_dimension}px on each side."
        )
    options = _normalize_job_options(
        config,
        source_width=source_width,
        source_height=source_height,
        target_size=target_size,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
        steps=steps,
        seed=seed,
        device=device,
        strip_background=strip_background,
        skip_phase_rerank=skip_phase_rerank,
    )
    return options, source_width, source_height


def _execute_job(job: GuiJob) -> None:
    if job.check_cancelled():
        raise PipelineCancelled(job.cancellation_message)
    source_rgba = _decode_rgba(job.spool_path.read_bytes())
    job.mark_running()
    run_pipeline_rgba(
        source_rgba,
        target_size=job.options["target_size"],
        target_width=job.options["target_width"],
        target_height=job.options["target_height"],
        phase_x=job.options["phase_x"],
        phase_y=job.options["phase_y"],
        palette_mode="off",
        seed=job.options["seed"],
        steps=job.options["steps"],
        device=job.options["device"],
        strip_background=job.options["strip_background"],
        enable_phase_rerank=not job.options["skip_phase_rerank"],
        observer=job.observe,
    )
    job.mark_completed()


class QueueFullError(RuntimeError):
    """Raised when the hosted demo queue has no remaining waiting slots."""


class GuiJobManager:
    def __init__(self, config: HostedDemoConfig) -> None:
        self.config = config
        self.jobs: dict[str, GuiJob] = {}
        self._queued_job_ids: deque[str] = deque()
        self._active_job_id: str | None = None
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
            self.config.spool_dir.mkdir(parents=True, exist_ok=True)
            self._purge_spool_dir()
            self._worker_thread = threading.Thread(target=self._worker_loop, name="repixelizer-gui-worker", daemon=True)
            self._monitor_thread = threading.Thread(target=self._monitor_loop, name="repixelizer-gui-monitor", daemon=True)
            self._worker_thread.start()
            self._monitor_thread.start()

    def stop(self) -> None:
        with self._condition:
            if not self._started:
                return
            self._stop_event.set()
            self._condition.notify_all()
        for thread in (self._worker_thread, self._monitor_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        with self._condition:
            self._started = False
        self._purge_spool_dir()

    def submit_job(self, *, filename: str, raw: bytes, options: dict[str, Any]) -> GuiJob:
        self.start()
        suffix = Path(filename or "input.png").suffix or ".png"
        job = GuiJob(
            job_id=str(uuid.uuid4()),
            filename=filename or "input.png",
            options=options,
            spool_path=self.config.spool_dir / f"{uuid.uuid4().hex}{suffix}",
            phase_field_preview_stride=self.config.phase_field_preview_stride,
        )
        job.spool_path.write_bytes(raw)
        with self._condition:
            if len(self._queued_job_ids) >= self.config.queue_capacity:
                self._cleanup_spool_file(job)
                raise QueueFullError(f"Queue is full. {self.config.queue_capacity} waiting jobs are already lined up.")
            self.jobs[job.job_id] = job
            self._queued_job_ids.append(job.job_id)
            job.publish("job_state", {"status": job.status})
            self._publish_queue_state_locked()
            self._condition.notify_all()
        return job

    def get_job(self, job_id: str) -> GuiJob | None:
        with self._condition:
            return self.jobs.get(job_id)

    def get_job_state_payload(self, job_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            payload = _job_state_payload(job)
            payload.update(self._queue_state_payload_locked(job.job_id))
            return payload

    def get_queue_summary(self) -> dict[str, Any]:
        with self._condition:
            return {
                "queueDepth": self._queue_depth_locked(),
                "waitingCount": len(self._queued_job_ids),
                "queueCapacity": self.config.queue_capacity,
                "hasActiveJob": self._active_job_id is not None,
                "activeStatus": None if self._active_job_id is None else self.jobs[self._active_job_id].status,
            }

    def heartbeat(self, job_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.status in {"queued", "running"}:
                job.touch_heartbeat()
            payload = _job_state_payload(job)
            payload.update(self._queue_state_payload_locked(job.job_id))
            return payload

    def cancel_job(self, job_id: str, reason: str) -> dict[str, Any] | None:
        with self._condition:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.status == "queued":
                self._remove_from_queue_locked(job_id)
                job.request_cancel(reason)
                job.mark_canceled(reason)
                self._cleanup_spool_file(job)
                self._publish_queue_state_locked()
                payload = _job_state_payload(job)
                payload.update(self._queue_state_payload_locked(job.job_id))
                self._condition.notify_all()
                return payload
            if job.status == "running":
                job.request_cancel(reason)
                job.publish("job_state", {"status": job.status, "message": reason, "cancelRequested": True})
                payload = _job_state_payload(job)
                payload.update(self._queue_state_payload_locked(job.job_id))
                return payload
            payload = _job_state_payload(job)
            payload.update(self._queue_state_payload_locked(job.job_id))
            return payload

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while not self._stop_event.is_set() and not self._queued_job_ids:
                    self._condition.wait(timeout=0.5)
                if self._stop_event.is_set():
                    return
                job_id = self._queued_job_ids.popleft()
                job = self.jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                self._active_job_id = job_id
                self._publish_queue_state_locked()
            try:
                _execute_job(job)
            except PipelineCancelled as exc:
                job.mark_canceled(str(exc))
            except Exception as exc:  # pragma: no cover - exercised through manual GUI runs
                job.mark_failed(str(exc))
            finally:
                self._cleanup_spool_file(job)
                with self._condition:
                    if self._active_job_id == job.job_id:
                        self._active_job_id = None
                    self._publish_queue_state_locked()
                    self._condition.notify_all()

    def _monitor_loop(self) -> None:
        poll_interval = min(1.0, max(0.25, self.config.heartbeat_interval_seconds / 2))
        while not self._stop_event.wait(timeout=poll_interval):
            now = time.time()
            with self._condition:
                changed = False
                for job_id in list(self._queued_job_ids):
                    job = self.jobs.get(job_id)
                    if job is None:
                        self._remove_from_queue_locked(job_id)
                        changed = True
                        continue
                    if job.is_stale(now=now, stale_after_seconds=self.config.stale_after_seconds):
                        self._remove_from_queue_locked(job_id)
                        job.request_cancel("Queued job canceled after the browser stopped heartbeating.")
                        job.mark_canceled(job.cancellation_message)
                        self._cleanup_spool_file(job)
                        changed = True
                if self._active_job_id is not None:
                    active_job = self.jobs.get(self._active_job_id)
                    if (
                        active_job is not None
                        and active_job.is_stale(now=now, stale_after_seconds=self.config.stale_after_seconds)
                        and not active_job.check_cancelled()
                    ):
                        active_job.request_cancel("Running job canceled after the browser stopped heartbeating.")
                        active_job.publish(
                            "job_state",
                            {
                                "status": active_job.status,
                                "message": active_job.cancellation_message,
                                "cancelRequested": True,
                            },
                        )
                if changed:
                    self._publish_queue_state_locked()
                    self._condition.notify_all()

    def _remove_from_queue_locked(self, job_id: str) -> None:
        with contextlib.suppress(ValueError):
            self._queued_job_ids.remove(job_id)

    def _cleanup_spool_file(self, job: GuiJob) -> None:
        with contextlib.suppress(FileNotFoundError):
            job.spool_path.unlink()

    def _purge_spool_dir(self) -> None:
        if not self.config.spool_dir.exists():
            return
        for entry in self.config.spool_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                with contextlib.suppress(FileNotFoundError):
                    entry.unlink()

    def _queue_depth_locked(self) -> int:
        return len(self._queued_job_ids) + (1 if self._active_job_id is not None else 0)

    def _queue_position_locked(self, job_id: str) -> int | None:
        if self._active_job_id == job_id:
            return 1
        if job_id not in self._queued_job_ids:
            return None
        offset = 1 if self._active_job_id is not None else 0
        return offset + list(self._queued_job_ids).index(job_id) + 1

    def _queue_state_payload_locked(self, job_id: str) -> dict[str, Any]:
        return {
            "queuePosition": self._queue_position_locked(job_id),
            "queueDepth": self._queue_depth_locked(),
            "waitingCount": len(self._queued_job_ids),
            "queueCapacity": self.config.queue_capacity,
            "heartbeatIntervalSeconds": self.config.heartbeat_interval_seconds,
            "staleAfterSeconds": self.config.stale_after_seconds,
        }

    def _publish_queue_state_locked(self) -> None:
        targets: list[str] = []
        if self._active_job_id is not None:
            targets.append(self._active_job_id)
        targets.extend(self._queued_job_ids)
        for job_id in targets:
            job = self.jobs.get(job_id)
            if job is None or job.status not in {"queued", "running"}:
                continue
            job.publish("queue_state", self._queue_state_payload_locked(job_id))


def _static_dir() -> Path:
    return Path(__file__).with_name("gui_static")


def _versioned_gui_index(static_dir: Path) -> str:
    index_path = static_dir / "index.html"
    styles_path = static_dir / "styles.css"
    script_path = static_dir / "app.js"
    styles_version = int(styles_path.stat().st_mtime_ns)
    script_version = int(script_path.stat().st_mtime_ns)
    html = index_path.read_text(encoding="utf-8")
    html = html.replace("./styles.css", f"./styles.css?v={styles_version}")
    html = html.replace("./app.js", f"./app.js?v={script_version}")
    return html


def create_app():
    deps = _require_gui_dependencies()
    FastAPI = deps["FastAPI"]
    File = deps["File"]
    Form = deps["Form"]
    HTTPException = deps["HTTPException"]
    UploadFile = deps["UploadFile"]
    HTMLResponse = deps["HTMLResponse"]
    JSONResponse = deps["JSONResponse"]
    RedirectResponse = deps["RedirectResponse"]
    StreamingResponse = deps["StreamingResponse"]
    StaticFiles = deps["StaticFiles"]

    config = HostedDemoConfig.from_env()
    manager = GuiJobManager(config)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="Repixelizer GUI", version="0.1.0", lifespan=lifespan)
    static_dir = _static_dir()

    @app.middleware("http")
    async def disable_gui_asset_caching(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/app"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def get_config():
        return JSONResponse(config.public_payload())

    @app.get("/api/queue")
    def get_queue():
        return JSONResponse(manager.get_queue_summary())

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        payload = manager.get_job_state_payload(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return JSONResponse(payload)

    @app.post("/api/jobs")
    async def create_job(
        image: UploadFile = File(...),
        target_size: int | None = Form(default=None),
        target_width: int | None = Form(default=None),
        target_height: int | None = Form(default=None),
        phase_x: float | None = Form(default=None),
        phase_y: float | None = Form(default=None),
        steps: int | None = Form(default=None),
        seed: int = Form(default=7),
        device: str = Form(default="auto"),
        strip_background: bool = Form(default=False),
        skip_phase_rerank: bool = Form(default=False),
    ):
        raw = await image.read()
        filename = image.filename or "input.png"
        try:
            options, _source_width, _source_height = _validate_upload_request(
                config,
                raw=raw,
                filename=filename,
                target_size=target_size,
                target_width=target_width,
                target_height=target_height,
                phase_x=phase_x,
                phase_y=phase_y,
                steps=steps,
                seed=seed,
                device=device,
                strip_background=strip_background,
                skip_phase_rerank=skip_phase_rerank,
            )
            job = manager.submit_job(filename=filename, raw=raw, options=options)
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            detail = str(exc)
            status_code = 413 if "too large" in detail.lower() else 422
            raise HTTPException(status_code=status_code, detail=detail) from exc
        state_payload = manager.get_job_state_payload(job.job_id)
        assert state_payload is not None
        return JSONResponse(
            {
                "jobId": job.job_id,
                "status": state_payload["status"],
                "eventsUrl": f"/api/jobs/{job.job_id}/events",
                "stateUrl": f"/api/jobs/{job.job_id}",
                "queuePosition": state_payload["queuePosition"],
                "queueDepth": state_payload["queueDepth"],
                "queueCapacity": state_payload["queueCapacity"],
                "heartbeatIntervalSeconds": state_payload["heartbeatIntervalSeconds"],
                "staleAfterSeconds": state_payload["staleAfterSeconds"],
            }
        )

    @app.post("/api/jobs/{job_id}/heartbeat")
    def job_heartbeat(job_id: str):
        payload = manager.heartbeat(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return JSONResponse(payload)

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str):
        payload = manager.cancel_job(job_id, "Canceled because the browser left or explicitly bailed.")
        if payload is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return JSONResponse(payload)

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")

        def generate():
            index = 0
            while True:
                pending = job.wait_for_events(index)
                for record in pending:
                    index += 1
                    yield (
                        f"id: {record['id']}\n"
                        f"event: {record['event']}\n"
                        f"data: {json.dumps(record['payload'])}\n\n"
                    )
                if job.status in {"completed", "failed", "canceled"} and index >= len(job.events):
                    break
                yield ": keep-alive\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    if static_dir.exists():
        @app.get("/app")
        @app.get("/app/")
        def gui_index():
            return HTMLResponse(_versioned_gui_index(static_dir))

        app.mount("/app", StaticFiles(directory=static_dir, html=False), name="app")

        @app.get("/")
        def root():
            return RedirectResponse(url="/app/")
    else:  # pragma: no cover - only relevant before frontend assets exist
        @app.get("/")
        def root_missing():
            return HTMLResponse("<h1>GUI assets are missing.</h1><p>Build the TypeScript frontend first.</p>")

    return app


def main(*, host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> int:
    _require_gui_dependencies()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only when uvicorn missing
        raise RuntimeError("The GUI requires uvicorn. Install the project dependencies first.") from exc
    uvicorn.run("repixelizer.gui:create_app", factory=True, host=host, port=port, reload=reload)
    return 0
