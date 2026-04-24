from __future__ import annotations

import base64
import io
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .diagnostics import _displacement_preview_rgba
from .inference import inference_to_json
from .pipeline import run_pipeline_rgba
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
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    current_source_rgba: np.ndarray | None = None
    current_inference: InferenceResult | None = None
    run_summary: dict[str, Any] | None = None
    _condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

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
        self.publish("job_state", {"status": self.status})

    def mark_completed(self) -> None:
        self.status = "completed"
        self.publish("job_state", {"status": self.status})

    def mark_failed(self, message: str) -> None:
        self.status = "failed"
        self.error = message
        self.publish("job_failed", {"status": self.status, "message": message})

    def wait_for_events(self, index: int, timeout: float = 0.5) -> list[dict[str, Any]]:
        with self._condition:
            if index >= len(self.events) and self.status not in {"completed", "failed"}:
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
        "status": job.status,
        "error": job.error,
        "createdAt": job.created_at,
        "eventCount": len(job.events),
        "runSummary": job.run_summary,
    }


def _run_job(job: GuiJob, source_rgba: np.ndarray, options: dict[str, Any]) -> None:
    try:
        job.mark_running()
        run_pipeline_rgba(
            source_rgba,
            target_size=options["target_size"],
            target_width=options["target_width"],
            target_height=options["target_height"],
            phase_x=options["phase_x"],
            phase_y=options["phase_y"],
            palette_mode="off",
            seed=options["seed"],
            steps=options["steps"],
            device=options["device"],
            strip_background=options["strip_background"],
            enable_phase_rerank=not options["skip_phase_rerank"],
            observer=job.observe,
        )
    except Exception as exc:  # pragma: no cover - exercised through manual GUI runs
        job.mark_failed(str(exc))
        return
    job.mark_completed()


def _static_dir() -> Path:
    return Path(__file__).with_name("gui_static")


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

    app = FastAPI(title="Repixelizer GUI", version="0.1.0")
    jobs: dict[str, GuiJob] = {}
    static_dir = _static_dir()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return JSONResponse(_job_state_payload(job))

    @app.post("/api/jobs")
    async def create_job(
        image: UploadFile = File(...),
        target_size: int | None = Form(default=None),
        target_width: int | None = Form(default=None),
        target_height: int | None = Form(default=None),
        phase_x: float | None = Form(default=None),
        phase_y: float | None = Form(default=None),
        steps: int = Form(default=48),
        seed: int = Form(default=7),
        device: str = Form(default="auto"),
        strip_background: bool = Form(default=False),
        skip_phase_rerank: bool = Form(default=False),
    ):
        raw = await image.read()
        source_rgba = _decode_rgba(raw)
        job = GuiJob(job_id=str(uuid.uuid4()), filename=image.filename or "input.png")
        jobs[job.job_id] = job
        options = {
            "target_size": target_size,
            "target_width": target_width,
            "target_height": target_height,
            "phase_x": phase_x,
            "phase_y": phase_y,
            "steps": max(0, int(steps)),
            "seed": int(seed),
            "device": device,
            "strip_background": bool(strip_background),
            "skip_phase_rerank": bool(skip_phase_rerank),
        }
        worker = threading.Thread(target=_run_job, args=(job, source_rgba, options), daemon=True)
        worker.start()
        return JSONResponse(
            {
                "jobId": job.job_id,
                "status": job.status,
                "eventsUrl": f"/api/jobs/{job.job_id}/events",
                "stateUrl": f"/api/jobs/{job.job_id}",
            }
        )

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str):
        job = jobs.get(job_id)
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
                if job.status in {"completed", "failed"} and index >= len(job.events):
                    break
                yield ": keep-alive\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    if static_dir.exists():
        app.mount("/app", StaticFiles(directory=static_dir, html=True), name="app")

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
