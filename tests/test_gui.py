from __future__ import annotations

import importlib.util
import io
from pathlib import Path

from PIL import Image
from starlette.datastructures import UploadFile

from repixelizer.pipeline import run_pipeline_rgba
from repixelizer.synthetic import fake_pixelize, make_emblem


def test_run_pipeline_rgba_emits_observer_events_for_gui() -> None:
    source = make_emblem(20, 20)
    fake = fake_pixelize(source, upscale=8, phase_x=0.15, phase_y=0.2, blur_radius=0.45, seed=3)
    events: list[str] = []

    def observer(event: str, payload) -> None:
        del payload
        events.append(event)

    run_pipeline_rgba(
        fake,
        target_width=20,
        target_height=20,
        phase_x=0.0,
        phase_y=0.0,
        steps=2,
        device="cpu",
        enable_phase_rerank=False,
        observer=observer,
    )

    assert events[0] == "source_loaded"
    assert events.count("stage_started") == 6
    assert events.index("inference_candidates_ready") < events.index("analysis_completed")
    assert events.index("analysis_completed") < events.index("phase_selection_completed")
    assert events.index("phase_selection_completed") < events.index("phase_field_prepared")
    assert "phase_field_prepared" in events
    assert "phase_field_initial" in events
    assert events.count("phase_field_step") == 2
    assert events[-4:] == ["cleanup_completed", "stage_started", "palette_completed", "pipeline_completed"]


def test_repo_gui_runner_dispatches_to_gui_main(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_gui.py"
    spec = importlib.util.spec_from_file_location("repixelizer_run_gui_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    called = {}

    def fake_gui_main(*, host: str, port: int, reload: bool) -> int:
        called["host"] = host
        called["port"] = port
        called["reload"] = reload
        return 0

    monkeypatch.setattr("repixelizer.gui.main", fake_gui_main)
    exit_code = module.main(["--host", "127.0.0.1", "--port", "8123", "--reload"])
    assert exit_code == 0
    assert called == {"host": "127.0.0.1", "port": 8123, "reload": True}


def test_create_job_upload_field_validates_without_forward_ref_errors() -> None:
    from repixelizer.gui import create_app

    image = Image.new("RGBA", (1, 1), (12, 34, 56, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    app = create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/api/jobs")
    image_field = next(field for field in route.dependant.body_params if field.name == "image")
    upload = UploadFile(filename="tiny.png", file=io.BytesIO(buffer.getvalue()))
    value, errors = image_field.validate(upload, {}, loc=("body", "image"))

    assert errors == []
    assert value.filename == "tiny.png"
