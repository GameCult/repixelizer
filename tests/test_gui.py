from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
from PIL import Image
from starlette.datastructures import UploadFile

import repixelizer.inference as inference_module
import repixelizer.pipeline as pipeline_module
from repixelizer.pipeline import run_pipeline_rgba
from repixelizer.synthetic import fake_pixelize, make_emblem
from repixelizer.params import SolverHyperParams
from repixelizer.types import InferenceCandidate, InferenceResult, PhaseFieldSourceAnalysis, SolverArtifacts


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


def test_infer_lattice_emits_search_progress_events(monkeypatch) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setattr(inference_module, "_require_torch", lambda: (_FakeTorch(), object()))
    monkeypatch.setattr(inference_module, "_estimate_lattice_spacing", lambda rgba: ((None, 0.0), (None, 0.0)))
    monkeypatch.setattr(inference_module, "_hint_target_sizes_from_spacing", lambda width, height, spacing_x, spacing_y: [])
    monkeypatch.setattr(inference_module, "_estimate_lattice_prior_details", lambda rgba: (4.0, 4.0, 0.5))
    monkeypatch.setattr(
        inference_module,
        "_candidate_dims",
        lambda width, height, target_size, *, hinted_sizes=None: [(10, 8), (12, 10)],
    )

    def fake_score_phase_group(
        rgba,
        *,
        target_width,
        target_height,
        prior_cell_x,
        prior_cell_y,
        prior_reliability,
        phase_x_values,
        phase_y_values,
        device,
    ):
        del rgba, prior_cell_x, prior_cell_y, prior_reliability, phase_x_values, phase_y_values, device
        return [
            InferenceCandidate(
                target_width=target_width,
                target_height=target_height,
                phase_x=0.0,
                phase_y=0.0,
                score=0.9 if target_width == 12 else 0.6,
                breakdown={},
            )
        ]

    monkeypatch.setattr(inference_module, "_score_phase_group", fake_score_phase_group)
    monkeypatch.setattr(inference_module, "_top_candidates_by_size", lambda candidates, limit: candidates[:limit])
    monkeypatch.setattr(
        inference_module,
        "_rerank_size_candidates_with_source_evidence",
        lambda rgba, candidates: sorted(candidates, key=lambda candidate: candidate.score, reverse=True),
    )

    events: list[tuple[str, dict[str, object]]] = []

    def observer(event: str, payload: dict[str, object]) -> None:
        events.append((event, payload))

    result = inference_module.infer_lattice(np.zeros((16, 20, 4), dtype=np.uint8), observer=observer)

    assert result.target_width == 12
    assert [event for event, _payload in events] == [
        "lattice_search_started",
        "lattice_search_progress",
        "lattice_search_progress",
    ]
    assert events[0][1]["candidate_count"] == 2
    assert events[1][1]["completed_candidates"] == 1
    assert events[2][1]["completed_candidates"] == 2
    assert events[2][1]["target_width"] == 12


def test_phase_rerank_emits_candidate_progress_events(monkeypatch) -> None:
    inference = InferenceResult(
        target_width=16,
        target_height=16,
        phase_x=0.0,
        phase_y=0.0,
        confidence=0.0,
        top_candidates=[
            InferenceCandidate(target_width=16, target_height=16, phase_x=0.0, phase_y=0.0, score=0.75, breakdown={}),
            InferenceCandidate(target_width=18, target_height=18, phase_x=0.1, phase_y=-0.1, score=0.73, breakdown={}),
        ],
    )
    analysis = PhaseFieldSourceAnalysis(edge_map=np.zeros((4, 4), dtype=np.float32))

    def fake_run_reconstruction(source, *, observer=None, inference, **kwargs):
        del source, kwargs
        if observer is not None:
            observer("phase_field_initial", {"step": 0, "total_steps": 2})
            observer("phase_field_step", {"step": 1, "total_steps": 2, "loss": 0.25})
            observer("phase_field_step", {"step": 2, "total_steps": 2, "loss": 0.125})
        target_rgba = np.zeros((inference.target_height, inference.target_width, 4), dtype=np.uint8)
        artifacts = SolverArtifacts(
            target_rgba=target_rgba,
            uv_field=np.zeros((inference.target_height, inference.target_width, 2), dtype=np.float32),
            guidance_strength=np.zeros((inference.target_height, inference.target_width), dtype=np.float32),
            initial_rgba=target_rgba.copy(),
            loss_history=[0.25, 0.125],
        )
        return artifacts, {}

    monkeypatch.setattr(pipeline_module, "_run_reconstruction", fake_run_reconstruction)
    monkeypatch.setattr(pipeline_module, "source_lattice_consistency_breakdown", lambda *args, **kwargs: {"score": 0.8})
    monkeypatch.setattr(
        pipeline_module,
        "nearest_resize",
        lambda rgba, width, height: np.zeros((height, width, 4), dtype=np.uint8),
    )
    monkeypatch.setattr(pipeline_module, "foreground_edge_position_error", lambda preview, source: 0.12)
    monkeypatch.setattr(pipeline_module, "foreground_stroke_wobble_error", lambda preview, source: 0.08)
    monkeypatch.setattr(pipeline_module, "foreground_edge_concentration", lambda rgba: 0.9)

    events: list[tuple[str, dict[str, object]]] = []

    def observer(event: str, payload: dict[str, object]) -> None:
        events.append((event, payload))

    pipeline_module._select_phase_candidate_with_reconstruction(
        np.zeros((16, 16, 4), dtype=np.uint8),
        inference,
        analysis=analysis,
        steps=2,
        seed=1,
        device="cpu",
        solver_params=SolverHyperParams(phase_rerank_preview_steps=2, phase_rerank_confidence_threshold=1.0),
        observer=observer,
    )

    event_names = [event for event, _payload in events]
    assert event_names[0] == "phase_rerank_started"
    assert event_names.count("phase_rerank_candidate_started") == 2
    assert event_names.count("phase_rerank_candidate_step") == 6
    assert event_names.count("phase_rerank_candidate_completed") == 2
    first_step_payload = next(payload for event, payload in events if event == "phase_rerank_candidate_step")
    assert first_step_payload["candidate_index"] == 1
    assert first_step_payload["total_steps"] == 2
