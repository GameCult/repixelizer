from __future__ import annotations

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

    assert events[:4] == [
        "source_loaded",
        "inference_candidates_ready",
        "analysis_completed",
        "phase_selection_completed",
    ]
    assert "phase_field_prepared" in events
    assert "phase_field_initial" in events
    assert events.count("phase_field_step") == 2
    assert events[-3:] == ["cleanup_completed", "palette_completed", "pipeline_completed"]
