from __future__ import annotations

import numpy as np
import pytest

from repixelizer.analysis import analyze_phase_field_source
from repixelizer.observe import PipelineCancelled
from repixelizer.phase_field import optimize_phase_field
from repixelizer.synthetic import fake_pixelize, make_emblem
from repixelizer.types import InferenceResult


def test_optimize_phase_field_emits_displacement_diagnostics_and_source_colors() -> None:
    source = np.zeros((12, 12, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[2:10, 2:10, :3] = np.asarray([0.8, 0.2, 0.1], dtype=np.float32)
    source[4:8, 4:8, :3] = np.asarray([0.1, 0.1, 0.9], dtype=np.float32)

    inference = InferenceResult(
        target_width=6,
        target_height=6,
        phase_x=0.0,
        phase_y=0.0,
        confidence=1.0,
        top_candidates=[],
    )
    artifacts = optimize_phase_field(
        source,
        inference=inference,
        analysis=analyze_phase_field_source(source, seed=7),
        steps=4,
        seed=7,
        device="cpu",
    )

    displacement = artifacts.stage_diagnostics["displacements"]
    assert {"initial_output", "final_output"} <= set(displacement.keys())
    assert displacement["final_output"]["displacement_x"].shape == (6, 6)
    assert displacement["final_output"]["displacement_y"].shape == (6, 6)
    assert artifacts.stage_diagnostics["phase_field"]["mean_displacement_px"] >= 0.0

    source_colors = {tuple(px.tolist()) for px in np.clip(np.rint(source * 255.0), 0, 255).astype(np.uint8).reshape(-1, 4)}
    output_colors = {tuple(px.tolist()) for px in np.clip(np.rint(artifacts.target_rgba * 255.0), 0, 255).astype(np.uint8).reshape(-1, 4)}
    assert output_colors <= source_colors


def test_optimize_phase_field_moves_off_zero_when_phase_is_wrong() -> None:
    source = make_emblem(18, 18)
    fake = fake_pixelize(
        source,
        upscale=8,
        phase_x=0.18,
        phase_y=-0.14,
        blur_radius=0.35,
        warp_strength=0.12,
        warp_detail=4,
        seed=4,
    )
    inference = InferenceResult(
        target_width=18,
        target_height=18,
        phase_x=0.0,
        phase_y=0.0,
        confidence=1.0,
        top_candidates=[],
    )
    artifacts = optimize_phase_field(
        fake,
        inference=inference,
        analysis=analyze_phase_field_source(fake, seed=7),
        steps=6,
        seed=7,
        device="cpu",
    )

    assert artifacts.stage_diagnostics["phase_field"]["mean_displacement_px"] > 0.01


def test_optimize_phase_field_honors_cooperative_cancellation() -> None:
    source = make_emblem(12, 12)
    inference = InferenceResult(
        target_width=12,
        target_height=12,
        phase_x=0.0,
        phase_y=0.0,
        confidence=1.0,
        top_candidates=[],
    )

    class CancelObserver:
        def __call__(self, event: str, payload: dict[str, object]) -> None:
            del event, payload

        def __init__(self) -> None:
            self.calls = 0

        def check_cancelled(self) -> bool:
            self.calls += 1
            return self.calls > 1

    with pytest.raises(PipelineCancelled):
        optimize_phase_field(
            source,
            inference=inference,
            analysis=analyze_phase_field_source(source, seed=7),
            steps=4,
            seed=7,
            device="cpu",
            observer=CancelObserver(),
        )
