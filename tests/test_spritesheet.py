from __future__ import annotations

from pathlib import Path

import numpy as np

from repixelizer.io import save_rgba
from repixelizer.spritesheet import detect_sprite_regions, run_spritesheet
from repixelizer.types import CleanupArtifacts, InferenceResult, PhaseFieldSourceAnalysis, RunResult, SolverArtifacts


def _sheet_with_four_alpha_sprites() -> np.ndarray:
    sheet = np.zeros((40, 50, 4), dtype=np.float32)
    boxes = [
        (4, 5, 14, 15, (1.0, 0.0, 0.0)),
        (30, 6, 42, 17, (0.0, 1.0, 0.0)),
        (5, 25, 18, 36, (0.0, 0.0, 1.0)),
        (31, 24, 44, 35, (1.0, 1.0, 0.0)),
    ]
    for left, top, right, bottom, color in boxes:
        sheet[top:bottom, left:right, :3] = color
        sheet[top:bottom, left:right, 3] = 1.0
    return sheet


def test_detect_sprite_regions_returns_reading_order_regions() -> None:
    regions = detect_sprite_regions(_sheet_with_four_alpha_sprites())

    assert len(regions) == 4
    assert [region.index for region in regions] == [0, 1, 2, 3]
    assert regions[0].left < regions[1].left
    assert regions[0].top < regions[2].top


def test_spritesheet_runs_pipeline_per_detected_region(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "sheet.png"
    output_path = tmp_path / "out.png"
    diagnostics_dir = tmp_path / "diagnostics"
    save_rgba(input_path, _sheet_with_four_alpha_sprites())
    crop_shapes: list[tuple[int, int]] = []

    def fake_run_pipeline_rgba(source, **kwargs):
        crop_shapes.append(source.shape[:2])
        index = len(crop_shapes)
        rgba = np.zeros((3, 4, 4), dtype=np.float32)
        rgba[..., 0] = index / 4.0
        rgba[..., 3] = 1.0
        inference = InferenceResult(target_width=4, target_height=3, confidence=1.0)
        solver = SolverArtifacts(
            target_rgba=rgba,
            uv_field=np.zeros((3, 4, 2), dtype=np.float32),
            signal_strength=np.zeros((3, 4), dtype=np.float32),
            initial_rgba=rgba.copy(),
            loss_history=[],
        )
        return RunResult(
            source_rgba=source,
            output_rgba=rgba,
            inference=inference,
            analysis=PhaseFieldSourceAnalysis(edge_map=np.zeros(source.shape[:2], dtype=np.float32)),
            solver=solver,
            cleanup=CleanupArtifacts(cleaned_rgba=rgba, isolated_heatmap=np.zeros((3, 4), dtype=np.float32)),
            palette_result=None,
            diagnostics={},
        )

    monkeypatch.setattr("repixelizer.spritesheet.run_pipeline_rgba", fake_run_pipeline_rgba)

    result = run_spritesheet(
        input_path,
        output_path,
        sprite_count=4,
        diagnostics_dir=diagnostics_dir,
        steps=0,
        device="cpu",
    )

    assert output_path.exists()
    assert (diagnostics_dir / "spritesheet.json").exists()
    assert len(crop_shapes) == 4
    assert result.output_rgba.shape[:2] == (7, 9)
    assert result.diagnostics["sprite_count_detected"] == 4


def test_detect_sprite_regions_can_select_requested_largest_regions() -> None:
    sheet = _sheet_with_four_alpha_sprites()
    sheet[20, 24, :3] = 1.0
    sheet[20, 24, 3] = 1.0

    regions = detect_sprite_regions(sheet, sprite_count=4)

    assert len(regions) == 4
    assert all(region.foreground_pixels > 1 for region in regions)


def test_auto_detect_ignores_wide_stray_components() -> None:
    sheet = _sheet_with_four_alpha_sprites()
    sheet[19:22, 2:40, :3] = 1.0
    sheet[19:22, 2:40, 3] = 1.0

    regions = detect_sprite_regions(sheet)

    assert len(regions) == 4
    assert all(region.height > 5 for region in regions)


def test_spritesheet_auto_mode_pins_shared_density(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "sheet.png"
    output_path = tmp_path / "out.png"
    save_rgba(input_path, _sheet_with_four_alpha_sprites())
    pinned_targets: list[tuple[int | None, int | None]] = []
    probe_targets = iter([(2, 2), (3, 3), (2, 2), (2, 2)])

    class DummyInference:
        def __init__(self, width: int, height: int) -> None:
            self.target_width = width
            self.target_height = height

    def fake_infer_autocorr_lattice(source, **kwargs):
        del source, kwargs
        width, height = next(probe_targets)
        return DummyInference(width, height)

    def fake_run_pipeline_rgba(source, **kwargs):
        pinned_targets.append((kwargs["target_width"], kwargs["target_height"]))
        rgba = np.zeros((kwargs["target_height"], kwargs["target_width"], 4), dtype=np.float32)
        rgba[..., 3] = 1.0
        inference = InferenceResult(
            target_width=kwargs["target_width"],
            target_height=kwargs["target_height"],
            confidence=1.0,
        )
        solver = SolverArtifacts(
            target_rgba=rgba,
            uv_field=np.zeros((*rgba.shape[:2], 2), dtype=np.float32),
            signal_strength=np.zeros(rgba.shape[:2], dtype=np.float32),
            initial_rgba=rgba.copy(),
            loss_history=[],
        )
        return RunResult(
            source_rgba=source,
            output_rgba=rgba,
            inference=inference,
            analysis=PhaseFieldSourceAnalysis(edge_map=np.zeros(source.shape[:2], dtype=np.float32)),
            solver=solver,
            cleanup=CleanupArtifacts(cleaned_rgba=rgba, isolated_heatmap=np.zeros(rgba.shape[:2], dtype=np.float32)),
            palette_result=None,
            diagnostics={},
        )

    monkeypatch.setattr("repixelizer.spritesheet.infer_autocorr_lattice", fake_infer_autocorr_lattice)
    monkeypatch.setattr("repixelizer.spritesheet.run_pipeline_rgba", fake_run_pipeline_rgba)

    result = run_spritesheet(input_path, output_path, steps=0, device="cpu")

    assert output_path.exists()
    assert len(pinned_targets) == 4
    assert all(width is not None and height is not None for width, height in pinned_targets)
    assert result.diagnostics["shared_density"]["max_relative_deviation"] < 0.12
