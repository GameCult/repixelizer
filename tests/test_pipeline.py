from __future__ import annotations

from pathlib import Path

from repixelizer.baselines import naive_resize_baseline
from repixelizer.metrics import coherence_breakdown
from repixelizer.pipeline import run_pipeline
from repixelizer.synthetic import fake_pixelize, make_emblem, make_sprite


def test_pipeline_writes_output_and_diagnostics(tmp_path: Path) -> None:
    source = make_emblem(32, 32)
    fake = fake_pixelize(source, upscale=10, phase_x=0.15, phase_y=0.25, blur_radius=0.5)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    diagnostics_dir = tmp_path / "diagnostics"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(input_path, output_path, diagnostics_dir=diagnostics_dir, steps=24)
    assert output_path.exists()
    assert (diagnostics_dir / "run.json").exists()
    assert result.output_rgba.shape[0] == result.inference.target_height
    assert result.output_rgba.shape[1] == result.inference.target_width


def test_pipeline_beats_naive_on_coherence_for_synthetic_emblem(tmp_path: Path) -> None:
    source = make_emblem(32, 32)
    fake = fake_pixelize(source, upscale=12, phase_x=0.2, phase_y=0.35, blur_radius=0.65)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(input_path, output_path, steps=32)
    naive = naive_resize_baseline(fake, width=result.inference.target_width, height=result.inference.target_height)
    optimized_score = coherence_breakdown(result.output_rgba)["coherence_score"]
    naive_score = coherence_breakdown(naive)["coherence_score"]
    assert optimized_score >= naive_score


def test_pipeline_preserves_transparency_for_sprite(tmp_path: Path) -> None:
    source = make_sprite(24, 24)
    fake = fake_pixelize(source, upscale=9, phase_x=0.1, phase_y=0.22, blur_radius=0.45)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(input_path, output_path, steps=20)
    alpha = result.output_rgba[..., 3]
    assert alpha.min() < 0.05
    assert alpha.max() > 0.9
