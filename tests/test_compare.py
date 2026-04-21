from __future__ import annotations

from pathlib import Path

from repixelizer.compare import run_compare
from repixelizer.io import save_rgba
from repixelizer.synthetic import fake_pixelize, make_emblem


def test_compare_writes_comparison_artifacts(tmp_path: Path) -> None:
    source = make_emblem(32, 32)
    fake = fake_pixelize(source, upscale=10, phase_x=0.2, phase_y=0.3, blur_radius=0.5)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    diagnostics_dir = tmp_path / "diagnostics"
    save_rgba(input_path, fake)

    payload = run_compare(input_path, output_path, diagnostics_dir=diagnostics_dir, steps=20)
    assert output_path.exists()
    assert (diagnostics_dir / "compare-sheet.png").exists()
    assert (diagnostics_dir / "compare.csv").exists()
    assert len(payload["rows"]) == 3
