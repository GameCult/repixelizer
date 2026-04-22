from __future__ import annotations

from pathlib import Path

from repixelizer.cli import main
from repixelizer.io import save_rgba
from repixelizer.synthetic import fake_pixelize, make_emblem


def test_default_command_runs_without_explicit_run_subcommand(tmp_path: Path) -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, phase_x=0.2, phase_y=0.25, blur_radius=0.4)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    save_rgba(input_path, fake)
    exit_code = main([str(input_path), "--out", str(output_path), "--steps", "8"])
    assert exit_code == 0
    assert output_path.exists()


def test_default_command_accepts_strip_background_flag(tmp_path: Path) -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, phase_x=0.2, phase_y=0.25, blur_radius=0.4)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    save_rgba(input_path, fake)
    exit_code = main([str(input_path), "--out", str(output_path), "--steps", "0", "--strip-background"])
    assert exit_code == 0
    assert output_path.exists()
