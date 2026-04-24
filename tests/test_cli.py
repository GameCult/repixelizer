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


def test_default_command_accepts_fixed_lattice_flags(tmp_path: Path) -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, phase_x=0.2, phase_y=0.25, blur_radius=0.4)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    save_rgba(input_path, fake)
    exit_code = main(
        [
            str(input_path),
            "--out",
            str(output_path),
            "--steps",
            "0",
            "--target-width",
            "16",
            "--target-height",
            "16",
            "--phase-x",
            "0.2",
            "--phase-y",
            "0.25",
            "--skip-phase-rerank",
        ]
    )
    assert exit_code == 0
    assert output_path.exists()


def test_gui_command_dispatches_to_gui_main(monkeypatch) -> None:
    called = {}

    def fake_gui_main(*, host: str, port: int, reload: bool) -> int:
        called["host"] = host
        called["port"] = port
        called["reload"] = reload
        return 0

    monkeypatch.setattr("repixelizer.cli.gui_main", fake_gui_main)
    exit_code = main(["gui", "--host", "127.0.0.1", "--port", "8123", "--reload"])
    assert exit_code == 0
    assert called == {"host": "127.0.0.1", "port": 8123, "reload": True}
