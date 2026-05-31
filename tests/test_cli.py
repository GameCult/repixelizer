from __future__ import annotations

from pathlib import Path

from repixelizer.cli import main
from repixelizer.io import save_rgba
from repixelizer.synthetic import fake_pixelize, make_emblem


def test_default_command_runs_without_explicit_run_subcommand(tmp_path: Path) -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, blur_radius=0.4)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    save_rgba(input_path, fake)
    exit_code = main([str(input_path), "--out", str(output_path), "--steps", "8"])
    assert exit_code == 0
    assert output_path.exists()


def test_default_command_accepts_strip_background_flag(tmp_path: Path) -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, blur_radius=0.4)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    save_rgba(input_path, fake)
    exit_code = main([str(input_path), "--out", str(output_path), "--steps", "0", "--strip-background"])
    assert exit_code == 0
    assert output_path.exists()


def test_default_command_accepts_fixed_lattice_flags(tmp_path: Path) -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, blur_radius=0.4)
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
            "--skip-candidate-rerank",
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


def test_spritesheet_command_dispatches_to_spritesheet_runner(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_run_spritesheet(input_path, output_path, **kwargs):
        called["input_path"] = input_path
        called["output_path"] = output_path
        called["sprite_count"] = kwargs["sprite_count"]
        called["sheet_columns"] = kwargs["sheet_columns"]
        called["sheet_rows"] = kwargs["sheet_rows"]
        called["export_sprites_dir"] = kwargs["export_sprites_dir"]
        called["target_width"] = kwargs["target_width"]
        called["target_height"] = kwargs["target_height"]
        called["enable_candidate_rerank"] = kwargs["enable_candidate_rerank"]

    monkeypatch.setattr("repixelizer.cli.run_spritesheet", fake_run_spritesheet)
    input_path = tmp_path / "sheet.png"
    output_path = tmp_path / "out.png"

    exit_code = main(
        [
            "spritesheet",
            str(input_path),
            "--out",
            str(output_path),
            "--sprite-count",
            "6",
            "--sheet-columns",
            "3",
            "--sheet-rows",
            "2",
            "--export-sprites-dir",
            str(tmp_path / "sprites"),
            "--target-width",
            "32",
            "--target-height",
            "32",
            "--skip-candidate-rerank",
        ]
    )

    assert exit_code == 0
    assert called == {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sprite_count": 6,
        "sheet_columns": 3,
        "sheet_rows": 2,
        "export_sprites_dir": str(tmp_path / "sprites"),
        "target_width": 32,
        "target_height": 32,
        "enable_candidate_rerank": False,
    }
