from __future__ import annotations

import json
from pathlib import Path

from repixelizer.cli import main as cli_main
from repixelizer.eve_surface import build_repixelizer_eve_surface


def test_repixelizer_eve_surface_preserves_retro_style_tokens(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_QUEUE_CAPACITY", "7")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "off")

    payload = build_repixelizer_eve_surface(
        updated_at="2026-06-04T00:00:00Z",
        public_base_url="https://repixelizer.example",
    )

    assert payload["schema"] == "gamecult.eve.surface.v1"
    assert payload["providerId"] == "repixelizer"
    assert payload["surface"]["root"]["props"]["styleProfile"] == "repixelizer.retro.pixel"
    assert payload["runtime"]["hostedDemo"] is True
    assert payload["runtime"]["queue"]["queueCapacity"] == 7

    tokens = payload["surface"]["styles"]["tokens"]
    assert "Press Start 2P" in tokens["fontTitle"]
    assert "VT323" in tokens["fontBody"]
    assert tokens["colorBackgroundTop"] == "#01040b"
    assert tokens["colorAccent"] == "#ffd84a"
    assert tokens["imageRendering"] == "pixelated"
    assert tokens["scanlineOverlay"] is True

    commands = {entry["command"]: entry for entry in payload["commands"]}
    assert commands["repixelizer.job.submit"]["route"] == "/api/jobs"
    assert commands["repixelizer.job.cancel_own"]["method"] == "DELETE"

    android = payload["loweringComparison"]["periwinkleAndroidKotlin"]
    assert android["status"] == "surface-tree-renderer-present-style-token-gap"
    assert any("Press Start 2P" in gap for gap in android["losesUntilRendererParity"])


def test_eve_surface_cli_writes_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "off")
    out = tmp_path / "repixelizer.eve-surface.json"

    exit_code = cli_main(
        [
            "eve-surface",
            "--out",
            str(out),
            "--updated-at",
            "2026-06-04T00:00:00Z",
            "--public-base-url",
            "https://repixelizer.example",
        ]
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["updatedAt"] == "2026-06-04T00:00:00Z"
    assert payload["surface"]["styles"]["assets"]["appUrl"] == "https://repixelizer.example/app/"
