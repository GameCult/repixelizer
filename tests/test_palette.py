from __future__ import annotations

from pathlib import Path

from repixelizer.palette import load_palette, quantize_rgba
from repixelizer.synthetic import make_emblem


def test_load_gpl_palette(tmp_path: Path) -> None:
    palette_path = tmp_path / "palette.gpl"
    palette_path.write_text(
        "GIMP Palette\nName: Test\nColumns: 2\n255 0 0 Red\n0 255 0 Green\n",
        encoding="utf-8",
    )
    palette = load_palette(palette_path)
    assert palette == [(255, 0, 0), (0, 255, 0)]


def test_strict_quantization_stays_in_palette() -> None:
    rgba = make_emblem(8, 8)
    palette = [(255, 0, 0), (0, 255, 0), (255, 255, 255)]
    result = quantize_rgba(rgba, mode="strict", palette=palette)
    assert result is not None
    unique = {
        tuple((pixel[:3] * 255.0).round().astype(int))
        for pixel in result.rgba.reshape(-1, 4)
        if pixel[3] > 0.05
    }
    assert unique.issubset(set(palette))
