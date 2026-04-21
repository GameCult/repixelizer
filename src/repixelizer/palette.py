from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .types import PaletteResult


def _parse_hex_color(value: str) -> tuple[int, int, int] | None:
    token = value.strip().lstrip("#")
    if len(token) != 6:
        return None
    try:
        return int(token[0:2], 16), int(token[2:4], 16), int(token[4:6], 16)
    except ValueError:
        return None


def load_palette(path: str | Path) -> list[tuple[int, int, int]]:
    palette_path = Path(path)
    if palette_path.suffix.lower() == ".json":
        data = json.loads(palette_path.read_text(encoding="utf-8"))
        return [tuple(color) for color in data["palette"]]
    colors: list[tuple[int, int, int]] = []
    for raw_line in palette_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("GIMP Palette") or line.startswith("Name:") or line.startswith("Columns:"):
            continue
        if line.startswith("Channels:"):
            continue
        hex_color = _parse_hex_color(line)
        if hex_color is not None:
            colors.append(hex_color)
            continue
        parts = line.split()
        if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
            colors.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not colors:
        raise ValueError(f"No palette colors found in {palette_path}")
    return colors


def save_palette_report(path: str | Path, palette: list[tuple[int, int, int]]) -> None:
    report = {"palette": palette}
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def derive_palette(rgba: np.ndarray, max_colors: int = 32) -> list[tuple[int, int, int]]:
    image = Image.fromarray(np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8), mode="RGBA")
    opaque_rgb = np.asarray(image)[..., :3]
    unique = np.unique(opaque_rgb.reshape(-1, 3), axis=0)
    colors = int(np.clip(unique.shape[0], 8, max_colors))
    quantized = image.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()[: colors * 3]
    return [tuple(palette[index : index + 3]) for index in range(0, len(palette), 3)]


def quantize_rgba(
    rgba: np.ndarray,
    mode: str,
    palette: list[tuple[int, int, int]] | None = None,
) -> PaletteResult | None:
    if mode == "off":
        return None
    if mode == "strict" and not palette:
        raise ValueError("Strict palette mode requires a palette file.")
    use_palette = palette or derive_palette(rgba)
    rgb = np.clip(np.rint(rgba[..., :3] * 255.0), 0, 255).astype(np.int16)
    alpha = rgba[..., 3:4]
    palette_array = np.asarray(use_palette, dtype=np.int16)
    flat = rgb.reshape(-1, 3)
    distances = np.sum((flat[:, None, :] - palette_array[None, :, :]) ** 2, axis=-1)
    indices = np.argmin(distances, axis=1)
    quantized_rgb = palette_array[indices].reshape(rgb.shape).astype(np.float32) / 255.0
    quantized_rgba = np.concatenate([quantized_rgb, alpha], axis=-1)
    return PaletteResult(rgba=quantized_rgba, palette=use_palette, indexed_rgba=None, indexed_png_path=None)
