from __future__ import annotations

import numpy as np

from .io import box_resize
from .palette import derive_palette


def naive_resize_baseline(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    return box_resize(rgba, width=width, height=height)


def _nearest_palette_color(pixel: np.ndarray, palette: np.ndarray) -> np.ndarray:
    distances = np.sum((palette - pixel[None, :]) ** 2, axis=1)
    return palette[int(np.argmin(distances))]


def error_diffusion_baseline(
    rgba: np.ndarray,
    width: int,
    height: int,
    palette: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    resized = box_resize(rgba, width=width, height=height).copy()
    palette_array = np.asarray(palette or derive_palette(resized), dtype=np.float32) / 255.0
    rgb = resized[..., :3].copy()
    alpha = resized[..., 3:4].copy()
    height_px, width_px = rgb.shape[:2]
    for y in range(height_px):
        serpentine = y % 2 == 1
        x_range = range(width_px - 1, -1, -1) if serpentine else range(width_px)
        for x in x_range:
            old = rgb[y, x].copy()
            new = _nearest_palette_color(old, palette_array)
            rgb[y, x] = new
            error = old - new
            neighbors = (
                [(-1, 1, 7 / 16), (0, 1, 5 / 16), (-1, 0, 3 / 16), (1, 0, 1 / 16)]
                if serpentine
                else [(1, 0, 7 / 16), (0, 1, 5 / 16), (-1, 1, 3 / 16), (1, 1, 1 / 16)]
            )
            for dx, dy, weight in neighbors:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width_px and 0 <= ny < height_px:
                    rgb[ny, nx] = np.clip(rgb[ny, nx] + error * weight, 0.0, 1.0)
    return np.concatenate([rgb, alpha], axis=-1)
