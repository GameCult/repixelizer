from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgba(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgba(path: str | Path, rgba: np.ndarray) -> None:
    clipped = np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="RGBA").save(path)


def premultiply(rgba: np.ndarray) -> np.ndarray:
    premult = rgba.copy()
    premult[..., :3] *= premult[..., 3:4]
    return premult


def unpremultiply(rgba: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = rgba.copy()
    alpha = np.maximum(out[..., 3:4], eps)
    out[..., :3] = np.where(out[..., 3:4] > eps, out[..., :3] / alpha, 0.0)
    return np.clip(out, 0.0, 1.0)


def nearest_resize(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8), mode="RGBA")
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) / 255.0


def box_resize(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8), mode="RGBA")
    resized = image.resize((width, height), resample=Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.float32) / 255.0


def bilinear_resize(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8), mode="RGBA")
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0
