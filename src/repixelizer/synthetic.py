from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_emblem(width: int = 32, height: int = 32) -> np.ndarray:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 6, 27, 28), radius=8, fill=(194, 24, 24, 255), outline=(255, 220, 96, 255), width=2)
    draw.polygon([(16, 3), (20, 9), (16, 29), (12, 9)], fill=(185, 220, 255, 255), outline=(40, 52, 80, 255))
    draw.ellipse((4, 4, 15, 15), fill=(24, 188, 120, 255))
    draw.ellipse((8, 7, 12, 11), fill=(240, 255, 255, 255))
    return np.asarray(image, dtype=np.float32) / 255.0


def make_sprite(width: int = 24, height: int = 24) -> np.ndarray:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 4, 15, 18), fill=(48, 88, 220, 255), outline=(244, 248, 255, 255))
    draw.rectangle((9, 2, 14, 7), fill=(255, 210, 162, 255), outline=(72, 32, 24, 255))
    draw.line((10, 18, 8, 22), fill=(42, 42, 42, 255), width=1)
    draw.line((13, 18, 15, 22), fill=(42, 42, 42, 255), width=1)
    draw.line((8, 10, 4, 14), fill=(255, 210, 162, 255), width=1)
    draw.line((15, 10, 19, 6), fill=(255, 210, 162, 255), width=1)
    return np.asarray(image, dtype=np.float32) / 255.0


def fake_pixelize(
    rgba: np.ndarray,
    upscale: int = 14,
    phase_x: float = 0.22,
    phase_y: float = 0.31,
    blur_radius: float = 0.7,
) -> np.ndarray:
    source = Image.fromarray(np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8), mode="RGBA")
    width, height = source.size
    target = source.resize((width * upscale, height * upscale), resample=Image.Resampling.NEAREST)
    shift_x = int(round(upscale * phase_x))
    shift_y = int(round(upscale * phase_y))
    shifted = Image.new("RGBA", target.size, (0, 0, 0, 0))
    shifted.alpha_composite(target, (shift_x, shift_y))
    blurred = shifted.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0
