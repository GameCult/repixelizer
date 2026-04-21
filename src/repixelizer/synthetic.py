from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .io import premultiply, unpremultiply


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
    warp_strength: float = 0.0,
    warp_detail: int = 4,
    warp_sample_mode: str = "bilinear",
    seed: int = 0,
) -> np.ndarray:
    source = Image.fromarray(np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8), mode="RGBA")
    width, height = source.size
    target = source.resize((width * upscale, height * upscale), resample=Image.Resampling.NEAREST)
    target_rgba = np.asarray(target, dtype=np.float32) / 255.0
    height_px, width_px = target_rgba.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width_px, dtype=np.float32), np.arange(height_px, dtype=np.float32))
    sample_x = grid_x - phase_x * float(upscale)
    sample_y = grid_y - phase_y * float(upscale)
    if warp_strength > 0.0:
        rng = np.random.default_rng(seed)
        strength_px = float(upscale) * warp_strength
        warp_x, warp_y = _warp_fields(width_px, height_px, upscale=upscale, detail=warp_detail, rng=rng)
        sample_x -= warp_x * strength_px
        sample_y -= warp_y * strength_px
    warped = _sample_rgba(target_rgba, sample_x, sample_y, mode=warp_sample_mode)
    if blur_radius <= 0.0:
        return warped
    blurred = Image.fromarray(np.clip(np.rint(warped * 255.0), 0, 255).astype(np.uint8), mode="RGBA").filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _warp_fields(width: int, height: int, *, upscale: int, detail: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    smooth_tile = max(float(upscale * (detail + 2)), 2.0)
    block_tile = max(float(upscale * max(1.5, 7.5 - detail)), 2.0)
    patch_tile = max(float(upscale * max(1.0, 5.5 - detail * 0.5)), 2.0)
    stripe_tile = max(float(upscale * max(1.0, 6.5 - detail)), 2.0)
    smooth_blur = max(0.0, upscale * 0.2)
    block_blur = max(0.0, upscale * 0.32)
    patch_blur = max(0.0, upscale * 0.12)
    stripe_blur = max(0.0, upscale * 0.16)
    warp_x = (
        _noise_field(width, height, tile_size=smooth_tile, rng=rng, blur_radius=smooth_blur) * 0.15
        + _noise_field(
            width,
            height,
            tile_size=block_tile,
            rng=rng,
            resample=Image.Resampling.NEAREST,
            blur_radius=block_blur,
        )
        * 0.45
        + _noise_field(
            width,
            height,
            tile_size=patch_tile,
            rng=rng,
            resample=Image.Resampling.NEAREST,
            blur_radius=patch_blur,
        )
        * 0.25
        + _stripe_field(
            width,
            height,
            tile_size=stripe_tile,
            rng=rng,
            axis="x",
            blur_radius=stripe_blur,
        )
        * 0.15
    )
    warp_y = (
        _noise_field(width, height, tile_size=smooth_tile, rng=rng, blur_radius=smooth_blur) * 0.15
        + _noise_field(
            width,
            height,
            tile_size=block_tile,
            rng=rng,
            resample=Image.Resampling.NEAREST,
            blur_radius=block_blur,
        )
        * 0.45
        + _noise_field(
            width,
            height,
            tile_size=patch_tile,
            rng=rng,
            resample=Image.Resampling.NEAREST,
            blur_radius=patch_blur,
        )
        * 0.25
        + _stripe_field(
            width,
            height,
            tile_size=stripe_tile,
            rng=rng,
            axis="y",
            blur_radius=stripe_blur,
        )
        * 0.15
    )
    return np.clip(warp_x, -1.25, 1.25), np.clip(warp_y, -1.25, 1.25)


def _noise_field(
    width: int,
    height: int,
    *,
    tile_size: float,
    rng: np.random.Generator,
    resample: int = Image.Resampling.BILINEAR,
    blur_radius: float = 0.0,
) -> np.ndarray:
    coarse_width = max(2, int(np.ceil(width / max(tile_size, 1.0))) + 1)
    coarse_height = max(2, int(np.ceil(height / max(tile_size, 1.0))) + 1)
    noise = (rng.random((coarse_height, coarse_width)) * 255.0).astype(np.uint8)
    image = Image.fromarray(noise, mode="L").resize((width, height), resample=resample)
    if blur_radius > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(image, dtype=np.float32) / 255.0 * 2.0 - 1.0


def _stripe_field(
    width: int,
    height: int,
    *,
    tile_size: float,
    rng: np.random.Generator,
    axis: str,
    blur_radius: float,
) -> np.ndarray:
    if axis == "x":
        coarse_height = max(2, int(np.ceil(height / max(tile_size, 1.0))) + 1)
        noise = (rng.random((coarse_height, 1)) * 255.0).astype(np.uint8)
        image = Image.fromarray(noise, mode="L").resize((1, height), resample=Image.Resampling.NEAREST)
        if blur_radius > 0.0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        values = np.asarray(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
        return np.repeat(values, width, axis=1)
    coarse_width = max(2, int(np.ceil(width / max(tile_size, 1.0))) + 1)
    noise = (rng.random((1, coarse_width)) * 255.0).astype(np.uint8)
    image = Image.fromarray(noise, mode="L").resize((width, 1), resample=Image.Resampling.NEAREST)
    if blur_radius > 0.0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    values = np.asarray(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return np.repeat(values, height, axis=0)


def _sample_rgba(rgba: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "nearest":
        return _nearest_sample_rgba(rgba, sample_x, sample_y)
    if mode == "bilinear":
        return _bilinear_sample_rgba(rgba, sample_x, sample_y)
    raise ValueError(f"Unsupported warp sample mode: {mode}")


def _nearest_sample_rgba(rgba: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[:2]
    x = np.clip(np.rint(sample_x), 0, max(0, width - 1)).astype(np.int32)
    y = np.clip(np.rint(sample_y), 0, max(0, height - 1)).astype(np.int32)
    return rgba[y, x]


def _bilinear_sample_rgba(rgba: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    premult = premultiply(rgba)
    height, width = premult.shape[:2]
    clamped_x = np.clip(sample_x, 0.0, max(0.0, width - 1))
    clamped_y = np.clip(sample_y, 0.0, max(0.0, height - 1))
    x0 = np.floor(clamped_x).astype(np.int32)
    y0 = np.floor(clamped_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, max(0, width - 1))
    y1 = np.clip(y0 + 1, 0, max(0, height - 1))
    wx = (clamped_x - x0).astype(np.float32)[..., None]
    wy = (clamped_y - y0).astype(np.float32)[..., None]
    top = premult[y0, x0] * (1.0 - wx) + premult[y0, x1] * wx
    bottom = premult[y1, x0] * (1.0 - wx) + premult[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return unpremultiply(np.clip(sampled, 0.0, 1.0))
