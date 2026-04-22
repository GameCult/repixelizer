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
    artifact_density: float = 0.0,
    artifact_strength: float = 0.0,
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
    else:
        rng = np.random.default_rng(seed)
    warped = _sample_rgba(target_rgba, sample_x, sample_y, mode=warp_sample_mode)
    if artifact_density > 0.0 and artifact_strength > 0.0:
        warped = _inject_fake_cell_artifacts(
            warped,
            upscale=upscale,
            density=artifact_density,
            strength=artifact_strength,
            rng=rng,
        )
    if blur_radius <= 0.0:
        return warped
    blurred = Image.fromarray(np.clip(np.rint(warped * 255.0), 0, 255).astype(np.uint8), mode="RGBA").filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _inject_fake_cell_artifacts(
    rgba: np.ndarray,
    *,
    upscale: int,
    density: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    premult = premultiply(rgba)
    out = premult.copy()
    height_px, width_px = premult.shape[:2]
    cell_w = max(1, int(upscale))
    cell_h = max(1, int(upscale))
    cells_x = max(1, width_px // cell_w)
    cells_y = max(1, height_px // cell_h)
    density = float(np.clip(density, 0.0, 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))
    op_count = max(1, int(round(cells_x * cells_y * density * 1.8)))
    max_sliver = max(1, int(round(cell_w * (0.08 + 0.18 * strength))))
    max_patch = max(1, int(round(cell_w * (0.10 + 0.22 * strength))))
    blend_alpha = 0.55 + 0.35 * strength

    def cell_bounds(cx: int, cy: int) -> tuple[int, int, int, int]:
        x0 = cx * cell_w
        y0 = cy * cell_h
        return x0, y0, min(width_px, x0 + cell_w), min(height_px, y0 + cell_h)

    def neighbors(cx: int, cy: int) -> list[tuple[int, int]]:
        items: list[tuple[int, int]] = []
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                if ox == 0 and oy == 0:
                    continue
                nx = cx + ox
                ny = cy + oy
                if 0 <= nx < cells_x and 0 <= ny < cells_y:
                    items.append((nx, ny))
        return items

    for _ in range(op_count):
        cx = int(rng.integers(0, cells_x))
        cy = int(rng.integers(0, cells_y))
        x0, y0, x1, y1 = cell_bounds(cx, cy)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        choices = neighbors(cx, cy)
        if not choices:
            continue
        nx, ny = choices[int(rng.integers(0, len(choices)))]
        nx0, ny0, nx1, ny1 = cell_bounds(nx, ny)
        op = str(rng.choice(["sliver", "ghost", "facet"]))

        if op == "sliver":
            strip = int(rng.integers(1, max_sliver + 1))
            dx = nx - cx
            dy = ny - cy
            if abs(dx) >= abs(dy):
                strip = min(strip, x1 - x0, nx1 - nx0)
                if strip < 1:
                    continue
                if dx < 0:
                    src = premult[ny0:ny1, nx1 - strip : nx1]
                    out[y0:y1, x0 : x0 + strip] = src[: y1 - y0, :strip]
                else:
                    src = premult[ny0:ny1, nx0 : nx0 + strip]
                    out[y0:y1, x1 - strip : x1] = src[: y1 - y0, :strip]
            else:
                strip = min(strip, y1 - y0, ny1 - ny0)
                if strip < 1:
                    continue
                if dy < 0:
                    src = premult[ny1 - strip : ny1, nx0:nx1]
                    out[y0 : y0 + strip, x0:x1] = src[:strip, : x1 - x0]
                else:
                    src = premult[ny0 : ny0 + strip, nx0:nx1]
                    out[y1 - strip : y1, x0:x1] = src[:strip, : x1 - x0]
            continue

        if op == "ghost":
            patch_w = min(x1 - x0 - 1, nx1 - nx0, max_patch)
            patch_h = min(y1 - y0 - 1, ny1 - ny0, max_patch)
            if patch_w < 1 or patch_h < 1:
                continue
            patch_w = int(rng.integers(1, patch_w + 1))
            patch_h = int(rng.integers(1, patch_h + 1))
            dst_x = int(rng.integers(x0, x1 - patch_w + 1))
            dst_y = int(rng.integers(y0, y1 - patch_h + 1))
            src_x = int(rng.integers(nx0, nx1 - patch_w + 1))
            src_y = int(rng.integers(ny0, ny1 - patch_h + 1))
            patch = premult[src_y : src_y + patch_h, src_x : src_x + patch_w]
            existing = out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w]
            out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = existing * (1.0 - blend_alpha) + patch * blend_alpha
            continue

        yy, xx = np.mgrid[0 : y1 - y0, 0 : x1 - x0]
        band = max(1.0, float(max_sliver))
        slope = float(rng.uniform(-0.55, 0.55))
        bias = float(rng.uniform(-0.2 * cell_w, 1.2 * cell_w))
        if rng.random() < 0.5:
            seam = xx.astype(np.float32) - (yy.astype(np.float32) * slope + bias)
        else:
            seam = xx.astype(np.float32) + (yy.astype(np.float32) * slope - bias)
        mask = np.abs(seam) <= band
        patch = premult[ny0:ny1, nx0:nx1][: y1 - y0, : x1 - x0]
        existing = out[y0:y1, x0:x1]
        mix = np.where(mask[..., None], patch, existing)
        blend = 0.35 * strength
        out[y0:y1, x0:x1] = existing * (1.0 - blend) + mix * blend

    return unpremultiply(np.clip(out, 0.0, 1.0))


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
