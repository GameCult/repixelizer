from __future__ import annotations

from collections import deque

import numpy as np


def strip_edge_background(
    rgba: np.ndarray,
    *,
    border_width: int = 8,
    quantize_step: int = 8,
    representative_count: int = 6,
    color_threshold: float = 20.0 / 255.0,
    chroma_floor: float = 6.0 / 255.0,
    luminance_slack: float = 18.0 / 255.0,
    enclosed_min_span: int = 24,
    enclosed_max_aspect: float = 4.0,
    monochrome_bucket_fraction: float = 0.7,
    fringe_color_threshold: float = 28.0 / 255.0,
    fringe_chroma_slack: float = 10.0 / 255.0,
    fringe_luminance_slack: float = 10.0 / 255.0,
) -> np.ndarray:
    image = rgba.copy()
    if image.size == 0:
        return image
    border_rgb = _border_rgb_samples(image, border_width=border_width)
    if border_rgb.size == 0:
        return image

    rgb = image[..., :3]
    candidate = _monochrome_candidate_mask(
        rgb,
        border_rgb=border_rgb,
        quantize_step=quantize_step,
        color_threshold=max(color_threshold, 14.0 / 255.0),
        bucket_fraction=monochrome_bucket_fraction,
    )
    if candidate is None:
        representatives = _background_representatives(
            border_rgb,
            quantize_step=quantize_step,
            representative_count=representative_count,
        )
        border_luminance = _luminance(border_rgb)
        border_chroma = border_rgb.max(axis=1) - border_rgb.min(axis=1)
        luminance_min = max(0.0, float(np.percentile(border_luminance, 1.0) - luminance_slack))
        luminance_max = min(1.0, float(np.percentile(border_luminance, 99.0) + luminance_slack))
        chroma_limit = max(chroma_floor, float(np.percentile(border_chroma, 99.0) + 2.0 / 255.0))
        luminance = _luminance(rgb)
        chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
        flat_rgb = rgb.reshape(-1, 3)
        distances = np.linalg.norm(
            flat_rgb[:, None, :] - representatives[None, :, :],
            axis=-1,
        ).min(axis=1).reshape(rgb.shape[:2])
        candidate = (
            (distances <= color_threshold)
            & (chroma <= chroma_limit)
            & (luminance >= luminance_min)
            & (luminance <= luminance_max)
        )
        relaxed_candidate = (
            (distances <= max(fringe_color_threshold, color_threshold))
            & (chroma <= chroma_limit + fringe_chroma_slack)
            & (luminance >= max(0.0, luminance_min - fringe_luminance_slack))
            & (luminance <= min(1.0, luminance_max + fringe_luminance_slack))
        )
    else:
        border_luminance = _luminance(border_rgb)
        luminance_min = max(0.0, float(np.percentile(border_luminance, 1.0) - luminance_slack))
        luminance_max = min(1.0, float(np.percentile(border_luminance, 99.0) + luminance_slack))
        chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
        luminance = _luminance(rgb)
        dominant_color = _dominant_border_color(border_rgb, quantize_step=quantize_step)
        relaxed_distance = np.linalg.norm(rgb - dominant_color[None, None, :], axis=-1)
        relaxed_candidate = (
            (relaxed_distance <= max(fringe_color_threshold, color_threshold))
            & (chroma <= max(chroma_floor + fringe_chroma_slack, 16.0 / 255.0))
            & (luminance >= max(0.0, luminance_min - fringe_luminance_slack))
            & (luminance <= min(1.0, luminance_max + fringe_luminance_slack))
        )

    edge_connected = _edge_connected_mask(candidate)
    remove_mask = edge_connected | _enclosed_background_mask(
        candidate & ~edge_connected,
        min_span=enclosed_min_span,
        max_aspect=enclosed_max_aspect,
    )
    remove_mask = _grow_from_seed(relaxed_candidate, remove_mask)
    image[remove_mask, 3] = 0.0
    image[remove_mask, :3] = 0.0
    return image


def _border_rgb_samples(rgba: np.ndarray, *, border_width: int) -> np.ndarray:
    height, width = rgba.shape[:2]
    border = max(1, min(border_width, height // 2, width // 2))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    opaque = rgba[..., 3] > 0.95
    return rgba[..., :3][mask & opaque]


def _background_representatives(
    border_rgb: np.ndarray,
    *,
    quantize_step: int,
    representative_count: int,
) -> np.ndarray:
    step = max(1, quantize_step)
    border8 = np.clip(np.rint(border_rgb * 255.0), 0, 255).astype(np.int16)
    buckets = border8 // step
    keys, inverse, counts = np.unique(buckets, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1]
    representatives: list[np.ndarray] = []
    for bucket_index in order[: max(1, representative_count)]:
        members = border_rgb[inverse == bucket_index]
        representatives.append(members.mean(axis=0))
    return np.stack(representatives, axis=0).astype(np.float32)


def _monochrome_candidate_mask(
    rgb: np.ndarray,
    *,
    border_rgb: np.ndarray,
    quantize_step: int,
    color_threshold: float,
    bucket_fraction: float,
) -> np.ndarray | None:
    step = max(1, quantize_step)
    border8 = np.clip(np.rint(border_rgb * 255.0), 0, 255).astype(np.int16)
    buckets = border8 // step
    _, inverse, counts = np.unique(buckets, axis=0, return_inverse=True, return_counts=True)
    dominant_index = int(np.argmax(counts))
    dominant_fraction = float(counts[dominant_index] / max(1, border8.shape[0]))
    if dominant_fraction < bucket_fraction:
        return None
    dominant_color = border_rgb[inverse == dominant_index].mean(axis=0)
    distances = np.linalg.norm(rgb - dominant_color[None, None, :], axis=-1)
    return distances <= color_threshold


def _dominant_border_color(border_rgb: np.ndarray, *, quantize_step: int) -> np.ndarray:
    step = max(1, quantize_step)
    border8 = np.clip(np.rint(border_rgb * 255.0), 0, 255).astype(np.int16)
    buckets = border8 // step
    _, inverse, counts = np.unique(buckets, axis=0, return_inverse=True, return_counts=True)
    dominant_index = int(np.argmax(counts))
    return border_rgb[inverse == dominant_index].mean(axis=0)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _edge_connected_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if mask[0, x]:
            queue.append((0, x))
            visited[0, x] = True
        if mask[height - 1, x] and not visited[height - 1, x]:
            queue.append((height - 1, x))
            visited[height - 1, x] = True
    for y in range(height):
        if mask[y, 0] and not visited[y, 0]:
            queue.append((y, 0))
            visited[y, 0] = True
        if mask[y, width - 1] and not visited[y, width - 1]:
            queue.append((y, width - 1))
            visited[y, width - 1] = True
    while queue:
        y, x = queue.popleft()
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return visited


def _enclosed_background_mask(
    mask: np.ndarray,
    *,
    min_span: int,
    max_aspect: float,
) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    selected = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            coords: list[tuple[int, int]] = []
            left = right = x
            top = bottom = y
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                left = min(left, cx)
                right = max(right, cx)
                top = min(top, cy)
                bottom = max(bottom, cy)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            span_x = right - left + 1
            span_y = bottom - top + 1
            short_span = min(span_x, span_y)
            long_span = max(span_x, span_y)
            aspect = long_span / max(1, short_span)
            if short_span < min_span or aspect > max_aspect:
                continue
            for cy, cx in coords:
                selected[cy, cx] = True
    return selected


def _grow_from_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    grown = seed.copy()
    queue: deque[tuple[int, int]] = deque(map(tuple, np.argwhere(seed)))
    while queue:
        y, x = queue.popleft()
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if mask[ny, nx] and not grown[ny, nx]:
                    grown[ny, nx] = True
                    queue.append((ny, nx))
    return grown
