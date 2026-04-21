from __future__ import annotations

import numpy as np


def luminance(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., 0] * 0.2126 + rgba[..., 1] * 0.7152 + rgba[..., 2] * 0.0722


def alpha_crispness(rgba: np.ndarray) -> float:
    alpha = rgba[..., 3]
    return float(1.0 - np.mean(4.0 * alpha * (1.0 - alpha)))


def isolated_pixel_rate(rgba: np.ndarray, threshold: float = 0.15) -> float:
    color = rgba[..., :4]
    diffs = []
    neighbors = []
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        shifted = np.roll(color, shift=(dy, dx), axis=(0, 1))
        dist = np.linalg.norm(color - shifted, axis=-1)
        diffs.append(dist)
        neighbors.append(dist > threshold)
    stacked = np.stack(neighbors, axis=0)
    isolated = np.sum(stacked, axis=0) >= 3
    return float(np.mean(isolated))


def cluster_continuity(rgba: np.ndarray) -> float:
    color = rgba[..., :4]
    sims = []
    for dy, dx in ((0, 1), (1, 0)):
        shifted = np.roll(color, shift=(dy, dx), axis=(0, 1))
        dist = np.linalg.norm(color - shifted, axis=-1)
        sims.append(np.exp(-dist * 8.0))
    return float(np.mean(np.stack(sims, axis=0)))


def color_chatter(rgba: np.ndarray) -> float:
    color = rgba[..., :3]
    padded = np.pad(color, ((1, 1), (1, 1), (0, 0)), mode="edge")
    windows = []
    for y in range(3):
        for x in range(3):
            windows.append(padded[y : y + color.shape[0], x : x + color.shape[1]])
    stack = np.stack(windows, axis=0)
    med = np.median(stack, axis=0)
    dev = np.linalg.norm(color - med, axis=-1)
    return float(np.mean(dev))


def outline_straightness(rgba: np.ndarray) -> float:
    mask = rgba[..., 3] >= 0.5
    if not np.any(mask):
        return 1.0
    a = mask[:-1, :-1]
    b = mask[:-1, 1:]
    c = mask[1:, :-1]
    d = mask[1:, 1:]
    block_sum = a.astype(np.int8) + b.astype(np.int8) + c.astype(np.int8) + d.astype(np.int8)
    corner_blocks = np.logical_or(block_sum == 1, block_sum == 3)
    edge_pixels = np.logical_xor(mask, np.roll(mask, 1, axis=0)) | np.logical_xor(mask, np.roll(mask, 1, axis=1))
    edge_count = np.maximum(1, np.count_nonzero(edge_pixels))
    return float(1.0 - np.count_nonzero(corner_blocks) / edge_count)


def reconstruction_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def coherence_breakdown(rgba: np.ndarray) -> dict[str, float]:
    breakdown = {
        "cluster_continuity": cluster_continuity(rgba),
        "alpha_crispness": alpha_crispness(rgba),
        "outline_straightness": outline_straightness(rgba),
        "isolated_penalty": isolated_pixel_rate(rgba),
        "color_chatter": color_chatter(rgba),
    }
    breakdown["coherence_score"] = (
        0.25 * breakdown["cluster_continuity"]
        + 0.20 * breakdown["alpha_crispness"]
        + 0.20 * breakdown["outline_straightness"]
        + 0.20 * (1.0 - breakdown["isolated_penalty"])
        + 0.15 * (1.0 - min(1.0, breakdown["color_chatter"]))
    )
    return breakdown
