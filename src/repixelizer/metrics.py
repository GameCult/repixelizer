from __future__ import annotations

import numpy as np

from .io import premultiply


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
    return float(np.mean(np.abs(premultiply(a) - premultiply(b))))


def sprite_mask(a: np.ndarray, b: np.ndarray, alpha_threshold: float = 0.05, halo: int = 1) -> np.ndarray:
    mask = np.maximum(a[..., 3], b[..., 3]) >= alpha_threshold
    if np.any(mask) and halo > 0:
        mask = _dilate_mask(mask, radius=halo)
    return mask


def foreground_coverage(a: np.ndarray, b: np.ndarray, alpha_threshold: float = 0.05, halo: int = 1) -> float:
    return float(np.mean(sprite_mask(a, b, alpha_threshold=alpha_threshold, halo=halo)))


def foreground_reconstruction_error(
    a: np.ndarray,
    b: np.ndarray,
    alpha_threshold: float = 0.05,
    halo: int = 1,
) -> float:
    mask = sprite_mask(a, b, alpha_threshold=alpha_threshold, halo=halo)
    if not np.any(mask):
        return reconstruction_error(a, b)
    diff = np.mean(np.abs(premultiply(a) - premultiply(b)), axis=-1)
    return float(np.mean(diff[mask]))


def foreground_adjacency_error(
    a: np.ndarray,
    b: np.ndarray,
    alpha_threshold: float = 0.05,
    halo: int = 1,
) -> float:
    premul_a = premultiply(a)
    premul_b = premultiply(b)
    mask = sprite_mask(a, b, alpha_threshold=alpha_threshold, halo=halo)
    diffs: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for axis in (0, 1):
        delta_a = np.diff(premul_a, axis=axis)
        delta_b = np.diff(premul_b, axis=axis)
        if axis == 0:
            edge_mask = mask[1:, :] | mask[:-1, :]
        else:
            edge_mask = mask[:, 1:] | mask[:, :-1]
        diffs.append(np.mean(np.abs(delta_a - delta_b), axis=-1))
        masks.append(edge_mask)

    weights = np.concatenate([edge_mask.reshape(-1) for edge_mask in masks], axis=0)
    values = np.concatenate([diff.reshape(-1) for diff in diffs], axis=0)
    if not np.any(weights):
        return 0.0
    return float(np.mean(values[weights]))


def foreground_adjacency_strength(
    rgba: np.ndarray,
    alpha_threshold: float = 0.05,
) -> float:
    premul = premultiply(rgba)
    mask = rgba[..., 3] >= alpha_threshold
    strengths: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for axis in (0, 1):
        delta = np.mean(np.abs(np.diff(premul, axis=axis)), axis=-1)
        if axis == 0:
            edge_mask = mask[1:, :] | mask[:-1, :]
        else:
            edge_mask = mask[:, 1:] | mask[:, :-1]
        strengths.append(delta.reshape(-1))
        weights.append(edge_mask.reshape(-1))

    values = np.concatenate(strengths, axis=0)
    support = np.concatenate(weights, axis=0)
    if not np.any(support):
        return 0.0
    return float(np.mean(values[support]))


def foreground_motif_error(
    a: np.ndarray,
    b: np.ndarray,
    alpha_threshold: float = 0.05,
    halo: int = 1,
) -> float:
    premul_a = premultiply(a)
    premul_b = premultiply(b)
    mask = sprite_mask(a, b, alpha_threshold=alpha_threshold, halo=halo)
    if premul_a.shape[0] < 2 or premul_a.shape[1] < 2:
        return foreground_adjacency_error(a, b, alpha_threshold=alpha_threshold, halo=halo)

    blocks_a = _motif_blocks(premul_a)
    blocks_b = _motif_blocks(premul_b)
    block_mask = mask[:-1, :-1] | mask[:-1, 1:] | mask[1:, :-1] | mask[1:, 1:]
    if not np.any(block_mask):
        return 0.0
    diff = np.mean(np.abs(blocks_a - blocks_b), axis=(-2, -1))
    return float(np.mean(diff[block_mask]))


def source_lattice_consistency_breakdown(
    source_rgba: np.ndarray,
    output_rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
    alpha_threshold: float = 0.05,
) -> dict[str, float]:
    lattice_source, dispersion = lattice_source_rgba(
        source_rgba,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
        alpha_threshold=alpha_threshold,
    )
    cell_error = foreground_reconstruction_error(output_rgba, lattice_source, alpha_threshold=alpha_threshold, halo=0)
    adjacency_error = foreground_adjacency_error(output_rgba, lattice_source, alpha_threshold=alpha_threshold, halo=0)
    motif_error = foreground_motif_error(output_rgba, lattice_source, alpha_threshold=alpha_threshold, halo=0)
    score = (
        cell_error * 0.34
        + adjacency_error * 0.24
        + motif_error * 0.24
        + dispersion * 0.18
    )
    return {
        "cell_error": cell_error,
        "adjacency_error": adjacency_error,
        "motif_error": motif_error,
        "cell_dispersion": dispersion,
        "score": score,
    }


def source_lattice_evidence_breakdown(
    source_rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
    alpha_threshold: float = 0.05,
) -> dict[str, float]:
    lattice_source, dispersion = lattice_source_rgba(
        source_rgba,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
        alpha_threshold=alpha_threshold,
    )
    adjacency_strength = foreground_adjacency_strength(lattice_source, alpha_threshold=alpha_threshold)
    evidence_score = adjacency_strength - dispersion * 0.8
    return {
        "cell_dispersion": dispersion,
        "adjacency_strength": adjacency_strength,
        "score": evidence_score,
    }


def exact_match_ratio(a: np.ndarray, b: np.ndarray) -> float:
    a8 = np.clip(np.rint(premultiply(a) * 255.0), 0, 255).astype(np.uint8)
    b8 = np.clip(np.rint(premultiply(b) * 255.0), 0, 255).astype(np.uint8)
    return float(np.mean(np.all(a8 == b8, axis=-1)))


def foreground_exact_match_ratio(
    a: np.ndarray,
    b: np.ndarray,
    alpha_threshold: float = 0.05,
    halo: int = 1,
) -> float:
    mask = sprite_mask(a, b, alpha_threshold=alpha_threshold, halo=halo)
    if not np.any(mask):
        return 1.0
    a8 = np.clip(np.rint(premultiply(a) * 255.0), 0, 255).astype(np.uint8)
    b8 = np.clip(np.rint(premultiply(b) * 255.0), 0, 255).astype(np.uint8)
    return float(np.mean(np.all(a8[mask] == b8[mask], axis=-1)))


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


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    windows = []
    for dy in range(radius * 2 + 1):
        for dx in range(radius * 2 + 1):
            windows.append(padded[dy : dy + height, dx : dx + width])
    return np.any(np.stack(windows, axis=0), axis=0)


def _motif_blocks(premul: np.ndarray) -> np.ndarray:
    blocks = np.stack(
        [
            premul[:-1, :-1, :],
            premul[:-1, 1:, :],
            premul[1:, :-1, :],
            premul[1:, 1:, :],
        ],
        axis=-2,
    )
    centered = blocks - np.mean(blocks, axis=-2, keepdims=True)
    scale = np.maximum(1e-4, np.mean(np.abs(centered), axis=(-2, -1), keepdims=True))
    return centered / scale


def lattice_source_rgba(
    source_rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
    alpha_threshold: float = 0.05,
) -> tuple[np.ndarray, float]:
    indices = _lattice_indices(
        height=source_rgba.shape[0],
        width=source_rgba.shape[1],
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
    )
    cell_count = max(1, target_width * target_height)
    premul = premultiply(source_rgba)
    flat_idx = indices.reshape(-1)
    flat_premul = premul.reshape(-1, premul.shape[-1])
    counts = np.bincount(flat_idx, minlength=cell_count).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    channel_sums = [
        np.bincount(flat_idx, weights=flat_premul[:, channel], minlength=cell_count).astype(np.float32)
        for channel in range(flat_premul.shape[-1])
    ]
    mean_premul = np.stack(channel_sums, axis=-1) / safe_counts[:, None]
    mean_premul = mean_premul.reshape(target_height, target_width, premul.shape[-1])
    mean_rgba = _unpremultiply_array(mean_premul)

    per_pixel_mean = mean_premul.reshape(-1, premul.shape[-1])[flat_idx]
    pixel_diff = np.mean(np.abs(flat_premul - per_pixel_mean), axis=-1)
    source_alpha = source_rgba.reshape(-1, source_rgba.shape[-1])[:, 3]
    lattice_alpha = mean_rgba.reshape(-1, mean_rgba.shape[-1])[flat_idx, 3]
    support_mask = np.maximum(source_alpha, lattice_alpha) >= alpha_threshold
    if np.any(support_mask):
        dispersion = float(np.mean(pixel_diff[support_mask]))
    else:
        dispersion = float(np.mean(pixel_diff)) if pixel_diff.size else 0.0
    return mean_rgba.astype(np.float32), dispersion


def _lattice_indices(
    *,
    height: int,
    width: int,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
) -> np.ndarray:
    cell_x = width / max(1, target_width)
    cell_y = height / max(1, target_height)
    xs = (np.arange(width, dtype=np.float32) + 0.5) / cell_x - phase_x
    ys = (np.arange(height, dtype=np.float32) + 0.5) / cell_y - phase_y
    x_idx = np.clip(np.floor(xs).astype(np.int32), 0, max(0, target_width - 1))
    y_idx = np.clip(np.floor(ys).astype(np.int32), 0, max(0, target_height - 1))
    return y_idx[:, None] * target_width + x_idx[None, :]


def _unpremultiply_array(premul: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = premul.copy()
    alpha = np.maximum(out[..., 3:4], eps)
    out[..., :3] = np.where(out[..., 3:4] > eps, out[..., :3] / alpha, 0.0)
    return np.clip(out, 0.0, 1.0)
