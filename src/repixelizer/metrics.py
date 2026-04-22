from __future__ import annotations

import numpy as np

from .io import premultiply
from .source_reference import build_source_lattice_reference


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


def foreground_edge_position_error(
    a: np.ndarray,
    b: np.ndarray,
    alpha_threshold: float = 0.05,
    halo: int = 1,
    radius: int = 1,
    distance_penalty: float = 0.35,
    edge_threshold: float = 0.02,
) -> float:
    edge_a = _edge_strength_map(a)
    edge_b = _edge_strength_map(b)
    mask = sprite_mask(a, b, alpha_threshold=alpha_threshold, halo=halo)
    support_mask = mask & ((edge_a >= edge_threshold) | (edge_b >= edge_threshold))
    if not np.any(support_mask):
        return 0.0

    support_from_b = _best_local_edge_support(edge_b, radius=radius, distance_penalty=distance_penalty)
    support_from_a = _best_local_edge_support(edge_a, radius=radius, distance_penalty=distance_penalty)
    deficit_a = np.maximum(0.0, edge_a - support_from_b)
    deficit_b = np.maximum(0.0, edge_b - support_from_a)
    return float(np.mean((deficit_a[support_mask] + deficit_b[support_mask]) * 0.5))


def foreground_edge_concentration(
    rgba: np.ndarray,
    alpha_threshold: float = 0.05,
) -> float:
    premul = premultiply(rgba)
    alpha = rgba[..., 3]

    delta_x = np.mean(np.abs(premul[:, 1:, :] - premul[:, :-1, :]), axis=-1)
    delta_y = np.mean(np.abs(premul[1:, :, :] - premul[:-1, :, :]), axis=-1)
    mask_x = (alpha[:, 1:] >= alpha_threshold) | (alpha[:, :-1] >= alpha_threshold)
    mask_y = (alpha[1:, :] >= alpha_threshold) | (alpha[:-1, :] >= alpha_threshold)

    values = np.concatenate([delta_x[mask_x], delta_y[mask_y]], axis=0)
    if values.size == 0:
        return 1.0
    total = float(np.sum(values))
    if total <= 1e-6:
        return 0.0
    return float(np.sum(values * values) / total)


def foreground_stroke_wobble_error(
    a: np.ndarray,
    b: np.ndarray,
    alpha_threshold: float = 0.05,
    halo: int = 1,
    radius: int = 2,
    line_threshold: float = 0.12,
) -> float:
    edge_a = _edge_strength_map(a)
    edge_b = _edge_strength_map(b)
    strength_a, centroid_a = _line_profile_signature(edge_a, radius=radius)
    strength_b, centroid_b = _line_profile_signature(edge_b, radius=radius)

    orientation = np.argmax(strength_b, axis=0)
    source_strength = np.take_along_axis(strength_b, orientation[None, ...], axis=0)[0]
    source_centroid = np.take_along_axis(centroid_b, orientation[None, ...], axis=0)[0]
    output_centroid = np.take_along_axis(centroid_a, orientation[None, ...], axis=0)[0]

    support_mask = sprite_mask(a, b, alpha_threshold=alpha_threshold, halo=halo) & (source_strength >= line_threshold)
    if not np.any(support_mask):
        return 0.0
    return float(np.mean(np.abs(output_centroid[support_mask] - source_centroid[support_mask])))


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
    reference = build_source_lattice_reference(
        source_rgba,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
        alpha_threshold=alpha_threshold,
    )
    cell_error = foreground_reconstruction_error(output_rgba, reference.mean_rgba, alpha_threshold=alpha_threshold, halo=0)
    adjacency_error = foreground_adjacency_error(output_rgba, reference.sharp_rgba, alpha_threshold=alpha_threshold, halo=0)
    motif_error = foreground_motif_error(output_rgba, reference.sharp_rgba, alpha_threshold=alpha_threshold, halo=0)
    score = (
        cell_error * 0.34
        + adjacency_error * 0.24
        + motif_error * 0.24
        + reference.dispersion * 0.18
    )
    return {
        "cell_error": cell_error,
        "adjacency_error": adjacency_error,
        "motif_error": motif_error,
        "cell_dispersion": reference.dispersion,
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
    reference = build_source_lattice_reference(
        source_rgba,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
        alpha_threshold=alpha_threshold,
    )
    adjacency_strength = foreground_adjacency_strength(reference.sharp_rgba, alpha_threshold=alpha_threshold)
    evidence_score = adjacency_strength - reference.dispersion * 0.8
    return {
        "cell_dispersion": reference.dispersion,
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


def _edge_strength_map(rgba: np.ndarray) -> np.ndarray:
    premul = premultiply(rgba)
    right = np.zeros(rgba.shape[:2], dtype=np.float32)
    down = np.zeros(rgba.shape[:2], dtype=np.float32)
    right[:, :-1] = np.mean(np.abs(premul[:, 1:, :] - premul[:, :-1, :]), axis=-1)
    down[:-1, :] = np.mean(np.abs(premul[1:, :, :] - premul[:-1, :, :]), axis=-1)
    edge = np.maximum.reduce(
        [
            right,
            np.pad(right[:, :-1], ((0, 0), (1, 0))),
            down,
            np.pad(down[:-1, :], ((1, 0), (0, 0))),
        ]
    )
    return edge.astype(np.float32)


def _best_local_edge_support(edge: np.ndarray, *, radius: int, distance_penalty: float) -> np.ndarray:
    if radius <= 0:
        return edge
    height, width = edge.shape
    best = np.zeros_like(edge, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            weight = 1.0 / (1.0 + np.hypot(dx, dy) * distance_penalty)
            shifted = np.zeros_like(edge, dtype=np.float32)
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_y0 = max(0, dy)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            dst_x0 = max(0, dx)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = edge[src_y0:src_y1, src_x0:src_x1] * weight
            best = np.maximum(best, shifted)
    return best


def _line_profile_signature(edge: np.ndarray, *, radius: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    profiles = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        stack = np.stack([_shift2d(edge, int(dy * offset), int(dx * offset)) for offset in offsets], axis=0)
        energy = np.sum(stack, axis=0)
        safe_energy = np.maximum(energy, 1e-6)
        centroid = np.sum(stack * offsets[:, None, None], axis=0) / safe_energy
        spread = np.sum(stack * np.abs(offsets[:, None, None] - centroid[None, :, :]), axis=0) / safe_energy
        strength = energy / (1.0 + spread * 2.0)
        profiles.append((strength.astype(np.float32), centroid.astype(np.float32)))
    strengths = np.stack([item[0] for item in profiles], axis=0)
    centroids = np.stack([item[1] for item in profiles], axis=0)
    return strengths, centroids


def _shift2d(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = arr.shape
    shifted = np.zeros_like(arr, dtype=np.float32)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return shifted


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
    reference = build_source_lattice_reference(
        source_rgba,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
        alpha_threshold=alpha_threshold,
    )
    return reference.mean_rgba, reference.dispersion
