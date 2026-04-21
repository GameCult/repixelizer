from __future__ import annotations

import numpy as np

from .metrics import luminance
from .types import SourceAnalysis


def _compute_edge_map(rgba: np.ndarray) -> np.ndarray:
    lum = luminance(rgba)
    alpha = rgba[..., 3]
    dx = np.zeros_like(lum)
    dy = np.zeros_like(lum)
    dx[:, 1:] = np.abs(lum[:, 1:] - lum[:, :-1]) + np.abs(alpha[:, 1:] - alpha[:, :-1])
    dy[1:, :] = np.abs(lum[1:, :] - lum[:-1, :]) + np.abs(alpha[1:, :] - alpha[:-1, :])
    edge = np.sqrt(dx * dx + dy * dy)
    max_edge = float(np.max(edge))
    if max_edge > 0:
        edge /= max_edge
    return edge.astype(np.float32)


def _kmeans(data: np.ndarray, k: int, seed: int, iterations: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if data.shape[0] <= k:
        centers = data.copy()
        labels = np.arange(data.shape[0], dtype=np.int32)
        return centers, labels
    picks = rng.choice(data.shape[0], size=k, replace=False)
    centers = data[picks].copy()
    for _ in range(iterations):
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=-1)
        labels = np.argmin(distances, axis=1)
        new_centers = []
        for idx in range(k):
            members = data[labels == idx]
            if members.size == 0:
                new_centers.append(centers[idx])
            else:
                new_centers.append(np.mean(members, axis=0))
        centers = np.asarray(new_centers, dtype=np.float32)
    distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=-1)
    labels = np.argmin(distances, axis=1)
    return centers.astype(np.float32), labels.astype(np.int32)


def _colorize_clusters(cluster_map: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.zeros((*cluster_map.shape, 4), dtype=np.float32)
    palette = np.clip(centers[:, :3], 0.0, 1.0)
    preview = palette[np.clip(cluster_map, 0, len(palette) - 1)]
    alpha = np.ones((*cluster_map.shape, 1), dtype=np.float32)
    return np.concatenate([preview, alpha], axis=-1)


def analyze_source(rgba: np.ndarray, seed: int, cluster_count: int = 6) -> SourceAnalysis:
    edge_map = _compute_edge_map(rgba)
    alpha = rgba[..., 3]
    opaque = alpha > 0.05
    cluster_map = np.full(alpha.shape, -1, dtype=np.int32)
    centers = np.zeros((0, 4), dtype=np.float32)
    if np.any(opaque):
        samples = rgba[opaque]
        k = min(cluster_count, max(2, int(np.sqrt(samples.shape[0] // 64 + 1))))
        centers, labels = _kmeans(samples, k=k, seed=seed)
        cluster_map[opaque] = labels
    preview = _colorize_clusters(np.maximum(cluster_map, 0), centers if centers.size else np.zeros((1, 4), dtype=np.float32))
    preview[..., 3] = alpha
    return SourceAnalysis(
        edge_map=edge_map,
        cluster_map=cluster_map,
        cluster_centers=centers,
        alpha_map=alpha.astype(np.float32),
        cluster_preview=preview,
    )
