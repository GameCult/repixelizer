from __future__ import annotations

import numpy as np

from .metrics import luminance
from .types import ContinuousSourceAnalysis, TileGraphSourceAnalysis


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only when torch missing
        raise RuntimeError(
            "PyTorch is required for GPU-accelerated source analysis. Install project dependencies first."
        ) from exc
    return torch


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for source analysis, but this PyTorch build does not have a usable CUDA device. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return requested


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


def _compute_edge_map_torch(torch, rgba_t):
    lum = rgba_t[..., 0] * 0.2126 + rgba_t[..., 1] * 0.7152 + rgba_t[..., 2] * 0.0722
    alpha = rgba_t[..., 3]
    dx = torch.zeros_like(lum, dtype=torch.float32)
    dy = torch.zeros_like(lum, dtype=torch.float32)
    dx[:, 1:] = (lum[:, 1:] - lum[:, :-1]).abs() + (alpha[:, 1:] - alpha[:, :-1]).abs()
    dy[1:, :] = (lum[1:, :] - lum[:-1, :]).abs() + (alpha[1:, :] - alpha[:-1, :]).abs()
    edge = torch.sqrt(dx * dx + dy * dy)
    max_edge = torch.max(edge)
    if float(max_edge.item()) > 0.0:
        edge = edge / max_edge
    return edge.to(dtype=torch.float32)


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


def _kmeans_torch(torch, data_t, k: int, seed: int, iterations: int = 12):
    if data_t.shape[0] <= k:
        centers_t = data_t.clone()
        labels_t = torch.arange(data_t.shape[0], device=data_t.device, dtype=torch.long)
        return centers_t, labels_t
    generator = torch.Generator(device=data_t.device)
    generator.manual_seed(seed)
    picks = torch.randperm(data_t.shape[0], device=data_t.device, generator=generator)[:k]
    centers_t = data_t[picks].clone()
    for _ in range(iterations):
        distances = torch.linalg.vector_norm(data_t[:, None, :] - centers_t[None, :, :], dim=-1)
        labels_t = torch.argmin(distances, dim=1)
        new_centers = []
        for idx in range(k):
            members = data_t[labels_t == idx]
            if members.numel() == 0:
                new_centers.append(centers_t[idx])
            else:
                new_centers.append(members.mean(dim=0))
        centers_t = torch.stack(new_centers, dim=0).to(dtype=torch.float32)
    distances = torch.linalg.vector_norm(data_t[:, None, :] - centers_t[None, :, :], dim=-1)
    labels_t = torch.argmin(distances, dim=1)
    return centers_t, labels_t


def analyze_continuous_source(
    rgba: np.ndarray,
    seed: int,
    cluster_count: int = 6,
    device: str | None = None,
) -> ContinuousSourceAnalysis:
    if device is not None:
        torch = _require_torch()
        resolved_device = _resolve_device(torch, device)
        rgba_t = torch.from_numpy(rgba).to(device=resolved_device, dtype=torch.float32)
        edge_map_t = _compute_edge_map_torch(torch, rgba_t)
        alpha_t = rgba_t[..., 3]
        cluster_map_t = torch.full(alpha_t.shape, -1, device=resolved_device, dtype=torch.long)
        opaque_t = alpha_t > 0.05
        if bool(torch.any(opaque_t).item()):
            samples_t = rgba_t[opaque_t]
            k = min(cluster_count, max(2, int(np.sqrt(int(samples_t.shape[0]) // 64 + 1))))
            _centers_t, labels_t = _kmeans_torch(torch, samples_t, k=k, seed=seed)
            cluster_map_t[opaque_t] = labels_t
        return ContinuousSourceAnalysis(
            edge_map=edge_map_t.detach().cpu().numpy().astype(np.float32),
            cluster_map=cluster_map_t.detach().cpu().numpy().astype(np.int32),
        )

    edge_map = _compute_edge_map(rgba)
    alpha = rgba[..., 3]
    cluster_map = np.full(alpha.shape, -1, dtype=np.int32)
    opaque = alpha > 0.05
    if np.any(opaque):
        samples = rgba[opaque]
        k = min(cluster_count, max(2, int(np.sqrt(samples.shape[0] // 64 + 1))))
        _centers, labels = _kmeans(samples, k=k, seed=seed)
        cluster_map[opaque] = labels
    return ContinuousSourceAnalysis(
        edge_map=edge_map,
        cluster_map=cluster_map,
    )


def analyze_tile_graph_source(
    rgba: np.ndarray,
    *,
    device: str | None = None,
) -> TileGraphSourceAnalysis:
    if device is not None:
        torch = _require_torch()
        resolved_device = _resolve_device(torch, device)
        rgba_t = torch.from_numpy(rgba).to(device=resolved_device, dtype=torch.float32)
        edge_map_t = _compute_edge_map_torch(torch, rgba_t)
        return TileGraphSourceAnalysis(
            edge_map=edge_map_t.detach().cpu().numpy().astype(np.float32),
        )

    return TileGraphSourceAnalysis(
        edge_map=_compute_edge_map(rgba),
    )
