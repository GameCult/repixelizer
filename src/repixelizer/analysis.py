from __future__ import annotations

import numpy as np

from .metrics import luminance
from .types import PhaseFieldSourceAnalysis


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


def analyze_phase_field_source(
    rgba: np.ndarray,
    seed: int,
    device: str | None = None,
) -> PhaseFieldSourceAnalysis:
    del seed
    if device is not None:
        torch = _require_torch()
        resolved_device = _resolve_device(torch, device)
        rgba_t = torch.from_numpy(rgba).to(device=resolved_device, dtype=torch.float32)
        edge_map_t = _compute_edge_map_torch(torch, rgba_t)
        return PhaseFieldSourceAnalysis(
            edge_map=edge_map_t.detach().cpu().numpy().astype(np.float32),
        )

    edge_map = _compute_edge_map(rgba)
    return PhaseFieldSourceAnalysis(
        edge_map=edge_map,
    )
