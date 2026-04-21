from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import premultiply, unpremultiply
from .types import InferenceResult, SolverArtifacts, SourceAnalysis


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - exercised only when torch missing
        raise RuntimeError(
            "PyTorch is required for the continuous optimization stage. Install project dependencies first."
        ) from exc
    return torch, F


def _make_regular_uv(height: int, width: int, target_height: int, target_width: int, phase_x: float, phase_y: float) -> np.ndarray:
    cell_x = width / target_width
    cell_y = height / target_height
    xs = (np.arange(target_width, dtype=np.float32) + 0.5 + phase_x) * cell_x - 0.5
    ys = (np.arange(target_height, dtype=np.float32) + 0.5 + phase_y) * cell_y - 0.5
    xs = np.clip(xs, 0.0, max(0.0, width - 1))
    ys = np.clip(ys, 0.0, max(0.0, height - 1))
    grid_x, grid_y = np.meshgrid(xs, ys)
    uv = np.stack([grid_x, grid_y], axis=-1)
    uv[..., 0] = (uv[..., 0] / max(1.0, width - 1)) * 2.0 - 1.0
    uv[..., 1] = (uv[..., 1] / max(1.0, height - 1)) * 2.0 - 1.0
    return uv.astype(np.float32)


def _cluster_boundary_map(cluster_map: np.ndarray) -> np.ndarray:
    if cluster_map.size == 0:
        return np.zeros_like(cluster_map, dtype=np.float32)
    boundary = np.zeros(cluster_map.shape, dtype=np.float32)
    boundary[:, 1:] = np.maximum(boundary[:, 1:], (cluster_map[:, 1:] != cluster_map[:, :-1]).astype(np.float32))
    boundary[1:, :] = np.maximum(boundary[1:, :], (cluster_map[1:, :] != cluster_map[:-1, :]).astype(np.float32))
    return boundary


def optimize_uv_field(
    rgba: np.ndarray,
    inference: InferenceResult,
    analysis: SourceAnalysis,
    steps: int,
    seed: int,
    device: str,
) -> SolverArtifacts:
    torch, F = _require_torch()
    torch.manual_seed(seed)
    source = premultiply(rgba)
    height, width = source.shape[:2]
    source_t = torch.from_numpy(source.transpose(2, 0, 1)[None, ...]).to(device=device, dtype=torch.float32)
    edge = np.maximum(analysis.edge_map, _cluster_boundary_map(analysis.cluster_map))
    edge_t = torch.from_numpy(edge[None, None, ...]).to(device=device, dtype=torch.float32)
    uv0 = _make_regular_uv(
        height=height,
        width=width,
        target_height=inference.target_height,
        target_width=inference.target_width,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
    )
    uv0_t = torch.from_numpy(uv0[None, ...]).to(device=device, dtype=torch.float32)
    uv = torch.nn.Parameter(uv0_t.clone())
    optimizer = torch.optim.Adam([uv], lr=0.02)
    loss_history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        sampled = F.grid_sample(source_t, uv, align_corners=True, mode="bilinear", padding_mode="border")
        reconstruction = F.interpolate(sampled, size=(height, width), mode="nearest")
        recon_loss = torch.mean(torch.abs(reconstruction - source_t))

        dx = uv[:, :, 1:, :] - uv[:, :, :-1, :]
        dy = uv[:, 1:, :, :] - uv[:, :-1, :, :]
        dx0 = uv0_t[:, :, 1:, :] - uv0_t[:, :, :-1, :]
        dy0 = uv0_t[:, 1:, :, :] - uv0_t[:, :-1, :, :]

        guide_small = F.interpolate(edge_t, size=(inference.target_height, inference.target_width), mode="bilinear", align_corners=True)
        guide_x = 1.0 - guide_small[:, :, :, 1:]
        guide_y = 1.0 - guide_small[:, :, 1:, :]

        smooth_loss = torch.mean((dx - dx0) ** 2 * guide_x.permute(0, 2, 3, 1)) + torch.mean(
            (dy - dy0) ** 2 * guide_y.permute(0, 2, 3, 1)
        )

        target_rgba = sampled.permute(0, 2, 3, 1)
        neighbor_mean = (
            torch.roll(target_rgba, 1, dims=1)
            + torch.roll(target_rgba, -1, dims=1)
            + torch.roll(target_rgba, 1, dims=2)
            + torch.roll(target_rgba, -1, dims=2)
        ) / 4.0
        speckle = torch.relu(torch.linalg.norm(target_rgba - neighbor_mean, dim=-1) - 0.08)
        anti_speckle = torch.mean(speckle * (1.0 - guide_small[:, 0]))

        alpha = target_rgba[..., 3]
        alpha_crisp = torch.mean(alpha * (1.0 - alpha))

        loss = recon_loss + smooth_loss * 0.35 + anti_speckle * 0.25 + alpha_crisp * 0.05
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            uv.clamp_(-1.0, 1.0)
        loss_history.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        sampled = F.grid_sample(source_t, uv, align_corners=True, mode="bilinear", padding_mode="border")
        target_rgba = sampled[0].permute(1, 2, 0).detach().cpu().numpy()
        initial_rgba = F.grid_sample(source_t, uv0_t, align_corners=True, mode="bilinear", padding_mode="border")[0].permute(1, 2, 0)
        initial_rgba_np = initial_rgba.detach().cpu().numpy()
        guidance = guide_small[0, 0].detach().cpu().numpy()

    return SolverArtifacts(
        target_rgba=unpremultiply(target_rgba),
        uv_field=uv.detach().cpu().numpy()[0],
        guidance_strength=guidance.astype(np.float32),
        initial_rgba=unpremultiply(initial_rgba_np),
        loss_history=loss_history,
    )
