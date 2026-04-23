from __future__ import annotations

import numpy as np

from .io import premultiply, unpremultiply
from .types import SourceLatticeReference, TileGraphSourceReference


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only when torch missing
        raise RuntimeError(
            "PyTorch is required for GPU-accelerated source lattice references. Install project dependencies first."
        ) from exc
    return torch


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for source lattice references, but this PyTorch build does not have a usable CUDA device. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return requested


def _default_edge_hint(source_rgba: np.ndarray) -> np.ndarray:
    rgb = source_rgba[..., :3]
    alpha = source_rgba[..., 3]
    lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    dx = np.zeros_like(lum, dtype=np.float32)
    dy = np.zeros_like(lum, dtype=np.float32)
    dx[:, 1:] = np.abs(lum[:, 1:] - lum[:, :-1]) + np.abs(alpha[:, 1:] - alpha[:, :-1])
    dy[1:, :] = np.abs(lum[1:, :] - lum[:-1, :]) + np.abs(alpha[1:, :] - alpha[:-1, :])
    edge = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
    max_edge = float(np.max(edge))
    if max_edge > 0.0:
        edge /= max_edge
    return edge.astype(np.float32)


def _default_edge_hint_torch(torch, source_t):
    rgb = source_t[..., :3]
    alpha = source_t[..., 3]
    lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    dx = torch.zeros_like(lum, dtype=torch.float32)
    dy = torch.zeros_like(lum, dtype=torch.float32)
    dx[:, 1:] = (lum[:, 1:] - lum[:, :-1]).abs() + (alpha[:, 1:] - alpha[:, :-1]).abs()
    dy[1:, :] = (lum[1:, :] - lum[:-1, :]).abs() + (alpha[1:, :] - alpha[:-1, :]).abs()
    edge = torch.sqrt(dx * dx + dy * dy)
    max_edge = torch.max(edge)
    if float(max_edge.item()) > 0.0:
        edge = edge / max_edge
    return edge.to(dtype=torch.float32)


def lattice_indices(
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


def _lattice_indices_torch(
    torch,
    *,
    height: int,
    width: int,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
    device: str,
):
    cell_x = width / max(1, target_width)
    cell_y = height / max(1, target_height)
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / cell_x - phase_x
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / cell_y - phase_y
    x_idx = torch.floor(xs).to(dtype=torch.long).clamp(0, max(0, target_width - 1))
    y_idx = torch.floor(ys).to(dtype=torch.long).clamp(0, max(0, target_height - 1))
    return y_idx[:, None] * target_width + x_idx[None, :]


def _unpremultiply_torch(torch, premul_t):
    alpha = premul_t[..., 3:4].clamp_min(1e-6)
    rgb = torch.where(alpha > 1e-6, premul_t[..., :3] / alpha, torch.zeros_like(premul_t[..., :3]))
    return torch.cat([rgb, premul_t[..., 3:4]], dim=-1).clamp(0.0, 1.0)


def _scatter_amax_1d(torch, indices_t, values_t, *, length: int):
    out_t = torch.full((length,), -float("inf"), device=values_t.device, dtype=torch.float32)
    out_t.scatter_reduce_(0, indices_t, values_t.to(dtype=torch.float32), reduce="amax", include_self=True)
    return torch.where(torch.isfinite(out_t), out_t, torch.zeros_like(out_t))


def _argbest_linear_indices(torch, indices_t, scores_t, *, length: int, maximize: bool):
    if maximize:
        best_scores_t = torch.full((length,), -float("inf"), device=scores_t.device, dtype=torch.float32)
        best_scores_t.scatter_reduce_(0, indices_t, scores_t.to(dtype=torch.float32), reduce="amax", include_self=True)
        match_t = scores_t >= (best_scores_t[indices_t] - 1e-8)
    else:
        best_scores_t = torch.full((length,), float("inf"), device=scores_t.device, dtype=torch.float32)
        best_scores_t.scatter_reduce_(0, indices_t, scores_t.to(dtype=torch.float32), reduce="amin", include_self=True)
        match_t = scores_t <= (best_scores_t[indices_t] + 1e-8)
    linear_t = torch.arange(indices_t.shape[0], device=indices_t.device, dtype=torch.long)
    sentinel = torch.full_like(linear_t, indices_t.shape[0])
    masked_linear_t = torch.where(match_t, linear_t, sentinel)
    best_linear_t = torch.full((length,), indices_t.shape[0], device=indices_t.device, dtype=torch.long)
    best_linear_t.scatter_reduce_(0, indices_t, masked_linear_t, reduce="amin", include_self=True)
    return best_linear_t


def build_source_lattice_reference(
    source_rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
    alpha_threshold: float = 0.05,
    edge_hint: np.ndarray | None = None,
    edge_grad_x_hint: np.ndarray | None = None,
    edge_grad_y_hint: np.ndarray | None = None,
    device: str | None = None,
) -> SourceLatticeReference:
    payload = _build_reference_payload(
        source_rgba,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
        alpha_threshold=alpha_threshold,
        edge_hint=edge_hint,
        edge_grad_x_hint=edge_grad_x_hint,
        edge_grad_y_hint=edge_grad_y_hint,
        device=device,
    )
    return SourceLatticeReference(**payload)


def build_tile_graph_source_reference(
    source_rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
    alpha_threshold: float = 0.05,
    edge_hint: np.ndarray | None = None,
    device: str | None = None,
) -> TileGraphSourceReference:
    payload = _build_reference_payload(
        source_rgba,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
        alpha_threshold=alpha_threshold,
        edge_hint=edge_hint,
        device=device,
    )
    return TileGraphSourceReference(
        sharp_rgba=payload["sharp_rgba"],
        sharp_x=payload["sharp_x"],
        sharp_y=payload["sharp_y"],
        edge_peak_x=payload["edge_peak_x"],
        edge_peak_y=payload["edge_peak_y"],
        edge_strength=payload["edge_strength"],
    )


def _build_reference_payload(
    source_rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
    alpha_threshold: float = 0.05,
    edge_hint: np.ndarray | None = None,
    edge_grad_x_hint: np.ndarray | None = None,
    edge_grad_y_hint: np.ndarray | None = None,
    device: str | None = None,
) -> dict[str, np.ndarray | float]:
    if device is not None:
        torch = _require_torch()
        resolved_device = _resolve_device(torch, device)
        source_t = torch.from_numpy(source_rgba).to(device=resolved_device, dtype=torch.float32)
        height = source_rgba.shape[0]
        width = source_rgba.shape[1]
        edge_hint_t = (
            torch.from_numpy(edge_hint.astype(np.float32)).to(device=resolved_device, dtype=torch.float32)
            if edge_hint is not None
            else _default_edge_hint_torch(torch, source_t)
        )
        indices_t = _lattice_indices_torch(
            torch,
            height=height,
            width=width,
            target_width=target_width,
            target_height=target_height,
            phase_x=phase_x,
            phase_y=phase_y,
            device=resolved_device,
        )
        cell_count = max(1, target_width * target_height)
        premul_t = source_t.clone()
        premul_t[..., :3] = premul_t[..., :3] * premul_t[..., 3:4]
        flat_idx_t = indices_t.reshape(-1).to(dtype=torch.long)
        flat_premul_t = premul_t.reshape(-1, premul_t.shape[-1])
        flat_alpha_t = source_t.reshape(-1, source_t.shape[-1])[:, 3]
        flat_edge_t = edge_hint_t.reshape(-1)

        counts_t = torch.bincount(flat_idx_t, minlength=cell_count).to(dtype=torch.float32)
        safe_counts_t = counts_t.clamp_min(1.0)
        channel_sums_t = torch.stack(
            [torch.bincount(flat_idx_t, weights=flat_premul_t[:, channel], minlength=cell_count) for channel in range(flat_premul_t.shape[-1])],
            dim=-1,
        ).to(dtype=torch.float32)
        mean_premul_flat_t = channel_sums_t / safe_counts_t[:, None]
        mean_premul_t = mean_premul_flat_t.reshape(target_height, target_width, premul_t.shape[-1])
        mean_rgba_t = _unpremultiply_torch(torch, mean_premul_t)

        per_pixel_mean_t = mean_premul_flat_t[flat_idx_t]
        pixel_diff_t = (flat_premul_t - per_pixel_mean_t).abs().mean(dim=-1)
        cell_diff_sums_t = torch.bincount(flat_idx_t, weights=pixel_diff_t, minlength=cell_count).to(dtype=torch.float32)
        cell_dispersion_t = (cell_diff_sums_t / safe_counts_t).reshape(target_height, target_width)

        lattice_alpha_t = mean_rgba_t.reshape(-1, mean_rgba_t.shape[-1])[flat_idx_t, 3]
        support_mask_t = torch.maximum(flat_alpha_t, lattice_alpha_t) >= alpha_threshold
        if bool(torch.any(support_mask_t).item()):
            dispersion = float(pixel_diff_t[support_mask_t].mean().item())
        else:
            dispersion = float(pixel_diff_t.mean().item()) if pixel_diff_t.numel() else 0.0

        cell_x = width / max(1, target_width)
        cell_y = height / max(1, target_height)
        default_x_t = torch.clamp(
            torch.round((torch.arange(target_width, device=resolved_device, dtype=torch.float32) + 0.5 + phase_x) * cell_x - 0.5),
            0,
            max(0, width - 1),
        ).to(dtype=torch.long)
        default_y_t = torch.clamp(
            torch.round((torch.arange(target_height, device=resolved_device, dtype=torch.float32) + 0.5 + phase_y) * cell_y - 0.5),
            0,
            max(0, height - 1),
        ).to(dtype=torch.long)
        sharp_x_flat_t = default_x_t[None, :].expand(target_height, target_width).reshape(-1).clone()
        sharp_y_flat_t = default_y_t[:, None].expand(target_height, target_width).reshape(-1).clone()

        pixel_y_t, pixel_x_t = torch.meshgrid(
            torch.arange(height, device=resolved_device, dtype=torch.long),
            torch.arange(width, device=resolved_device, dtype=torch.long),
            indexing="ij",
        )
        flat_x_t = pixel_x_t.reshape(-1)
        flat_y_t = pixel_y_t.reshape(-1)
        exemplar_cost_t = pixel_diff_t + torch.clamp(per_pixel_mean_t[:, 3] - flat_premul_t[:, 3], min=0.0) * 1e-3

        edge_score_t = flat_edge_t + flat_alpha_t.to(dtype=torch.float32) * 0.05
        edge_peak_x_flat_t = sharp_x_flat_t.clone()
        edge_peak_y_flat_t = sharp_y_flat_t.clone()
        edge_strength_flat_t = _scatter_amax_1d(torch, flat_idx_t, flat_edge_t, length=cell_count)
        edge_grad_x_flat_t = torch.zeros(cell_count, device=resolved_device, dtype=torch.float32)
        edge_grad_y_flat_t = torch.zeros(cell_count, device=resolved_device, dtype=torch.float32)
        best_edge_linear_t = _argbest_linear_indices(torch, flat_idx_t, edge_score_t, length=cell_count, maximize=True)
        valid_edge_mask_t = best_edge_linear_t < flat_idx_t.shape[0]
        valid_edge_cells_t = torch.nonzero(valid_edge_mask_t, as_tuple=False).reshape(-1)
        if valid_edge_cells_t.numel() > 0:
            valid_edge_linear_t = best_edge_linear_t[valid_edge_cells_t]
            edge_peak_x_flat_t[valid_edge_cells_t] = flat_x_t[valid_edge_linear_t]
            edge_peak_y_flat_t[valid_edge_cells_t] = flat_y_t[valid_edge_linear_t]
        if edge_grad_x_hint is not None or edge_grad_y_hint is not None:
            edge_peak_index_flat_t = edge_peak_y_flat_t * width + edge_peak_x_flat_t
            if edge_grad_x_hint is not None:
                edge_grad_x_hint_t = torch.from_numpy(edge_grad_x_hint.astype(np.float32)).to(device=resolved_device, dtype=torch.float32).reshape(-1)
                edge_grad_x_flat_t = edge_grad_x_hint_t[edge_peak_index_flat_t]
            if edge_grad_y_hint is not None:
                edge_grad_y_hint_t = torch.from_numpy(edge_grad_y_hint.astype(np.float32)).to(device=resolved_device, dtype=torch.float32).reshape(-1)
                edge_grad_y_flat_t = edge_grad_y_hint_t[edge_peak_index_flat_t]

        sharp_premul_flat_t = mean_premul_flat_t.clone()
        best_exemplar_linear_t = _argbest_linear_indices(torch, flat_idx_t, exemplar_cost_t, length=cell_count, maximize=False)
        valid_exemplar_mask_t = best_exemplar_linear_t < flat_idx_t.shape[0]
        valid_exemplar_cells_t = torch.nonzero(valid_exemplar_mask_t, as_tuple=False).reshape(-1)
        if valid_exemplar_cells_t.numel() > 0:
            valid_exemplar_linear_t = best_exemplar_linear_t[valid_exemplar_cells_t]
            sharp_premul_flat_t[valid_exemplar_cells_t] = flat_premul_t[valid_exemplar_linear_t]
            sharp_x_flat_t[valid_exemplar_cells_t] = flat_x_t[valid_exemplar_linear_t]
            sharp_y_flat_t[valid_exemplar_cells_t] = flat_y_t[valid_exemplar_linear_t]
        sharp_premul_t = sharp_premul_flat_t.reshape(target_height, target_width, premul_t.shape[-1])
        sharp_rgba_t = _unpremultiply_torch(torch, sharp_premul_t)

        cell_alpha_max_t = torch.zeros(cell_count, device=resolved_device, dtype=torch.float32)
        cell_alpha_max_t.scatter_reduce_(0, flat_idx_t, flat_alpha_t, reduce="amax", include_self=True)
        expected_pixels_per_cell = float(source_rgba.shape[0] * source_rgba.shape[1]) / float(cell_count)
        cell_support_t = (counts_t / max(expected_pixels_per_cell, 1.0)).reshape(target_height, target_width)

        return {
            "mean_rgba": mean_rgba_t.detach().cpu().numpy().astype(np.float32),
            "sharp_rgba": sharp_rgba_t.detach().cpu().numpy().astype(np.float32),
            "dispersion": dispersion,
            "cell_dispersion": cell_dispersion_t.detach().cpu().numpy().astype(np.float32),
            "cell_support": cell_support_t.detach().cpu().numpy().astype(np.float32),
            "cell_alpha_max": cell_alpha_max_t.reshape(target_height, target_width).detach().cpu().numpy().astype(np.float32),
            "sharp_x": sharp_x_flat_t.reshape(target_height, target_width).detach().cpu().numpy().astype(np.int32),
            "sharp_y": sharp_y_flat_t.reshape(target_height, target_width).detach().cpu().numpy().astype(np.int32),
            "edge_peak_x": edge_peak_x_flat_t.reshape(target_height, target_width).detach().cpu().numpy().astype(np.int32),
            "edge_peak_y": edge_peak_y_flat_t.reshape(target_height, target_width).detach().cpu().numpy().astype(np.int32),
            "edge_strength": edge_strength_flat_t.reshape(target_height, target_width).detach().cpu().numpy().astype(np.float32),
            "edge_grad_x": edge_grad_x_flat_t.reshape(target_height, target_width).detach().cpu().numpy().astype(np.float32),
            "edge_grad_y": edge_grad_y_flat_t.reshape(target_height, target_width).detach().cpu().numpy().astype(np.float32),
        }

    height = source_rgba.shape[0]
    width = source_rgba.shape[1]
    edge_hint = edge_hint.astype(np.float32) if edge_hint is not None else _default_edge_hint(source_rgba)
    indices = lattice_indices(
        height=height,
        width=width,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
    )
    cell_count = max(1, target_width * target_height)
    premul = premultiply(source_rgba)
    flat_idx = indices.reshape(-1)
    flat_premul = premul.reshape(-1, premul.shape[-1])
    flat_alpha = source_rgba.reshape(-1, source_rgba.shape[-1])[:, 3]
    flat_edge = edge_hint.reshape(-1).astype(np.float32)

    counts = np.bincount(flat_idx, minlength=cell_count).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    channel_sums = [
        np.bincount(flat_idx, weights=flat_premul[:, channel], minlength=cell_count).astype(np.float32)
        for channel in range(flat_premul.shape[-1])
    ]
    mean_premul_flat = np.stack(channel_sums, axis=-1) / safe_counts[:, None]
    mean_premul = mean_premul_flat.reshape(target_height, target_width, premul.shape[-1])
    mean_rgba = unpremultiply(mean_premul)

    per_pixel_mean = mean_premul_flat[flat_idx]
    pixel_diff = np.mean(np.abs(flat_premul - per_pixel_mean), axis=-1)
    cell_diff_sums = np.bincount(flat_idx, weights=pixel_diff, minlength=cell_count).astype(np.float32)
    cell_dispersion = (cell_diff_sums / safe_counts).reshape(target_height, target_width)

    lattice_alpha = mean_rgba.reshape(-1, mean_rgba.shape[-1])[flat_idx, 3]
    support_mask = np.maximum(flat_alpha, lattice_alpha) >= alpha_threshold
    if np.any(support_mask):
        dispersion = float(np.mean(pixel_diff[support_mask]))
    else:
        dispersion = float(np.mean(pixel_diff)) if pixel_diff.size else 0.0

    cell_x = width / max(1, target_width)
    cell_y = height / max(1, target_height)
    default_x = np.clip(
        np.rint((np.arange(target_width, dtype=np.float32) + 0.5 + phase_x) * cell_x - 0.5),
        0,
        max(0, width - 1),
    )
    default_y = np.clip(
        np.rint((np.arange(target_height, dtype=np.float32) + 0.5 + phase_y) * cell_y - 0.5),
        0,
        max(0, height - 1),
    )
    sharp_x_flat = np.tile(default_x[None, :], (target_height, 1)).reshape(-1).astype(np.int32)
    sharp_y_flat = np.tile(default_y[:, None], (1, target_width)).reshape(-1).astype(np.int32)

    pixel_y, pixel_x = np.indices((height, width), dtype=np.int32)
    flat_x = pixel_x.reshape(-1)
    flat_y = pixel_y.reshape(-1)
    exemplar_cost = pixel_diff + np.maximum(0.0, per_pixel_mean[:, 3] - flat_premul[:, 3]) * 1e-3

    edge_score = flat_edge + flat_alpha.astype(np.float32) * 0.05
    edge_order = np.lexsort((-edge_score.astype(np.float32), flat_idx.astype(np.int64)))
    edge_peak_x_flat = sharp_x_flat.copy()
    edge_peak_y_flat = sharp_y_flat.copy()
    edge_strength_flat = np.zeros(cell_count, dtype=np.float32)
    edge_grad_x_flat = np.zeros(cell_count, dtype=np.float32)
    edge_grad_y_flat = np.zeros(cell_count, dtype=np.float32)
    best_edge_order = np.asarray([], dtype=np.int64)
    best_edge_cells = np.asarray([], dtype=np.int64)
    if edge_order.size > 0:
        sorted_edge_cells = flat_idx[edge_order]
        first_edge_positions = np.concatenate(([0], np.flatnonzero(np.diff(sorted_edge_cells)) + 1))
        best_edge_order = edge_order[first_edge_positions]
        best_edge_cells = flat_idx[best_edge_order]
        edge_peak_x_flat[best_edge_cells] = flat_x[best_edge_order]
        edge_peak_y_flat[best_edge_cells] = flat_y[best_edge_order]
        edge_strength_flat[best_edge_cells] = flat_edge[best_edge_order]
        if edge_grad_x_hint is not None:
            edge_grad_x_flat[best_edge_cells] = edge_grad_x_hint.reshape(-1)[best_edge_order].astype(np.float32)
        if edge_grad_y_hint is not None:
            edge_grad_y_flat[best_edge_cells] = edge_grad_y_hint.reshape(-1)[best_edge_order].astype(np.float32)

    np.maximum.at(edge_strength_flat, flat_idx, flat_edge)
    if edge_grad_x_hint is not None or edge_grad_y_hint is not None:
        edge_peak_index_flat = edge_peak_y_flat * width + edge_peak_x_flat
        if edge_grad_x_hint is not None:
            edge_grad_x_flat = edge_grad_x_hint.reshape(-1)[edge_peak_index_flat].astype(np.float32)
        if edge_grad_y_hint is not None:
            edge_grad_y_flat = edge_grad_y_hint.reshape(-1)[edge_peak_index_flat].astype(np.float32)

    order = np.lexsort((exemplar_cost.astype(np.float32), flat_idx.astype(np.int64)))
    sharp_premul_flat = mean_premul_flat.copy()
    if order.size > 0:
        sorted_cells = flat_idx[order]
        first_positions = np.concatenate(([0], np.flatnonzero(np.diff(sorted_cells)) + 1))
        best_order = order[first_positions]
        best_cells = flat_idx[best_order]
        sharp_premul_flat[best_cells] = flat_premul[best_order]
        sharp_x_flat[best_cells] = flat_x[best_order]
        sharp_y_flat[best_cells] = flat_y[best_order]

    sharp_premul = sharp_premul_flat.reshape(target_height, target_width, premul.shape[-1])
    sharp_rgba = unpremultiply(sharp_premul)

    cell_alpha_max = np.zeros(cell_count, dtype=np.float32)
    np.maximum.at(cell_alpha_max, flat_idx, flat_alpha)
    expected_pixels_per_cell = float(source_rgba.shape[0] * source_rgba.shape[1]) / float(cell_count)
    cell_support = (counts / max(expected_pixels_per_cell, 1.0)).reshape(target_height, target_width)

    return {
        "mean_rgba": mean_rgba.astype(np.float32),
        "sharp_rgba": sharp_rgba.astype(np.float32),
        "dispersion": dispersion,
        "cell_dispersion": cell_dispersion.astype(np.float32),
        "cell_support": cell_support.astype(np.float32),
        "cell_alpha_max": cell_alpha_max.reshape(target_height, target_width).astype(np.float32),
        "sharp_x": sharp_x_flat.reshape(target_height, target_width).astype(np.int32),
        "sharp_y": sharp_y_flat.reshape(target_height, target_width).astype(np.int32),
        "edge_peak_x": edge_peak_x_flat.reshape(target_height, target_width).astype(np.int32),
        "edge_peak_y": edge_peak_y_flat.reshape(target_height, target_width).astype(np.int32),
        "edge_strength": edge_strength_flat.reshape(target_height, target_width).astype(np.float32),
        "edge_grad_x": edge_grad_x_flat.reshape(target_height, target_width).astype(np.float32),
        "edge_grad_y": edge_grad_y_flat.reshape(target_height, target_width).astype(np.float32),
    }
