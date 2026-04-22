from __future__ import annotations

import numpy as np

from .io import premultiply, unpremultiply
from .types import SourceLatticeReference


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


def _reference_deltas(reference: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    height = reference.shape[0]
    width = reference.shape[1]
    delta_x = reference[:, 1:, :] - reference[:, :-1, :] if width > 1 else None
    delta_y = reference[1:, :, :] - reference[:-1, :, :] if height > 1 else None
    delta_diag = reference[1:, 1:, :] - reference[:-1, :-1, :] if height > 1 and width > 1 else None
    delta_anti = reference[1:, :-1, :] - reference[:-1, 1:, :] if height > 1 and width > 1 else None
    return delta_x, delta_y, delta_diag, delta_anti


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
) -> SourceLatticeReference:
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

    delta_x, delta_y, delta_diag, delta_anti = _reference_deltas(sharp_premul)
    return SourceLatticeReference(
        mean_rgba=mean_rgba.astype(np.float32),
        sharp_rgba=sharp_rgba.astype(np.float32),
        dispersion=dispersion,
        lattice_indices=indices.astype(np.int32),
        cell_dispersion=cell_dispersion.astype(np.float32),
        cell_counts=counts.reshape(target_height, target_width).astype(np.float32),
        cell_support=cell_support.astype(np.float32),
        cell_alpha_max=cell_alpha_max.reshape(target_height, target_width).astype(np.float32),
        sharp_x=sharp_x_flat.reshape(target_height, target_width).astype(np.int32),
        sharp_y=sharp_y_flat.reshape(target_height, target_width).astype(np.int32),
        edge_peak_x=edge_peak_x_flat.reshape(target_height, target_width).astype(np.int32),
        edge_peak_y=edge_peak_y_flat.reshape(target_height, target_width).astype(np.int32),
        edge_strength=edge_strength_flat.reshape(target_height, target_width).astype(np.float32),
        edge_grad_x=edge_grad_x_flat.reshape(target_height, target_width).astype(np.float32),
        edge_grad_y=edge_grad_y_flat.reshape(target_height, target_width).astype(np.float32),
        delta_x=delta_x.astype(np.float32) if delta_x is not None else None,
        delta_y=delta_y.astype(np.float32) if delta_y is not None else None,
        delta_diag=delta_diag.astype(np.float32) if delta_diag is not None else None,
        delta_anti=delta_anti.astype(np.float32) if delta_anti is not None else None,
    )
