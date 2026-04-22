from __future__ import annotations

import numpy as np

from .io import premultiply, unpremultiply
from .types import SourceLatticeReference


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
) -> SourceLatticeReference:
    indices = lattice_indices(
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
    flat_alpha = source_rgba.reshape(-1, source_rgba.shape[-1])[:, 3]

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

    exemplar_cost = pixel_diff + np.maximum(0.0, per_pixel_mean[:, 3] - flat_premul[:, 3]) * 1e-3
    order = np.lexsort((exemplar_cost.astype(np.float32), flat_idx.astype(np.int64)))
    sharp_premul_flat = mean_premul_flat.copy()
    if order.size > 0:
        sorted_cells = flat_idx[order]
        first_positions = np.concatenate(([0], np.flatnonzero(np.diff(sorted_cells)) + 1))
        best_order = order[first_positions]
        best_cells = flat_idx[best_order]
        sharp_premul_flat[best_cells] = flat_premul[best_order]
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
        cell_dispersion=cell_dispersion.astype(np.float32),
        cell_counts=counts.reshape(target_height, target_width).astype(np.float32),
        cell_support=cell_support.astype(np.float32),
        cell_alpha_max=cell_alpha_max.reshape(target_height, target_width).astype(np.float32),
        delta_x=delta_x.astype(np.float32) if delta_x is not None else None,
        delta_y=delta_y.astype(np.float32) if delta_y is not None else None,
        delta_diag=delta_diag.astype(np.float32) if delta_diag is not None else None,
        delta_anti=delta_anti.astype(np.float32) if delta_anti is not None else None,
    )
