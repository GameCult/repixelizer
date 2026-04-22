from __future__ import annotations

import numpy as np

from .io import premultiply, unpremultiply
from .params import SolverHyperParams
from .source_reference import build_source_lattice_reference
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


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build does not have a usable CUDA device. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return requested


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


def _make_patch_offsets(height: int, width: int, target_height: int, target_width: int) -> np.ndarray:
    cell_x = width / max(1, target_width)
    cell_y = height / max(1, target_height)
    fractions = np.asarray([-0.36, -0.12, 0.12, 0.36], dtype=np.float32)
    offsets: list[tuple[float, float]] = []
    for fy in fractions:
        for fx in fractions:
            offsets.append(
                (
                    (fx * cell_x / max(1.0, width - 1)) * 2.0,
                    (fy * cell_y / max(1.0, height - 1)) * 2.0,
                )
            )
    return np.asarray(offsets, dtype=np.float32)


def _sample_source(F, source_t, grid, *, mode: str = "bilinear"):
    sampled = F.grid_sample(source_t, grid, align_corners=True, mode=mode, padding_mode="border")
    return sampled.permute(0, 2, 3, 1)


def _sample_cell_patches(F, source_t, uv, offsets_t):
    patch_grid = uv[:, :, :, None, :] + offsets_t[None, None, None, :, :]
    patch_grid = patch_grid.clamp(-1.0, 1.0)
    height = patch_grid.shape[1]
    width = patch_grid.shape[2]
    samples = patch_grid.shape[3]
    flattened = patch_grid.reshape(1, height, width * samples, 2)
    sampled = _sample_source(F, source_t, flattened, mode="bilinear")
    return sampled.reshape(1, height, width, samples, sampled.shape[-1])


def _representative_colors(patches, solver_params: SolverHyperParams):
    patch_mean = patches.mean(dim=3, keepdim=True)
    distances = (patches - patch_mean).abs().mean(dim=-1)
    weights = distances.mul(-solver_params.representative_softmax_scale).softmax(dim=3)
    representative = (patches * weights[..., None]).sum(dim=3)
    coherence = (patches - representative[:, :, :, None, :]).abs().mean()
    return representative, coherence


def _exemplar_colors(patches):
    patch_mean = patches.mean(dim=3, keepdim=True)
    distances = (patches - patch_mean).abs().mean(dim=-1)
    selected = distances.argmin(dim=3)
    return patches.gather(
        dim=3,
        index=selected[..., None, None].expand(*selected.shape, 1, patches.shape[-1]),
    ).squeeze(3)


def _source_boundary_deltas(F, source_t, uv, axis: str, probe_scale: float = 0.22):
    if axis == "x":
        if uv.shape[2] < 2:
            return None
        delta = uv[:, :, 1:, :] - uv[:, :, :-1, :]
        midpoint = (uv[:, :, 1:, :] + uv[:, :, :-1, :]) * 0.5
    elif axis == "y":
        if uv.shape[1] < 2:
            return None
        delta = uv[:, 1:, :, :] - uv[:, :-1, :, :]
        midpoint = (uv[:, 1:, :, :] + uv[:, :-1, :, :]) * 0.5
    elif axis == "diag":
        if uv.shape[1] < 2 or uv.shape[2] < 2:
            return None
        delta = uv[:, 1:, 1:, :] - uv[:, :-1, :-1, :]
        midpoint = (uv[:, 1:, 1:, :] + uv[:, :-1, :-1, :]) * 0.5
    elif axis == "anti":
        if uv.shape[1] < 2 or uv.shape[2] < 2:
            return None
        delta = uv[:, 1:, :-1, :] - uv[:, :-1, 1:, :]
        midpoint = (uv[:, 1:, :-1, :] + uv[:, :-1, 1:, :]) * 0.5
    else:
        raise ValueError(f"Unsupported boundary axis: {axis}")
    left = (midpoint - delta * probe_scale).clamp(-1.0, 1.0)
    right = (midpoint + delta * probe_scale).clamp(-1.0, 1.0)
    source_left = _sample_source(F, source_t, left, mode="bilinear")
    source_right = _sample_source(F, source_t, right, mode="bilinear")
    return source_right - source_left


def _boundary_pattern_loss(F, source_t, uv, representative, axis: str, solver_params: SolverHyperParams):
    source_delta = _source_boundary_deltas(
        F,
        source_t,
        uv,
        axis=axis,
        probe_scale=solver_params.boundary_probe_scale,
    )
    if source_delta is None:
        return representative.new_tensor(0.0)
    if axis == "x":
        rep_delta = representative[:, :, 1:, :] - representative[:, :, :-1, :]
    elif axis == "y":
        rep_delta = representative[:, 1:, :, :] - representative[:, :-1, :, :]
    elif axis == "diag":
        rep_delta = representative[:, 1:, 1:, :] - representative[:, :-1, :-1, :]
    elif axis == "anti":
        rep_delta = representative[:, 1:, :-1, :] - representative[:, :-1, 1:, :]
    else:
        raise ValueError(f"Unsupported boundary axis: {axis}")
    magnitude_loss = (
        rep_delta.abs().mean(dim=-1) - source_delta.abs().mean(dim=-1)
    ).abs().mean()
    signed_loss = (rep_delta - source_delta).abs().mean()
    rep_norm = rep_delta / rep_delta.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
    source_norm = source_delta / source_delta.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
    direction_loss = (rep_norm - source_norm).abs().mean()
    return (
        signed_loss * solver_params.boundary_signed_weight
        + direction_loss * solver_params.boundary_direction_weight
        + magnitude_loss * solver_params.boundary_magnitude_weight
    )


def _adjacency_pattern_loss(representative, reference):
    losses = []
    if representative.shape[2] > 1:
        rep_dx = representative[:, :, 1:, :] - representative[:, :, :-1, :]
        ref_dx = reference[:, :, 1:, :] - reference[:, :, :-1, :]
        rep_dx_norm = rep_dx / rep_dx.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
        ref_dx_norm = ref_dx / ref_dx.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
        losses.append((rep_dx - ref_dx).abs().mean() * 0.65 + (rep_dx_norm - ref_dx_norm).abs().mean() * 0.35)
    if representative.shape[1] > 1:
        rep_dy = representative[:, 1:, :, :] - representative[:, :-1, :, :]
        ref_dy = reference[:, 1:, :, :] - reference[:, :-1, :, :]
        rep_dy_norm = rep_dy / rep_dy.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
        ref_dy_norm = ref_dy / ref_dy.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
        losses.append((rep_dy - ref_dy).abs().mean() * 0.65 + (rep_dy_norm - ref_dy_norm).abs().mean() * 0.35)
    if not losses:
        return representative.new_tensor(0.0)
    return sum(losses) / len(losses)


def _motif_pattern_loss(representative, reference):
    if representative.shape[1] < 2 or representative.shape[2] < 2:
        return _adjacency_pattern_loss(representative, reference)
    import torch

    def motif_blocks(tensor):
        blocks = torch.stack(
            [
                tensor[:, :-1, :-1, :],
                tensor[:, :-1, 1:, :],
                tensor[:, 1:, :-1, :],
                tensor[:, 1:, 1:, :],
            ],
            dim=-2,
        )
        centered = blocks - blocks.mean(dim=-2, keepdim=True)
        scale = centered.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        return centered / scale

    return (motif_blocks(representative) - motif_blocks(reference)).abs().mean()


def _normalized_motif_blocks(torch, blocks):
    centered = blocks - blocks.mean(dim=-2, keepdim=True)
    scale = centered.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    return centered / scale


def _normalized_line_triplets(torch, triplets):
    centered = triplets - triplets.mean(dim=-2, keepdim=True)
    scale = centered.abs().mean(dim=(-2, -1), keepdim=True)
    return centered / scale.clamp_min(1e-4), scale


def _pairwise_candidate_energy(
    candidate_colors,
    context_colors,
    desired_delta_x,
    desired_delta_y,
    desired_delta_diag,
    desired_delta_anti,
    *,
    orthogonal_weight: float,
    diagonal_weight: float,
):
    output_height = context_colors.shape[0]
    output_width = context_colors.shape[1]
    energy = candidate_colors.new_zeros(candidate_colors.shape[:3])

    if output_width > 1:
        left_context = context_colors.new_zeros(context_colors.shape)
        left_context[:, 1:, :] = context_colors[:, :-1, :]
        left_mask = context_colors.new_zeros((output_height, output_width, 1))
        left_mask[:, 1:, :] = 1.0
        left_desired = context_colors.new_zeros(context_colors.shape)
        left_desired[:, 1:, :] = desired_delta_x
        left_error = ((candidate_colors - left_context[..., None, :]) - left_desired[..., None, :]).abs().mean(dim=-1)
        energy = energy + left_error * left_mask * orthogonal_weight

        right_context = context_colors.new_zeros(context_colors.shape)
        right_context[:, :-1, :] = context_colors[:, 1:, :]
        right_mask = context_colors.new_zeros((output_height, output_width, 1))
        right_mask[:, :-1, :] = 1.0
        right_desired = context_colors.new_zeros(context_colors.shape)
        right_desired[:, :-1, :] = desired_delta_x
        right_error = ((right_context[..., None, :] - candidate_colors) - right_desired[..., None, :]).abs().mean(dim=-1)
        energy = energy + right_error * right_mask * orthogonal_weight

    if output_height > 1:
        up_context = context_colors.new_zeros(context_colors.shape)
        up_context[1:, :, :] = context_colors[:-1, :, :]
        up_mask = context_colors.new_zeros((output_height, output_width, 1))
        up_mask[1:, :, :] = 1.0
        up_desired = context_colors.new_zeros(context_colors.shape)
        up_desired[1:, :, :] = desired_delta_y
        up_error = ((candidate_colors - up_context[..., None, :]) - up_desired[..., None, :]).abs().mean(dim=-1)
        energy = energy + up_error * up_mask * orthogonal_weight

        down_context = context_colors.new_zeros(context_colors.shape)
        down_context[:-1, :, :] = context_colors[1:, :, :]
        down_mask = context_colors.new_zeros((output_height, output_width, 1))
        down_mask[:-1, :, :] = 1.0
        down_desired = context_colors.new_zeros(context_colors.shape)
        down_desired[:-1, :, :] = desired_delta_y
        down_error = ((down_context[..., None, :] - candidate_colors) - down_desired[..., None, :]).abs().mean(dim=-1)
        energy = energy + down_error * down_mask * orthogonal_weight

    if output_height > 1 and output_width > 1:
        diag_context = context_colors.new_zeros(context_colors.shape)
        diag_context[1:, 1:, :] = context_colors[:-1, :-1, :]
        diag_mask = context_colors.new_zeros((output_height, output_width, 1))
        diag_mask[1:, 1:, :] = 1.0
        diag_desired = context_colors.new_zeros(context_colors.shape)
        diag_desired[1:, 1:, :] = desired_delta_diag
        diag_error = ((candidate_colors - diag_context[..., None, :]) - diag_desired[..., None, :]).abs().mean(dim=-1)
        energy = energy + diag_error * diag_mask * diagonal_weight

        anti_context = context_colors.new_zeros(context_colors.shape)
        anti_context[1:, :-1, :] = context_colors[:-1, 1:, :]
        anti_mask = context_colors.new_zeros((output_height, output_width, 1))
        anti_mask[1:, :-1, :] = 1.0
        anti_desired = context_colors.new_zeros(context_colors.shape)
        anti_desired[1:, :-1, :] = desired_delta_anti
        anti_error = ((candidate_colors - anti_context[..., None, :]) - anti_desired[..., None, :]).abs().mean(dim=-1)
        energy = energy + anti_error * anti_mask * diagonal_weight

    return energy


def _reference_deltas(reference):
    height = reference.shape[0]
    width = reference.shape[1]
    delta_x = reference[:, 1:, :] - reference[:, :-1, :] if width > 1 else None
    delta_y = reference[1:, :, :] - reference[:-1, :, :] if height > 1 else None
    delta_diag = reference[1:, 1:, :] - reference[:-1, :-1, :] if height > 1 and width > 1 else None
    delta_anti = reference[1:, :-1, :] - reference[:-1, 1:, :] if height > 1 and width > 1 else None
    return delta_x, delta_y, delta_diag, delta_anti


def _blend_reference_delta(primary_delta, secondary_delta, secondary_weight: float):
    if primary_delta is None:
        return None
    if secondary_delta is None:
        return primary_delta
    secondary_weight = float(np.clip(secondary_weight, 0.0, 1.0))
    primary_weight = 1.0 - secondary_weight
    return primary_delta * primary_weight + secondary_delta * secondary_weight


def _build_source_reliability(reference) -> np.ndarray:
    baseline_dispersion = max(reference.dispersion, 1e-4)
    dispersion_scale = reference.cell_dispersion / baseline_dispersion
    dispersion_confidence = np.exp(-np.maximum(0.0, dispersion_scale - 1.0))
    support_confidence = np.clip(reference.cell_support, 0.0, 1.0)
    alpha_confidence = np.clip(np.maximum(reference.cell_alpha_max, reference.sharp_rgba[..., 3]), 0.0, 1.0)
    reliability = np.clip(0.2 + support_confidence * 0.5 + alpha_confidence * 0.3, 0.0, 1.0)
    return (reliability * dispersion_confidence).astype(np.float32)


def _reference_match_energy(
    representative_energy,
    source_energy,
    source_reliability,
    *,
    representative_weight: float,
    source_weight: float,
):
    source_mix = source_reliability[..., None] * source_weight
    representative_mix = representative_weight + (1.0 - source_reliability[..., None]) * source_weight
    normalization = (source_mix + representative_mix).clamp_min(1e-4)
    return (representative_energy * representative_mix + source_energy * source_mix) / normalization


def _edge_reliability(source_reliability, axis: str):
    if axis == "x":
        return (source_reliability[:, 1:] + source_reliability[:, :-1]) * 0.5
    if axis == "y":
        return (source_reliability[1:, :] + source_reliability[:-1, :]) * 0.5
    if axis == "diag":
        return (source_reliability[1:, 1:] + source_reliability[:-1, :-1]) * 0.5
    if axis == "anti":
        return (source_reliability[1:, :-1] + source_reliability[:-1, 1:]) * 0.5
    raise ValueError(f"Unsupported edge axis: {axis}")


def _blend_reference_delta_map(
    representative_delta,
    source_delta,
    source_reliability,
    *,
    axis: str,
    representative_weight: float,
    source_weight: float,
):
    if representative_delta is None:
        return source_delta
    if source_delta is None:
        return representative_delta
    edge_reliability = _edge_reliability(source_reliability, axis=axis)
    source_mix = edge_reliability * source_weight
    representative_mix = representative_weight + (1.0 - edge_reliability) * source_weight
    normalization = (source_mix + representative_mix).clamp_min(1e-4)
    return (
        representative_delta * representative_mix[..., None] + source_delta * source_mix[..., None]
    ) / normalization[..., None]


def _motif_candidate_energy(torch, candidate_colors, context_colors, anchor):
    if candidate_colors.shape[0] < 2 or candidate_colors.shape[1] < 2:
        return candidate_colors.new_zeros(candidate_colors.shape[:3])

    energy = candidate_colors.new_zeros(candidate_colors.shape[:3])
    contributions = candidate_colors.new_zeros(candidate_colors.shape[:3])

    def add_block_error(
        y_slice,
        x_slice,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        anchor_top_left,
        anchor_top_right,
        anchor_bottom_left,
        anchor_bottom_right,
    ):
        k = candidate_colors[y_slice, x_slice].shape[2]

        def ensure_candidate_dim(tensor):
            if tensor.ndim == 4:
                return tensor
            return tensor[..., None, :].expand(-1, -1, k, -1)

        candidate_block = torch.stack(
            [
                ensure_candidate_dim(top_left),
                ensure_candidate_dim(top_right),
                ensure_candidate_dim(bottom_left),
                ensure_candidate_dim(bottom_right),
            ],
            dim=-2,
        )
        anchor_block = torch.stack(
            [
                anchor_top_left,
                anchor_top_right,
                anchor_bottom_left,
                anchor_bottom_right,
            ],
            dim=-2,
        )
        error = (
            _normalized_motif_blocks(torch, candidate_block)
            - _normalized_motif_blocks(torch, anchor_block[..., None, :, :])
        ).abs().mean(dim=(-2, -1))
        energy[y_slice, x_slice] = energy[y_slice, x_slice] + error
        contributions[y_slice, x_slice] = contributions[y_slice, x_slice] + 1.0

    add_block_error(
        slice(1, None),
        slice(1, None),
        context_colors[:-1, :-1, :],
        context_colors[:-1, 1:, :],
        context_colors[1:, :-1, :],
        candidate_colors[1:, 1:, :, :],
        anchor[:-1, :-1, :],
        anchor[:-1, 1:, :],
        anchor[1:, :-1, :],
        anchor[1:, 1:, :],
    )
    add_block_error(
        slice(1, None),
        slice(None, -1),
        context_colors[:-1, :-1, :],
        context_colors[:-1, 1:, :],
        candidate_colors[1:, :-1, :, :],
        context_colors[1:, 1:, :],
        anchor[:-1, :-1, :],
        anchor[:-1, 1:, :],
        anchor[1:, :-1, :],
        anchor[1:, 1:, :],
    )
    add_block_error(
        slice(None, -1),
        slice(1, None),
        context_colors[:-1, :-1, :],
        candidate_colors[:-1, 1:, :, :],
        context_colors[1:, :-1, :],
        context_colors[1:, 1:, :],
        anchor[:-1, :-1, :],
        anchor[:-1, 1:, :],
        anchor[1:, :-1, :],
        anchor[1:, 1:, :],
    )
    add_block_error(
        slice(None, -1),
        slice(None, -1),
        candidate_colors[:-1, :-1, :, :],
        context_colors[:-1, 1:, :],
        context_colors[1:, :-1, :],
        context_colors[1:, 1:, :],
        anchor[:-1, :-1, :],
        anchor[:-1, 1:, :],
        anchor[1:, :-1, :],
        anchor[1:, 1:, :],
    )

    return energy / contributions.clamp_min(1.0)


def _line_candidate_energy(torch, candidate_colors, context_colors, anchor):
    if candidate_colors.shape[0] < 3 and candidate_colors.shape[1] < 3:
        return candidate_colors.new_zeros(candidate_colors.shape[:3])

    energy = candidate_colors.new_zeros(candidate_colors.shape[:3])
    contributions = candidate_colors.new_zeros(candidate_colors.shape[:3])

    def add_line_error(y_slice, x_slice, prev_context, next_context, anchor_prev, anchor_center, anchor_next):
        candidate_triplet = torch.stack(
            [
                prev_context[..., None, :].expand(-1, -1, candidate_colors[y_slice, x_slice].shape[2], -1),
                candidate_colors[y_slice, x_slice],
                next_context[..., None, :].expand(-1, -1, candidate_colors[y_slice, x_slice].shape[2], -1),
            ],
            dim=-2,
        )
        anchor_triplet = torch.stack(
            [
                anchor_prev,
                anchor_center,
                anchor_next,
            ],
            dim=-2,
        )
        candidate_norm, _ = _normalized_line_triplets(torch, candidate_triplet)
        anchor_norm, anchor_scale = _normalized_line_triplets(torch, anchor_triplet)
        weight = anchor_scale[..., 0, 0].clamp_max(1.0)
        error = (candidate_norm - anchor_norm[..., None, :, :]).abs().mean(dim=(-2, -1))
        energy[y_slice, x_slice] = energy[y_slice, x_slice] + error * weight[..., None]
        contributions[y_slice, x_slice] = contributions[y_slice, x_slice] + weight[..., None]

    if candidate_colors.shape[1] >= 3:
        add_line_error(
            slice(None),
            slice(1, -1),
            context_colors[:, :-2, :],
            context_colors[:, 2:, :],
            anchor[:, :-2, :],
            anchor[:, 1:-1, :],
            anchor[:, 2:, :],
        )

    if candidate_colors.shape[0] >= 3:
        add_line_error(
            slice(1, -1),
            slice(None),
            context_colors[:-2, :, :],
            context_colors[2:, :, :],
            anchor[:-2, :, :],
            anchor[1:-1, :, :],
            anchor[2:, :, :],
        )

    if candidate_colors.shape[0] >= 3 and candidate_colors.shape[1] >= 3:
        add_line_error(
            slice(1, -1),
            slice(1, -1),
            context_colors[:-2, :-2, :],
            context_colors[2:, 2:, :],
            anchor[:-2, :-2, :],
            anchor[1:-1, 1:-1, :],
            anchor[2:, 2:, :],
        )
        add_line_error(
            slice(1, -1),
            slice(1, -1),
            context_colors[:-2, 2:, :],
            context_colors[2:, :-2, :],
            anchor[:-2, 2:, :],
            anchor[1:-1, 1:-1, :],
            anchor[2:, :-2, :],
        )

    return energy / contributions.clamp_min(1e-4)


def _line_pattern_loss(torch, tensor, reference):
    losses = []

    def add_line_loss(prev_tensor, center_tensor, next_tensor, prev_ref, center_ref, next_ref):
        tensor_triplet = torch.stack([prev_tensor, center_tensor, next_tensor], dim=-2)
        ref_triplet = torch.stack([prev_ref, center_ref, next_ref], dim=-2)
        tensor_norm, _ = _normalized_line_triplets(torch, tensor_triplet)
        ref_norm, ref_scale = _normalized_line_triplets(torch, ref_triplet)
        weight = ref_scale[..., 0, 0].clamp_max(1.0)
        losses.append(((tensor_norm - ref_norm).abs().mean(dim=(-2, -1)) * weight).sum() / weight.sum().clamp_min(1e-4))

    if tensor.shape[2] >= 3:
        add_line_loss(
            tensor[:, :, :-2, :],
            tensor[:, :, 1:-1, :],
            tensor[:, :, 2:, :],
            reference[:, :, :-2, :],
            reference[:, :, 1:-1, :],
            reference[:, :, 2:, :],
        )
    if tensor.shape[1] >= 3:
        add_line_loss(
            tensor[:, :-2, :, :],
            tensor[:, 1:-1, :, :],
            tensor[:, 2:, :, :],
            reference[:, :-2, :, :],
            reference[:, 1:-1, :, :],
            reference[:, 2:, :, :],
        )
    if tensor.shape[1] >= 3 and tensor.shape[2] >= 3:
        add_line_loss(
            tensor[:, :-2, :-2, :],
            tensor[:, 1:-1, 1:-1, :],
            tensor[:, 2:, 2:, :],
            reference[:, :-2, :-2, :],
            reference[:, 1:-1, 1:-1, :],
            reference[:, 2:, 2:, :],
        )
        add_line_loss(
            tensor[:, :-2, 2:, :],
            tensor[:, 1:-1, 1:-1, :],
            tensor[:, 2:, :-2, :],
            reference[:, :-2, 2:, :],
            reference[:, 1:-1, 1:-1, :],
            reference[:, 2:, :-2, :],
        )

    if not losses:
        return tensor.new_tensor(0.0)
    return sum(losses) / len(losses)


def _relax_candidate_selection(
    torch,
    candidate_colors,
    base_energy,
    anchor,
    source_reference,
    desired_delta_x,
    desired_delta_y,
    desired_delta_diag,
    desired_delta_anti,
    solver_params: SolverHyperParams,
    *,
    iterations: int,
):
    if iterations <= 0:
        selected = torch.argmin(base_energy, dim=-1)
        return selected, _select_colors(candidate_colors, selected), [], selected.detach()

    start_temp = max(1e-3, solver_params.relax_start_temperature)
    end_temp = max(1e-3, min(start_temp, solver_params.relax_end_temperature))
    damping = float(np.clip(solver_params.relax_damping, 0.0, 0.95))

    probs = (-base_energy / start_temp).softmax(dim=2)
    loss_history: list[float] = []
    final_energy = base_energy
    source_delta_x, source_delta_y, source_delta_diag, source_delta_anti = _reference_deltas(source_reference)

    for step in range(iterations):
        context_colors = (candidate_colors * probs[..., None]).sum(dim=2)
        final_energy = base_energy.clone()
        final_energy = final_energy + _pairwise_candidate_energy(
            candidate_colors,
            context_colors,
            desired_delta_x,
            desired_delta_y,
            desired_delta_diag,
            desired_delta_anti,
            orthogonal_weight=solver_params.relax_orthogonal_weight,
            diagonal_weight=solver_params.relax_diagonal_weight,
        )
        if solver_params.relax_source_adjacency_weight > 0.0:
            source_energy = _pairwise_candidate_energy(
                candidate_colors,
                context_colors,
                source_delta_x,
                source_delta_y,
                source_delta_diag,
                source_delta_anti,
                orthogonal_weight=solver_params.relax_orthogonal_weight,
                diagonal_weight=solver_params.relax_diagonal_weight,
            )
            final_energy = final_energy + source_energy * solver_params.relax_source_adjacency_weight
        if solver_params.relax_motif_weight > 0.0:
            motif_energy = _motif_candidate_energy(torch, candidate_colors, context_colors, anchor)
            final_energy = final_energy + motif_energy * solver_params.relax_motif_weight
        if solver_params.relax_source_motif_weight > 0.0:
            source_motif_energy = _motif_candidate_energy(torch, candidate_colors, context_colors, source_reference)
            final_energy = final_energy + source_motif_energy * solver_params.relax_source_motif_weight
        if solver_params.relax_line_weight > 0.0:
            line_energy = _line_candidate_energy(torch, candidate_colors, context_colors, anchor)
            final_energy = final_energy + line_energy * solver_params.relax_line_weight
        if solver_params.relax_source_line_weight > 0.0:
            source_line_energy = _line_candidate_energy(torch, candidate_colors, context_colors, source_reference)
            final_energy = final_energy + source_line_energy * solver_params.relax_source_line_weight

        alpha = 1.0 if iterations <= 1 else step / float(iterations - 1)
        temperature = start_temp + (end_temp - start_temp) * alpha
        updated_probs = (-final_energy / max(temperature, 1e-3)).softmax(dim=2)
        if damping > 0.0:
            probs = updated_probs * (1.0 - damping) + probs * damping
            probs = probs / probs.sum(dim=2, keepdim=True).clamp_min(1e-8)
        else:
            probs = updated_probs
        loss_history.append(float((final_energy * probs).sum(dim=2).mean().detach().cpu().item()))

    relaxed_context = (candidate_colors * probs[..., None]).sum(dim=2)
    handoff_energy = final_energy + (
        (candidate_colors - relaxed_context[..., None, :]).abs().mean(dim=-1) * solver_params.relax_handoff_weight
    )
    handoff_energy = handoff_energy + _pairwise_candidate_energy(
        candidate_colors,
        relaxed_context,
        desired_delta_x,
        desired_delta_y,
        desired_delta_diag,
        desired_delta_anti,
        orthogonal_weight=solver_params.relax_orthogonal_weight,
        diagonal_weight=solver_params.relax_diagonal_weight,
    )
    if solver_params.relax_source_adjacency_weight > 0.0:
        handoff_source_energy = _pairwise_candidate_energy(
            candidate_colors,
            relaxed_context,
            source_delta_x,
            source_delta_y,
            source_delta_diag,
            source_delta_anti,
            orthogonal_weight=solver_params.relax_orthogonal_weight,
            diagonal_weight=solver_params.relax_diagonal_weight,
        )
        handoff_energy = handoff_energy + handoff_source_energy * solver_params.relax_source_adjacency_weight
    if solver_params.relax_motif_weight > 0.0:
        handoff_energy = handoff_energy + _motif_candidate_energy(
            torch,
            candidate_colors,
            relaxed_context,
            relaxed_context,
        ) * solver_params.relax_motif_weight
    if solver_params.relax_source_motif_weight > 0.0:
        handoff_energy = handoff_energy + _motif_candidate_energy(
            torch,
            candidate_colors,
            relaxed_context,
            source_reference,
        ) * solver_params.relax_source_motif_weight
    if solver_params.relax_line_weight > 0.0:
        handoff_energy = handoff_energy + _line_candidate_energy(
            torch,
            candidate_colors,
            relaxed_context,
            relaxed_context,
        ) * solver_params.relax_line_weight
    if solver_params.relax_source_line_weight > 0.0:
        handoff_energy = handoff_energy + _line_candidate_energy(
            torch,
            candidate_colors,
            relaxed_context,
            source_reference,
        ) * solver_params.relax_source_line_weight
    selected = torch.argmin(handoff_energy, dim=2)
    mode_selected = torch.argmax(probs, dim=2)
    return selected, relaxed_context.detach(), loss_history, mode_selected.detach()


def _snap_output_to_source_pixels(
    torch,
    source_t,
    uv_t,
    representative_t,
    source_reference_t,
    source_reliability_t,
    solver_params: SolverHyperParams,
    *,
    cell_x: float,
    cell_y: float,
) -> np.ndarray:
    source_hw = source_t[0].permute(1, 2, 0)
    height = source_hw.shape[0]
    width = source_hw.shape[1]
    uv = uv_t[0]
    representative = representative_t[0]
    source_reference = source_reference_t[0]
    source_reliability = source_reliability_t[0]
    output_height = representative.shape[0]
    output_width = representative.shape[1]

    fractions = torch.tensor([-0.42, -0.21, 0.0, 0.21, 0.42], device=uv.device, dtype=uv.dtype)
    offset_y, offset_x = torch.meshgrid(fractions, fractions, indexing="ij")
    offset_x = (offset_x.reshape(-1) * cell_x).to(dtype=uv.dtype)
    offset_y = (offset_y.reshape(-1) * cell_y).to(dtype=uv.dtype)

    xs = (uv[..., 0] + 1.0) * 0.5 * max(1.0, float(width - 1))
    ys = (uv[..., 1] + 1.0) * 0.5 * max(1.0, float(height - 1))
    candidate_x = torch.round(xs[..., None] + offset_x).clamp(0, max(0, width - 1)).to(dtype=torch.long)
    candidate_y = torch.round(ys[..., None] + offset_y).clamp(0, max(0, height - 1)).to(dtype=torch.long)
    candidate_colors = source_hw[candidate_y, candidate_x]

    representative_match = (candidate_colors - representative[..., None, :]).abs().mean(dim=-1)
    source_match = (candidate_colors - source_reference[..., None, :]).abs().mean(dim=-1)
    base_energy = _reference_match_energy(
        representative_match,
        source_match,
        source_reliability,
        representative_weight=solver_params.snap_representative_match_weight,
        source_weight=solver_params.snap_source_match_weight,
    ) * solver_params.snap_base_match_weight
    representative_delta_x = representative[:, 1:, :] - representative[:, :-1, :] if output_width > 1 else None
    representative_delta_y = representative[1:, :, :] - representative[:-1, :, :] if output_height > 1 else None
    representative_delta_diag = (
        representative[1:, 1:, :] - representative[:-1, :-1, :] if output_height > 1 and output_width > 1 else None
    )
    representative_delta_anti = (
        representative[1:, :-1, :] - representative[:-1, 1:, :] if output_height > 1 and output_width > 1 else None
    )
    source_delta_x, source_delta_y, source_delta_diag, source_delta_anti = _reference_deltas(source_reference)
    desired_delta_x = _blend_reference_delta_map(
        representative_delta_x,
        source_delta_x,
        source_reliability,
        axis="x",
        representative_weight=solver_params.snap_representative_delta_weight,
        source_weight=solver_params.snap_source_delta_weight,
    )
    desired_delta_y = _blend_reference_delta_map(
        representative_delta_y,
        source_delta_y,
        source_reliability,
        axis="y",
        representative_weight=solver_params.snap_representative_delta_weight,
        source_weight=solver_params.snap_source_delta_weight,
    )
    desired_delta_diag = _blend_reference_delta_map(
        representative_delta_diag,
        source_delta_diag,
        source_reliability,
        axis="diag",
        representative_weight=solver_params.snap_representative_delta_weight,
        source_weight=solver_params.snap_source_delta_weight,
    )
    desired_delta_anti = _blend_reference_delta_map(
        representative_delta_anti,
        source_delta_anti,
        source_reliability,
        axis="anti",
        representative_weight=solver_params.snap_representative_delta_weight,
        source_weight=solver_params.snap_source_delta_weight,
    )

    selected = torch.argmin(base_energy, dim=-1)
    ranking_energy = base_energy
    for _ in range(4):
        selected_colors = torch.gather(
            candidate_colors,
            dim=2,
            index=selected[..., None, None].expand(output_height, output_width, 1, candidate_colors.shape[-1]),
        ).squeeze(2)
        energy = base_energy.clone()
        if output_width > 1:
            left_selected = torch.zeros_like(selected_colors)
            left_selected[:, 1:, :] = selected_colors[:, :-1, :]
            left_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            left_mask[:, 1:, :] = 1.0
            left_desired = torch.zeros_like(selected_colors)
            left_desired[:, 1:, :] = desired_delta_x
            left_error = ((candidate_colors - left_selected[..., None, :]) - left_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + left_error * left_mask * solver_params.snap_neighbor_weight

            right_selected = torch.zeros_like(selected_colors)
            right_selected[:, :-1, :] = selected_colors[:, 1:, :]
            right_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            right_mask[:, :-1, :] = 1.0
            right_desired = torch.zeros_like(selected_colors)
            right_desired[:, :-1, :] = desired_delta_x
            right_error = ((right_selected[..., None, :] - candidate_colors) - right_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + right_error * right_mask * solver_params.snap_neighbor_weight
        if output_height > 1:
            up_selected = torch.zeros_like(selected_colors)
            up_selected[1:, :, :] = selected_colors[:-1, :, :]
            up_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            up_mask[1:, :, :] = 1.0
            up_desired = torch.zeros_like(selected_colors)
            up_desired[1:, :, :] = desired_delta_y
            up_error = ((candidate_colors - up_selected[..., None, :]) - up_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + up_error * up_mask * solver_params.snap_neighbor_weight

            down_selected = torch.zeros_like(selected_colors)
            down_selected[:-1, :, :] = selected_colors[1:, :, :]
            down_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            down_mask[:-1, :, :] = 1.0
            down_desired = torch.zeros_like(selected_colors)
            down_desired[:-1, :, :] = desired_delta_y
            down_error = ((down_selected[..., None, :] - candidate_colors) - down_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + down_error * down_mask * solver_params.snap_neighbor_weight
        if output_height > 1 and output_width > 1:
            diag_selected = torch.zeros_like(selected_colors)
            diag_selected[1:, 1:, :] = selected_colors[:-1, :-1, :]
            diag_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            diag_mask[1:, 1:, :] = 1.0
            diag_desired = torch.zeros_like(selected_colors)
            diag_desired[1:, 1:, :] = desired_delta_diag
            diag_error = ((candidate_colors - diag_selected[..., None, :]) - diag_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + diag_error * diag_mask * solver_params.snap_diagonal_weight

            anti_selected = torch.zeros_like(selected_colors)
            anti_selected[1:, :-1, :] = selected_colors[:-1, 1:, :]
            anti_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            anti_mask[1:, :-1, :] = 1.0
            anti_desired = torch.zeros_like(selected_colors)
            anti_desired[1:, :-1, :] = desired_delta_anti
            anti_error = ((candidate_colors - anti_selected[..., None, :]) - anti_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + anti_error * anti_mask * solver_params.snap_diagonal_weight
        ranking_energy = energy
        selected = torch.argmin(energy, dim=-1)

    hardened_selected, use_opaque, use_transparent = _harden_binary_alpha_selection(
        torch,
        candidate_colors,
        selected,
        ranking_energy,
        representative[..., 3],
        solver_params=solver_params,
    )
    final_colors = _select_colors(candidate_colors, hardened_selected)
    return _finalize_output_rgba(final_colors, use_opaque, use_transparent)


def _select_colors(candidate_colors, selected):
    output_height = candidate_colors.shape[0]
    output_width = candidate_colors.shape[1]
    return candidate_colors.gather(
        dim=2,
        index=selected[..., None, None].expand(output_height, output_width, 1, candidate_colors.shape[-1]),
    ).squeeze(2)


def _harden_binary_alpha_selection(
    torch,
    candidate_colors,
    selected,
    ranking_energy,
    representative_alpha,
    solver_params: SolverHyperParams,
):
    selected_colors = _select_colors(candidate_colors, selected)
    selected_alpha = selected_colors[..., 3]
    partial_mask = (
        selected_alpha > solver_params.alpha_transparent_threshold
    ) & (
        selected_alpha < solver_params.alpha_opaque_threshold
    )
    prefer_foreground = (selected_alpha >= solver_params.alpha_foreground_threshold) | (
        representative_alpha >= solver_params.alpha_representative_foreground_threshold
    )

    opaque_mask = candidate_colors[..., 3] >= solver_params.alpha_opaque_threshold
    transparent_mask = candidate_colors[..., 3] <= solver_params.alpha_transparent_threshold
    large_penalty = ranking_energy.new_full(ranking_energy.shape, 1e6)
    opaque_scores = torch.where(opaque_mask, ranking_energy, ranking_energy + large_penalty)
    transparent_scores = torch.where(transparent_mask, ranking_energy, ranking_energy + large_penalty)

    opaque_idx = torch.argmin(opaque_scores, dim=2)
    transparent_idx = torch.argmin(transparent_scores, dim=2)
    opaque_available = torch.any(opaque_mask, dim=2)
    transparent_available = torch.any(transparent_mask, dim=2)

    hardened = selected.clone()
    use_opaque = partial_mask & prefer_foreground & opaque_available
    use_transparent = partial_mask & (~prefer_foreground) & transparent_available
    hardened = torch.where(use_opaque, opaque_idx, hardened)
    hardened = torch.where(use_transparent, transparent_idx, hardened)
    return hardened, use_opaque, use_transparent


def _finalize_output_rgba(best_colors, force_opaque_mask, force_transparent_mask) -> np.ndarray:
    rgba = unpremultiply(best_colors.detach().cpu().numpy().clip(0.0, 1.0))
    if force_opaque_mask is not None:
        opaque = force_opaque_mask.detach().cpu().numpy()
        rgba[..., 3][opaque] = 1.0
    if force_transparent_mask is not None:
        transparent = force_transparent_mask.detach().cpu().numpy()
        rgba[..., 3][transparent] = 0.0
        rgba[..., :3][transparent] = 0.0
    return rgba


def _structure_score(
    torch,
    F,
    source_t,
    uv_t,
    candidate_t,
    representative_t,
    source_reference_t,
    anchor_t,
    solver_params: SolverHyperParams,
):
    boundary_terms = [
        _boundary_pattern_loss(F, source_t, uv_t, candidate_t, "x", solver_params),
        _boundary_pattern_loss(F, source_t, uv_t, candidate_t, "y", solver_params),
        _boundary_pattern_loss(F, source_t, uv_t, candidate_t, "diag", solver_params),
        _boundary_pattern_loss(F, source_t, uv_t, candidate_t, "anti", solver_params),
    ]
    boundary = sum(boundary_terms) / len(boundary_terms)
    anchor_adj = _adjacency_pattern_loss(candidate_t, anchor_t)
    anchor_motif = _motif_pattern_loss(candidate_t, anchor_t)
    anchor_line = _line_pattern_loss(torch, candidate_t, anchor_t)
    source_adj = _adjacency_pattern_loss(candidate_t, source_reference_t)
    source_motif = _motif_pattern_loss(candidate_t, source_reference_t)
    source_line = _line_pattern_loss(torch, candidate_t, source_reference_t)
    representative_match = (candidate_t - representative_t).abs().mean()
    return (
        boundary * solver_params.structure_boundary_weight
        + anchor_adj * solver_params.structure_anchor_adjacency_weight
        + anchor_motif * solver_params.structure_anchor_motif_weight
        + anchor_line * solver_params.structure_anchor_line_weight
        + source_adj * solver_params.structure_source_adjacency_weight
        + source_motif * solver_params.structure_source_motif_weight
        + source_line * solver_params.structure_source_line_weight
        + representative_match * solver_params.structure_representative_weight
    )


def _discrete_refine_output(
    torch,
    F,
    source_t,
    uv_t,
    representative_t,
    source_reference_t,
    source_reliability_t,
    anchor_t,
    source_delta_x_t,
    source_delta_y_t,
    source_delta_diag_t,
    source_delta_anti_t,
    solver_params: SolverHyperParams,
    *,
    cell_x: float,
    cell_y: float,
    iterations: int,
) -> tuple[np.ndarray, list[float]]:
    source_hw = source_t[0].permute(1, 2, 0)
    height = source_hw.shape[0]
    width = source_hw.shape[1]
    uv = uv_t[0]
    representative = representative_t[0]
    source_reference = source_reference_t[0]
    source_reliability = source_reliability_t[0]
    anchor = anchor_t[0].permute(1, 2, 0)
    output_height = representative.shape[0]
    output_width = representative.shape[1]

    candidate_levels = max(3, int(solver_params.refine_candidate_levels))
    if candidate_levels % 2 == 0:
        candidate_levels += 1
    candidate_extent = max(0.05, float(solver_params.refine_candidate_extent))
    fraction_values = np.linspace(-candidate_extent, candidate_extent, candidate_levels, dtype=np.float32)
    fractions = torch.tensor(fraction_values, device=uv.device, dtype=uv.dtype)
    offset_y, offset_x = torch.meshgrid(fractions, fractions, indexing="ij")
    offset_x = (offset_x.reshape(-1) * cell_x).to(dtype=uv.dtype)
    offset_y = (offset_y.reshape(-1) * cell_y).to(dtype=uv.dtype)

    xs = (uv[..., 0] + 1.0) * 0.5 * max(1.0, float(width - 1))
    ys = (uv[..., 1] + 1.0) * 0.5 * max(1.0, float(height - 1))
    candidate_x = torch.round(xs[..., None] + offset_x).clamp(0, max(0, width - 1)).to(dtype=torch.long)
    candidate_y = torch.round(ys[..., None] + offset_y).clamp(0, max(0, height - 1)).to(dtype=torch.long)
    candidate_colors = source_hw[candidate_y, candidate_x]
    anchor_energy = (candidate_colors - anchor[..., None, :]).abs().mean(dim=-1)
    rep_energy = (candidate_colors - representative[..., None, :]).abs().mean(dim=-1)
    source_reference_energy = (candidate_colors - source_reference[..., None, :]).abs().mean(dim=-1)
    alpha_energy = (candidate_colors[..., 3] - anchor[..., None, 3]).abs()
    distance_energy = ((offset_x / max(cell_x, 1e-4)) ** 2 + (offset_y / max(cell_y, 1e-4)) ** 2).reshape(1, 1, -1)
    match_energy = _reference_match_energy(
        rep_energy,
        source_reference_energy,
        source_reliability,
        representative_weight=solver_params.refine_representative_match_weight,
        source_weight=solver_params.refine_source_match_weight,
    )
    base_energy = (
        anchor_energy * solver_params.refine_anchor_weight
        + match_energy * solver_params.refine_representative_weight
        + alpha_energy * solver_params.refine_alpha_weight
        + distance_energy * solver_params.refine_distance_weight
    )
    relax_base_energy = (
        anchor_energy * solver_params.refine_anchor_weight * solver_params.relax_anchor_scale
        + match_energy * solver_params.refine_representative_weight
        + alpha_energy * solver_params.refine_alpha_weight
        + distance_energy * solver_params.refine_distance_weight
    )

    anchor_delta_x = anchor[:, 1:, :] - anchor[:, :-1, :] if output_width > 1 else None
    anchor_delta_y = anchor[1:, :, :] - anchor[:-1, :, :] if output_height > 1 else None
    anchor_delta_diag = anchor[1:, 1:, :] - anchor[:-1, :-1, :] if output_height > 1 and output_width > 1 else None
    anchor_delta_anti = anchor[1:, :-1, :] - anchor[:-1, 1:, :] if output_height > 1 and output_width > 1 else None
    source_delta_x = source_delta_x_t[0] if source_delta_x_t is not None else None
    source_delta_y = source_delta_y_t[0] if source_delta_y_t is not None else None
    source_delta_diag = source_delta_diag_t[0] if source_delta_diag_t is not None else None
    source_delta_anti = source_delta_anti_t[0] if source_delta_anti_t is not None else None
    source_ref_dx, source_ref_dy, source_ref_diag, source_ref_anti = _reference_deltas(source_reference)

    def desired_delta(anchor_delta, source_delta):
        if anchor_delta is None:
            return None
        if source_delta is None:
            return anchor_delta
        source_weight = solver_params.refine_source_delta_weight
        anchor_weight = 1.0 - source_weight
        return anchor_delta * anchor_weight + source_delta * source_weight

    desired_delta_x = desired_delta(anchor_delta_x, source_delta_x)
    desired_delta_y = desired_delta(anchor_delta_y, source_delta_y)
    desired_delta_diag = desired_delta(anchor_delta_diag, source_delta_diag)
    desired_delta_anti = desired_delta(anchor_delta_anti, source_delta_anti)

    passes = max(0, iterations)
    relax_iterations = min(max(0, solver_params.relax_iterations), passes) if passes > 0 else 0
    selected, relaxed_context, relax_history, relaxed_mode_selected = _relax_candidate_selection(
        torch,
        candidate_colors,
        relax_base_energy,
        anchor,
        source_reference,
        desired_delta_x,
        desired_delta_y,
        desired_delta_diag,
        desired_delta_anti,
        solver_params,
        iterations=relax_iterations,
    )
    relaxed_mode_colors = _select_colors(candidate_colors, relaxed_mode_selected)
    refine_base_energy = base_energy + (
        (candidate_colors - relaxed_context[..., None, :]).abs().mean(dim=-1) * solver_params.relax_handoff_weight
    )
    if solver_params.refine_relaxed_mode_weight > 0.0:
        refine_base_energy = refine_base_energy + (
            (candidate_colors - relaxed_mode_colors[..., None, :]).abs().mean(dim=-1)
            * solver_params.refine_relaxed_mode_weight
        )
    start_candidates = [selected, relaxed_mode_selected]
    selected = selected.clone()
    best_selected = selected.clone()
    best_score = float("inf")
    loss_history: list[float] = list(relax_history)
    anchor_hw = anchor_t.permute(0, 2, 3, 1)

    def candidate_energy(context_colors):
        energy = refine_base_energy.clone()
        energy = energy + _pairwise_candidate_energy(
            candidate_colors,
            context_colors,
            desired_delta_x,
            desired_delta_y,
            desired_delta_diag,
            desired_delta_anti,
            orthogonal_weight=solver_params.refine_orthogonal_weight,
            diagonal_weight=solver_params.refine_diagonal_weight,
        )
        if solver_params.refine_motif_weight > 0.0:
            motif_energy = _motif_candidate_energy(torch, candidate_colors, context_colors, anchor)
            energy = energy + motif_energy * solver_params.refine_motif_weight
        if solver_params.structure_source_motif_weight > 0.0:
            source_motif_energy = _motif_candidate_energy(torch, candidate_colors, context_colors, source_reference)
            energy = energy + source_motif_energy * solver_params.structure_source_motif_weight
        if solver_params.refine_line_weight > 0.0:
            line_energy = _line_candidate_energy(torch, candidate_colors, context_colors, anchor)
            energy = energy + line_energy * solver_params.refine_line_weight
        if solver_params.structure_source_line_weight > 0.0:
            source_line_energy = _line_candidate_energy(torch, candidate_colors, context_colors, source_reference)
            energy = energy + source_line_energy * solver_params.structure_source_line_weight
        if solver_params.structure_source_adjacency_weight > 0.0:
            source_adjacency_energy = _pairwise_candidate_energy(
                candidate_colors,
                context_colors,
                source_ref_dx,
                source_ref_dy,
                source_ref_diag,
                source_ref_anti,
                orthogonal_weight=solver_params.refine_orthogonal_weight,
                diagonal_weight=solver_params.refine_diagonal_weight,
            )
            energy = energy + source_adjacency_energy * solver_params.structure_source_adjacency_weight
        return energy

    for start_selected in start_candidates:
        start_colors = _select_colors(candidate_colors, start_selected)
        start_score = float(
            _structure_score(
                torch,
                F,
                source_t,
                uv_t,
                start_colors[None, ...],
                representative_t,
                source_reference_t,
                anchor_hw,
                solver_params,
            ).detach().cpu().item()
        )
        if start_score < best_score:
            best_score = start_score
            best_selected = start_selected.clone()
            selected = start_selected.clone()

    for step in range(passes + 1):
        selected_colors = _select_colors(candidate_colors, selected)
        energy = candidate_energy(selected_colors)

        candidate_t = selected_colors[None, ...]
        score = float(
            _structure_score(
                torch,
                F,
                source_t,
                uv_t,
                candidate_t,
                representative_t,
                source_reference_t,
                anchor_hw,
                solver_params,
            ).detach().cpu().item()
        )
        loss_history.append(score)
        if score < best_score:
            best_score = score
            best_selected = selected.clone()

        if step >= passes:
            break
        next_selected = torch.argmin(energy, dim=-1)
        if torch.equal(next_selected, selected):
            selected = next_selected
            break
        selected = next_selected

    final_candidates = [best_selected, relaxed_mode_selected]
    final_selected = best_selected
    final_score = best_score
    for candidate_selected in final_candidates:
        candidate_colors_selected = _select_colors(candidate_colors, candidate_selected)
        candidate_score = float(
            _structure_score(
                torch,
                F,
                source_t,
                uv_t,
                candidate_colors_selected[None, ...],
                representative_t,
                source_reference_t,
                anchor_hw,
                solver_params,
            ).detach().cpu().item()
        )
        if candidate_score < final_score:
            final_score = candidate_score
            final_selected = candidate_selected.clone()

    final_selected_colors = _select_colors(candidate_colors, final_selected)
    final_ranking_energy = candidate_energy(final_selected_colors)
    hardened_selected, use_opaque, use_transparent = _harden_binary_alpha_selection(
        torch,
        candidate_colors,
        final_selected,
        final_ranking_energy,
        representative[..., 3],
        solver_params=solver_params,
    )
    best_colors = _select_colors(candidate_colors, hardened_selected)
    return _finalize_output_rgba(best_colors, use_opaque, use_transparent), loss_history


def optimize_uv_field(
    rgba: np.ndarray,
    inference: InferenceResult,
    analysis: SourceAnalysis,
    steps: int,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
) -> SolverArtifacts:
    torch, F = _require_torch()
    device = _resolve_device(torch, device)
    solver_params = solver_params or SolverHyperParams()
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    source = premultiply(rgba)
    height, width = source.shape[:2]
    cell_x = width / max(1, inference.target_width)
    cell_y = height / max(1, inference.target_height)
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
    offsets_t = torch.from_numpy(
        _make_patch_offsets(height=height, width=width, target_height=inference.target_height, target_width=inference.target_width)
    ).to(device=device, dtype=torch.float32)
    guide_small = F.interpolate(edge_t, size=(inference.target_height, inference.target_width), mode="bilinear", align_corners=True)
    source_lattice_reference = build_source_lattice_reference(
        rgba,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
        alpha_threshold=solver_params.alpha_transparent_threshold,
    )
    initial_patches = _sample_cell_patches(F, source_t, uv0_t, offsets_t)
    initial_representative_t, _ = _representative_colors(initial_patches, solver_params)
    initial_representative_t = initial_representative_t.detach()
    initial_source_reference_t = torch.from_numpy(premultiply(source_lattice_reference.sharp_rgba)[None, ...]).to(
        device=device,
        dtype=torch.float32,
    )
    source_reliability_t = torch.from_numpy(_build_source_reliability(source_lattice_reference)[None, ...]).to(
        device=device,
        dtype=torch.float32,
    )
    source_delta_x_t = (
        torch.from_numpy(source_lattice_reference.delta_x[None, ...]).to(device=device, dtype=torch.float32)
        if source_lattice_reference.delta_x is not None
        else None
    )
    source_delta_y_t = (
        torch.from_numpy(source_lattice_reference.delta_y[None, ...]).to(device=device, dtype=torch.float32)
        if source_lattice_reference.delta_y is not None
        else None
    )
    source_delta_diag_t = (
        torch.from_numpy(source_lattice_reference.delta_diag[None, ...]).to(device=device, dtype=torch.float32)
        if source_lattice_reference.delta_diag is not None
        else None
    )
    source_delta_anti_t = (
        torch.from_numpy(source_lattice_reference.delta_anti[None, ...]).to(device=device, dtype=torch.float32)
        if source_lattice_reference.delta_anti is not None
        else None
    )
    snap_rgba = _snap_output_to_source_pixels(
        torch,
        source_t,
        uv0_t,
        initial_representative_t,
        initial_source_reference_t,
        source_reliability_t,
        solver_params,
        cell_x=cell_x,
        cell_y=cell_y,
    )
    snap_t = torch.from_numpy(premultiply(snap_rgba).transpose(2, 0, 1)[None, ...]).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        target_rgba, loss_history = _discrete_refine_output(
            torch,
            F,
            source_t,
            uv0_t,
            initial_representative_t,
            initial_source_reference_t,
            source_reliability_t,
            snap_t,
            source_delta_x_t,
            source_delta_y_t,
            source_delta_diag_t,
            source_delta_anti_t,
            solver_params,
            cell_x=cell_x,
            cell_y=cell_y,
            iterations=steps,
        )
        initial_rgba = snap_rgba
        uv_np = uv0_t.detach().cpu().numpy()[0]
        guidance = guide_small[0, 0].detach().cpu().numpy()

    return SolverArtifacts(
        target_rgba=target_rgba,
        uv_field=uv_np,
        guidance_strength=guidance.astype(np.float32),
        initial_rgba=initial_rgba,
        loss_history=loss_history,
    )
