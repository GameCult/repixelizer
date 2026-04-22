from __future__ import annotations

import numpy as np

from .io import premultiply, unpremultiply
from .params import SolverHyperParams
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


def _source_boundary_deltas(F, source_t, uv, axis: str, probe_scale: float = 0.22):
    if axis == "x":
        if uv.shape[2] < 2:
            return None
        delta = uv[:, :, 1:, :] - uv[:, :, :-1, :]
        midpoint = (uv[:, :, 1:, :] + uv[:, :, :-1, :]) * 0.5
    else:
        if uv.shape[1] < 2:
            return None
        delta = uv[:, 1:, :, :] - uv[:, :-1, :, :]
        midpoint = (uv[:, 1:, :, :] + uv[:, :-1, :, :]) * 0.5
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
    else:
        rep_delta = representative[:, 1:, :, :] - representative[:, :-1, :, :]
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


def _relax_candidate_selection(
    torch,
    candidate_colors,
    base_energy,
    anchor,
    desired_delta_x,
    desired_delta_y,
    desired_delta_diag,
    desired_delta_anti,
    solver_params: SolverHyperParams,
    *,
    iterations: int,
):
    if iterations <= 0:
        return torch.argmin(base_energy, dim=-1), []

    start_temp = max(1e-3, solver_params.relax_start_temperature)
    end_temp = max(1e-3, min(start_temp, solver_params.relax_end_temperature))
    damping = float(np.clip(solver_params.relax_damping, 0.0, 0.95))

    probs = (-base_energy / start_temp).softmax(dim=2)
    loss_history: list[float] = []
    final_energy = base_energy

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
            orthogonal_weight=solver_params.refine_orthogonal_weight,
            diagonal_weight=solver_params.refine_diagonal_weight,
        )
        if solver_params.refine_motif_weight > 0.0:
            motif_energy = _motif_candidate_energy(torch, candidate_colors, context_colors, anchor)
            final_energy = final_energy + motif_energy * solver_params.refine_motif_weight

        alpha = 1.0 if iterations <= 1 else step / float(iterations - 1)
        temperature = start_temp + (end_temp - start_temp) * alpha
        updated_probs = (-final_energy / max(temperature, 1e-3)).softmax(dim=2)
        if damping > 0.0:
            probs = updated_probs * (1.0 - damping) + probs * damping
            probs = probs / probs.sum(dim=2, keepdim=True).clamp_min(1e-8)
        else:
            probs = updated_probs
        loss_history.append(float((final_energy * probs).sum(dim=2).mean().detach().cpu().item()))

    return torch.argmax(probs, dim=2), loss_history


def _snap_output_to_source_pixels(
    torch,
    source_t,
    uv_t,
    representative_t,
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

    base_energy = (candidate_colors - representative[..., None, :]).abs().mean(dim=-1) * solver_params.snap_base_match_weight
    representative_delta_x = representative[:, 1:, :] - representative[:, :-1, :] if output_width > 1 else None
    representative_delta_y = representative[1:, :, :] - representative[:-1, :, :] if output_height > 1 else None
    representative_delta_diag = (
        representative[1:, 1:, :] - representative[:-1, :-1, :] if output_height > 1 and output_width > 1 else None
    )
    representative_delta_anti = (
        representative[1:, :-1, :] - representative[:-1, 1:, :] if output_height > 1 and output_width > 1 else None
    )

    selected = torch.argmin(base_energy, dim=-1)
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
            left_desired[:, 1:, :] = representative_delta_x
            left_error = ((candidate_colors - left_selected[..., None, :]) - left_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + left_error * left_mask * solver_params.snap_neighbor_weight

            right_selected = torch.zeros_like(selected_colors)
            right_selected[:, :-1, :] = selected_colors[:, 1:, :]
            right_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            right_mask[:, :-1, :] = 1.0
            right_desired = torch.zeros_like(selected_colors)
            right_desired[:, :-1, :] = representative_delta_x
            right_error = ((right_selected[..., None, :] - candidate_colors) - right_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + right_error * right_mask * solver_params.snap_neighbor_weight
        if output_height > 1:
            up_selected = torch.zeros_like(selected_colors)
            up_selected[1:, :, :] = selected_colors[:-1, :, :]
            up_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            up_mask[1:, :, :] = 1.0
            up_desired = torch.zeros_like(selected_colors)
            up_desired[1:, :, :] = representative_delta_y
            up_error = ((candidate_colors - up_selected[..., None, :]) - up_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + up_error * up_mask * solver_params.snap_neighbor_weight

            down_selected = torch.zeros_like(selected_colors)
            down_selected[:-1, :, :] = selected_colors[1:, :, :]
            down_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            down_mask[:-1, :, :] = 1.0
            down_desired = torch.zeros_like(selected_colors)
            down_desired[:-1, :, :] = representative_delta_y
            down_error = ((down_selected[..., None, :] - candidate_colors) - down_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + down_error * down_mask * solver_params.snap_neighbor_weight
        if output_height > 1 and output_width > 1:
            diag_selected = torch.zeros_like(selected_colors)
            diag_selected[1:, 1:, :] = selected_colors[:-1, :-1, :]
            diag_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            diag_mask[1:, 1:, :] = 1.0
            diag_desired = torch.zeros_like(selected_colors)
            diag_desired[1:, 1:, :] = representative_delta_diag
            diag_error = ((candidate_colors - diag_selected[..., None, :]) - diag_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + diag_error * diag_mask * solver_params.snap_diagonal_weight

            anti_selected = torch.zeros_like(selected_colors)
            anti_selected[1:, :-1, :] = selected_colors[:-1, 1:, :]
            anti_mask = torch.zeros((output_height, output_width, 1), device=uv.device, dtype=uv.dtype)
            anti_mask[1:, :-1, :] = 1.0
            anti_desired = torch.zeros_like(selected_colors)
            anti_desired[1:, :-1, :] = representative_delta_anti
            anti_error = ((candidate_colors - anti_selected[..., None, :]) - anti_desired[..., None, :]).abs().mean(dim=-1)
            energy = energy + anti_error * anti_mask * solver_params.snap_diagonal_weight
        selected = torch.argmin(energy, dim=-1)

    hardened_selected, use_opaque, use_transparent = _harden_binary_alpha_selection(
        torch,
        candidate_colors,
        selected,
        base_energy,
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
    F,
    source_t,
    uv_t,
    candidate_t,
    representative_t,
    anchor_t,
    solver_params: SolverHyperParams,
):
    boundary = _boundary_pattern_loss(F, source_t, uv_t, candidate_t, "x", solver_params) + _boundary_pattern_loss(
        F,
        source_t,
        uv_t,
        candidate_t,
        "y",
        solver_params,
    )
    anchor_adj = _adjacency_pattern_loss(candidate_t, anchor_t)
    anchor_motif = _motif_pattern_loss(candidate_t, anchor_t)
    representative_match = (candidate_t - representative_t).abs().mean()
    return (
        boundary * solver_params.structure_boundary_weight
        + anchor_adj * solver_params.structure_anchor_adjacency_weight
        + anchor_motif * solver_params.structure_anchor_motif_weight
        + representative_match * solver_params.structure_representative_weight
    )


def _discrete_refine_output(
    torch,
    F,
    source_t,
    uv_t,
    representative_t,
    anchor_t,
    source_delta_x_t,
    source_delta_y_t,
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
    anchor = anchor_t[0].permute(1, 2, 0)
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
    anchor_energy = (candidate_colors - anchor[..., None, :]).abs().mean(dim=-1)
    rep_energy = (candidate_colors - representative[..., None, :]).abs().mean(dim=-1)
    alpha_energy = (candidate_colors[..., 3] - anchor[..., None, 3]).abs()
    distance_energy = ((offset_x / max(cell_x, 1e-4)) ** 2 + (offset_y / max(cell_y, 1e-4)) ** 2).reshape(1, 1, -1)
    base_energy = (
        anchor_energy * solver_params.refine_anchor_weight
        + rep_energy * solver_params.refine_representative_weight
        + alpha_energy * solver_params.refine_alpha_weight
        + distance_energy * solver_params.refine_distance_weight
    )

    anchor_delta_x = anchor[:, 1:, :] - anchor[:, :-1, :] if output_width > 1 else None
    anchor_delta_y = anchor[1:, :, :] - anchor[:-1, :, :] if output_height > 1 else None
    anchor_delta_diag = anchor[1:, 1:, :] - anchor[:-1, :-1, :] if output_height > 1 and output_width > 1 else None
    anchor_delta_anti = anchor[1:, :-1, :] - anchor[:-1, 1:, :] if output_height > 1 and output_width > 1 else None
    source_delta_x = source_delta_x_t[0] if source_delta_x_t is not None else None
    source_delta_y = source_delta_y_t[0] if source_delta_y_t is not None else None

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
    desired_delta_diag = anchor_delta_diag
    desired_delta_anti = anchor_delta_anti

    passes = max(0, iterations)
    relax_iterations = min(max(0, solver_params.relax_iterations), passes) if passes > 0 else 0
    selected, relax_history = _relax_candidate_selection(
        torch,
        candidate_colors,
        anchor_energy + distance_energy * 0.05,
        anchor,
        desired_delta_x,
        desired_delta_y,
        desired_delta_diag,
        desired_delta_anti,
        solver_params,
        iterations=relax_iterations,
    )
    best_selected = selected.clone()
    best_score = float("inf")
    loss_history: list[float] = list(relax_history)
    anchor_hw = anchor_t.permute(0, 2, 3, 1)

    for step in range(passes + 1):
        selected_colors = _select_colors(candidate_colors, selected)
        energy = base_energy.clone()
        energy = energy + _pairwise_candidate_energy(
            candidate_colors,
            selected_colors,
            desired_delta_x,
            desired_delta_y,
            desired_delta_diag,
            desired_delta_anti,
            orthogonal_weight=solver_params.refine_orthogonal_weight,
            diagonal_weight=solver_params.refine_diagonal_weight,
        )
        if solver_params.refine_motif_weight > 0.0:
            motif_energy = _motif_candidate_energy(torch, candidate_colors, selected_colors, anchor)
            energy = energy + motif_energy * solver_params.refine_motif_weight

        candidate_t = selected_colors[None, ...]
        score = float(
            _structure_score(
                F,
                source_t,
                uv_t,
                candidate_t,
                representative_t,
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

    hardened_selected, use_opaque, use_transparent = _harden_binary_alpha_selection(
        torch,
        candidate_colors,
        best_selected,
        base_energy,
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
    initial_patches = _sample_cell_patches(F, source_t, uv0_t, offsets_t)
    initial_representative_t, _ = _representative_colors(initial_patches, solver_params)
    initial_representative_t = initial_representative_t.detach()
    source_delta_x_t = _source_boundary_deltas(F, source_t, uv0_t, axis="x")
    source_delta_y_t = _source_boundary_deltas(F, source_t, uv0_t, axis="y")
    snap_rgba = _snap_output_to_source_pixels(
        torch,
        source_t,
        uv0_t,
        initial_representative_t,
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
            snap_t,
            source_delta_x_t,
            source_delta_y_t,
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
