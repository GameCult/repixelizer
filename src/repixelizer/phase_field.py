from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import premultiply
from .observe import PipelineObserver, check_observer_cancelled, emit_observer, observer_attribute
from .types import InferenceResult, PhaseFieldSourceAnalysis, SolverArtifacts
from .params import SolverHyperParams


@dataclass(slots=True)
class _PhaseFieldPrep:
    source_t: object
    edge_t: object
    uv0_px_t: object
    uv0_norm: np.ndarray
    base_x_t: object
    base_y_t: object
    guidance: np.ndarray
    cell_x: float
    cell_y: float
    patch_offsets_t: object
    feature_t: object
    width: int
    height: int

def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - exercised only when torch missing
        raise RuntimeError(
            "PyTorch is required for the phase-field optimization stage. Install project dependencies first."
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


def _make_regular_uv_px(
    *,
    height: int,
    width: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    cell_x = width / target_width
    cell_y = height / target_height
    xs = (np.arange(target_width, dtype=np.float32) + 0.5) * cell_x - 0.5
    ys = (np.arange(target_height, dtype=np.float32) + 0.5) * cell_y - 0.5
    xs = np.clip(xs, 0.0, max(0.0, width - 1))
    ys = np.clip(ys, 0.0, max(0.0, height - 1))
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x, grid_y], axis=-1).astype(np.float32)


def _pixel_to_normalized(coords_px, *, width: int, height: int):
    coords = coords_px.clone()
    coords[..., 0] = (coords[..., 0] / max(1.0, float(width - 1))) * 2.0 - 1.0
    coords[..., 1] = (coords[..., 1] / max(1.0, float(height - 1))) * 2.0 - 1.0
    return coords


def _sample_rgba(F, source_t, coords_px, *, width: int, height: int):
    grid = _pixel_to_normalized(coords_px, width=width, height=height).clamp(-1.0, 1.0)
    sampled = F.grid_sample(source_t, grid, align_corners=True, mode="bilinear", padding_mode="border")
    return sampled.permute(0, 2, 3, 1)


def _sample_scalar(F, scalar_t, coords_px, *, width: int, height: int):
    grid = _pixel_to_normalized(coords_px, width=width, height=height).clamp(-1.0, 1.0)
    sampled = F.grid_sample(scalar_t, grid, align_corners=True, mode="bilinear", padding_mode="border")
    return sampled[:, 0]


def _sample_patch_rgba(F, source_t, coords_px, offsets_t, *, width: int, height: int):
    patch_px = coords_px[:, :, :, None, :] + offsets_t[None, None, None, :, :]
    patch_px = patch_px.new_empty((*patch_px.shape[:-1], 2))
    patch_px[..., 0] = (coords_px[:, :, :, None, 0] + offsets_t[None, None, None, :, 0]).clamp(
        0.0, max(0.0, float(width - 1))
    )
    patch_px[..., 1] = (coords_px[:, :, :, None, 1] + offsets_t[None, None, None, :, 1]).clamp(
        0.0, max(0.0, float(height - 1))
    )
    patch_grid = _pixel_to_normalized(patch_px, width=width, height=height).clamp(-1.0, 1.0)
    batch, out_h, out_w, samples, _ = patch_grid.shape
    flattened = patch_grid.reshape(batch, out_h, out_w * samples, 2)
    sampled = F.grid_sample(source_t, flattened, align_corners=True, mode="bilinear", padding_mode="border")
    sampled = sampled.permute(0, 2, 3, 1)
    return sampled.reshape(batch, out_h, out_w, samples, sampled.shape[-1])


def _sample_patch_scalar(F, scalar_t, coords_px, offsets_t, *, width: int, height: int):
    patch_px = coords_px[:, :, :, None, :] + offsets_t[None, None, None, :, :]
    patch_px = patch_px.new_empty((*patch_px.shape[:-1], 2))
    patch_px[..., 0] = (coords_px[:, :, :, None, 0] + offsets_t[None, None, None, :, 0]).clamp(
        0.0, max(0.0, float(width - 1))
    )
    patch_px[..., 1] = (coords_px[:, :, :, None, 1] + offsets_t[None, None, None, :, 1]).clamp(
        0.0, max(0.0, float(height - 1))
    )
    patch_grid = _pixel_to_normalized(patch_px, width=width, height=height).clamp(-1.0, 1.0)
    batch, out_h, out_w, samples, _ = patch_grid.shape
    flattened = patch_grid.reshape(batch, out_h, out_w * samples, 2)
    sampled = F.grid_sample(scalar_t, flattened, align_corners=True, mode="bilinear", padding_mode="border")
    sampled = sampled[:, 0]
    return sampled.reshape(batch, out_h, out_w, samples)

def _displacement_diagnostics(uv_field: np.ndarray, selected_x, selected_y, *, width: int, height: int) -> dict[str, np.ndarray | float]:
    uv_x = (uv_field[..., 0] + 1.0) * 0.5 * max(1.0, float(width - 1))
    uv_y = (uv_field[..., 1] + 1.0) * 0.5 * max(1.0, float(height - 1))
    displacement_x = selected_x.astype(np.float32) - uv_x.astype(np.float32)
    displacement_y = selected_y.astype(np.float32) - uv_y.astype(np.float32)
    magnitude = np.sqrt(np.square(displacement_x) + np.square(displacement_y)).astype(np.float32)

    orthogonal_jitter_terms: list[np.ndarray] = []
    local_residual_terms: list[np.ndarray] = []
    for field in (displacement_x, displacement_y):
        if field.shape[1] > 1:
            orthogonal_jitter_terms.append(np.abs(field[:, 1:] - field[:, :-1]))
        if field.shape[0] > 1:
            orthogonal_jitter_terms.append(np.abs(field[1:, :] - field[:-1, :]))
        blurred = field.copy()
        if field.shape[1] > 2:
            blurred[:, 1:-1] = (field[:, :-2] + field[:, 1:-1] + field[:, 2:]) / 3.0
        if field.shape[0] > 2:
            blurred[1:-1, :] = (blurred[:-2, :] + blurred[1:-1, :] + blurred[2:, :]) / 3.0
        local_residual_terms.append(np.abs(field - blurred))

    orthogonal_jitter = (
        float(np.mean(np.concatenate([term.reshape(-1) for term in orthogonal_jitter_terms])))
        if orthogonal_jitter_terms
        else 0.0
    )
    local_residual = (
        float(np.mean(np.concatenate([term.reshape(-1) for term in local_residual_terms])))
        if local_residual_terms
        else 0.0
    )
    rounded_offsets = np.stack([np.rint(displacement_x), np.rint(displacement_y)], axis=-1).reshape(-1, 2)
    if rounded_offsets.size == 0:
        dominant_offset_ratio = 1.0
    else:
        _, counts = np.unique(rounded_offsets, axis=0, return_counts=True)
        dominant_offset_ratio = float(np.max(counts) / max(1, rounded_offsets.shape[0]))
    return {
        "displacement_x": displacement_x,
        "displacement_y": displacement_y,
        "mean_magnitude_px": float(np.mean(magnitude)),
        "orthogonal_jitter_px": orthogonal_jitter,
        "local_residual_px": local_residual,
        "dominant_offset_ratio": dominant_offset_ratio,
    }


def _project_displacements_in_place(
    torch,
    disp_t,
    base_x_t,
    base_y_t,
    *,
    min_dx: float,
    min_dy: float,
    max_dx: float,
    max_dy: float,
    width: int,
    height: int,
) -> None:
    with torch.no_grad():
        pos_x = (base_x_t + disp_t[..., 0]).clamp(0.0, max(0.0, float(width - 1)))
        pos_y = (base_y_t + disp_t[..., 1]).clamp(0.0, max(0.0, float(height - 1)))

        for index in range(1, pos_x.shape[2]):
            pos_x[:, :, index] = torch.maximum(pos_x[:, :, index], pos_x[:, :, index - 1] + min_dx)
        for index in range(pos_x.shape[2] - 2, -1, -1):
            pos_x[:, :, index] = torch.minimum(pos_x[:, :, index], pos_x[:, :, index + 1] - min_dx)
        for index in range(1, pos_y.shape[1]):
            pos_y[:, index, :] = torch.maximum(pos_y[:, index, :], pos_y[:, index - 1, :] + min_dy)
        for index in range(pos_y.shape[1] - 2, -1, -1):
            pos_y[:, index, :] = torch.minimum(pos_y[:, index, :], pos_y[:, index + 1, :] - min_dy)

        pos_x = pos_x.clamp(0.0, max(0.0, float(width - 1)))
        pos_y = pos_y.clamp(0.0, max(0.0, float(height - 1)))
        disp_t[..., 0].copy_((pos_x - base_x_t).clamp(-max_dx, max_dx))
        disp_t[..., 1].copy_((pos_y - base_y_t).clamp(-max_dy, max_dy))


def _prepare_phase_field(
    torch,
    F,
    rgba: np.ndarray,
    inference: InferenceResult,
    analysis: PhaseFieldSourceAnalysis,
    solver_params: SolverHyperParams,
    *,
    device: str,
) -> _PhaseFieldPrep:
    source = premultiply(rgba)
    height, width = source.shape[:2]
    cell_x = width / max(1, inference.target_width)
    cell_y = height / max(1, inference.target_height)
    source_t = torch.from_numpy(source.transpose(2, 0, 1)[None, ...]).to(device=device, dtype=torch.float32)
    edge_t = torch.from_numpy(analysis.edge_map[None, None, ...]).to(device=device, dtype=torch.float32)
    luma = (
        source[..., 0] * np.float32(0.2126)
        + source[..., 1] * np.float32(0.7152)
        + source[..., 2] * np.float32(0.0722)
    ).astype(np.float32)
    padded_luma = np.pad(luma, 1, mode="edge")
    local_mean = (
        padded_luma[:-2, :-2]
        + padded_luma[:-2, 1:-1]
        + padded_luma[:-2, 2:]
        + padded_luma[1:-1, :-2]
        + padded_luma[1:-1, 1:-1]
        + padded_luma[1:-1, 2:]
        + padded_luma[2:, :-2]
        + padded_luma[2:, 1:-1]
        + padded_luma[2:, 2:]
    ) / np.float32(9.0)
    feature = np.abs(luma - local_mean).astype(np.float32)
    feature_t = torch.from_numpy(feature[None, None, ...]).to(device=device, dtype=torch.float32)
    uv0_px = _make_regular_uv_px(
        height=height,
        width=width,
        target_height=inference.target_height,
        target_width=inference.target_width,
    )
    uv0_px_t = torch.from_numpy(uv0_px[None, ...]).to(device=device, dtype=torch.float32)
    base_x_t = uv0_px_t[..., 0]
    base_y_t = uv0_px_t[..., 1]
    uv0_norm = uv0_px.copy()
    uv0_norm[..., 0] = (uv0_norm[..., 0] / max(1.0, float(width - 1))) * 2.0 - 1.0
    uv0_norm[..., 1] = (uv0_norm[..., 1] / max(1.0, float(height - 1))) * 2.0 - 1.0
    offsets = np.asarray(
        [
            [0.0, 0.0],
            [-solver_params.phase_field_patch_extent * cell_x, 0.0],
            [solver_params.phase_field_patch_extent * cell_x, 0.0],
            [0.0, -solver_params.phase_field_patch_extent * cell_y],
            [0.0, solver_params.phase_field_patch_extent * cell_y],
            [-solver_params.phase_field_patch_extent * cell_x, -solver_params.phase_field_patch_extent * cell_y],
            [solver_params.phase_field_patch_extent * cell_x, -solver_params.phase_field_patch_extent * cell_y],
            [-solver_params.phase_field_patch_extent * cell_x, solver_params.phase_field_patch_extent * cell_y],
            [solver_params.phase_field_patch_extent * cell_x, solver_params.phase_field_patch_extent * cell_y],
        ],
        dtype=np.float32,
    )
    patch_offsets_t = torch.from_numpy(offsets).to(device=device, dtype=torch.float32)
    guide_small = F.interpolate(
        edge_t,
        size=(inference.target_height, inference.target_width),
        mode="bilinear",
        align_corners=True,
    )
    return _PhaseFieldPrep(
        source_t=source_t,
        edge_t=edge_t,
        uv0_px_t=uv0_px_t,
        uv0_norm=uv0_norm,
        base_x_t=base_x_t,
        base_y_t=base_y_t,
        guidance=guide_small[0, 0].detach().cpu().numpy().astype(np.float32),
        cell_x=cell_x,
        cell_y=cell_y,
        patch_offsets_t=patch_offsets_t,
        feature_t=feature_t,
        width=width,
        height=height,
    )


def _phase_field_loss(torch, F, prep: _PhaseFieldPrep, disp_t, solver_params: SolverHyperParams):
    pos_px = prep.uv0_px_t + disp_t
    sampled_rgba = _sample_rgba(F, prep.source_t, pos_px, width=prep.width, height=prep.height)
    patch_rgba = _sample_patch_rgba(
        F,
        prep.source_t,
        pos_px,
        prep.patch_offsets_t,
        width=prep.width,
        height=prep.height,
    )
    patch_edge = _sample_patch_scalar(
        F,
        prep.edge_t,
        pos_px,
        prep.patch_offsets_t,
        width=prep.width,
        height=prep.height,
    )

    center_rgba = patch_rgba[..., 0, :]
    neighbor_rgba = patch_rgba[..., 1:, :]
    local_coherence = (neighbor_rgba - center_rgba[..., None, :]).abs().mean()
    local_edge = patch_edge.mean()

    disp_norm = disp_t.clone()
    disp_norm[..., 0] = disp_norm[..., 0] / max(prep.cell_x, 1e-4)
    disp_norm[..., 1] = disp_norm[..., 1] / max(prep.cell_y, 1e-4)
    pos_mid_x = (pos_px[:, :, 1:, :] + pos_px[:, :, :-1, :]) * 0.5 if pos_px.shape[2] > 1 else None
    pos_mid_y = (pos_px[:, 1:, :, :] + pos_px[:, :-1, :, :]) * 0.5 if pos_px.shape[1] > 1 else None

    smoothness = sampled_rgba.new_tensor(0.0)
    if pos_mid_x is not None:
        edge_x = _sample_scalar(F, prep.edge_t, pos_mid_x, width=prep.width, height=prep.height)
        weight_x = torch.exp(-solver_params.phase_field_edge_gate_strength * edge_x)
        delta_x = disp_norm[:, :, 1:, :] - disp_norm[:, :, :-1, :]
        smoothness = smoothness + (weight_x * torch.sqrt(delta_x.square().sum(dim=-1) + 1e-6)).mean()
    if pos_mid_y is not None:
        edge_y = _sample_scalar(F, prep.edge_t, pos_mid_y, width=prep.width, height=prep.height)
        weight_y = torch.exp(-solver_params.phase_field_edge_gate_strength * edge_y)
        delta_y = disp_norm[:, 1:, :, :] - disp_norm[:, :-1, :, :]
        smoothness = smoothness + (weight_y * torch.sqrt(delta_y.square().sum(dim=-1) + 1e-6)).mean()

    grid_alignment = sampled_rgba.new_tensor(0.0)
    if pos_px.shape[2] > 1:
        step_x = (pos_px[:, :, 1:, 0] - pos_px[:, :, :-1, 0]) / max(prep.cell_x, 1e-4)
        row_y_delta = (pos_px[:, :, 1:, 1] - pos_px[:, :, :-1, 1]) / max(prep.cell_y, 1e-4)
        grid_alignment = grid_alignment + (step_x - 1.0).square().mean() + row_y_delta.square().mean()
    if pos_px.shape[1] > 1:
        step_y = (pos_px[:, 1:, :, 1] - pos_px[:, :-1, :, 1]) / max(prep.cell_y, 1e-4)
        col_x_delta = (pos_px[:, 1:, :, 0] - pos_px[:, :-1, :, 0]) / max(prep.cell_x, 1e-4)
        grid_alignment = grid_alignment + (step_y - 1.0).square().mean() + col_x_delta.square().mean()

    collapse = sampled_rgba.new_tensor(0.0)
    min_dx = solver_params.phase_field_min_spacing_ratio * prep.cell_x
    min_dy = solver_params.phase_field_min_spacing_ratio * prep.cell_y
    if pos_px.shape[2] > 1:
        step_x = pos_px[:, :, 1:, 0] - pos_px[:, :, :-1, 0]
        collapse = collapse + torch.relu(min_dx - step_x).square().mean()
    if pos_px.shape[1] > 1:
        step_y = pos_px[:, 1:, :, 1] - pos_px[:, :-1, :, 1]
        collapse = collapse + torch.relu(min_dy - step_y).square().mean()

    magnitude = (
        (disp_t[..., 0] / max(prep.cell_x, 1e-4)).square()
        + (disp_t[..., 1] / max(prep.cell_y, 1e-4)).square()
    ).mean()

    loss = (
        local_coherence * solver_params.phase_field_data_coherence_weight
        + local_edge * solver_params.phase_field_data_edge_weight
        + smoothness * solver_params.phase_field_smoothness_weight
        + grid_alignment * solver_params.phase_field_grid_alignment_weight
        + collapse * solver_params.phase_field_collapse_weight
        + magnitude * solver_params.phase_field_magnitude_weight
    )
    terms = {
        "local_coherence": local_coherence.detach(),
        "local_edge": local_edge.detach(),
        "smoothness": smoothness.detach(),
        "grid_alignment": grid_alignment.detach(),
        "collapse": collapse.detach(),
        "magnitude": magnitude.detach(),
    }
    return loss, sampled_rgba, terms


def _local_candidate_displacement_step_in_place(
    torch,
    F,
    prep: _PhaseFieldPrep,
    disp_t,
    solver_params: SolverHyperParams,
    *,
    min_dx: float,
    min_dy: float,
    max_dx: float,
    max_dy: float,
) -> None:
    radius_ratio = float(solver_params.phase_field_local_search_radius_ratio)
    blend = float(solver_params.phase_field_local_search_blend)
    if radius_ratio <= 0.0 or blend <= 0.0:
        return

    with torch.no_grad():
        device = disp_t.device
        steps = torch.linspace(-1.0, 1.0, 5, device=device, dtype=disp_t.dtype)
        offset_y, offset_x = torch.meshgrid(steps, steps, indexing="ij")
        offsets_t = torch.stack(
            [
                offset_x.reshape(-1) * (radius_ratio * prep.cell_x),
                offset_y.reshape(-1) * (radius_ratio * prep.cell_y),
            ],
            dim=-1,
        )

        pos_px = prep.uv0_px_t + disp_t
        candidate_pos = pos_px[:, :, :, None, :] + offsets_t[None, None, None, :, :]
        candidate_pos = candidate_pos.clone()
        candidate_pos[..., 0] = candidate_pos[..., 0].clamp(0.0, max(0.0, float(prep.width - 1)))
        candidate_pos[..., 1] = candidate_pos[..., 1].clamp(0.0, max(0.0, float(prep.height - 1)))

        batch, out_h, out_w, candidates, _ = candidate_pos.shape
        flat_pos = candidate_pos.reshape(batch, out_h, out_w * candidates, 2)
        patch_rgba = _sample_patch_rgba(
            F,
            prep.source_t,
            flat_pos,
            prep.patch_offsets_t,
            width=prep.width,
            height=prep.height,
        ).reshape(batch, out_h, out_w, candidates, -1, 4)
        patch_edge = _sample_patch_scalar(
            F,
            prep.edge_t,
            flat_pos,
            prep.patch_offsets_t,
            width=prep.width,
            height=prep.height,
        ).reshape(batch, out_h, out_w, candidates, -1)
        candidate_feature = _sample_scalar(
            F,
            prep.feature_t,
            flat_pos,
            width=prep.width,
            height=prep.height,
        ).reshape(batch, out_h, out_w, candidates)

        center_rgba = patch_rgba[..., 0, :]
        neighbor_rgba = patch_rgba[..., 1:, :]
        coherence_score = (neighbor_rgba - center_rgba[..., None, :]).abs().mean(dim=(-1, -2))
        edge_score = patch_edge.mean(dim=-1)
        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=disp_t.dtype)
        center_luma = (center_rgba[..., :3] * luma_weights).sum(dim=-1)
        local_luma_mean = center_luma.mean(dim=-1, keepdim=True)
        feature_score = (center_luma - local_luma_mean).abs() + candidate_feature
        current_pos = pos_px.detach()
        grid_score = torch.zeros_like(coherence_score)
        grid_neighbors = torch.zeros_like(coherence_score)
        if out_w > 1:
            left_x = torch.empty_like(current_pos[..., 0])
            left_y = torch.empty_like(current_pos[..., 1])
            right_x = torch.empty_like(current_pos[..., 0])
            right_y = torch.empty_like(current_pos[..., 1])
            left_x[:, :, 0] = current_pos[:, :, 0, 0] - prep.cell_x
            left_x[:, :, 1:] = current_pos[:, :, :-1, 0]
            left_y[:, :, 0] = current_pos[:, :, 0, 1]
            left_y[:, :, 1:] = current_pos[:, :, :-1, 1]
            right_x[:, :, -1] = current_pos[:, :, -1, 0] + prep.cell_x
            right_x[:, :, :-1] = current_pos[:, :, 1:, 0]
            right_y[:, :, -1] = current_pos[:, :, -1, 1]
            right_y[:, :, :-1] = current_pos[:, :, 1:, 1]
            candidate_x = candidate_pos[..., 0]
            candidate_y = candidate_pos[..., 1]
            grid_score = grid_score + ((candidate_x - left_x[..., None] - prep.cell_x) / max(prep.cell_x, 1e-4)).square()
            grid_score = grid_score + ((right_x[..., None] - candidate_x - prep.cell_x) / max(prep.cell_x, 1e-4)).square()
            grid_score = grid_score + ((candidate_y - left_y[..., None]) / max(prep.cell_y, 1e-4)).square()
            grid_score = grid_score + ((candidate_y - right_y[..., None]) / max(prep.cell_y, 1e-4)).square()
            grid_neighbors = grid_neighbors + 4.0
        if out_h > 1:
            up_x = torch.empty_like(current_pos[..., 0])
            up_y = torch.empty_like(current_pos[..., 1])
            down_x = torch.empty_like(current_pos[..., 0])
            down_y = torch.empty_like(current_pos[..., 1])
            up_x[:, 0, :] = current_pos[:, 0, :, 0]
            up_x[:, 1:, :] = current_pos[:, :-1, :, 0]
            up_y[:, 0, :] = current_pos[:, 0, :, 1] - prep.cell_y
            up_y[:, 1:, :] = current_pos[:, :-1, :, 1]
            down_x[:, -1, :] = current_pos[:, -1, :, 0]
            down_x[:, :-1, :] = current_pos[:, 1:, :, 0]
            down_y[:, -1, :] = current_pos[:, -1, :, 1] + prep.cell_y
            down_y[:, :-1, :] = current_pos[:, 1:, :, 1]
            candidate_x = candidate_pos[..., 0]
            candidate_y = candidate_pos[..., 1]
            grid_score = grid_score + ((candidate_x - up_x[..., None]) / max(prep.cell_x, 1e-4)).square()
            grid_score = grid_score + ((candidate_x - down_x[..., None]) / max(prep.cell_x, 1e-4)).square()
            grid_score = grid_score + ((candidate_y - up_y[..., None] - prep.cell_y) / max(prep.cell_y, 1e-4)).square()
            grid_score = grid_score + ((down_y[..., None] - candidate_y - prep.cell_y) / max(prep.cell_y, 1e-4)).square()
            grid_neighbors = grid_neighbors + 4.0
        grid_score = grid_score / grid_neighbors.clamp_min(1.0)
        move_score = (
            (offsets_t[:, 0] / max(prep.cell_x, 1e-4)).square()
            + (offsets_t[:, 1] / max(prep.cell_y, 1e-4)).square()
        )
        score = (
            coherence_score * solver_params.phase_field_data_coherence_weight
            + edge_score * solver_params.phase_field_data_edge_weight
            + move_score[None, None, None, :] * solver_params.phase_field_local_search_move_weight
            + grid_score * solver_params.phase_field_local_search_grid_weight
            - feature_score * solver_params.phase_field_local_search_feature_weight
        )
        best_index = torch.argmin(score, dim=-1)
        gather_index = best_index[..., None, None].expand(batch, out_h, out_w, 1, 2)
        best_pos = torch.gather(candidate_pos, dim=3, index=gather_index).squeeze(3)
        target_disp = best_pos - prep.uv0_px_t
        clamped_blend = max(0.0, min(1.0, blend))
        disp_t.mul_(1.0 - clamped_blend).add_(target_disp * clamped_blend)

    _project_displacements_in_place(
        torch,
        disp_t,
        prep.base_x_t,
        prep.base_y_t,
        min_dx=min_dx,
        min_dy=min_dy,
        max_dx=max_dx,
        max_dy=max_dy,
        width=prep.width,
        height=prep.height,
    )


def _nearest_source_rgba(source_rgba: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    return source_rgba[sample_y, sample_x]


def _observer_snapshot(
    prep: _PhaseFieldPrep,
    rgba: np.ndarray,
    pos_px: np.ndarray,
) -> dict[str, np.ndarray]:
    sample_x = np.rint(pos_px[..., 0]).astype(np.int32).clip(0, prep.width - 1)
    sample_y = np.rint(pos_px[..., 1]).astype(np.int32).clip(0, prep.height - 1)
    displacement = _displacement_diagnostics(
        prep.uv0_norm,
        sample_x,
        sample_y,
        width=prep.width,
        height=prep.height,
    )
    return {
        "target_rgba": _nearest_source_rgba(rgba, sample_x, sample_y),
        "sample_x": sample_x,
        "sample_y": sample_y,
        "pos_x_px": pos_px[..., 0].astype(np.float32),
        "pos_y_px": pos_px[..., 1].astype(np.float32),
        "displacement_x": displacement["displacement_x"],
        "displacement_y": displacement["displacement_y"],
    }


def _materialize_phase_terms(terms: dict[str, object]) -> dict[str, float]:
    materialized: dict[str, float] = {}
    for key, value in terms.items():
        if hasattr(value, "detach"):
            materialized[key] = float(value.detach().cpu().item())
        else:
            materialized[key] = float(value)
    return materialized


def _materialize_loss_history(torch, values: list[object]) -> list[float]:
    if not values:
        return []
    normalized = []
    for value in values:
        if hasattr(value, "detach"):
            normalized.append(value.detach().reshape(()))
        else:
            normalized.append(torch.tensor(float(value)))
    stacked = torch.stack(normalized)
    return [float(item) for item in stacked.cpu().tolist()]


def _sample_bilinear_np(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = field.shape[:2]
    x = np.clip(x, 0.0, max(0.0, float(width - 1)))
    y = np.clip(y, 0.0, max(0.0, float(height - 1)))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    top = field[y0, x0] * (1.0 - wx) + field[y0, x1] * wx
    bottom = field[y1, x0] * (1.0 - wx) + field[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _resize_scalar_bilinear_np(field: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = field.shape[:2]
    xs = (np.arange(width, dtype=np.float32) + 0.5) * (src_w / max(1, width)) - 0.5
    ys = (np.arange(height, dtype=np.float32) + 0.5) * (src_h / max(1, height)) - 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    return _sample_bilinear_np(field, grid_x, grid_y).astype(np.float32)


def _feature_map_from_rgba(rgba: np.ndarray) -> np.ndarray:
    source = premultiply(rgba)
    luma = (
        source[..., 0] * np.float32(0.2126)
        + source[..., 1] * np.float32(0.7152)
        + source[..., 2] * np.float32(0.0722)
    ).astype(np.float32)
    padded = np.pad(luma, 1, mode="edge")
    local_mean = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / np.float32(9.0)
    return np.abs(luma - local_mean).astype(np.float32)


def _box_mean_np(field: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return field.astype(np.float32, copy=True)
    height, width = field.shape[:2]
    padded = np.pad(field.astype(np.float32), radius, mode="edge")
    integral = np.pad(
        padded,
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0.0,
    ).cumsum(axis=0).cumsum(axis=1)
    size = radius * 2 + 1
    total = (
        integral[size : size + height, size : size + width]
        - integral[:height, size : size + width]
        - integral[size : size + height, :width]
        + integral[:height, :width]
    )
    return (total / np.float32(size * size)).astype(np.float32)


def _local_color_coherence_map_from_rgba(
    rgba: np.ndarray,
    *,
    cell_x: float,
    cell_y: float,
    radius_ratio: float,
) -> np.ndarray:
    radius = max(1, int(round(min(cell_x, cell_y) * max(0.0, float(radius_ratio)))))
    source = premultiply(rgba).astype(np.float32)
    variance = np.zeros(source.shape[:2], dtype=np.float32)
    for channel in range(3):
        values = source[..., channel]
        mean = _box_mean_np(values, radius)
        mean_square = _box_mean_np(values * values, radius)
        variance += np.maximum(0.0, mean_square - mean * mean)
    return np.sqrt(variance / np.float32(3.0)).astype(np.float32)


def _local_luma_mean_map_from_rgba(
    rgba: np.ndarray,
    *,
    cell_x: float,
    cell_y: float,
    radius_ratio: float,
) -> np.ndarray:
    radius = max(1, int(round(min(cell_x, cell_y) * max(0.0, float(radius_ratio)))))
    source = premultiply(rgba).astype(np.float32)
    luma = (
        source[..., 0] * np.float32(0.2126)
        + source[..., 1] * np.float32(0.7152)
        + source[..., 2] * np.float32(0.0722)
    ).astype(np.float32)
    return _box_mean_np(luma, radius)


def _downsample_half_np(field: np.ndarray) -> np.ndarray:
    height, width = field.shape[:2]
    padded = np.pad(field.astype(np.float32), ((0, height % 2), (0, width % 2)), mode="edge")
    return (
        padded[0::2, 0::2]
        + padded[1::2, 0::2]
        + padded[0::2, 1::2]
        + padded[1::2, 1::2]
    ) * np.float32(0.25)


def _resize_scalar_to_shape_np(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    src_h, src_w = field.shape[:2]
    xs = (np.arange(width, dtype=np.float32) + 0.5) * (src_w / max(1, width)) - 0.5
    ys = (np.arange(height, dtype=np.float32) + 0.5) * (src_h / max(1, height)) - 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    return _sample_bilinear_np(field, grid_x, grid_y).astype(np.float32)


def _gradient_magnitude_np(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field.astype(np.float32), 1, mode="edge")
    dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * np.float32(0.5)
    dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * np.float32(0.5)
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _laplacian_abs_np(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field.astype(np.float32), 1, mode="edge")
    laplacian = (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        - padded[1:-1, 1:-1] * np.float32(4.0)
    )
    return np.abs(laplacian).astype(np.float32)


def _normalize_percentile_np(values: np.ndarray, *, low_percentile: float = 5.0, high_percentile: float = 99.0) -> np.ndarray:
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if high <= low + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - np.float32(low)) / np.float32(high - low), 0.0, 1.0).astype(np.float32)


def _source_signal_map_np(
    rgba: np.ndarray,
    edge_map: np.ndarray,
    solver_params: SolverHyperParams,
    *,
    cell_x: float,
    cell_y: float,
) -> np.ndarray:
    source = premultiply(rgba).astype(np.float32)
    luma = (
        source[..., 0] * np.float32(0.2126)
        + source[..., 1] * np.float32(0.7152)
        + source[..., 2] * np.float32(0.0722)
    ).astype(np.float32)
    full_shape = luma.shape
    point_energy = np.zeros(full_shape, dtype=np.float32)
    context_energy = np.zeros(full_shape, dtype=np.float32)
    level_field = luma
    levels = max(1, int(solver_params.phase_field_signal_pyramid_levels))
    for level in range(levels):
        gradient = _gradient_magnitude_np(level_field)
        curvature = _laplacian_abs_np(level_field)
        boundary = (
            gradient * np.float32(solver_params.phase_field_signal_boundary_weight)
            + curvature * np.float32(solver_params.phase_field_signal_curvature_weight)
        ).astype(np.float32)
        boundary = _normalize_percentile_np(boundary, low_percentile=10.0, high_percentile=98.0)
        context_radius = max(1, int(round(1.5 + level)))
        context = _box_mean_np(boundary, context_radius)
        level_weight = np.float32(1.0 / (1 << level))
        point_energy += _resize_scalar_to_shape_np(boundary, full_shape) * level_weight
        context_energy += _resize_scalar_to_shape_np(context, full_shape) * level_weight
        if min(level_field.shape[:2]) <= 8:
            break
        level_field = _downsample_half_np(level_field)

    point_energy = _normalize_percentile_np(point_energy, low_percentile=5.0, high_percentile=99.0)
    context_energy = _normalize_percentile_np(context_energy, low_percentile=5.0, high_percentile=99.0)
    quiet_here = np.exp(-point_energy * np.float32(solver_params.phase_field_signal_self_penalty)).astype(np.float32)
    context = np.power(context_energy, np.float32(solver_params.phase_field_signal_context_power)).astype(np.float32)
    signal = _normalize_percentile_np(quiet_here * context, low_percentile=5.0, high_percentile=99.0)
    return np.power(signal, np.float32(solver_params.phase_field_signal_peak_power)).astype(np.float32)


def _candidate_grid_score_np(
    pos: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    *,
    cell_x: float,
    cell_y: float,
) -> np.ndarray:
    pos_x = pos[..., 0]
    pos_y = pos[..., 1]
    score = np.zeros_like(candidate_x, dtype=np.float32)
    neighbors = np.zeros_like(candidate_x, dtype=np.float32)

    if pos.shape[1] > 1:
        left_x = np.empty_like(pos_x)
        right_x = np.empty_like(pos_x)
        left_y = np.empty_like(pos_y)
        right_y = np.empty_like(pos_y)
        left_x[:, 0] = pos_x[:, 0] - cell_x
        left_x[:, 1:] = pos_x[:, :-1]
        right_x[:, -1] = pos_x[:, -1] + cell_x
        right_x[:, :-1] = pos_x[:, 1:]
        left_y[:, 0] = pos_y[:, 0]
        left_y[:, 1:] = pos_y[:, :-1]
        right_y[:, -1] = pos_y[:, -1]
        right_y[:, :-1] = pos_y[:, 1:]
        score += np.square((candidate_x - left_x - cell_x) / max(cell_x, 1e-4))
        score += np.square((right_x - candidate_x - cell_x) / max(cell_x, 1e-4))
        score += np.square((candidate_y - left_y) / max(cell_y, 1e-4))
        score += np.square((candidate_y - right_y) / max(cell_y, 1e-4))
        neighbors += 4.0

    if pos.shape[0] > 1:
        up_x = np.empty_like(pos_x)
        down_x = np.empty_like(pos_x)
        up_y = np.empty_like(pos_y)
        down_y = np.empty_like(pos_y)
        up_x[0, :] = pos_x[0, :]
        up_x[1:, :] = pos_x[:-1, :]
        down_x[-1, :] = pos_x[-1, :]
        down_x[:-1, :] = pos_x[1:, :]
        up_y[0, :] = pos_y[0, :] - cell_y
        up_y[1:, :] = pos_y[:-1, :]
        down_y[-1, :] = pos_y[-1, :] + cell_y
        down_y[:-1, :] = pos_y[1:, :]
        score += np.square((candidate_x - up_x) / max(cell_x, 1e-4))
        score += np.square((candidate_x - down_x) / max(cell_x, 1e-4))
        score += np.square((candidate_y - up_y - cell_y) / max(cell_y, 1e-4))
        score += np.square((down_y - candidate_y - cell_y) / max(cell_y, 1e-4))
        neighbors += 4.0

    return score / np.maximum(neighbors, 1.0)


def _choose_local_preferred_positions_np(
    pos: np.ndarray,
    signal_map: np.ndarray,
    solver_params: SolverHyperParams,
    *,
    cell_x: float,
    cell_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    radius_x = float(solver_params.phase_field_local_search_radius_ratio) * cell_x
    radius_y = float(solver_params.phase_field_local_search_radius_ratio) * cell_y
    if radius_x <= 0.0 or radius_y <= 0.0:
        return pos.copy()

    offsets = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    best_score = np.full(pos.shape[:2], np.inf, dtype=np.float32)
    best = pos.copy()
    for oy in offsets:
        for ox in offsets:
            candidate_x = np.clip(pos[..., 0] + ox * radius_x, 0.0, max(0.0, float(width - 1)))
            candidate_y = np.clip(pos[..., 1] + oy * radius_y, 0.0, max(0.0, float(height - 1)))
            signal_score = _sample_bilinear_np(signal_map, candidate_x, candidate_y)
            move_score = np.float32(ox * ox + oy * oy) * np.float32(
                solver_params.phase_field_local_search_radius_ratio
                * solver_params.phase_field_local_search_radius_ratio
            )
            grid_score = _candidate_grid_score_np(pos, candidate_x, candidate_y, cell_x=cell_x, cell_y=cell_y)
            score = (
                move_score * np.float32(solver_params.phase_field_local_search_move_weight)
                + grid_score * np.float32(solver_params.phase_field_local_search_grid_weight)
                - signal_score * np.float32(solver_params.phase_field_signal_weight)
            )
            improved = score < best_score
            best_score = np.where(improved, score, best_score)
            best[..., 0] = np.where(improved, candidate_x, best[..., 0])
            best[..., 1] = np.where(improved, candidate_y, best[..., 1])
    return best.astype(np.float32)


def _relax_lattice_springs_np(
    pos: np.ndarray,
    solver_params: SolverHyperParams,
    *,
    cell_x: float,
    cell_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    out = pos.copy()
    spring_step = min(0.22, max(0.0, float(solver_params.phase_field_grid_alignment_weight)) * 0.06)
    if spring_step <= 0.0:
        return out
    for _ in range(3):
        force = np.zeros_like(out, dtype=np.float32)
        if out.shape[1] > 1:
            horizontal_error = out[:, 1:, :] - out[:, :-1, :]
            horizontal_error[..., 0] -= cell_x
            force[:, :-1, :] += horizontal_error
            force[:, 1:, :] -= horizontal_error
        if out.shape[0] > 1:
            vertical_error = out[1:, :, :] - out[:-1, :, :]
            vertical_error[..., 1] -= cell_y
            force[:-1, :, :] += vertical_error
            force[1:, :, :] -= vertical_error
        out += force * np.float32(spring_step)
        out[..., 0] = np.clip(out[..., 0], 0.0, max(0.0, float(width - 1)))
        out[..., 1] = np.clip(out[..., 1], 0.0, max(0.0, float(height - 1)))
    return out


def _explicit_solver_terms_np(
    pos: np.ndarray,
    signal_map: np.ndarray,
    solver_params: SolverHyperParams,
    *,
    cell_x: float,
    cell_y: float,
) -> dict[str, float]:
    signal = _sample_bilinear_np(signal_map, pos[..., 0], pos[..., 1])
    grid_alignment = 0.0
    if pos.shape[1] > 1:
        step_x = (pos[:, 1:, 0] - pos[:, :-1, 0]) / max(cell_x, 1e-4)
        row_y_delta = (pos[:, 1:, 1] - pos[:, :-1, 1]) / max(cell_y, 1e-4)
        grid_alignment += float(np.mean(np.square(step_x - 1.0)) + np.mean(np.square(row_y_delta)))
    if pos.shape[0] > 1:
        step_y = (pos[1:, :, 1] - pos[:-1, :, 1]) / max(cell_y, 1e-4)
        col_x_delta = (pos[1:, :, 0] - pos[:-1, :, 0]) / max(cell_x, 1e-4)
        grid_alignment += float(np.mean(np.square(step_y - 1.0)) + np.mean(np.square(col_x_delta)))
    local_signal = float(np.mean(signal))
    loss = (
        grid_alignment * float(solver_params.phase_field_grid_alignment_weight)
        - local_signal * float(solver_params.phase_field_signal_weight)
    )
    return {
        "local_signal": local_signal,
        "smoothness": 0.0,
        "grid_alignment": grid_alignment,
        "collapse": 0.0,
        "magnitude": 0.0,
        "explicit_loss": float(loss),
    }


def _observer_option(observer: PipelineObserver | None, name: str, default: object) -> object:
    return observer_attribute(observer, name, default)


def _observer_preview_stride(observer: PipelineObserver | None) -> int:
    raw = _observer_option(observer, "phase_field_preview_stride", 1)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _observer_needs_phase_field_snapshot(observer: PipelineObserver | None) -> bool:
    return bool(_observer_option(observer, "phase_field_include_snapshot", True))


def _should_emit_phase_field_step(step: int, total_steps: int, *, preview_stride: int) -> bool:
    if total_steps <= 0:
        return True
    if preview_stride <= 1:
        return True
    if step >= total_steps:
        return False
    return step % max(1, preview_stride) == 0


def optimize_phase_field(
    rgba: np.ndarray,
    inference: InferenceResult,
    analysis: PhaseFieldSourceAnalysis,
    steps: int,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
    observer: PipelineObserver | None = None,
) -> SolverArtifacts:
    solver_params = solver_params or SolverHyperParams()
    preview_stride = _observer_preview_stride(observer)
    include_snapshots = _observer_needs_phase_field_snapshot(observer)

    height, width = rgba.shape[:2]
    cell_x = width / max(1, inference.target_width)
    cell_y = height / max(1, inference.target_height)
    uv0_px = _make_regular_uv_px(
        height=height,
        width=width,
        target_height=inference.target_height,
        target_width=inference.target_width,
    )
    uv0_norm = uv0_px.copy()
    uv0_norm[..., 0] = (uv0_norm[..., 0] / max(1.0, float(width - 1))) * 2.0 - 1.0
    uv0_norm[..., 1] = (uv0_norm[..., 1] / max(1.0, float(height - 1))) * 2.0 - 1.0
    signal_map = _source_signal_map_np(
        rgba,
        analysis.edge_map,
        solver_params,
        cell_x=cell_x,
        cell_y=cell_y,
    )
    guidance = signal_map
    prep = _PhaseFieldPrep(
        source_t=None,
        edge_t=None,
        uv0_px_t=None,
        uv0_norm=uv0_norm,
        base_x_t=None,
        base_y_t=None,
        guidance=guidance,
        cell_x=cell_x,
        cell_y=cell_y,
        patch_offsets_t=None,
        feature_t=None,
        width=width,
        height=height,
    )
    prepared_payload = {
        "cell_x": float(cell_x),
        "cell_y": float(cell_y),
        "target_width": int(inference.target_width),
        "target_height": int(inference.target_height),
    }
    if include_snapshots:
        prepared_payload["uv0_px"] = uv0_px.astype(np.float32)
        prepared_payload["guidance"] = guidance.copy()
    emit_observer(observer, "phase_field_prepared", **prepared_payload)

    pos = uv0_px.astype(np.float32).copy()
    initial_x = np.rint(pos[..., 0]).astype(np.int32).clip(0, width - 1)
    initial_y = np.rint(pos[..., 1]).astype(np.int32).clip(0, height - 1)
    initial_rgba = _nearest_source_rgba(rgba, initial_x, initial_y)
    initial_payload = {
        "step": 0,
        "total_steps": int(max(0, steps)),
    }
    if include_snapshots:
        initial_payload.update(_observer_snapshot(prep, rgba, pos.astype(np.float32)))
    emit_observer(observer, "phase_field_initial", **initial_payload)

    rng = np.random.default_rng(seed)
    pos += rng.normal(0.0, 0.015, size=pos.shape).astype(np.float32) * np.float32(min(cell_x, cell_y))
    pos[..., 0] = np.clip(pos[..., 0], 0.0, max(0.0, float(width - 1)))
    pos[..., 1] = np.clip(pos[..., 1], 0.0, max(0.0, float(height - 1)))

    loss_history: list[float] = []
    final_terms: dict[str, float] = _explicit_solver_terms_np(
        pos,
        signal_map,
        solver_params,
        cell_x=cell_x,
        cell_y=cell_y,
    )
    local_search_interval = max(1, int(solver_params.phase_field_local_search_interval))
    local_blend = max(0.0, min(1.0, float(solver_params.phase_field_local_search_blend)))

    for step_index in range(max(0, steps)):
        check_observer_cancelled(observer)
        if step_index % local_search_interval == 0:
            preferred = _choose_local_preferred_positions_np(
                pos,
                signal_map,
                solver_params,
                cell_x=cell_x,
                cell_y=cell_y,
                width=width,
                height=height,
            )
            pos = (pos * np.float32(1.0 - local_blend) + preferred * np.float32(local_blend)).astype(np.float32)
        pos = _relax_lattice_springs_np(
            pos,
            solver_params,
            cell_x=cell_x,
            cell_y=cell_y,
            width=prep.width,
            height=prep.height,
        )
        final_terms = _explicit_solver_terms_np(
            pos,
            signal_map,
            solver_params,
            cell_x=cell_x,
            cell_y=cell_y,
        )
        loss_history.append(float(final_terms["explicit_loss"]))
        step_number = int(step_index + 1)
        if observer is not None and _should_emit_phase_field_step(step_number, int(max(0, steps)), preview_stride=preview_stride):
            payload = {
                "step": step_number,
                "total_steps": int(max(0, steps)),
                "loss": float(final_terms["explicit_loss"]),
                "terms": final_terms.copy(),
            }
            if include_snapshots:
                payload.update(_observer_snapshot(prep, rgba, pos.astype(np.float32)))
            emit_observer(observer, "phase_field_step", **payload)

    check_observer_cancelled(observer)
    if steps <= 0:
        loss_history.append(float(final_terms["explicit_loss"]))

    final_x_np = np.rint(pos[..., 0]).astype(np.int32).clip(0, width - 1)
    final_y_np = np.rint(pos[..., 1]).astype(np.int32).clip(0, height - 1)
    target_rgba = _nearest_source_rgba(rgba, final_x_np, final_y_np)
    final_disp = (pos - uv0_px).astype(np.float32)
    final_pos_norm = pos.copy()
    final_pos_norm[..., 0] = (final_pos_norm[..., 0] / max(1.0, float(width - 1))) * 2.0 - 1.0
    final_pos_norm[..., 1] = (final_pos_norm[..., 1] / max(1.0, float(height - 1))) * 2.0 - 1.0
    stage_diagnostics = {
        "displacements": {
            "initial_output": _displacement_diagnostics(
                uv0_norm,
                initial_x,
                initial_y,
                width=width,
                height=height,
            ),
            "final_output": _displacement_diagnostics(
                uv0_norm,
                final_x_np,
                final_y_np,
                width=width,
                height=height,
            ),
        },
        "phase_field": {
            "max_abs_dx_px": float(np.max(np.abs(final_disp[..., 0]))),
            "max_abs_dy_px": float(np.max(np.abs(final_disp[..., 1]))),
            "mean_displacement_px": float(np.mean(np.sqrt(np.square(final_disp[..., 0]) + np.square(final_disp[..., 1])))),
            **final_terms,
        },
    }
    final_payload = {
        "step": int(max(0, steps)),
        "total_steps": int(max(0, steps)),
        "terms": final_terms.copy(),
        "phase_metrics": stage_diagnostics["phase_field"].copy(),
        "loss_history": loss_history.copy(),
    }
    if loss_history:
        final_payload["loss"] = float(loss_history[-1])
    if include_snapshots:
        final_payload.update(_observer_snapshot(prep, rgba, pos.astype(np.float32)))
    emit_observer(observer, "phase_field_final", **final_payload)
    return SolverArtifacts(
        target_rgba=target_rgba,
        uv_field=final_pos_norm,
        guidance_strength=prep.guidance,
        initial_rgba=initial_rgba,
        loss_history=loss_history,
        stage_diagnostics=stage_diagnostics,
    )
