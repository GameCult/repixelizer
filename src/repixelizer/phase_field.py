from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import premultiply
from .types import ContinuousSourceAnalysis, InferenceResult, SolverArtifacts
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
    width: int
    height: int


def _charbonnier(torch, value, epsilon: float = 1e-4):
    return torch.sqrt(value.square() + epsilon)


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
    phase_x: float,
    phase_y: float,
) -> np.ndarray:
    cell_x = width / target_width
    cell_y = height / target_height
    xs = (np.arange(target_width, dtype=np.float32) + 0.5 + phase_x) * cell_x - 0.5
    ys = (np.arange(target_height, dtype=np.float32) + 0.5 + phase_y) * cell_y - 0.5
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


def _spacing_loss(torch, step, target_step: float):
    normalized = step / max(target_step, 1e-4)
    return _charbonnier(torch, normalized - 1.0).mean()


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
    max_step_x: float,
    max_step_y: float,
    max_dx: float,
    max_dy: float,
    width: int,
    height: int,
) -> None:
    with torch.no_grad():
        pos_x = (base_x_t + disp_t[..., 0]).clamp(0.0, max(0.0, float(width - 1)))
        pos_y = (base_y_t + disp_t[..., 1]).clamp(0.0, max(0.0, float(height - 1)))

        for index in range(1, pos_x.shape[2]):
            lower = pos_x[:, :, index - 1] + min_dx
            upper = pos_x[:, :, index - 1] + max_step_x
            pos_x[:, :, index] = torch.minimum(torch.maximum(pos_x[:, :, index], lower), upper)
        for index in range(pos_x.shape[2] - 2, -1, -1):
            lower = pos_x[:, :, index + 1] - max_step_x
            upper = pos_x[:, :, index + 1] - min_dx
            pos_x[:, :, index] = torch.maximum(torch.minimum(pos_x[:, :, index], upper), lower)
        for index in range(1, pos_y.shape[1]):
            lower = pos_y[:, index - 1, :] + min_dy
            upper = pos_y[:, index - 1, :] + max_step_y
            pos_y[:, index, :] = torch.minimum(torch.maximum(pos_y[:, index, :], lower), upper)
        for index in range(pos_y.shape[1] - 2, -1, -1):
            lower = pos_y[:, index + 1, :] - max_step_y
            upper = pos_y[:, index + 1, :] - min_dy
            pos_y[:, index, :] = torch.maximum(torch.minimum(pos_y[:, index, :], upper), lower)

        pos_x = pos_x.clamp(0.0, max(0.0, float(width - 1)))
        pos_y = pos_y.clamp(0.0, max(0.0, float(height - 1)))
        disp_t[..., 0].copy_((pos_x - base_x_t).clamp(-max_dx, max_dx))
        disp_t[..., 1].copy_((pos_y - base_y_t).clamp(-max_dy, max_dy))


def _prepare_phase_field(
    torch,
    F,
    rgba: np.ndarray,
    inference: InferenceResult,
    analysis: ContinuousSourceAnalysis,
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
    uv0_px = _make_regular_uv_px(
        height=height,
        width=width,
        target_height=inference.target_height,
        target_width=inference.target_width,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
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
    neighbor_weights = patch_rgba.new_tensor([1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75]).view(1, 1, 1, -1)
    color_delta = (neighbor_rgba - center_rgba[..., None, :]).abs().mean(dim=-1)
    local_coherence = (color_delta * neighbor_weights).sum() / neighbor_weights.sum().clamp_min(1e-6) / max(
        1.0, float(color_delta.shape[0] * color_delta.shape[1] * color_delta.shape[2])
    )
    center_edge = patch_edge[..., 0].mean()
    ring_edge = ((patch_edge[..., 1:] * neighbor_weights).sum() / neighbor_weights.sum().clamp_min(1e-6)) / max(
        1.0, float(patch_edge.shape[0] * patch_edge.shape[1] * patch_edge.shape[2])
    )
    local_edge = center_edge * solver_params.phase_field_data_center_edge_weight + ring_edge

    disp_norm = disp_t.clone()
    disp_norm[..., 0] = disp_norm[..., 0] / max(prep.cell_x, 1e-4)
    disp_norm[..., 1] = disp_norm[..., 1] / max(prep.cell_y, 1e-4)
    pos_mid_x = (pos_px[:, :, 1:, :] + pos_px[:, :, :-1, :]) * 0.5 if pos_px.shape[2] > 1 else None
    pos_mid_y = (pos_px[:, 1:, :, :] + pos_px[:, :-1, :, :]) * 0.5 if pos_px.shape[1] > 1 else None

    smoothness = sampled_rgba.new_tensor(0.0)
    if pos_mid_x is not None:
        edge_x = _sample_scalar(F, prep.edge_t, pos_mid_x, width=prep.width, height=prep.height)
        weight_x = torch.exp(-solver_params.phase_field_edge_gate_strength * edge_x.clamp_min(0.0))
        delta_x = disp_norm[:, :, 1:, :] - disp_norm[:, :, :-1, :]
        smoothness = smoothness + (weight_x * _charbonnier(torch, delta_x, epsilon=1e-4).mean(dim=-1)).mean()
    if pos_mid_y is not None:
        edge_y = _sample_scalar(F, prep.edge_t, pos_mid_y, width=prep.width, height=prep.height)
        weight_y = torch.exp(-solver_params.phase_field_edge_gate_strength * edge_y.clamp_min(0.0))
        delta_y = disp_norm[:, 1:, :, :] - disp_norm[:, :-1, :, :]
        smoothness = smoothness + (weight_y * _charbonnier(torch, delta_y, epsilon=1e-4).mean(dim=-1)).mean()

    collapse = sampled_rgba.new_tensor(0.0)
    spacing = sampled_rgba.new_tensor(0.0)
    min_dx = solver_params.phase_field_min_spacing_ratio * prep.cell_x
    min_dy = solver_params.phase_field_min_spacing_ratio * prep.cell_y
    if pos_px.shape[2] > 1:
        step_x = pos_px[:, :, 1:, 0] - pos_px[:, :, :-1, 0]
        collapse = collapse + torch.relu(min_dx - step_x).square().mean()
        spacing = spacing + _spacing_loss(torch, step_x, prep.cell_x)
    if pos_px.shape[1] > 1:
        step_y = pos_px[:, 1:, :, 1] - pos_px[:, :-1, :, 1]
        collapse = collapse + torch.relu(min_dy - step_y).square().mean()
        spacing = spacing + _spacing_loss(torch, step_y, prep.cell_y)

    magnitude = (
        (disp_t[..., 0] / max(prep.cell_x, 1e-4)).square()
        + (disp_t[..., 1] / max(prep.cell_y, 1e-4)).square()
    ).mean()

    loss = (
        local_coherence * solver_params.phase_field_data_coherence_weight
        + local_edge * solver_params.phase_field_data_edge_weight
        + smoothness * solver_params.phase_field_smoothness_weight
        + spacing * solver_params.phase_field_spacing_weight
        + collapse * solver_params.phase_field_collapse_weight
        + magnitude * solver_params.phase_field_magnitude_weight
    )
    terms = {
        "local_coherence": float(local_coherence.detach().cpu().item()),
        "local_edge": float(local_edge.detach().cpu().item()),
        "smoothness": float(smoothness.detach().cpu().item()),
        "spacing": float(spacing.detach().cpu().item()),
        "collapse": float(collapse.detach().cpu().item()),
        "magnitude": float(magnitude.detach().cpu().item()),
    }
    return loss, sampled_rgba, terms


def _nearest_source_rgba(source_rgba: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    return source_rgba[sample_y, sample_x]


def optimize_phase_field(
    rgba: np.ndarray,
    inference: InferenceResult,
    analysis: ContinuousSourceAnalysis,
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

    prep = _prepare_phase_field(
        torch,
        F,
        rgba,
        inference,
        analysis,
        solver_params,
        device=device,
    )

    disp_t = torch.zeros_like(prep.uv0_px_t, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([disp_t], lr=solver_params.phase_field_learning_rate)

    initial_x = np.rint(prep.uv0_px_t[0, ..., 0].detach().cpu().numpy()).astype(np.int32).clip(0, prep.width - 1)
    initial_y = np.rint(prep.uv0_px_t[0, ..., 1].detach().cpu().numpy()).astype(np.int32).clip(0, prep.height - 1)
    initial_rgba = _nearest_source_rgba(rgba, initial_x, initial_y)

    loss_history: list[float] = []
    final_terms: dict[str, float] = {}
    min_dx = solver_params.phase_field_min_spacing_ratio * prep.cell_x
    min_dy = solver_params.phase_field_min_spacing_ratio * prep.cell_y
    max_step_x = solver_params.phase_field_max_spacing_ratio * prep.cell_x
    max_step_y = solver_params.phase_field_max_spacing_ratio * prep.cell_y
    max_dx = solver_params.phase_field_max_displacement_ratio * prep.cell_x
    max_dy = solver_params.phase_field_max_displacement_ratio * prep.cell_y

    for _ in range(max(0, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss, _sampled_rgba, terms = _phase_field_loss(torch, F, prep, disp_t, solver_params)
        loss.backward()
        optimizer.step()
        _project_displacements_in_place(
            torch,
            disp_t,
            prep.base_x_t,
            prep.base_y_t,
            min_dx=min_dx,
            min_dy=min_dy,
            max_step_x=max_step_x,
            max_step_y=max_step_y,
            max_dx=max_dx,
            max_dy=max_dy,
            width=prep.width,
            height=prep.height,
        )
        loss_history.append(float(loss.detach().cpu().item()))
        final_terms = terms

    with torch.no_grad():
        if steps <= 0:
            loss, _sampled_rgba, final_terms = _phase_field_loss(torch, F, prep, disp_t, solver_params)
            loss_history.append(float(loss.detach().cpu().item()))
        final_px = prep.uv0_px_t + disp_t
        final_x = torch.round(final_px[..., 0]).clamp(0, prep.width - 1).to(dtype=torch.long)
        final_y = torch.round(final_px[..., 1]).clamp(0, prep.height - 1).to(dtype=torch.long)
        final_pos_norm = _pixel_to_normalized(final_px.clone(), width=prep.width, height=prep.height)[0].detach().cpu().numpy()

    final_x_np = final_x[0].detach().cpu().numpy().astype(np.int32)
    final_y_np = final_y[0].detach().cpu().numpy().astype(np.int32)
    target_rgba = _nearest_source_rgba(rgba, final_x_np, final_y_np)
    final_disp = disp_t[0].detach().cpu().numpy().astype(np.float32)
    stage_diagnostics = {
        "displacements": {
            "initial_output": _displacement_diagnostics(
                prep.uv0_norm,
                initial_x,
                initial_y,
                width=prep.width,
                height=prep.height,
            ),
            "final_output": _displacement_diagnostics(
                prep.uv0_norm,
                final_x_np,
                final_y_np,
                width=prep.width,
                height=prep.height,
            ),
        },
        "phase_field": {
            "max_abs_dx_px": float(np.max(np.abs(final_disp[..., 0]))),
            "max_abs_dy_px": float(np.max(np.abs(final_disp[..., 1]))),
            "mean_displacement_px": float(np.mean(np.sqrt(np.square(final_disp[..., 0]) + np.square(final_disp[..., 1])))),
            **final_terms,
        },
    }
    return SolverArtifacts(
        target_rgba=target_rgba,
        uv_field=final_pos_norm,
        guidance_strength=prep.guidance,
        initial_rgba=initial_rgba,
        loss_history=loss_history,
        stage_diagnostics=stage_diagnostics,
    )
