from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .params import SolverHyperParams
from .source_reference import build_source_lattice_reference
from .types import InferenceResult, SolverArtifacts, SourceAnalysis

_RIGHT = 0
_DOWN = 1
_LEFT = 2
_UP = 3
_DIRS = (
    (_RIGHT, 0, 1),
    (_DOWN, 1, 0),
    (_LEFT, 0, -1),
    (_UP, -1, 0),
)
_OPPOSITE = {
    _RIGHT: _LEFT,
    _DOWN: _UP,
    _LEFT: _RIGHT,
    _UP: _DOWN,
}


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only when torch missing
        raise RuntimeError(
            "PyTorch is required for the tile-graph solver. Install project dependencies first."
        ) from exc
    return torch


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build does not have a usable CUDA device. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return requested


@dataclass(slots=True)
class TileGraphModel:
    candidate_rgba: np.ndarray
    candidate_coords: np.ndarray
    candidate_area_ratio: np.ndarray
    candidate_coverage: np.ndarray
    candidate_deltas: np.ndarray
    cell_candidate_offsets: np.ndarray
    cell_candidate_indices: np.ndarray
    reference_mean_rgba: np.ndarray
    reference_sharp_rgba: np.ndarray
    reference_edge_rgba: np.ndarray
    edge_strength: np.ndarray
    component_count: int
    edge_density: float
    average_choices: float
    model_device: str


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


def _build_ranked_proposal_pool_torch(torch, cell_indices_t, proposal_score_t, *, cell_count: int, pool_size: int):
    if pool_size <= 0:
        return torch.full((cell_count, 0), -1, device=cell_indices_t.device, dtype=torch.long)
    if cell_indices_t.numel() == 0:
        return torch.full((cell_count, pool_size), -1, device=cell_indices_t.device, dtype=torch.long)
    score_lo_t = torch.min(proposal_score_t)
    score_hi_t = torch.max(proposal_score_t)
    normalized_score_t = (proposal_score_t - score_lo_t) / (score_hi_t - score_lo_t).clamp_min(1e-6)
    sort_key_t = cell_indices_t.to(dtype=torch.float64) * 2.0 + (1.0 - normalized_score_t.to(dtype=torch.float64))
    order_t = torch.argsort(sort_key_t)
    sorted_cells_t = cell_indices_t[order_t]
    positions_t = torch.arange(order_t.shape[0], device=cell_indices_t.device, dtype=torch.long)
    start_flags_t = torch.ones_like(sorted_cells_t, dtype=torch.bool)
    if sorted_cells_t.numel() > 1:
        start_flags_t[1:] = sorted_cells_t[1:] != sorted_cells_t[:-1]
    group_start_t = torch.where(start_flags_t, positions_t, torch.zeros_like(positions_t))
    group_start_t = torch.cummax(group_start_t, dim=0).values
    rank_t = positions_t - group_start_t
    keep_mask_t = rank_t < pool_size
    pool_t = torch.full((cell_count, pool_size), -1, device=cell_indices_t.device, dtype=torch.long)
    pool_t[sorted_cells_t[keep_mask_t], rank_t[keep_mask_t]] = order_t[keep_mask_t]
    return pool_t


def build_tile_graph_model(
    source_rgba: np.ndarray,
    *,
    inference: InferenceResult,
    analysis: SourceAnalysis,
    solver_params: SolverHyperParams | None = None,
    device: str = "cpu",
) -> TileGraphModel:
    torch = _require_torch()
    resolved_device = _resolve_device(torch, device)
    solver_params = solver_params or SolverHyperParams()
    source_reference = build_source_lattice_reference(
        source_rgba,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
        alpha_threshold=solver_params.alpha_transparent_threshold,
        edge_hint=analysis.edge_map,
        device=resolved_device,
    )
    height, width = source_rgba.shape[:2]
    cell_h = height / max(1, inference.target_height)
    cell_w = width / max(1, inference.target_width)
    output_area = max(1, inference.target_height * inference.target_width)
    max_candidates_per_coord = max(1, int(solver_params.tile_graph_max_candidates_per_coord))
    proposal_pool_size = max(max_candidates_per_coord, int(getattr(solver_params, "tile_graph_proposal_pool_size", 4)))

    source_t = torch.from_numpy(source_rgba).to(device=resolved_device, dtype=torch.float32)
    edge_t = torch.from_numpy(analysis.edge_map).to(device=resolved_device, dtype=torch.float32)
    cluster_t = torch.from_numpy(analysis.cluster_map).to(device=resolved_device, dtype=torch.long)
    flat_rgba_t = source_t.reshape(-1, source_t.shape[-1])
    flat_alpha_t = flat_rgba_t[:, 3]
    flat_edge_t = edge_t.reshape(-1)
    flat_cluster_t = cluster_t.reshape(-1)
    lattice_indices_t = torch.from_numpy(source_reference.lattice_indices).to(device=resolved_device, dtype=torch.long).reshape(-1)
    cell_support_flat_t = torch.from_numpy(source_reference.cell_support.reshape(-1)).to(device=resolved_device, dtype=torch.float32)
    cell_counts_flat_t = torch.from_numpy(source_reference.cell_counts.reshape(-1)).to(device=resolved_device, dtype=torch.float32)
    edge_strength_flat_t = torch.from_numpy(source_reference.edge_strength.reshape(-1)).to(device=resolved_device, dtype=torch.float32)
    sharp_x_flat_t = torch.from_numpy(source_reference.sharp_x.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    sharp_y_flat_t = torch.from_numpy(source_reference.sharp_y.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    edge_peak_x_flat_t = torch.from_numpy(source_reference.edge_peak_x.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    edge_peak_y_flat_t = torch.from_numpy(source_reference.edge_peak_y.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    sharp_linear_t = sharp_y_flat_t * width + sharp_x_flat_t
    edge_linear_t = edge_peak_y_flat_t * width + edge_peak_x_flat_t

    pixel_y_t, pixel_x_t = torch.meshgrid(
        torch.arange(height, device=resolved_device, dtype=torch.long),
        torch.arange(width, device=resolved_device, dtype=torch.long),
        indexing="ij",
    )
    flat_y_t = pixel_y_t.reshape(-1)
    flat_x_t = pixel_x_t.reshape(-1)
    sharp_rgba_flat_t = flat_rgba_t[sharp_linear_t]
    edge_rgba_flat_t = flat_rgba_t[edge_linear_t]
    sharp_cluster_flat_t = flat_cluster_t[sharp_linear_t]
    sharp_dist_t = (
        (flat_y_t.to(dtype=torch.float32) - sharp_y_flat_t[lattice_indices_t].to(dtype=torch.float32)).abs() / max(cell_h, 1.0)
        + (flat_x_t.to(dtype=torch.float32) - sharp_x_flat_t[lattice_indices_t].to(dtype=torch.float32)).abs() / max(cell_w, 1.0)
    )
    edge_focus_dist_t = (
        (flat_y_t.to(dtype=torch.float32) - edge_peak_y_flat_t[lattice_indices_t].to(dtype=torch.float32)).abs() / max(cell_h, 1.0)
        + (flat_x_t.to(dtype=torch.float32) - edge_peak_x_flat_t[lattice_indices_t].to(dtype=torch.float32)).abs() / max(cell_w, 1.0)
    )
    same_cluster_t = (
        (flat_cluster_t >= 0) & (flat_cluster_t == sharp_cluster_flat_t[lattice_indices_t])
    ).to(dtype=torch.float32)
    sharp_color_error_t = (flat_rgba_t - sharp_rgba_flat_t[lattice_indices_t]).abs().mean(dim=-1)
    edge_color_error_t = (flat_rgba_t - edge_rgba_flat_t[lattice_indices_t]).abs().mean(dim=-1)
    edge_cell_mask_t = edge_strength_flat_t[lattice_indices_t] >= solver_params.source_edge_detail_threshold
    proposal_color_error_t = torch.where(
        edge_cell_mask_t,
        torch.minimum(sharp_color_error_t, edge_color_error_t),
        sharp_color_error_t,
    )
    edge_cell_boost_t = (
        (1.0 - edge_focus_dist_t.clamp(0.0, 1.0))
        * edge_cell_mask_t.to(dtype=torch.float32)
    )
    proposal_score_t = (
        flat_alpha_t
        + flat_edge_t * getattr(solver_params, "tile_graph_proposal_edge_weight", 0.40)
        + same_cluster_t * getattr(solver_params, "tile_graph_proposal_cluster_weight", 0.12)
        + edge_cell_boost_t * getattr(solver_params, "tile_graph_proposal_edge_weight", 0.40) * 0.5
        + cell_support_flat_t[lattice_indices_t] * 0.08
        - sharp_dist_t * getattr(solver_params, "tile_graph_proposal_distance_weight", 0.22)
        - proposal_color_error_t * getattr(solver_params, "tile_graph_proposal_color_weight", 0.10)
    )
    ranked_pool_t = _build_ranked_proposal_pool_torch(
        torch,
        lattice_indices_t,
        proposal_score_t,
        cell_count=output_area,
        pool_size=proposal_pool_size,
    )
    edge_ranked_pool_t = _build_ranked_proposal_pool_torch(
        torch,
        lattice_indices_t,
        flat_edge_t + flat_alpha_t * 0.05,
        cell_count=output_area,
        pool_size=proposal_pool_size,
    )

    cluster_count = max(1, int(max(analysis.cluster_centers.shape[0], 1)))
    pixel_coverage_t = torch.maximum(flat_alpha_t.clamp(0.0, 1.0), flat_alpha_t.new_zeros(flat_alpha_t.shape))
    valid_cluster_mask_t = flat_cluster_t >= 0
    if bool(valid_cluster_mask_t.any().item()):
        cluster_keys_t = lattice_indices_t[valid_cluster_mask_t] * cluster_count + flat_cluster_t[valid_cluster_mask_t]
        cluster_counts_t = torch.bincount(cluster_keys_t, minlength=output_area * cluster_count).to(dtype=torch.float32)
        pixel_coverage_t = torch.zeros_like(flat_alpha_t, dtype=torch.float32)
        pixel_coverage_t[valid_cluster_mask_t] = (
            cluster_counts_t[cluster_keys_t] / cell_counts_flat_t[lattice_indices_t[valid_cluster_mask_t]].clamp_min(1.0)
        )
        pixel_coverage_t = torch.maximum(pixel_coverage_t, flat_alpha_t * 0.5)

    sharp_linear_np = sharp_linear_t.detach().cpu().numpy().astype(np.int64)
    edge_linear_np = edge_linear_t.detach().cpu().numpy().astype(np.int64)
    edge_neighbor_linear_np: list[np.ndarray] = []
    cell_ids_t = torch.arange(output_area, device=resolved_device, dtype=torch.long)
    for _, dy, dx in _DIRS:
        neighbor_y_t = (edge_peak_y_flat_t + dy).clamp(0, max(0, height - 1))
        neighbor_x_t = (edge_peak_x_flat_t + dx).clamp(0, max(0, width - 1))
        neighbor_linear_t = neighbor_y_t * width + neighbor_x_t
        same_cell_neighbor_t = lattice_indices_t[neighbor_linear_t] == cell_ids_t
        masked_neighbor_linear_t = torch.where(
            same_cell_neighbor_t,
            neighbor_linear_t,
            torch.full_like(neighbor_linear_t, -1),
        )
        edge_neighbor_linear_np.append(masked_neighbor_linear_t.detach().cpu().numpy().astype(np.int64))
    ranked_pool_np = ranked_pool_t.detach().cpu().numpy().astype(np.int64)
    edge_ranked_pool_np = edge_ranked_pool_t.detach().cpu().numpy().astype(np.int64)
    pixel_coverage_np = pixel_coverage_t.detach().cpu().numpy().astype(np.float32)
    edge_strength_np = source_reference.edge_strength.reshape(-1).astype(np.float32)
    flat_alpha_np = source_rgba.reshape(-1, source_rgba.shape[-1])[:, 3].astype(np.float32)

    candidate_linear: list[int] = []
    candidate_coords: list[tuple[int, int]] = []
    candidate_area_ratio: list[float] = []
    candidate_coverage: list[float] = []
    choice_counts_list: list[int] = []
    edge_candidate_cap = max(max_candidates_per_coord, int(getattr(solver_params, "tile_graph_edge_candidates_per_coord", max_candidates_per_coord)))
    for flat_index in range(output_area):
        coord_y = flat_index // inference.target_width
        coord_x = flat_index % inference.target_width
        seen_pixels: set[int] = set()
        cell_start = len(candidate_linear)
        edge_cell = edge_strength_np[flat_index] >= solver_params.source_edge_detail_threshold
        allowed_candidates = edge_candidate_cap if edge_cell else max_candidates_per_coord
        proposal_pixels = [int(sharp_linear_np[flat_index])]
        if edge_cell:
            proposal_pixels.append(int(edge_linear_np[flat_index]))
            for neighbor_linear in edge_neighbor_linear_np:
                proposal_pixels.append(int(neighbor_linear[flat_index]))
            proposal_pixels.extend(int(pixel_idx) for pixel_idx in edge_ranked_pool_np[flat_index].tolist() if int(pixel_idx) >= 0)
        proposal_pixels.extend(int(pixel_idx) for pixel_idx in ranked_pool_np[flat_index].tolist() if int(pixel_idx) >= 0)
        for pixel_index in proposal_pixels:
            if pixel_index < 0 or pixel_index in seen_pixels:
                continue
            is_primary = len(seen_pixels) == 0
            coverage = float(pixel_coverage_np[pixel_index])
            if (
                not is_primary
                and coverage < solver_params.tile_graph_window_coverage_threshold
                and not edge_cell
            ):
                continue
            seen_pixels.add(pixel_index)
            candidate_linear.append(pixel_index)
            candidate_coords.append((coord_y, coord_x))
            candidate_area_ratio.append(1.0)
            candidate_coverage.append(max(coverage, float(flat_alpha_np[pixel_index])))
            if len(seen_pixels) >= allowed_candidates:
                break
        if len(candidate_linear) == cell_start:
            pixel_index = int(sharp_linear_np[flat_index])
            candidate_linear.append(pixel_index)
            candidate_coords.append((coord_y, coord_x))
            candidate_area_ratio.append(1.0)
            candidate_coverage.append(max(float(pixel_coverage_np[pixel_index]), float(flat_alpha_np[pixel_index])))
        choice_counts_list.append(len(candidate_linear) - cell_start)

    candidate_linear_t = torch.as_tensor(candidate_linear, device=resolved_device, dtype=torch.long)
    center_rgba_t = flat_rgba_t[candidate_linear_t]
    center_y_t = flat_y_t[candidate_linear_t].to(dtype=torch.float32)
    center_x_t = flat_x_t[candidate_linear_t].to(dtype=torch.float32)
    candidate_deltas_t = torch.zeros((candidate_linear_t.shape[0], 4, 4), device=resolved_device, dtype=torch.float32)
    for direction, dy, dx in _DIRS:
        sample_y_t = torch.round(center_y_t + dy * cell_h).clamp(0, max(0, height - 1)).to(dtype=torch.long)
        sample_x_t = torch.round(center_x_t + dx * cell_w).clamp(0, max(0, width - 1)).to(dtype=torch.long)
        candidate_deltas_t[:, direction] = source_t[sample_y_t, sample_x_t] - center_rgba_t

    candidate_rgba_np = center_rgba_t.detach().cpu().numpy().astype(np.float32)
    candidate_coords_np = np.asarray(candidate_coords, dtype=np.int32)
    candidate_area_ratio_np = np.asarray(candidate_area_ratio, dtype=np.float32)
    candidate_coverage_np = np.asarray(candidate_coverage, dtype=np.float32)
    candidate_deltas_np = candidate_deltas_t.detach().cpu().numpy().astype(np.float32)

    cell_candidate_offsets = np.zeros(output_area + 1, dtype=np.int32)
    cell_candidate_offsets[1:] = np.cumsum(np.asarray(choice_counts_list, dtype=np.int32), dtype=np.int32)
    cell_candidate_indices_np = np.arange(candidate_rgba_np.shape[0], dtype=np.int32)
    choice_counts = np.diff(cell_candidate_offsets)
    average_choices = float(np.mean(choice_counts)) if choice_counts.size else 0.0

    extracted_choice_grid = np.maximum(choice_counts.reshape(inference.target_height, inference.target_width) - 1, 0)
    edge_support = 0.0
    edge_total = 0
    if inference.target_width > 1:
        edge_support += float(
            np.mean(
                (extracted_choice_grid[:, :-1] > 0).astype(np.float32)
                * (extracted_choice_grid[:, 1:] > 0).astype(np.float32)
            )
        )
        edge_total += 1
    if inference.target_height > 1:
        edge_support += float(
            np.mean(
                (extracted_choice_grid[:-1, :] > 0).astype(np.float32)
                * (extracted_choice_grid[1:, :] > 0).astype(np.float32)
            )
        )
        edge_total += 1
    edge_density = edge_support / max(1, edge_total)

    return TileGraphModel(
        candidate_rgba=candidate_rgba_np,
        candidate_coords=candidate_coords_np,
        candidate_area_ratio=candidate_area_ratio_np,
        candidate_coverage=candidate_coverage_np,
        candidate_deltas=candidate_deltas_np,
        cell_candidate_offsets=cell_candidate_offsets,
        cell_candidate_indices=cell_candidate_indices_np,
        reference_mean_rgba=source_reference.mean_rgba.astype(np.float32),
        reference_sharp_rgba=source_reference.sharp_rgba.astype(np.float32),
        reference_edge_rgba=edge_rgba_flat_t.reshape(inference.target_height, inference.target_width, -1).detach().cpu().numpy().astype(np.float32),
        edge_strength=source_reference.edge_strength.astype(np.float32),
        component_count=int(np.unique(analysis.cluster_map[analysis.alpha_map >= solver_params.alpha_transparent_threshold]).size),
        edge_density=edge_density,
        average_choices=average_choices,
        model_device=resolved_device,
    )


def _cell_candidate_span(model: TileGraphModel, y: int, x: int) -> tuple[int, int]:
    width = model.reference_sharp_rgba.shape[1]
    flat_index = y * width + x
    return int(model.cell_candidate_offsets[flat_index]), int(model.cell_candidate_offsets[flat_index + 1])


def _build_choice_grid(model: TileGraphModel) -> tuple[np.ndarray, np.ndarray]:
    height, width = model.reference_sharp_rgba.shape[:2]
    choice_counts = np.diff(model.cell_candidate_offsets)
    max_choices = int(np.max(choice_counts)) if choice_counts.size else 1
    choice_indices = np.zeros((height, width, max_choices), dtype=np.int64)
    choice_mask = np.zeros((height, width, max_choices), dtype=bool)
    for flat_index, count in enumerate(choice_counts.tolist()):
        if count <= 0:
            continue
        y = flat_index // width
        x = flat_index % width
        start = int(model.cell_candidate_offsets[flat_index])
        end = int(model.cell_candidate_offsets[flat_index + 1])
        choice_indices[y, x, :count] = model.cell_candidate_indices[start:end]
        choice_mask[y, x, :count] = True
    return choice_indices, choice_mask


def _tile_graph_unary_cost(model: TileGraphModel, solver_params: SolverHyperParams) -> np.ndarray:
    height, width = model.reference_sharp_rgba.shape[:2]
    unary_cost = np.full(model.cell_candidate_indices.shape[0], np.inf, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            start, end = _cell_candidate_span(model, y, x)
            if end <= start:
                continue
            indices = model.cell_candidate_indices[start:end]
            reference_sharp = model.reference_sharp_rgba[y, x]
            reference_mean = model.reference_mean_rgba[y, x]
            reference_edge = model.reference_edge_rgba[y, x]
            candidates = model.candidate_rgba[indices]
            sharp_error = np.mean(np.abs(reference_sharp[None, :] - candidates), axis=-1)
            mean_error = np.mean(np.abs(reference_mean[None, :] - candidates), axis=-1)
            edge_error = np.mean(np.abs(reference_edge[None, :] - candidates), axis=-1)
            if float(model.edge_strength[y, x]) >= solver_params.source_edge_detail_threshold:
                color_error = np.minimum(sharp_error, edge_error) + mean_error * solver_params.tile_graph_edge_mean_weight
            else:
                color_error = (
                    sharp_error * solver_params.tile_graph_nonedge_sharp_weight
                    + mean_error * solver_params.tile_graph_nonedge_mean_weight
                )
            area_error = np.abs(np.log(np.clip(model.candidate_area_ratio[indices], 1e-4, None) + 1e-4))
            alpha_error = np.abs(candidates[:, 3] - reference_mean[3])
            coverage_error = 1.0 - np.clip(model.candidate_coverage[indices], 0.0, 1.0)
            unary_cost[start:end] = (
                color_error
                + area_error * solver_params.tile_graph_area_weight
                + alpha_error * solver_params.tile_graph_alpha_weight
                + coverage_error * solver_params.tile_graph_coverage_weight
            ).astype(np.float32)
    return unary_cost


def _tile_graph_unary_cost_torch(
    torch,
    model: TileGraphModel,
    choice_indices_t,
    choice_mask_t,
    *,
    device: str,
    solver_params: SolverHyperParams,
):
    candidate_rgba_t = torch.from_numpy(model.candidate_rgba).to(device=device, dtype=torch.float32)
    candidate_deltas_t = torch.from_numpy(model.candidate_deltas).to(device=device, dtype=torch.float32)
    candidate_area_t = torch.from_numpy(model.candidate_area_ratio).to(device=device, dtype=torch.float32)
    candidate_coverage_t = torch.from_numpy(model.candidate_coverage).to(device=device, dtype=torch.float32)
    reference_sharp_t = torch.from_numpy(model.reference_sharp_rgba).to(device=device, dtype=torch.float32)
    reference_mean_t = torch.from_numpy(model.reference_mean_rgba).to(device=device, dtype=torch.float32)
    reference_edge_t = torch.from_numpy(model.reference_edge_rgba).to(device=device, dtype=torch.float32)
    edge_strength_t = torch.from_numpy(model.edge_strength).to(device=device, dtype=torch.float32)

    choice_rgba_t = candidate_rgba_t[choice_indices_t]
    choice_deltas_t = candidate_deltas_t[choice_indices_t]
    sharp_error_t = (reference_sharp_t[..., None, :] - choice_rgba_t).abs().mean(dim=-1)
    mean_error_t = (reference_mean_t[..., None, :] - choice_rgba_t).abs().mean(dim=-1)
    edge_error_t = (reference_edge_t[..., None, :] - choice_rgba_t).abs().mean(dim=-1)
    edge_cell_mask_t = edge_strength_t[..., None] >= solver_params.source_edge_detail_threshold
    color_error = torch.where(
        edge_cell_mask_t,
        torch.minimum(sharp_error_t, edge_error_t) + mean_error_t * solver_params.tile_graph_edge_mean_weight,
        sharp_error_t * solver_params.tile_graph_nonedge_sharp_weight
        + mean_error_t * solver_params.tile_graph_nonedge_mean_weight,
    )
    area_error = torch.log(candidate_area_t[choice_indices_t].clamp_min(1e-4) + 1e-4).abs()
    alpha_error = (choice_rgba_t[..., 3] - reference_mean_t[..., None, 3]).abs()
    coverage_error = 1.0 - candidate_coverage_t[choice_indices_t].clamp(0.0, 1.0)
    unary_cost_t = (
        color_error
        + area_error * solver_params.tile_graph_area_weight
        + alpha_error * solver_params.tile_graph_alpha_weight
        + coverage_error * solver_params.tile_graph_coverage_weight
    )
    unary_cost_t = unary_cost_t.masked_fill(~choice_mask_t, float("inf"))
    return unary_cost_t, candidate_rgba_t, candidate_deltas_t, choice_rgba_t, choice_deltas_t


def _pair_penalty_selected_torch(torch, candidate_rgba_t, candidate_deltas_t, left_indices_t, right_indices_t, direction: int, weight: float):
    left_rgba = candidate_rgba_t[left_indices_t]
    right_rgba = candidate_rgba_t[right_indices_t]
    observed_delta = right_rgba - left_rgba
    expected = candidate_deltas_t[left_indices_t, direction]
    reverse = candidate_deltas_t[right_indices_t, _OPPOSITE[direction]]
    delta_penalty = (observed_delta - expected).abs().mean(dim=-1)
    reverse_penalty = (-observed_delta - reverse).abs().mean(dim=-1)
    return (delta_penalty + reverse_penalty) * (0.5 * weight)


def _pair_penalty_option_right_torch(
    torch,
    fixed_left_indices_t,
    option_rgba_t,
    option_deltas_t,
    candidate_rgba_t,
    candidate_deltas_t,
    *,
    weight: float,
):
    fixed_left_rgba = candidate_rgba_t[fixed_left_indices_t]
    expected = candidate_deltas_t[fixed_left_indices_t, _RIGHT][..., None, :]
    reverse = option_deltas_t[..., _LEFT, :]
    observed_delta = option_rgba_t - fixed_left_rgba[..., None, :]
    delta_penalty = (observed_delta - expected).abs().mean(dim=-1)
    reverse_penalty = (-observed_delta - reverse).abs().mean(dim=-1)
    return (delta_penalty + reverse_penalty) * (0.5 * weight)


def _pair_penalty_option_left_torch(
    torch,
    option_rgba_t,
    option_deltas_t,
    fixed_right_indices_t,
    candidate_rgba_t,
    candidate_deltas_t,
    *,
    weight: float,
):
    fixed_right_rgba = candidate_rgba_t[fixed_right_indices_t]
    expected = option_deltas_t[..., _RIGHT, :]
    reverse = candidate_deltas_t[fixed_right_indices_t, _LEFT][..., None, :]
    observed_delta = fixed_right_rgba[..., None, :] - option_rgba_t
    delta_penalty = (observed_delta - expected).abs().mean(dim=-1)
    reverse_penalty = (-observed_delta - reverse).abs().mean(dim=-1)
    return (delta_penalty + reverse_penalty) * (0.5 * weight)


def _pair_penalty_option_down_torch(
    torch,
    fixed_up_indices_t,
    option_rgba_t,
    option_deltas_t,
    candidate_rgba_t,
    candidate_deltas_t,
    *,
    weight: float,
):
    fixed_up_rgba = candidate_rgba_t[fixed_up_indices_t]
    expected = candidate_deltas_t[fixed_up_indices_t, _DOWN][..., None, :]
    reverse = option_deltas_t[..., _UP, :]
    observed_delta = option_rgba_t - fixed_up_rgba[..., None, :]
    delta_penalty = (observed_delta - expected).abs().mean(dim=-1)
    reverse_penalty = (-observed_delta - reverse).abs().mean(dim=-1)
    return (delta_penalty + reverse_penalty) * (0.5 * weight)


def _pair_penalty_option_up_torch(
    torch,
    option_rgba_t,
    option_deltas_t,
    fixed_down_indices_t,
    candidate_rgba_t,
    candidate_deltas_t,
    *,
    weight: float,
):
    fixed_down_rgba = candidate_rgba_t[fixed_down_indices_t]
    expected = option_deltas_t[..., _DOWN, :]
    reverse = candidate_deltas_t[fixed_down_indices_t, _UP][..., None, :]
    observed_delta = fixed_down_rgba[..., None, :] - option_rgba_t
    delta_penalty = (observed_delta - expected).abs().mean(dim=-1)
    reverse_penalty = (-observed_delta - reverse).abs().mean(dim=-1)
    return (delta_penalty + reverse_penalty) * (0.5 * weight)


def _assignment_score_torch(torch, selected_t, unary_cost_t, choice_indices_t, candidate_rgba_t, candidate_deltas_t, solver_params: SolverHyperParams) -> float:
    selected_choice_t = (choice_indices_t == selected_t[..., None]).to(dtype=torch.int64).argmax(dim=-1)
    gathered = torch.take_along_dim(unary_cost_t, selected_choice_t[..., None], dim=2)[..., 0]
    score = float(gathered.mean().item())
    if selected_t.shape[1] > 1:
        score += float(
            _pair_penalty_selected_torch(
                torch,
                candidate_rgba_t,
                candidate_deltas_t,
                selected_t[:, :-1],
                selected_t[:, 1:],
                _RIGHT,
                solver_params.tile_graph_delta_weight,
            ).mean().item()
        )
    if selected_t.shape[0] > 1:
        score += float(
            _pair_penalty_selected_torch(
                torch,
                candidate_rgba_t,
                candidate_deltas_t,
                selected_t[:-1, :],
                selected_t[1:, :],
                _DOWN,
                solver_params.tile_graph_delta_weight,
            ).mean().item()
        )
    return score


def _assignment_rgba(model: TileGraphModel, selected: np.ndarray) -> np.ndarray:
    return model.candidate_rgba[selected]


def _pair_penalty(model: TileGraphModel, left_idx: int, right_idx: int, direction: int, solver_params: SolverHyperParams) -> float:
    observed_delta = model.candidate_rgba[right_idx] - model.candidate_rgba[left_idx]
    expected = model.candidate_deltas[left_idx, direction]
    reverse = model.candidate_deltas[right_idx, _OPPOSITE[direction]]
    delta_penalty = np.mean(np.abs(observed_delta - expected))
    reverse_penalty = np.mean(np.abs(-observed_delta - reverse))
    return float((delta_penalty + reverse_penalty) * 0.5 * solver_params.tile_graph_delta_weight)


def _assignment_score(selected: np.ndarray, model: TileGraphModel, unary_cost: np.ndarray, solver_params: SolverHyperParams) -> float:
    height, width = selected.shape
    gathered = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            start, end = _cell_candidate_span(model, y, x)
            indices = model.cell_candidate_indices[start:end]
            match = np.flatnonzero(indices == selected[y, x])
            if match.size == 0:
                continue
            gathered[y, x] = unary_cost[start + int(match[0])]
    score = float(np.mean(gathered))
    if width > 1:
        penalties = []
        for y in range(height):
            for x in range(width - 1):
                penalties.append(_pair_penalty(model, int(selected[y, x]), int(selected[y, x + 1]), _RIGHT, solver_params))
        if penalties:
            score += float(np.mean(np.asarray(penalties, dtype=np.float32)))
    if height > 1:
        penalties = []
        for y in range(height - 1):
            for x in range(width):
                penalties.append(_pair_penalty(model, int(selected[y, x]), int(selected[y + 1, x]), _DOWN, solver_params))
        if penalties:
            score += float(np.mean(np.asarray(penalties, dtype=np.float32)))
    return score


def optimize_tile_graph(
    rgba: np.ndarray,
    inference: InferenceResult,
    analysis: SourceAnalysis,
    steps: int,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
) -> tuple[SolverArtifacts, dict[str, Any]]:
    del steps
    torch = _require_torch()
    resolved_device = _resolve_device(torch, device)
    solver_params = solver_params or SolverHyperParams()
    torch.manual_seed(seed)
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_tile_graph_model(
        rgba,
        inference=inference,
        analysis=analysis,
        solver_params=solver_params,
        device=resolved_device,
    )
    choice_indices_np, choice_mask_np = _build_choice_grid(model)
    choice_indices_t = torch.from_numpy(choice_indices_np).to(device=resolved_device, dtype=torch.long)
    choice_mask_t = torch.from_numpy(choice_mask_np).to(device=resolved_device, dtype=torch.bool)
    unary_cost_t, candidate_rgba_t, candidate_deltas_t, choice_rgba_t, choice_deltas_t = _tile_graph_unary_cost_torch(
        torch,
        model,
        choice_indices_t,
        choice_mask_t,
        device=resolved_device,
        solver_params=solver_params,
    )
    initial_choice_t = torch.argmin(unary_cost_t, dim=2)
    selected_t = torch.take_along_dim(choice_indices_t, initial_choice_t[..., None], dim=2)[..., 0]
    initial_selected_t = selected_t.clone()
    loss_history = [_assignment_score_torch(torch, selected_t, unary_cost_t, choice_indices_t, candidate_rgba_t, candidate_deltas_t, solver_params)]
    grid_y_t, grid_x_t = torch.meshgrid(
        torch.arange(inference.target_height, device=resolved_device, dtype=torch.long),
        torch.arange(inference.target_width, device=resolved_device, dtype=torch.long),
        indexing="ij",
    )
    for _ in range(max(0, int(solver_params.tile_graph_iterations))):
        changed = 0
        for parity in (0, 1):
            parity_mask_t = ((grid_y_t + grid_x_t) & 1) == parity
            local_cost_t = unary_cost_t.clone()
            if selected_t.shape[1] > 1:
                local_cost_t[:, 1:, :] += _pair_penalty_option_right_torch(
                    torch,
                    selected_t[:, :-1],
                    choice_rgba_t[:, 1:, :, :],
                    choice_deltas_t[:, 1:, :, :, :],
                    candidate_rgba_t,
                    candidate_deltas_t,
                    weight=solver_params.tile_graph_delta_weight,
                )
                local_cost_t[:, :-1, :] += _pair_penalty_option_left_torch(
                    torch,
                    choice_rgba_t[:, :-1, :, :],
                    choice_deltas_t[:, :-1, :, :, :],
                    selected_t[:, 1:],
                    candidate_rgba_t,
                    candidate_deltas_t,
                    weight=solver_params.tile_graph_delta_weight,
                )
            if selected_t.shape[0] > 1:
                local_cost_t[1:, :, :] += _pair_penalty_option_down_torch(
                    torch,
                    selected_t[:-1, :],
                    choice_rgba_t[1:, :, :, :],
                    choice_deltas_t[1:, :, :, :, :],
                    candidate_rgba_t,
                    candidate_deltas_t,
                    weight=solver_params.tile_graph_delta_weight,
                )
                local_cost_t[:-1, :, :] += _pair_penalty_option_up_torch(
                    torch,
                    choice_rgba_t[:-1, :, :, :],
                    choice_deltas_t[:-1, :, :, :, :],
                    selected_t[1:, :],
                    candidate_rgba_t,
                    candidate_deltas_t,
                    weight=solver_params.tile_graph_delta_weight,
                )
            best_choice_t = torch.argmin(local_cost_t, dim=2)
            best_selected_t = torch.take_along_dim(choice_indices_t, best_choice_t[..., None], dim=2)[..., 0]
            update_mask_t = parity_mask_t & (best_selected_t != selected_t)
            changed += int(update_mask_t.sum().item())
            selected_t = torch.where(parity_mask_t, best_selected_t, selected_t)
        loss_history.append(
            _assignment_score_torch(
                torch,
                selected_t,
                unary_cost_t,
                choice_indices_t,
                candidate_rgba_t,
                candidate_deltas_t,
                solver_params,
            )
        )
        if changed == 0:
            break

    initial_selected = initial_selected_t.detach().cpu().numpy().astype(np.int32)
    selected = selected_t.detach().cpu().numpy().astype(np.int32)
    initial_rgba = _assignment_rgba(model, initial_selected).astype(np.float32)
    target_rgba = _assignment_rgba(model, selected).astype(np.float32)
    diagnostics = {
        "mode": "tile-graph",
        "tile_graph_model_device": model.model_device,
        "tile_graph_solver_device": resolved_device,
        "tile_graph_proposal_mode": "lattice-topk",
        "tile_graph_component_count": model.component_count,
        "tile_graph_candidate_count": int(model.candidate_rgba.shape[0]),
        "tile_graph_edge_density": model.edge_density,
        "tile_graph_average_choices": model.average_choices,
        "tile_graph_initial_score": float(loss_history[0]),
        "tile_graph_final_score": float(loss_history[-1]),
    }
    return (
        SolverArtifacts(
            target_rgba=target_rgba,
            uv_field=_make_regular_uv(
                height=rgba.shape[0],
                width=rgba.shape[1],
                target_height=inference.target_height,
                target_width=inference.target_width,
                phase_x=inference.phase_x,
                phase_y=inference.phase_y,
            ),
            guidance_strength=np.zeros((inference.target_height, inference.target_width), dtype=np.float32),
            initial_rgba=initial_rgba,
            loss_history=[float(value) for value in loss_history],
        ),
        diagnostics,
    )
