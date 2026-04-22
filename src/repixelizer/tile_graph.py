from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import source_lattice_consistency_breakdown
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


def _rgba_luminance(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., 0] * 0.2126 + rgba[..., 1] * 0.7152 + rgba[..., 2] * 0.0722


def _segment_atomic_components_in_cell(
    *,
    linear_indices: np.ndarray,
    flat_y: np.ndarray,
    flat_x: np.ndarray,
    flat_rgba: np.ndarray,
    flat_premul: np.ndarray,
    flat_edge: np.ndarray,
    color_threshold: float,
    alpha_threshold: float,
) -> list[dict[str, float | int | np.ndarray]]:
    if linear_indices.size == 0:
        return []
    ys = flat_y[linear_indices]
    xs = flat_x[linear_indices]
    premul = flat_premul[linear_indices]
    rgba = flat_rgba[linear_indices]
    edges = flat_edge[linear_indices]
    y0 = int(np.min(ys))
    x0 = int(np.min(xs))
    local_y = ys - y0
    local_x = xs - x0
    local_index = -np.ones((int(np.max(local_y)) + 1, int(np.max(local_x)) + 1), dtype=np.int32)
    local_index[local_y, local_x] = np.arange(linear_indices.shape[0], dtype=np.int32)
    visited = np.zeros(linear_indices.shape[0], dtype=bool)
    components: list[dict[str, float | int | np.ndarray]] = []

    for seed in range(linear_indices.shape[0]):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = [seed]
        members: list[int] = []
        seed_premul = premul[seed]
        seed_alpha = float(rgba[seed, 3])
        while queue:
            pos = queue.pop()
            members.append(pos)
            cy = int(local_y[pos])
            cx = int(local_x[pos])
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or nx < 0 or ny >= local_index.shape[0] or nx >= local_index.shape[1]:
                    continue
                neighbor = int(local_index[ny, nx])
                if neighbor < 0 or visited[neighbor]:
                    continue
                color_diff = float(np.mean(np.abs(premul[neighbor] - seed_premul)))
                alpha_diff = abs(float(rgba[neighbor, 3]) - seed_alpha)
                if color_diff > color_threshold or alpha_diff > alpha_threshold:
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        member_array = np.asarray(members, dtype=np.int32)
        member_linear = linear_indices[member_array]
        member_y = ys[member_array].astype(np.float32)
        member_x = xs[member_array].astype(np.float32)
        centroid_y = float(np.mean(member_y))
        centroid_x = float(np.mean(member_x))
        centroid_dist = (member_y - centroid_y) ** 2 + (member_x - centroid_x) ** 2
        centroid_pick = int(np.argmin(centroid_dist))
        edge_pick = int(np.argmax(edges[member_array]))
        edge_peak = float(np.max(edges[member_array]))
        rep_slot = edge_pick if edge_peak > 0.0 else centroid_pick
        rep_linear = int(member_linear[rep_slot])
        rep_rgba = rgba[member_array][rep_slot].astype(np.float32)
        rgba_mean = np.mean(rgba[member_array], axis=0).astype(np.float32)
        components.append(
            {
                "rep_linear": rep_linear,
                "rep_rgba": rep_rgba,
                "coverage": float(member_array.size / max(1, linear_indices.size)),
                "edge_peak": edge_peak,
                "alpha_mean": float(np.mean(rgba[member_array, 3])),
                "luminance": float(_rgba_luminance(rgba_mean[None, :])[0]),
                "size": int(member_array.size),
            }
        )
    return components


def _select_atomic_cell_candidates(
    *,
    components: list[dict[str, float | int | np.ndarray]],
    allowed_candidates: int,
    min_component_coverage: float,
    reference_rgba: np.ndarray,
    edge_reference_rgba: np.ndarray,
    prefer_edge: bool,
) -> list[dict[str, float | int | np.ndarray]]:
    if not components:
        return []
    picked: list[dict[str, float | int | np.ndarray]] = []
    seen_linear: set[int] = set()

    def add(component: dict[str, float | int | np.ndarray]) -> None:
        rep_linear = int(component["rep_linear"])
        if rep_linear in seen_linear:
            return
        if float(component["coverage"]) < min_component_coverage and picked:
            return
        seen_linear.add(rep_linear)
        picked.append(component)

    by_area = sorted(components, key=lambda comp: (-float(comp["coverage"]), -float(comp["edge_peak"])))
    by_edge = sorted(components, key=lambda comp: (-float(comp["edge_peak"]), -float(comp["coverage"])))
    by_reference = sorted(
        components,
        key=lambda comp: float(np.mean(np.abs(np.asarray(comp["rep_rgba"], dtype=np.float32) - reference_rgba))),
    )
    by_edge_reference = sorted(
        components,
        key=lambda comp: float(np.mean(np.abs(np.asarray(comp["rep_rgba"], dtype=np.float32) - edge_reference_rgba))),
    )

    add(by_area[0])
    add(by_reference[0])
    if prefer_edge:
        add(by_edge_reference[0])
        add(by_edge[0])
    for component in by_area:
        add(component)
        if len(picked) >= allowed_candidates:
            break
    return picked[:allowed_candidates]


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

    source_t = torch.from_numpy(source_rgba).to(device=resolved_device, dtype=torch.float32)
    flat_rgba_t = source_t.reshape(-1, source_t.shape[-1])
    pixel_y_t, pixel_x_t = torch.meshgrid(
        torch.arange(height, device=resolved_device, dtype=torch.long),
        torch.arange(width, device=resolved_device, dtype=torch.long),
        indexing="ij",
    )
    flat_y_t = pixel_y_t.reshape(-1)
    flat_x_t = pixel_x_t.reshape(-1)
    lattice_indices_t = torch.from_numpy(source_reference.lattice_indices).to(device=resolved_device, dtype=torch.long).reshape(-1)
    edge_strength_flat_t = torch.from_numpy(source_reference.edge_strength.reshape(-1)).to(device=resolved_device, dtype=torch.float32)
    sharp_x_flat_t = torch.from_numpy(source_reference.sharp_x.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    sharp_y_flat_t = torch.from_numpy(source_reference.sharp_y.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    edge_peak_x_flat_t = torch.from_numpy(source_reference.edge_peak_x.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    edge_peak_y_flat_t = torch.from_numpy(source_reference.edge_peak_y.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    sharp_linear_t = sharp_y_flat_t * width + sharp_x_flat_t
    edge_linear_t = edge_peak_y_flat_t * width + edge_peak_x_flat_t

    flat_rgba_np = source_rgba.reshape(-1, source_rgba.shape[-1]).astype(np.float32)
    flat_premul_np = flat_rgba_np.copy()
    flat_premul_np[:, :3] *= flat_premul_np[:, 3:4]
    flat_edge_np = analysis.edge_map.reshape(-1).astype(np.float32)
    flat_idx_np = source_reference.lattice_indices.reshape(-1).astype(np.int32)
    counts_np = np.bincount(flat_idx_np, minlength=output_area).astype(np.int32)
    offsets_np = np.zeros(output_area + 1, dtype=np.int32)
    offsets_np[1:] = np.cumsum(counts_np, dtype=np.int32)
    order_np = np.argsort(flat_idx_np, kind="stable")
    flat_y_np, flat_x_np = np.divmod(np.arange(flat_idx_np.size, dtype=np.int64), width)
    flat_y_np = flat_y_np.astype(np.int32)
    flat_x_np = flat_x_np.astype(np.int32)
    sharp_linear_np = sharp_linear_t.detach().cpu().numpy().astype(np.int64)
    edge_linear_np = edge_linear_t.detach().cpu().numpy().astype(np.int64)
    flat_alpha_np = flat_rgba_np[:, 3].astype(np.float32)
    edge_rgba_np = flat_rgba_np[edge_linear_np]
    edge_strength_np = source_reference.edge_strength.reshape(-1).astype(np.float32)

    candidate_linear: list[int] = []
    candidate_coords: list[tuple[int, int]] = []
    candidate_area_ratio: list[float] = []
    candidate_coverage: list[float] = []
    choice_counts_list: list[int] = []
    component_total = 0
    edge_candidate_cap = max(max_candidates_per_coord, int(getattr(solver_params, "tile_graph_edge_candidates_per_coord", max_candidates_per_coord)))
    for flat_index in range(output_area):
        coord_y = flat_index // inference.target_width
        coord_x = flat_index % inference.target_width
        seen_pixels: set[int] = set()
        cell_start = len(candidate_linear)
        edge_cell = edge_strength_np[flat_index] >= solver_params.source_edge_detail_threshold
        allowed_candidates = edge_candidate_cap if edge_cell else max_candidates_per_coord

        def add_candidate(pixel_index: int, coverage: float) -> None:
            if pixel_index in seen_pixels or len(seen_pixels) >= allowed_candidates:
                return
            seen_pixels.add(pixel_index)
            candidate_linear.append(pixel_index)
            candidate_coords.append((coord_y, coord_x))
            candidate_area_ratio.append(1.0)
            candidate_coverage.append(float(np.clip(coverage, 0.0, 1.0)))

        linear_indices = order_np[offsets_np[flat_index] : offsets_np[flat_index + 1]]
        components = _segment_atomic_components_in_cell(
            linear_indices=linear_indices,
            flat_y=flat_y_np,
            flat_x=flat_x_np,
            flat_rgba=flat_rgba_np,
            flat_premul=flat_premul_np,
            flat_edge=flat_edge_np,
            color_threshold=float(getattr(solver_params, "tile_graph_component_color_threshold", 0.055)),
            alpha_threshold=float(getattr(solver_params, "tile_graph_component_alpha_threshold", 0.12)),
        )
        component_total += len(components)
        selected_components = _select_atomic_cell_candidates(
            components=components,
            allowed_candidates=allowed_candidates,
            min_component_coverage=float(getattr(solver_params, "tile_graph_component_min_coverage", 0.02)),
            reference_rgba=source_reference.sharp_rgba.reshape(-1, 4)[flat_index],
            edge_reference_rgba=edge_rgba_np[flat_index],
            prefer_edge=edge_cell,
        )
        add_candidate(int(sharp_linear_np[flat_index]), max(0.05, float(flat_alpha_np[int(sharp_linear_np[flat_index])])))
        if edge_cell:
            add_candidate(int(edge_linear_np[flat_index]), max(0.05, float(flat_alpha_np[int(edge_linear_np[flat_index])])))
        for component in selected_components:
            pixel_index = int(component["rep_linear"])
            add_candidate(pixel_index, float(component["coverage"]))
        if len(candidate_linear) == cell_start:
            add_candidate(int(sharp_linear_np[flat_index]), max(0.05, float(flat_alpha_np[int(sharp_linear_np[flat_index])])))
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

    edge_density = float(
        np.mean(source_reference.edge_strength >= solver_params.source_edge_detail_threshold)
    )

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
        reference_edge_rgba=edge_rgba_np.reshape(inference.target_height, inference.target_width, -1).astype(np.float32),
        edge_strength=source_reference.edge_strength.astype(np.float32),
        component_count=component_total,
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
    initial_source_fidelity = source_lattice_consistency_breakdown(
        rgba,
        initial_rgba,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
    )["score"]
    final_source_fidelity = source_lattice_consistency_breakdown(
        rgba,
        target_rgba,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
    )["score"]
    kept_initial_assignment = False
    if final_source_fidelity > initial_source_fidelity + 1e-6:
        target_rgba = initial_rgba.copy()
        final_source_fidelity = initial_source_fidelity
        kept_initial_assignment = True
    diagnostics = {
        "mode": "tile-graph",
        "tile_graph_model_device": model.model_device,
        "tile_graph_solver_device": resolved_device,
        "tile_graph_proposal_mode": "atomic-components",
        "tile_graph_component_count": model.component_count,
        "tile_graph_candidate_count": int(model.candidate_rgba.shape[0]),
        "tile_graph_edge_density": model.edge_density,
        "tile_graph_average_choices": model.average_choices,
        "tile_graph_initial_score": float(loss_history[0]),
        "tile_graph_final_score": float(loss_history[-1]),
        "tile_graph_initial_source_fidelity": float(initial_source_fidelity),
        "tile_graph_final_source_fidelity": float(final_source_fidelity),
        "tile_graph_kept_initial_assignment": kept_initial_assignment,
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
