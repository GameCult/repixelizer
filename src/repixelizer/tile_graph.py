from __future__ import annotations

from collections import deque
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
    component_count: int
    edge_density: float
    average_choices: float


@dataclass(slots=True)
class _Component:
    label: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    available_mask: np.ndarray
    available_integral: np.ndarray
    area: int
    centroid_y: float
    centroid_x: float


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


def _component_integral(mask: np.ndarray) -> np.ndarray:
    integral = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int32), axis=0), axis=1)
    return integral


def _integral_sum(integral: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    return int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def _window_bounds(center_y: float, center_x: float, cell_h: float, cell_w: float, height: int, width: int) -> tuple[int, int, int, int]:
    y0 = int(np.floor(center_y - cell_h * 0.5))
    x0 = int(np.floor(center_x - cell_w * 0.5))
    y1 = int(np.ceil(center_y + cell_h * 0.5))
    x1 = int(np.ceil(center_x + cell_w * 0.5))
    y0 = max(0, min(height - 1, y0))
    x0 = max(0, min(width - 1, x0))
    y1 = max(y0 + 1, min(height, y1))
    x1 = max(x0 + 1, min(width, x1))
    return y0, x0, y1, x1


def _source_to_cell_coord(
    center_y: float,
    center_x: float,
    *,
    cell_h: float,
    cell_w: float,
    target_height: int,
    target_width: int,
    phase_y: float,
    phase_x: float,
) -> tuple[int, int]:
    x_idx = int(np.floor((center_x + 0.5) / max(cell_w, 1e-4) - phase_x))
    y_idx = int(np.floor((center_y + 0.5) / max(cell_h, 1e-4) - phase_y))
    x_idx = max(0, min(target_width - 1, x_idx))
    y_idx = max(0, min(target_height - 1, y_idx))
    return y_idx, x_idx


def _sample_source_pixel(source_rgba: np.ndarray, center_y: float, center_x: float) -> np.ndarray:
    y = int(np.clip(np.rint(center_y), 0, max(0, source_rgba.shape[0] - 1)))
    x = int(np.clip(np.rint(center_x), 0, max(0, source_rgba.shape[1] - 1)))
    return source_rgba[y, x].astype(np.float32)


def _window_component_coverage(component: _Component, center_y: float, center_x: float, cell_h: float, cell_w: float) -> float:
    y0, x0, y1, x1 = _window_bounds(
        center_y,
        center_x,
        cell_h,
        cell_w,
        component.available_mask.shape[0] + component.bbox[0],
        component.available_mask.shape[1] + component.bbox[1],
    )
    bbox_y0, bbox_x0, _, _ = component.bbox
    local_y0 = max(0, y0 - bbox_y0)
    local_x0 = max(0, x0 - bbox_x0)
    local_y1 = min(component.available_mask.shape[0], y1 - bbox_y0)
    local_x1 = min(component.available_mask.shape[1], x1 - bbox_x0)
    if local_y1 <= local_y0 or local_x1 <= local_x0:
        return 0.0
    covered = _integral_sum(component.available_integral, local_y0, local_x0, local_y1, local_x1)
    total = max(1, (local_y1 - local_y0) * (local_x1 - local_x0))
    return float(covered / total)


def _nearest_component_pixel(component: _Component, center_y: float, center_x: float) -> tuple[float, float]:
    bbox_y0, bbox_x0, _, _ = component.bbox
    local = component.coords - np.asarray([bbox_y0, bbox_x0], dtype=np.int32)[None, :]
    available = component.available_mask[local[:, 0], local[:, 1]]
    coords = component.coords[available]
    if coords.shape[0] == 0:
        return float(center_y), float(center_x)
    deltas = coords.astype(np.float32) - np.asarray([center_y, center_x], dtype=np.float32)[None, :]
    best = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    return float(coords[best, 0]), float(coords[best, 1])


def _consume_component_window(component: _Component, center_y: float, center_x: float, cell_h: float, cell_w: float) -> None:
    y0, x0, y1, x1 = _window_bounds(
        center_y,
        center_x,
        cell_h,
        cell_w,
        component.available_mask.shape[0] + component.bbox[0],
        component.available_mask.shape[1] + component.bbox[1],
    )
    bbox_y0, bbox_x0, _, _ = component.bbox
    local_y0 = max(0, y0 - bbox_y0)
    local_x0 = max(0, x0 - bbox_x0)
    local_y1 = min(component.available_mask.shape[0], y1 - bbox_y0)
    local_x1 = min(component.available_mask.shape[1], x1 - bbox_x0)
    if local_y1 <= local_y0 or local_x1 <= local_x0:
        return
    component.available_mask[local_y0:local_y1, local_x0:local_x1] = False
    component.available_integral = _component_integral(component.available_mask)


def _best_stepped_center(
    component: _Component,
    center_y: float,
    center_x: float,
    *,
    step_y: float,
    step_x: float,
    cell_h: float,
    cell_w: float,
) -> tuple[float, float, float] | None:
    offsets_y = np.asarray([0.0, -step_y * 0.35, step_y * 0.35], dtype=np.float32)
    offsets_x = np.asarray([0.0, -step_x * 0.35, step_x * 0.35], dtype=np.float32)
    best: tuple[float, float, float] | None = None
    for dy in offsets_y:
        for dx in offsets_x:
            probe_y, probe_x = _nearest_component_pixel(component, center_y + float(dy), center_x + float(dx))
            coverage = _window_component_coverage(component, probe_y, probe_x, cell_h, cell_w)
            if best is None or coverage > best[2]:
                best = (probe_y, probe_x, coverage)
    return best


def _directional_pixel(
    source_rgba: np.ndarray,
    component: _Component,
    center_y: float,
    center_x: float,
    *,
    dy: int,
    dx: int,
    step_y: float,
    step_x: float,
    cell_h: float,
    cell_w: float,
    coverage_threshold: float,
) -> np.ndarray:
    best = _best_stepped_center(
        component,
        center_y + dy * step_y,
        center_x + dx * step_x,
        step_y=step_y,
        step_x=step_x,
        cell_h=cell_h,
        cell_w=cell_w,
    )
    if best is not None:
        probe_y, probe_x, coverage = best
        if coverage >= coverage_threshold:
            return _sample_source_pixel(source_rgba, probe_y, probe_x)
    return _sample_source_pixel(source_rgba, center_y + dy * step_y, center_x + dx * step_x)


def _extract_components(cluster_map: np.ndarray, alpha_map: np.ndarray, alpha_threshold: float) -> list[_Component]:
    valid = alpha_map >= alpha_threshold
    visited = np.zeros(valid.shape, dtype=bool)
    components: list[_Component] = []
    height, width = valid.shape
    for y in range(height):
        for x in range(width):
            if not valid[y, x] or visited[y, x]:
                continue
            label = int(cluster_map[y, x])
            queue = deque([(y, x)])
            visited[y, x] = True
            coords: list[tuple[int, int]] = []
            min_y = max_y = y
            min_x = max_x = x
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not valid[ny, nx] or int(cluster_map[ny, nx]) != label:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            coords_array = np.asarray(coords, dtype=np.int32)
            mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=bool)
            local = coords_array - np.asarray([min_y, min_x], dtype=np.int32)[None, :]
            mask[local[:, 0], local[:, 1]] = True
            components.append(
                _Component(
                    label=label,
                    coords=coords_array,
                    bbox=(min_y, min_x, max_y + 1, max_x + 1),
                    mask=mask,
                    available_mask=mask.copy(),
                    available_integral=_component_integral(mask),
                    area=int(coords_array.shape[0]),
                    centroid_y=float(np.mean(coords_array[:, 0])),
                    centroid_x=float(np.mean(coords_array[:, 1])),
                )
            )
    return components


def build_tile_graph_model(
    source_rgba: np.ndarray,
    *,
    inference: InferenceResult,
    analysis: SourceAnalysis,
    solver_params: SolverHyperParams | None = None,
) -> TileGraphModel:
    solver_params = solver_params or SolverHyperParams()
    source_reference = build_source_lattice_reference(
        source_rgba,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
        alpha_threshold=solver_params.alpha_transparent_threshold,
        edge_hint=analysis.edge_map,
    )
    height, width = source_rgba.shape[:2]
    cell_h = height / max(1, inference.target_height)
    cell_w = width / max(1, inference.target_width)
    target_area = cell_h * cell_w
    output_area = max(1, inference.target_height * inference.target_width)
    extracted_candidate_limit = max(
        int(solver_params.tile_graph_max_candidates),
        output_area * max(1, int(solver_params.tile_graph_max_candidates_per_coord)),
    )
    components = _extract_components(analysis.cluster_map, analysis.alpha_map, solver_params.alpha_transparent_threshold)
    components.sort(key=lambda component: (abs(component.area / max(target_area, 1e-4) - 1.0), -component.area))

    candidate_rgba: list[np.ndarray] = []
    candidate_coords: list[tuple[int, int]] = []
    candidate_area_ratio: list[float] = []
    candidate_coverage: list[float] = []
    candidate_deltas: list[np.ndarray] = []

    extracted_coord_counts: dict[tuple[int, int], int] = {}
    step_y = max(1.0, cell_h)
    step_x = max(1.0, cell_w)
    for component in components:
        if not np.any(component.available_mask):
            continue
        area_ratio = float(component.area / max(target_area, 1e-4))
        if area_ratio < solver_params.tile_graph_component_min_area_ratio:
            continue
        per_component_limit = max(1, int(np.ceil(max(1.0, area_ratio))))
        seed_y, seed_x = _nearest_component_pixel(component, component.centroid_y, component.centroid_x)
        queue = deque([(seed_y, seed_x)])
        local_seen: set[tuple[int, int]] = set()
        added = 0
        while queue and added < per_component_limit and len(candidate_rgba) < extracted_candidate_limit:
            center_y, center_x = queue.popleft()
            coord = _source_to_cell_coord(
                center_y,
                center_x,
                cell_h=cell_h,
                cell_w=cell_w,
                target_height=inference.target_height,
                target_width=inference.target_width,
                phase_y=inference.phase_y,
                phase_x=inference.phase_x,
            )
            if coord in local_seen:
                continue
            if extracted_coord_counts.get(coord, 0) >= solver_params.tile_graph_max_candidates_per_coord:
                continue
            local_seen.add(coord)
            coverage = _window_component_coverage(component, center_y, center_x, cell_h, cell_w)
            if coverage < solver_params.tile_graph_window_coverage_threshold and added > 0:
                continue
            center_rgba = _sample_source_pixel(source_rgba, center_y, center_x)
            delta_features = np.zeros((4, 4), dtype=np.float32)
            for direction, dy, dx in _DIRS:
                neighbor_rgba = _directional_pixel(
                    source_rgba,
                    component,
                    center_y,
                    center_x,
                    dy=dy,
                    dx=dx,
                    step_y=step_y,
                    step_x=step_x,
                    cell_h=cell_h,
                    cell_w=cell_w,
                    coverage_threshold=solver_params.tile_graph_window_coverage_threshold,
                )
                delta_features[direction] = neighbor_rgba - center_rgba
            candidate_rgba.append(center_rgba)
            candidate_coords.append(coord)
            candidate_area_ratio.append(area_ratio)
            candidate_coverage.append(coverage)
            candidate_deltas.append(delta_features)
            extracted_coord_counts[coord] = extracted_coord_counts.get(coord, 0) + 1
            _consume_component_window(component, center_y, center_x, cell_h, cell_w)
            added += 1
            if area_ratio < solver_params.tile_graph_large_component_ratio:
                continue
            for _, dy, dx in _DIRS:
                best = _best_stepped_center(
                    component,
                    center_y + dy * step_y,
                    center_x + dx * step_x,
                    step_y=step_y,
                    step_x=step_x,
                    cell_h=cell_h,
                    cell_w=cell_w,
                )
                if best is None:
                    continue
                probe_y, probe_x, probe_coverage = best
                if probe_coverage < solver_params.tile_graph_window_coverage_threshold:
                    continue
                probe_coord = _source_to_cell_coord(
                    probe_y,
                    probe_x,
                    cell_h=cell_h,
                    cell_w=cell_w,
                    target_height=inference.target_height,
                    target_width=inference.target_width,
                    phase_y=inference.phase_y,
                    phase_x=inference.phase_x,
                )
                if (
                    probe_coord not in local_seen
                    and extracted_coord_counts.get(probe_coord, 0) < solver_params.tile_graph_max_candidates_per_coord
                ):
                    queue.append((probe_y, probe_x))

    def add_reference_candidate(coord_y: int, coord_x: int) -> None:
        coord = (coord_y, coord_x)
        delta_features = np.zeros((4, 4), dtype=np.float32)
        center_rgba = source_reference.sharp_rgba[coord_y, coord_x].astype(np.float32)
        for direction, dy, dx in _DIRS:
            neighbor_y = min(max(coord_y + dy, 0), inference.target_height - 1)
            neighbor_x = min(max(coord_x + dx, 0), inference.target_width - 1)
            delta_features[direction] = source_reference.sharp_rgba[neighbor_y, neighbor_x] - center_rgba
        candidate_rgba.append(center_rgba)
        candidate_coords.append(coord)
        candidate_area_ratio.append(1.0)
        candidate_coverage.append(1.0)
        candidate_deltas.append(delta_features)

    for coord_y in range(inference.target_height):
        for coord_x in range(inference.target_width):
            add_reference_candidate(coord_y, coord_x)

    candidate_rgba_np = np.asarray(candidate_rgba, dtype=np.float32)
    candidate_coords_np = np.asarray(candidate_coords, dtype=np.int32)
    candidate_area_ratio_np = np.asarray(candidate_area_ratio, dtype=np.float32)
    candidate_coverage_np = np.asarray(candidate_coverage, dtype=np.float32)
    candidate_deltas_np = np.asarray(candidate_deltas, dtype=np.float32)

    coord_to_indices: dict[tuple[int, int], list[int]] = {}
    for index, coord in enumerate(candidate_coords_np.tolist()):
        coord_to_indices.setdefault((int(coord[0]), int(coord[1])), []).append(index)
    cell_candidate_offsets = np.zeros(output_area + 1, dtype=np.int32)
    cell_candidate_indices: list[int] = []
    for flat_index in range(output_area):
        coord_y = flat_index // inference.target_width
        coord_x = flat_index % inference.target_width
        indices = coord_to_indices.get((coord_y, coord_x), [])
        cell_candidate_indices.extend(indices)
        cell_candidate_offsets[flat_index + 1] = len(cell_candidate_indices)
    cell_candidate_indices_np = np.asarray(cell_candidate_indices, dtype=np.int32)
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
        component_count=len(components),
        edge_density=edge_density,
        average_choices=average_choices,
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
            candidates = model.candidate_rgba[indices]
            color_error = (
                np.mean(np.abs(reference_sharp[None, :] - candidates), axis=-1) * 0.85
                + np.mean(np.abs(reference_mean[None, :] - candidates), axis=-1) * 0.15
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

    choice_rgba_t = candidate_rgba_t[choice_indices_t]
    choice_deltas_t = candidate_deltas_t[choice_indices_t]
    color_error = (
        (reference_sharp_t[..., None, :] - choice_rgba_t).abs().mean(dim=-1) * 0.85
        + (reference_mean_t[..., None, :] - choice_rgba_t).abs().mean(dim=-1) * 0.15
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
        "tile_graph_solver_device": resolved_device,
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
