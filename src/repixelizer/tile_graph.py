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


@dataclass(slots=True)
class TileGraphModel:
    candidate_rgba: np.ndarray
    candidate_coords: np.ndarray
    candidate_area_ratio: np.ndarray
    candidate_coverage: np.ndarray
    pair_penalty: np.ndarray
    reference_mean_rgba: np.ndarray
    reference_sharp_rgba: np.ndarray
    component_count: int
    edge_density: float


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
    components = _extract_components(analysis.cluster_map, analysis.alpha_map, solver_params.alpha_transparent_threshold)
    components.sort(key=lambda component: (abs(component.area / max(target_area, 1e-4) - 1.0), -component.area))

    candidate_rgba: list[np.ndarray] = [np.zeros(4, dtype=np.float32)]
    candidate_coords: list[tuple[int, int]] = [(0, 0)]
    candidate_area_ratio: list[float] = [0.0]
    candidate_coverage: list[float] = [1.0]
    candidate_deltas: list[np.ndarray] = [np.zeros((4, 4), dtype=np.float32)]

    coord_counts: dict[tuple[int, int], int] = {}
    step_y = max(1.0, cell_h)
    step_x = max(1.0, cell_w)
    for component in components:
        if not np.any(component.available_mask):
            continue
        area_ratio = float(component.area / max(target_area, 1e-4))
        if area_ratio < solver_params.tile_graph_component_min_area_ratio:
            continue
        per_component_limit = max(1, min(24, int(np.ceil(max(1.0, area_ratio)))))
        seed_y, seed_x = _nearest_component_pixel(component, component.centroid_y, component.centroid_x)
        queue = deque([(seed_y, seed_x)])
        local_seen: set[tuple[int, int]] = set()
        added = 0
        while queue and added < per_component_limit and len(candidate_rgba) < solver_params.tile_graph_max_candidates:
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
            if coord_counts.get(coord, 0) >= solver_params.tile_graph_max_candidates_per_coord:
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
            coord_counts[coord] = coord_counts.get(coord, 0) + 1
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
                    and coord_counts.get(probe_coord, 0) < solver_params.tile_graph_max_candidates_per_coord
                ):
                    queue.append((probe_y, probe_x))

    def add_reference_candidate(coord_y: int, coord_x: int) -> None:
        if len(candidate_rgba) >= solver_params.tile_graph_max_candidates:
            return
        coord = (coord_y, coord_x)
        if coord_counts.get(coord, 0) >= max(1, solver_params.tile_graph_max_candidates_per_coord):
            return
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
        coord_counts[coord] = coord_counts.get(coord, 0) + 1

    if len(candidate_rgba) < solver_params.tile_graph_max_candidates:
        remaining = solver_params.tile_graph_max_candidates - len(candidate_rgba)
        stride = max(1, int(np.sqrt((inference.target_height * inference.target_width) / max(1, remaining))))
        for coord_y in range(0, inference.target_height, stride):
            for coord_x in range(0, inference.target_width, stride):
                add_reference_candidate(coord_y, coord_x)
                if len(candidate_rgba) >= solver_params.tile_graph_max_candidates:
                    break
            if len(candidate_rgba) >= solver_params.tile_graph_max_candidates:
                break
    if len(candidate_rgba) < solver_params.tile_graph_max_candidates:
        priority = (
            source_reference.edge_strength
            + np.clip(source_reference.cell_dispersion / max(source_reference.dispersion, 1e-4), 0.0, 4.0) * 0.35
        )
        for flat_index in np.argsort(priority.reshape(-1))[::-1]:
            coord_y = int(flat_index // inference.target_width)
            coord_x = int(flat_index % inference.target_width)
            add_reference_candidate(coord_y, coord_x)
            if len(candidate_rgba) >= solver_params.tile_graph_max_candidates:
                break

    candidate_rgba_np = np.asarray(candidate_rgba, dtype=np.float32)
    candidate_coords_np = np.asarray(candidate_coords, dtype=np.int32)
    candidate_area_ratio_np = np.asarray(candidate_area_ratio, dtype=np.float32)
    candidate_coverage_np = np.asarray(candidate_coverage, dtype=np.float32)
    candidate_deltas_np = np.asarray(candidate_deltas, dtype=np.float32)

    candidate_count = candidate_rgba_np.shape[0]
    adjacency_counts = np.zeros((4, candidate_count, candidate_count), dtype=np.float32)
    coord_to_indices: dict[tuple[int, int], list[int]] = {}
    for index, coord in enumerate(candidate_coords_np.tolist()):
        coord_to_indices.setdefault((int(coord[0]), int(coord[1])), []).append(index)
    for index, (coord_y, coord_x) in enumerate(candidate_coords_np.tolist()):
        for direction, dy, dx in _DIRS:
            for neighbor_index in coord_to_indices.get((int(coord_y + dy), int(coord_x + dx)), []):
                adjacency_counts[direction, index, neighbor_index] += 1.0

    pair_penalty = np.zeros((4, candidate_count, candidate_count), dtype=np.float32)
    observed_delta = candidate_rgba_np[None, :, :] - candidate_rgba_np[:, None, :]
    for direction, _, _ in _DIRS:
        row_sum = adjacency_counts[direction].sum(axis=1, keepdims=True)
        probability = (adjacency_counts[direction] + 1.0) / (row_sum + float(candidate_count))
        count_penalty = -np.log(probability)
        count_penalty = count_penalty / max(float(np.max(count_penalty)), 1e-4)
        expected = candidate_deltas_np[:, direction, :][:, None, :]
        reverse = candidate_deltas_np[:, _OPPOSITE[direction], :][None, :, :]
        delta_penalty = np.mean(np.abs(observed_delta - expected), axis=-1)
        reverse_penalty = np.mean(np.abs(-observed_delta - reverse), axis=-1)
        delta_penalty = (delta_penalty + reverse_penalty) * 0.5
        delta_penalty = delta_penalty / max(float(np.max(delta_penalty)), 1e-4)
        pair_penalty[direction] = (
            count_penalty * solver_params.tile_graph_count_weight
            + delta_penalty * solver_params.tile_graph_delta_weight
        ).astype(np.float32)

    edge_density = float(np.mean(adjacency_counts > 0.0)) if candidate_count > 0 else 0.0
    return TileGraphModel(
        candidate_rgba=candidate_rgba_np,
        candidate_coords=candidate_coords_np,
        candidate_area_ratio=candidate_area_ratio_np,
        candidate_coverage=candidate_coverage_np,
        pair_penalty=pair_penalty,
        reference_mean_rgba=source_reference.mean_rgba.astype(np.float32),
        reference_sharp_rgba=source_reference.sharp_rgba.astype(np.float32),
        component_count=len(components),
        edge_density=edge_density,
    )


def _tile_graph_unary_cost(model: TileGraphModel, solver_params: SolverHyperParams) -> np.ndarray:
    height, width = model.reference_sharp_rgba.shape[:2]
    color_error = (
        np.mean(np.abs(model.reference_sharp_rgba[..., None, :] - model.candidate_rgba[None, None, :, :]), axis=-1) * 0.85
        + np.mean(np.abs(model.reference_mean_rgba[..., None, :] - model.candidate_rgba[None, None, :, :]), axis=-1) * 0.15
    )
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    coord_error = (
        np.abs(grid_y[..., None] - model.candidate_coords[None, None, :, 0].astype(np.float32))
        + np.abs(grid_x[..., None] - model.candidate_coords[None, None, :, 1].astype(np.float32))
    ) / max(1.0, float(height + width))
    nearest_coord_error = np.min(coord_error, axis=2, keepdims=True)
    coord_gate = coord_error <= nearest_coord_error + (
        float(solver_params.tile_graph_coord_gate_slack) / max(1.0, float(height + width))
    )
    coord_gate[..., 0] = model.reference_mean_rgba[..., 3] <= 0.35
    area_error = np.abs(np.log(np.clip(model.candidate_area_ratio, 1e-4, None) + 1e-4))[None, None, :]
    return (
        color_error
        + coord_error * solver_params.tile_graph_coordinate_weight
        + (~coord_gate).astype(np.float32) * solver_params.tile_graph_coord_gate_penalty
        + area_error * solver_params.tile_graph_area_weight
    ).astype(np.float32)


def _assignment_rgba(model: TileGraphModel, selected: np.ndarray) -> np.ndarray:
    return model.candidate_rgba[selected]


def _assignment_score(selected: np.ndarray, unary_cost: np.ndarray, pair_penalty: np.ndarray) -> float:
    gathered = np.take_along_axis(unary_cost, selected[..., None], axis=2)[..., 0]
    score = float(np.mean(gathered))
    if selected.shape[1] > 1:
        score += float(np.mean(pair_penalty[_RIGHT, selected[:, :-1], selected[:, 1:]]))
    if selected.shape[0] > 1:
        score += float(np.mean(pair_penalty[_DOWN, selected[:-1, :], selected[1:, :]]))
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
    del steps, seed, device
    solver_params = solver_params or SolverHyperParams()
    model = build_tile_graph_model(
        rgba,
        inference=inference,
        analysis=analysis,
        solver_params=solver_params,
    )
    unary_cost = _tile_graph_unary_cost(model, solver_params)
    selected = np.argmin(unary_cost, axis=2).astype(np.int32)
    initial_selected = selected.copy()
    loss_history = [_assignment_score(selected, unary_cost, model.pair_penalty)]
    for _ in range(max(0, int(solver_params.tile_graph_iterations))):
        changed = 0
        for parity in (0, 1):
            for y in range(selected.shape[0]):
                for x in range((y + parity) % 2, selected.shape[1], 2):
                    cost = unary_cost[y, x].copy()
                    if x > 0:
                        cost += model.pair_penalty[_RIGHT, selected[y, x - 1], :]
                    if x + 1 < selected.shape[1]:
                        cost += model.pair_penalty[_RIGHT, :, selected[y, x + 1]]
                    if y > 0:
                        cost += model.pair_penalty[_DOWN, selected[y - 1, x], :]
                    if y + 1 < selected.shape[0]:
                        cost += model.pair_penalty[_DOWN, :, selected[y + 1, x]]
                    best = int(np.argmin(cost))
                    if best != int(selected[y, x]):
                        selected[y, x] = best
                        changed += 1
        loss_history.append(_assignment_score(selected, unary_cost, model.pair_penalty))
        if changed == 0:
            break

    initial_rgba = _assignment_rgba(model, initial_selected).astype(np.float32)
    target_rgba = _assignment_rgba(model, selected).astype(np.float32)
    diagnostics = {
        "mode": "tile-graph",
        "tile_graph_component_count": model.component_count,
        "tile_graph_candidate_count": int(model.candidate_rgba.shape[0]),
        "tile_graph_edge_density": model.edge_density,
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
