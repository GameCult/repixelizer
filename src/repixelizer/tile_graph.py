from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .metrics import source_lattice_consistency_breakdown
from .params import SolverHyperParams
from .types import InferenceResult, SolverArtifacts, TileGraphSourceAnalysis

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
    candidate_area_ratio: np.ndarray
    candidate_coverage: np.ndarray
    candidate_edge_peak: np.ndarray
    candidate_neighbor_rgba: np.ndarray
    candidate_neighbor_mask: np.ndarray
    cell_candidate_offsets: np.ndarray
    cell_candidate_indices: np.ndarray
    cell_mean_rgba: np.ndarray
    cell_alpha_mean: np.ndarray
    cell_edge_strength: np.ndarray


@dataclass(slots=True)
class TileGraphBuildStats:
    component_count: int
    candidate_count: int
    edge_density: float
    average_choices: float
    model_device: str
    cache_hit: bool = False


_TILE_GRAPH_MODEL_CACHE_MAX_ENTRIES = 8
_TILE_GRAPH_MODEL_CACHE: OrderedDict[tuple[object, ...], TileGraphModel] = OrderedDict()
_TILE_GRAPH_BUILD_STATS_CACHE: OrderedDict[tuple[object, ...], TileGraphBuildStats] = OrderedDict()


def clear_tile_graph_model_cache() -> None:
    _TILE_GRAPH_MODEL_CACHE.clear()
    _TILE_GRAPH_BUILD_STATS_CACHE.clear()


def _hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()


def _tile_graph_model_cache_key(
    source_rgba: np.ndarray,
    *,
    inference: InferenceResult,
    solver_params: SolverHyperParams,
    device: str,
) -> tuple[object, ...]:
    build_param_names = (
        "alpha_transparent_threshold",
        "source_edge_detail_threshold",
        "tile_graph_max_candidates_per_coord",
        "tile_graph_edge_candidates_per_coord",
        "tile_graph_component_color_threshold",
        "tile_graph_component_alpha_threshold",
        "tile_graph_source_region_min_area_ratio",
        "tile_graph_source_region_window_coverage",
        "tile_graph_stroke_linearity_threshold",
        "tile_graph_stroke_step_scale",
        "tile_graph_stroke_minor_limit_scale",
        "tile_graph_area_weight",
        "tile_graph_alpha_weight",
        "tile_graph_coverage_weight",
        "tile_graph_edge_peak_weight",
        "tile_graph_adjacency_weight",
    )
    build_signature = tuple((name, getattr(solver_params, name)) for name in build_param_names)
    return (
        _hash_array(source_rgba),
        source_rgba.shape,
        str(source_rgba.dtype),
        int(inference.target_width),
        int(inference.target_height),
        float(inference.phase_x),
        float(inference.phase_y),
        device,
        build_signature,
    )


def _get_cached_tile_graph_model(cache_key: tuple[object, ...]) -> tuple[TileGraphModel, TileGraphBuildStats] | None:
    cached = _TILE_GRAPH_MODEL_CACHE.get(cache_key)
    cached_stats = _TILE_GRAPH_BUILD_STATS_CACHE.get(cache_key)
    if cached is None or cached_stats is None:
        return None
    _TILE_GRAPH_MODEL_CACHE.move_to_end(cache_key)
    _TILE_GRAPH_BUILD_STATS_CACHE.move_to_end(cache_key)
    return cached, cached_stats


def _store_cached_tile_graph_model(
    cache_key: tuple[object, ...],
    model: TileGraphModel,
    stats: TileGraphBuildStats,
) -> None:
    _TILE_GRAPH_MODEL_CACHE[cache_key] = model
    _TILE_GRAPH_MODEL_CACHE.move_to_end(cache_key)
    _TILE_GRAPH_BUILD_STATS_CACHE[cache_key] = stats
    _TILE_GRAPH_BUILD_STATS_CACHE.move_to_end(cache_key)
    while len(_TILE_GRAPH_MODEL_CACHE) > _TILE_GRAPH_MODEL_CACHE_MAX_ENTRIES:
        old_key, _old_model = _TILE_GRAPH_MODEL_CACHE.popitem(last=False)
        _TILE_GRAPH_BUILD_STATS_CACHE.pop(old_key, None)


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


def _component_axis_stats(member_x: np.ndarray, member_y: np.ndarray) -> tuple[float, float, float, float, float]:
    if member_x.size <= 1:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    centered_x = member_x.astype(np.float32) - float(np.mean(member_x))
    centered_y = member_y.astype(np.float32) - float(np.mean(member_y))
    covariance = np.asarray(
        [
            [float(np.mean(centered_x * centered_x)), float(np.mean(centered_x * centered_y))],
            [float(np.mean(centered_x * centered_y)), float(np.mean(centered_y * centered_y))],
        ],
        dtype=np.float32,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    minor_value = float(max(0.0, eigenvalues[order[0]]))
    major_value = float(max(0.0, eigenvalues[order[1]]))
    major_vector = eigenvectors[:, order[1]].astype(np.float32)
    norm = float(np.linalg.norm(major_vector))
    if norm <= 1e-6:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    major_vector /= norm
    axis_x = float(major_vector[0])
    axis_y = float(major_vector[1])
    projections = centered_x * axis_x + centered_y * axis_y
    perpendicular = centered_x * (-axis_y) + centered_y * axis_x
    major_span = float(np.max(projections) - np.min(projections)) if projections.size else 0.0
    minor_span = float(np.max(perpendicular) - np.min(perpendicular)) if perpendicular.size else 0.0
    linearity = float(np.clip(1.0 - (minor_value / max(major_value, 1e-6)), 0.0, 1.0))
    return axis_x, axis_y, linearity, major_span, minor_span


def _project_source_point_to_output_coord(
    *,
    x: float,
    y: float,
    cell_w: float,
    cell_h: float,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
) -> tuple[int, int, int]:
    coord_x = int(np.clip(np.floor((x + 0.5) / max(cell_w, 1e-6) - phase_x), 0, max(0, target_width - 1)))
    coord_y = int(np.clip(np.floor((y + 0.5) / max(cell_h, 1e-6) - phase_y), 0, max(0, target_height - 1)))
    return coord_y, coord_x, coord_y * target_width + coord_x


def _segment_atomic_source_regions_cpu(
    *,
    width: int,
    height: int,
    flat_rgba: np.ndarray,
    flat_premul: np.ndarray,
    flat_edge: np.ndarray,
    alpha_floor: float,
    color_threshold: float,
    alpha_threshold: float,
) -> list[dict[str, float | int | np.ndarray]]:
    alpha_map = flat_rgba[:, 3].reshape(height, width).astype(np.float32)
    opaque = alpha_map >= alpha_floor
    premul_map = flat_premul.reshape(height, width, flat_premul.shape[-1]).astype(np.float32)
    join_right = np.zeros((height, max(0, width - 1)), dtype=bool)
    join_down = np.zeros((max(0, height - 1), width), dtype=bool)
    if width > 1:
        right_color = np.mean(np.abs(premul_map[:, 1:, :] - premul_map[:, :-1, :]), axis=-1)
        right_alpha = np.abs(alpha_map[:, 1:] - alpha_map[:, :-1])
        join_right = opaque[:, 1:] & opaque[:, :-1] & (right_color <= color_threshold) & (right_alpha <= alpha_threshold)
    if height > 1:
        down_color = np.mean(np.abs(premul_map[1:, :, :] - premul_map[:-1, :, :]), axis=-1)
        down_alpha = np.abs(alpha_map[1:, :] - alpha_map[:-1, :])
        join_down = opaque[1:, :] & opaque[:-1, :] & (down_color <= color_threshold) & (down_alpha <= alpha_threshold)

    pixel_count = width * height
    visited = ~opaque.reshape(-1)
    components: list[dict[str, float | int | np.ndarray]] = []

    for seed in range(pixel_count):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = [seed]
        members: list[int] = []
        while queue:
            pos = queue.pop()
            members.append(pos)
            y = pos // width
            x = pos % width
            if x + 1 < width and join_right[y, x]:
                neighbor = pos + 1
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
            if x > 0 and join_right[y, x - 1]:
                neighbor = pos - 1
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
            if y + 1 < height and join_down[y, x]:
                neighbor = pos + width
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
            if y > 0 and join_down[y - 1, x]:
                neighbor = pos - width
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        member_array = np.asarray(members, dtype=np.int32)
        member_y = (member_array // width).astype(np.float32)
        member_x = (member_array % width).astype(np.float32)
        centroid_y = float(np.mean(member_y))
        centroid_x = float(np.mean(member_x))
        axis_x, axis_y, linearity, major_span, minor_span = _component_axis_stats(member_x, member_y)
        centroid_dist = (member_y - centroid_y) ** 2 + (member_x - centroid_x) ** 2
        centroid_pick = int(np.argmin(centroid_dist))
        edge_pick = int(np.argmax(flat_edge[member_array]))
        edge_peak = float(np.max(flat_edge[member_array]))
        components.append(
            {
                "member_linear": member_array,
                "edge_peak_linear": int(member_array[edge_pick]),
                "edge_peak": edge_peak,
                "centroid_linear": int(member_array[centroid_pick]),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "axis_x": axis_x,
                "axis_y": axis_y,
                "linearity": linearity,
                "major_span": major_span,
                "minor_span": minor_span,
                "size": int(member_array.size),
            }
        )
    return components


def _segment_atomic_source_regions(
    *,
    source_rgba: np.ndarray,
    edge_map: np.ndarray,
    alpha_floor: float,
    color_threshold: float,
    alpha_threshold: float,
    device: str,
) -> list[dict[str, float | int | np.ndarray]]:
    torch = _require_torch()
    resolved_device = _resolve_device(torch, device)
    height, width = source_rgba.shape[:2]
    if height <= 0 or width <= 0:
        return []

    if resolved_device == "cpu" and (height * width) <= 4096:
        flat_rgba = source_rgba.reshape(-1, source_rgba.shape[-1]).astype(np.float32)
        flat_premul = flat_rgba.copy()
        flat_premul[:, :3] *= flat_premul[:, 3:4]
        return _segment_atomic_source_regions_cpu(
            width=width,
            height=height,
            flat_rgba=flat_rgba,
            flat_premul=flat_premul,
            flat_edge=edge_map.reshape(-1).astype(np.float32),
            alpha_floor=alpha_floor,
            color_threshold=color_threshold,
            alpha_threshold=alpha_threshold,
        )

    source_t = torch.from_numpy(source_rgba).to(device=resolved_device, dtype=torch.float32)
    edge_t = torch.from_numpy(edge_map).to(device=resolved_device, dtype=torch.float32)
    alpha_map = source_t[..., 3]
    opaque = alpha_map >= alpha_floor
    if not bool(opaque.any().item()):
        return []

    premul = source_t.clone()
    premul[..., :3] *= premul[..., 3:4]
    if width > 1:
        right_color = torch.mean(torch.abs(premul[:, 1:, :] - premul[:, :-1, :]), dim=-1)
        right_alpha = torch.abs(alpha_map[:, 1:] - alpha_map[:, :-1])
        join_right = opaque[:, 1:] & opaque[:, :-1] & (right_color <= color_threshold) & (right_alpha <= alpha_threshold)
    else:
        join_right = torch.zeros((height, 0), device=resolved_device, dtype=torch.bool)
    if height > 1:
        down_color = torch.mean(torch.abs(premul[1:, :, :] - premul[:-1, :, :]), dim=-1)
        down_alpha = torch.abs(alpha_map[1:, :] - alpha_map[:-1, :])
        join_down = opaque[1:, :] & opaque[:-1, :] & (down_color <= color_threshold) & (down_alpha <= alpha_threshold)
    else:
        join_down = torch.zeros((0, width), device=resolved_device, dtype=torch.bool)

    sentinel = height * width
    labels = torch.arange(sentinel, device=resolved_device, dtype=torch.int64).reshape(height, width)
    labels = torch.where(opaque, labels, torch.full_like(labels, sentinel))
    max_iterations = max(4, height + width)
    for _ in range(max_iterations):
        previous = labels
        best = previous.clone()
        if width > 1:
            pair_min = torch.minimum(previous[:, :-1], previous[:, 1:])
            best[:, :-1] = torch.where(join_right, torch.minimum(best[:, :-1], pair_min), best[:, :-1])
            best[:, 1:] = torch.where(join_right, torch.minimum(best[:, 1:], pair_min), best[:, 1:])
        if height > 1:
            pair_min = torch.minimum(previous[:-1, :], previous[1:, :])
            best[:-1, :] = torch.where(join_down, torch.minimum(best[:-1, :], pair_min), best[:-1, :])
            best[1:, :] = torch.where(join_down, torch.minimum(best[1:, :], pair_min), best[1:, :])

        flat_best = best.reshape(-1)
        valid = flat_best < sentinel
        if bool(valid.any().item()):
            compressed = flat_best.clone()
            for _jump in range(16):
                parents = compressed[valid]
                grandparents = compressed[parents]
                updated = torch.minimum(parents, grandparents)
                if bool(torch.equal(updated, parents)):
                    break
                compressed[valid] = updated
            best = compressed.reshape(height, width)

        labels = best
        if bool(torch.equal(labels, previous)):
            break

    valid_linear_t = torch.nonzero(labels.reshape(-1) < sentinel, as_tuple=False).flatten()
    if valid_linear_t.numel() <= 0:
        return []

    component_labels_t = labels.reshape(-1)[valid_linear_t]
    _, inverse_t = torch.unique(component_labels_t, sorted=True, return_inverse=True)
    valid_linear = valid_linear_t.detach().cpu().numpy().astype(np.int32)
    inverse = inverse_t.detach().cpu().numpy().astype(np.int32)
    flat_edge = edge_t.reshape(-1).detach().cpu().numpy().astype(np.float32)

    order = np.argsort(inverse, kind="stable")
    sorted_linear = valid_linear[order]
    sorted_inverse = inverse[order]
    boundaries = np.flatnonzero(
        np.concatenate((
            np.asarray([True], dtype=bool),
            sorted_inverse[1:] != sorted_inverse[:-1],
        ))
    )
    ends = np.concatenate((boundaries[1:], np.asarray([sorted_inverse.shape[0]], dtype=np.int64)))

    components: list[dict[str, float | int | np.ndarray]] = []
    for start, end in zip(boundaries.tolist(), ends.tolist(), strict=False):
        member_array = sorted_linear[start:end].astype(np.int32, copy=False)
        member_y = (member_array // width).astype(np.float32)
        member_x = (member_array % width).astype(np.float32)
        centroid_y = float(np.mean(member_y))
        centroid_x = float(np.mean(member_x))
        axis_x, axis_y, linearity, major_span, minor_span = _component_axis_stats(member_x, member_y)
        centroid_dist = (member_y - centroid_y) ** 2 + (member_x - centroid_x) ** 2
        centroid_pick = int(np.argmin(centroid_dist))
        edge_pick = int(np.argmax(flat_edge[member_array]))
        edge_peak = float(np.max(flat_edge[member_array]))
        components.append(
            {
                "member_linear": member_array,
                "edge_peak_linear": int(member_array[edge_pick]),
                "edge_peak": edge_peak,
                "centroid_linear": int(member_array[centroid_pick]),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "axis_x": axis_x,
                "axis_y": axis_y,
                "linearity": linearity,
                "major_span": major_span,
                "minor_span": minor_span,
                "size": int(member_array.size),
            }
        )
    return components


def _stroke_window_mask(
    *,
    member_x: np.ndarray,
    member_y: np.ndarray,
    remaining: np.ndarray,
    center_x: float,
    center_y: float,
    axis_x: float,
    axis_y: float,
    major_half: float,
    minor_half: float,
) -> np.ndarray:
    delta_x = member_x - float(center_x)
    delta_y = member_y - float(center_y)
    major_delta = delta_x * axis_x + delta_y * axis_y
    minor_delta = delta_x * (-axis_y) + delta_y * axis_x
    return remaining & (np.abs(major_delta) <= major_half) & (np.abs(minor_delta) <= minor_half)


def _stroke_seed_centers(
    *,
    component: dict[str, float | int | np.ndarray],
    member_x: np.ndarray,
    member_y: np.ndarray,
    flat_x: np.ndarray,
    flat_y: np.ndarray,
    cell_w: float,
    cell_h: float,
    step_scale: float,
) -> list[tuple[float, float]]:
    axis_x = float(component["axis_x"])
    axis_y = float(component["axis_y"])
    centroid_x = float(component["centroid_x"])
    centroid_y = float(component["centroid_y"])
    projection = (member_x - centroid_x) * axis_x + (member_y - centroid_y) * axis_y
    if projection.size <= 0:
        return [(centroid_y, centroid_x)]
    step = max(
        1.0,
        (abs(axis_x) * cell_w + abs(axis_y) * cell_h) * float(step_scale),
        np.hypot(axis_x * cell_w, axis_y * cell_h) * 0.9,
    )
    if step <= 1e-6:
        return [(centroid_y, centroid_x)]
    low = float(np.min(projection))
    high = float(np.max(projection))
    edge_peak_linear = int(component["edge_peak_linear"])
    anchor_x = float(flat_x[edge_peak_linear])
    anchor_y = float(flat_y[edge_peak_linear])
    anchor_projection = (anchor_x - centroid_x) * axis_x + (anchor_y - centroid_y) * axis_y
    if not np.isfinite(anchor_projection):
        anchor_projection = 0.0

    seed_projections: list[float] = [anchor_projection]
    cursor = anchor_projection - step
    while cursor >= low - 0.5 * step:
        seed_projections.append(cursor)
        cursor -= step
    cursor = anchor_projection + step
    while cursor <= high + 0.5 * step:
        seed_projections.append(cursor)
        cursor += step
    if len(seed_projections) == 1 and abs(high - low) > 0.6 * step:
        center_projection = 0.5 * (low + high)
        seed_projections.extend([center_projection - 0.5 * step, center_projection + 0.5 * step])

    seen: set[tuple[int, int]] = set()
    centers: list[tuple[float, float]] = []
    perpendicular = (member_x - centroid_x) * (-axis_y) + (member_y - centroid_y) * axis_x
    for proj_center in sorted(seed_projections):
        score = np.abs(projection - proj_center) + np.abs(perpendicular) * 0.35
        pick = int(np.argmin(score))
        center = (float(member_y[pick]), float(member_x[pick]))
        key = (int(round(center[0] * 2.0)), int(round(center[1] * 2.0)))
        if key in seen:
            continue
        seen.add(key)
        centers.append(center)
    if not centers:
        centers.append((centroid_y, centroid_x))
    return centers


def _extract_source_region_tiles(
    *,
    components: list[dict[str, float | int | np.ndarray]],
    flat_rgba: np.ndarray,
    flat_edge: np.ndarray,
    flat_x: np.ndarray,
    flat_y: np.ndarray,
    cell_w: float,
    cell_h: float,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
    min_region_area_ratio: float,
    min_window_coverage: float,
    stroke_linearity_threshold: float,
    stroke_step_scale: float,
    stroke_minor_limit_scale: float,
) -> list[list[dict[str, float | int | np.ndarray]]]:
    output_area = max(1, target_width * target_height)
    cell_area = max(cell_w * cell_h, 1e-6)
    half_w = cell_w * 0.5
    half_h = cell_h * 0.5
    buckets: list[list[dict[str, float | int | np.ndarray]]] = [[] for _ in range(output_area)]
    min_region_pixels = max(1, int(np.ceil(cell_area * min_region_area_ratio)))
    ordered_components = sorted(
        components,
        key=lambda comp: (abs(float(comp["size"]) - cell_area), -float(comp["edge_peak"])),
    )
    primary_tiles: list[dict[str, float | int | np.ndarray]] = []

    def overlap_tiles_for_component(
        component: dict[str, float | int | np.ndarray],
        *,
        component_id: int,
    ) -> list[dict[str, float | int | np.ndarray]]:
        member_linear = np.asarray(component["member_linear"], dtype=np.int32)
        if member_linear.size <= 0:
            return []
        member_x = flat_x[member_linear].astype(np.float32)
        member_y = flat_y[member_linear].astype(np.float32)
        member_coord_x = np.clip(
            np.floor((member_x + 0.5) / max(cell_w, 1e-6) - phase_x).astype(np.int32),
            0,
            max(0, target_width - 1),
        )
        member_coord_y = np.clip(
            np.floor((member_y + 0.5) / max(cell_h, 1e-6) - phase_y).astype(np.int32),
            0,
            max(0, target_height - 1),
        )
        member_flat_index = member_coord_y * target_width + member_coord_x
        order = np.argsort(member_flat_index, kind="stable")
        sorted_flat_index = member_flat_index[order]
        starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(sorted_flat_index[1:] != sorted_flat_index[:-1]) + 1,
            )
        )
        ends = np.concatenate((starts[1:], np.asarray([sorted_flat_index.shape[0]], dtype=np.int64)))
        tiles: list[dict[str, float | int | np.ndarray]] = []
        for start, end in zip(starts.tolist(), ends.tolist(), strict=False):
            local_order = order[start:end]
            cell_linear = member_linear[local_order]
            cell_x = member_x[local_order]
            cell_y = member_y[local_order]
            cell_edge = flat_edge[cell_linear]
            flat_index = int(sorted_flat_index[start])
            center_x = float(np.mean(cell_x))
            center_y = float(np.mean(cell_y))
            edge_peak = float(np.max(cell_edge)) if cell_edge.size else 0.0
            if edge_peak > 0.0:
                rep_linear = int(cell_linear[int(np.argmax(cell_edge))])
            else:
                centroid_dist = (cell_x - center_x) ** 2 + (cell_y - center_y) ** 2
                rep_linear = int(cell_linear[int(np.argmin(centroid_dist))])
            tiles.append(
                {
                    "rep_linear": rep_linear,
                    "rep_rgba": flat_rgba[rep_linear].astype(np.float32),
                    "area_ratio": float(cell_linear.shape[0] / cell_area),
                    "coverage": float(np.clip(cell_linear.shape[0] / cell_area, 0.0, 1.0)),
                    "edge_peak": edge_peak,
                    "source_center_x": center_x,
                    "source_center_y": center_y,
                    "coord_x": int(flat_index % target_width),
                    "coord_y": int(flat_index // target_width),
                    "flat_index": flat_index,
                    "component_id": component_id,
                }
            )
        return tiles

    for component_id, component in enumerate(ordered_components):
        component_size = int(component["size"])
        if component_size < min_region_pixels:
            continue
        member_linear = np.asarray(component["member_linear"], dtype=np.int32)
        member_x = flat_x[member_linear].astype(np.float32)
        member_y = flat_y[member_linear].astype(np.float32)
        remaining = np.ones(member_linear.shape[0], dtype=bool)
        seed_queue: deque[tuple[float, float]] = deque()
        seen_seeds: set[tuple[int, int]] = set()
        accepted_any = False
        axis_x = float(component["axis_x"])
        axis_y = float(component["axis_y"])
        linearity = float(component["linearity"])
        major_span = float(component["major_span"])
        minor_span = float(component["minor_span"])
        stroke_component = (
            linearity >= stroke_linearity_threshold
            and major_span >= max(cell_w, cell_h) * 1.1
            and minor_span <= max(1.0, min(cell_w, cell_h) * stroke_minor_limit_scale)
        )
        stroke_major_half = max(
            0.5,
            0.5 * (abs(axis_x) * cell_w + abs(axis_y) * cell_h) * stroke_step_scale,
            0.45 * np.hypot(axis_x * cell_w, axis_y * cell_h),
        )
        stroke_minor_half = max(0.5, 0.5 * min(cell_w, cell_h) * stroke_minor_limit_scale)
        component_tiles: list[dict[str, float | int | np.ndarray]] = []

        def enqueue(seed_y: float, seed_x: float) -> None:
            key = (int(round(seed_y * 2.0)), int(round(seed_x * 2.0)))
            if key in seen_seeds:
                return
            seen_seeds.add(key)
            seed_queue.append((float(seed_y), float(seed_x)))

        if stroke_component:
            for seed_y, seed_x in _stroke_seed_centers(
                component=component,
                member_x=member_x,
                member_y=member_y,
                flat_x=flat_x,
                flat_y=flat_y,
                cell_w=cell_w,
                cell_h=cell_h,
                step_scale=stroke_step_scale,
            ):
                enqueue(seed_y, seed_x)
        else:
            enqueue(float(component["centroid_y"]), float(component["centroid_x"]))
            edge_peak_linear = int(component["edge_peak_linear"])
            enqueue(float(flat_y[edge_peak_linear]), float(flat_x[edge_peak_linear]))

        while seed_queue:
            center_y, center_x = seed_queue.popleft()
            if stroke_component:
                in_window = _stroke_window_mask(
                    member_x=member_x,
                    member_y=member_y,
                    remaining=remaining,
                    center_x=center_x,
                    center_y=center_y,
                    axis_x=axis_x,
                    axis_y=axis_y,
                    major_half=stroke_major_half,
                    minor_half=stroke_minor_half,
                )
                if not np.any(in_window):
                    in_window = remaining & (np.abs(member_x - center_x) <= half_w) & (np.abs(member_y - center_y) <= half_h)
            else:
                in_window = remaining & (np.abs(member_x - center_x) <= half_w) & (np.abs(member_y - center_y) <= half_h)
            footprint_count = int(np.count_nonzero(in_window))
            if footprint_count <= 0:
                continue
            area_ratio = float(footprint_count / cell_area)
            if area_ratio < min_window_coverage:
                continue

            accepted_any = True
            window_linear = member_linear[in_window]
            window_x = member_x[in_window]
            window_y = member_y[in_window]
            window_edge = flat_edge[window_linear]
            if stroke_component:
                accepted_center_x = float(center_x * 0.75 + np.mean(window_x) * 0.25)
                accepted_center_y = float(center_y * 0.75 + np.mean(window_y) * 0.25)
            else:
                accepted_center_x = float(np.mean(window_x))
                accepted_center_y = float(np.mean(window_y))
            edge_peak = float(np.max(window_edge)) if window_edge.size else 0.0
            if stroke_component:
                center_dist = (window_x - accepted_center_x) ** 2 + (window_y - accepted_center_y) ** 2
                rep_linear = int(window_linear[int(np.argmin(center_dist))])
            elif edge_peak > 0.0:
                rep_linear = int(window_linear[int(np.argmax(window_edge))])
            else:
                centroid_dist = (window_x - accepted_center_x) ** 2 + (window_y - accepted_center_y) ** 2
                rep_linear = int(window_linear[int(np.argmin(centroid_dist))])

            coord_y, coord_x, flat_index = _project_source_point_to_output_coord(
                x=accepted_center_x,
                y=accepted_center_y,
                cell_w=cell_w,
                cell_h=cell_h,
                target_width=target_width,
                target_height=target_height,
                phase_x=phase_x,
                phase_y=phase_y,
            )
            component_tiles.append(
                {
                    "rep_linear": rep_linear,
                    "rep_rgba": flat_rgba[rep_linear].astype(np.float32),
                    "area_ratio": area_ratio,
                    "coverage": float(np.clip(area_ratio, 0.0, 1.0)),
                    "edge_peak": edge_peak,
                    "source_center_x": accepted_center_x,
                    "source_center_y": accepted_center_y,
                    "coord_x": coord_x,
                    "coord_y": coord_y,
                    "flat_index": flat_index,
                    "component_id": component_id,
                }
            )
            remaining[in_window] = False
            if int(np.count_nonzero(remaining)) < min_region_pixels:
                break
            if not stroke_component:
                for _, dy, dx in _DIRS:
                    enqueue(accepted_center_y + dy * cell_h, accepted_center_x + dx * cell_w)

        if accepted_any:
            if stroke_component and component_tiles:
                dominant_vertical = abs(axis_y) >= abs(axis_x)
                grouped_tiles: dict[int, list[dict[str, float | int | np.ndarray]]] = {}
                for tile in component_tiles:
                    group_key = int(tile["coord_y"]) if dominant_vertical else int(tile["coord_x"])
                    grouped_tiles.setdefault(group_key, []).append(tile)
                filtered_tiles: list[dict[str, float | int | np.ndarray]] = []
                ordered_keys = sorted(grouped_tiles)
                secondary_step = (
                    axis_x / max(abs(axis_y), 1e-6)
                    if dominant_vertical
                    else axis_y / max(abs(axis_x), 1e-6)
                )
                previous_primary: int | None = None
                previous_secondary: float | None = None

                def tile_score(
                    tile: dict[str, float | int | np.ndarray],
                    *,
                    expected_secondary: float | None,
                ) -> tuple[float, float, float, float]:
                    continuous_coord = (
                        (float(tile["source_center_x"]) + 0.5) / max(cell_w, 1e-6) - phase_x
                        if dominant_vertical
                        else (float(tile["source_center_y"]) + 0.5) / max(cell_h, 1e-6) - phase_y
                    )
                    discrete_coord = float(tile["coord_x"] if dominant_vertical else tile["coord_y"])
                    path_error = abs(discrete_coord - expected_secondary) if expected_secondary is not None else 0.0
                    return (
                        path_error,
                        abs(discrete_coord - continuous_coord),
                        abs(float(tile["area_ratio"]) - 1.0),
                        -float(tile["coverage"]),
                    )

                for group_key in ordered_keys:
                    group_tiles = grouped_tiles[group_key]
                    expected_secondary = None
                    if previous_primary is not None and previous_secondary is not None:
                        expected_secondary = previous_secondary + secondary_step * float(group_key - previous_primary)

                    best_tile = min(
                        group_tiles,
                        key=lambda tile: tile_score(tile, expected_secondary=expected_secondary),
                    )
                    filtered_tiles.append(best_tile)
                    previous_primary = group_key
                    previous_secondary = float(best_tile["coord_x"] if dominant_vertical else best_tile["coord_y"])
                component_tiles = filtered_tiles
            for tile in component_tiles:
                primary_tiles.append(tile)
            continue

        component_area_ratio = float(component_size / cell_area)
        if component_area_ratio < min_window_coverage:
            continue
        centroid_linear = int(component["centroid_linear"])
        center_x = float(component["centroid_x"])
        center_y = float(component["centroid_y"])
        coord_y, coord_x, flat_index = _project_source_point_to_output_coord(
            x=center_x,
            y=center_y,
            cell_w=cell_w,
            cell_h=cell_h,
            target_width=target_width,
            target_height=target_height,
            phase_x=phase_x,
            phase_y=phase_y,
        )
        buckets[flat_index].append(
            {
                "rep_linear": centroid_linear,
                "rep_rgba": flat_rgba[centroid_linear].astype(np.float32),
                "area_ratio": component_area_ratio,
                "coverage": float(np.clip(component_area_ratio, 0.0, 1.0)),
                "edge_peak": float(component["edge_peak"]),
                "source_center_x": center_x,
                "source_center_y": center_y,
                "coord_x": coord_x,
                "coord_y": coord_y,
                "flat_index": flat_index,
                "component_id": component_id,
            }
        )

    for tile in primary_tiles:
        buckets[int(tile["flat_index"])].append(tile)
    for component_id, component in enumerate(ordered_components):
        for tile in overlap_tiles_for_component(component, component_id=component_id):
            flat_index = int(tile["flat_index"])
            if buckets[flat_index]:
                continue
            buckets[flat_index].append(tile)

    for flat_index, bucket in enumerate(buckets):
        if not bucket:
            continue
        coord_y = flat_index // target_width
        coord_x = flat_index % target_width
        for tile in bucket:
            neighbor_rgba = np.zeros((4, 4), dtype=np.float32)
            neighbor_mask = np.zeros((4,), dtype=bool)
            for direction, dy, dx in _DIRS:
                neighbor_x = coord_x + dx
                neighbor_y = coord_y + dy
                if neighbor_x < 0 or neighbor_x >= target_width or neighbor_y < 0 or neighbor_y >= target_height:
                    continue
                neighbor_bucket = buckets[neighbor_y * target_width + neighbor_x]
                if not neighbor_bucket:
                    continue
                expected_x = float(tile["source_center_x"]) + dx * cell_w
                expected_y = float(tile["source_center_y"]) + dy * cell_h

                def neighbor_score(candidate: dict[str, float | int | np.ndarray]) -> tuple[float, int, float, float, float]:
                    return (
                        abs(float(candidate["source_center_x"]) - expected_x)
                        + abs(float(candidate["source_center_y"]) - expected_y),
                        0 if int(candidate["component_id"]) == int(tile["component_id"]) else 1,
                        abs(float(candidate["area_ratio"]) - 1.0),
                        -float(candidate["coverage"]),
                        -float(candidate["edge_peak"]),
                    )

                best_neighbor = min(neighbor_bucket, key=neighbor_score)
                neighbor_rgba[direction] = np.asarray(best_neighbor["rep_rgba"], dtype=np.float32)
                neighbor_mask[direction] = True
            tile["neighbor_rgba"] = neighbor_rgba
            tile["neighbor_mask"] = neighbor_mask

    return buckets


def _select_source_region_candidates(
    *,
    candidates: list[dict[str, float | int | np.ndarray]],
    allowed_candidates: int,
) -> list[dict[str, float | int | np.ndarray]]:
    if not candidates or allowed_candidates <= 0:
        return []
    picked: list[dict[str, float | int | np.ndarray]] = []
    seen_linear: set[int] = set()

    def add(candidate: dict[str, float | int | np.ndarray]) -> None:
        rep_linear = int(candidate["rep_linear"])
        if rep_linear in seen_linear:
            return
        seen_linear.add(rep_linear)
        picked.append(candidate)

    by_area = sorted(
        candidates,
        key=lambda cand: (abs(float(cand["area_ratio"]) - 1.0), -float(cand["coverage"]), -float(cand["edge_peak"])),
    )
    by_edge = sorted(candidates, key=lambda cand: (-float(cand["edge_peak"]), abs(float(cand["area_ratio"]) - 1.0)))
    by_coverage = sorted(candidates, key=lambda cand: (-float(cand["coverage"]), abs(float(cand["area_ratio"]) - 1.0), -float(cand["edge_peak"])))

    add(by_area[0])
    add(by_coverage[0])
    add(by_edge[0])
    for candidate in by_area:
        add(candidate)
        if len(picked) >= allowed_candidates:
            break
    return picked[:allowed_candidates]


def build_tile_graph_model(
    source_rgba: np.ndarray,
    *,
    inference: InferenceResult,
    analysis: TileGraphSourceAnalysis,
    solver_params: SolverHyperParams | None = None,
    device: str = "cpu",
) -> tuple[TileGraphModel, TileGraphBuildStats]:
    torch = _require_torch()
    resolved_device = _resolve_device(torch, device)
    solver_params = solver_params or SolverHyperParams()
    cache_key = _tile_graph_model_cache_key(
        source_rgba,
        inference=inference,
        solver_params=solver_params,
        device=resolved_device,
    )
    cached_model = _get_cached_tile_graph_model(cache_key)
    if cached_model is not None:
        model, stats = cached_model
        return model, replace(stats, cache_hit=True)
    height, width = source_rgba.shape[:2]
    cell_h = height / max(1, inference.target_height)
    cell_w = width / max(1, inference.target_width)
    output_area = max(1, inference.target_height * inference.target_width)
    max_candidates_per_coord = max(1, int(solver_params.tile_graph_max_candidates_per_coord))

    flat_rgba_np = source_rgba.reshape(-1, source_rgba.shape[-1]).astype(np.float32)
    flat_edge_np = analysis.edge_map.reshape(-1).astype(np.float32)
    flat_y_np, flat_x_np = np.divmod(np.arange(flat_rgba_np.shape[0], dtype=np.int64), width)
    flat_y_np = flat_y_np.astype(np.int32)
    flat_x_np = flat_x_np.astype(np.int32)
    flat_alpha_np = flat_rgba_np[:, 3].astype(np.float32)
    projected_coord_x = np.clip(
        np.floor((flat_x_np.astype(np.float32) + 0.5) / max(cell_w, 1e-6) - inference.phase_x).astype(np.int32),
        0,
        max(0, inference.target_width - 1),
    )
    projected_coord_y = np.clip(
        np.floor((flat_y_np.astype(np.float32) + 0.5) / max(cell_h, 1e-6) - inference.phase_y).astype(np.int32),
        0,
        max(0, inference.target_height - 1),
    )
    projected_flat_index = projected_coord_y * inference.target_width + projected_coord_x
    cell_counts_flat = np.bincount(projected_flat_index, minlength=output_area).astype(np.float32)
    safe_cell_counts_flat = np.clip(cell_counts_flat, 1.0, None)
    cell_mean_rgba_flat = np.stack(
        [
            np.bincount(projected_flat_index, weights=flat_rgba_np[:, channel], minlength=output_area).astype(np.float32)
            / safe_cell_counts_flat
            for channel in range(flat_rgba_np.shape[-1])
        ],
        axis=-1,
    )
    cell_alpha_support_flat = np.zeros(output_area, dtype=np.float32)
    np.maximum.at(cell_alpha_support_flat, projected_flat_index, flat_alpha_np)
    cell_alpha_mean_flat = (
        np.bincount(projected_flat_index, weights=flat_alpha_np, minlength=output_area).astype(np.float32)
        / safe_cell_counts_flat
    )
    cell_edge_strength_flat = np.zeros(output_area, dtype=np.float32)
    np.maximum.at(cell_edge_strength_flat, projected_flat_index, flat_edge_np)
    global_components = _segment_atomic_source_regions(
        source_rgba=source_rgba.astype(np.float32),
        edge_map=analysis.edge_map.astype(np.float32),
        alpha_floor=solver_params.alpha_transparent_threshold,
        color_threshold=float(getattr(solver_params, "tile_graph_component_color_threshold", 0.055)),
        alpha_threshold=float(getattr(solver_params, "tile_graph_component_alpha_threshold", 0.12)),
        device=resolved_device,
    )
    region_buckets = _extract_source_region_tiles(
        components=global_components,
        flat_rgba=flat_rgba_np,
        flat_edge=flat_edge_np,
        flat_x=flat_x_np,
        flat_y=flat_y_np,
        cell_w=cell_w,
        cell_h=cell_h,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
        min_region_area_ratio=float(getattr(solver_params, "tile_graph_source_region_min_area_ratio", 0.06)),
        min_window_coverage=float(getattr(solver_params, "tile_graph_source_region_window_coverage", 0.12)),
        stroke_linearity_threshold=float(getattr(solver_params, "tile_graph_stroke_linearity_threshold", 0.72)),
        stroke_step_scale=float(getattr(solver_params, "tile_graph_stroke_step_scale", 0.95)),
        stroke_minor_limit_scale=float(getattr(solver_params, "tile_graph_stroke_minor_limit_scale", 0.55)),
    )

    candidate_rgba: list[np.ndarray] = []
    candidate_area_ratio: list[float] = []
    candidate_coverage: list[float] = []
    candidate_edge_peak: list[float] = []
    candidate_neighbor_rgba: list[np.ndarray] = []
    candidate_neighbor_mask: list[np.ndarray] = []
    choice_counts_list: list[int] = []
    component_total = len(global_components)
    edge_candidate_cap = max(max_candidates_per_coord, int(getattr(solver_params, "tile_graph_edge_candidates_per_coord", max_candidates_per_coord)))
    for flat_index in range(output_area):
        seen_pixels: set[int] = set()
        cell_start = len(candidate_rgba)
        edge_cell = cell_edge_strength_flat[flat_index] >= solver_params.source_edge_detail_threshold
        allowed_candidates = edge_candidate_cap if edge_cell else max_candidates_per_coord

        def add_candidate(
            rgba: np.ndarray,
            *,
            pixel_index: int | None,
            area_ratio: float,
            coverage: float,
            edge_peak: float,
            neighbor_rgba: np.ndarray | None = None,
            neighbor_mask: np.ndarray | None = None,
        ) -> None:
            if len(candidate_rgba) - cell_start >= allowed_candidates:
                return
            if pixel_index is not None:
                if pixel_index in seen_pixels:
                    return
                seen_pixels.add(pixel_index)
            candidate_rgba.append(np.asarray(rgba, dtype=np.float32))
            candidate_area_ratio.append(float(max(area_ratio, 1e-3)))
            candidate_coverage.append(float(np.clip(coverage, 0.0, 1.0)))
            candidate_edge_peak.append(float(np.clip(edge_peak, 0.0, 1.0)))
            candidate_neighbor_rgba.append(
                np.asarray(neighbor_rgba if neighbor_rgba is not None else np.zeros((4, 4), dtype=np.float32), dtype=np.float32)
            )
            candidate_neighbor_mask.append(
                np.asarray(neighbor_mask if neighbor_mask is not None else np.zeros((4,), dtype=bool), dtype=bool)
            )

        selected_regions = _select_source_region_candidates(
            candidates=region_buckets[flat_index],
            allowed_candidates=allowed_candidates,
        )
        for region in selected_regions:
            add_candidate(
                np.asarray(region["rep_rgba"], dtype=np.float32),
                pixel_index=int(region["rep_linear"]),
                area_ratio=float(region["area_ratio"]),
                coverage=float(region["coverage"]),
                edge_peak=float(region["edge_peak"]),
                neighbor_rgba=np.asarray(region.get("neighbor_rgba", np.zeros((4, 4), dtype=np.float32)), dtype=np.float32),
                neighbor_mask=np.asarray(region.get("neighbor_mask", np.zeros((4,), dtype=bool)), dtype=bool),
            )
        if cell_alpha_mean_flat[flat_index] < 0.98:
            add_candidate(
                np.zeros((4,), dtype=np.float32),
                pixel_index=None,
                area_ratio=1.0,
                coverage=float(np.clip(1.0 - cell_alpha_mean_flat[flat_index], 0.0, 1.0)),
                edge_peak=0.0,
            )
        if len(candidate_rgba) == cell_start:
            if cell_alpha_support_flat[flat_index] <= solver_params.alpha_transparent_threshold:
                add_candidate(
                    np.zeros((4,), dtype=np.float32),
                    pixel_index=None,
                    area_ratio=1.0,
                    coverage=1.0,
                    edge_peak=0.0,
                )
            else:
                coord_y = flat_index // inference.target_width
                coord_x = flat_index % inference.target_width
                raise RuntimeError(
                    "Tile-graph extraction produced no source-owned candidate "
                    f"for occupied output cell ({coord_x}, {coord_y})."
                )
        choice_counts_list.append(len(candidate_rgba) - cell_start)

    candidate_rgba_np = np.asarray(candidate_rgba, dtype=np.float32)
    candidate_area_ratio_np = np.asarray(candidate_area_ratio, dtype=np.float32)
    candidate_coverage_np = np.asarray(candidate_coverage, dtype=np.float32)
    candidate_edge_peak_np = np.asarray(candidate_edge_peak, dtype=np.float32)
    candidate_neighbor_rgba_np = np.asarray(candidate_neighbor_rgba, dtype=np.float32)
    candidate_neighbor_mask_np = np.asarray(candidate_neighbor_mask, dtype=bool)

    cell_candidate_offsets = np.zeros(output_area + 1, dtype=np.int32)
    cell_candidate_offsets[1:] = np.cumsum(np.asarray(choice_counts_list, dtype=np.int32), dtype=np.int32)
    cell_candidate_indices_np = np.arange(candidate_rgba_np.shape[0], dtype=np.int32)
    choice_counts = np.diff(cell_candidate_offsets)
    average_choices = float(np.mean(choice_counts)) if choice_counts.size else 0.0

    edge_density = float(
        np.mean(cell_edge_strength_flat >= solver_params.source_edge_detail_threshold)
    )
    base_model = TileGraphModel(
        candidate_rgba=candidate_rgba_np,
        candidate_area_ratio=candidate_area_ratio_np,
        candidate_coverage=candidate_coverage_np,
        candidate_edge_peak=candidate_edge_peak_np,
        candidate_neighbor_rgba=candidate_neighbor_rgba_np,
        candidate_neighbor_mask=candidate_neighbor_mask_np,
        cell_candidate_offsets=cell_candidate_offsets,
        cell_candidate_indices=cell_candidate_indices_np,
        cell_mean_rgba=cell_mean_rgba_flat.reshape(inference.target_height, inference.target_width, 4).astype(np.float32),
        cell_alpha_mean=cell_alpha_mean_flat.reshape(inference.target_height, inference.target_width).astype(np.float32),
        cell_edge_strength=cell_edge_strength_flat.reshape(inference.target_height, inference.target_width).astype(np.float32),
    )
    base_stats = TileGraphBuildStats(
        component_count=component_total,
        candidate_count=int(candidate_rgba_np.shape[0]),
        edge_density=edge_density,
        average_choices=average_choices,
        model_device=resolved_device,
        cache_hit=False,
    )
    _store_cached_tile_graph_model(cache_key, base_model, base_stats)
    return base_model, base_stats


def _cell_candidate_span(model: TileGraphModel, y: int, x: int) -> tuple[int, int]:
    width = model.cell_alpha_mean.shape[1]
    flat_index = y * width + x
    return int(model.cell_candidate_offsets[flat_index]), int(model.cell_candidate_offsets[flat_index + 1])


def _build_choice_grid(model: TileGraphModel) -> tuple[np.ndarray, np.ndarray]:
    height, width = model.cell_alpha_mean.shape[:2]
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
    candidate_area_t = torch.from_numpy(model.candidate_area_ratio).to(device=device, dtype=torch.float32)
    candidate_coverage_t = torch.from_numpy(model.candidate_coverage).to(device=device, dtype=torch.float32)
    candidate_edge_peak_t = torch.from_numpy(model.candidate_edge_peak).to(device=device, dtype=torch.float32)
    candidate_neighbor_rgba_t = torch.from_numpy(model.candidate_neighbor_rgba).to(device=device, dtype=torch.float32)
    candidate_neighbor_mask_t = torch.from_numpy(model.candidate_neighbor_mask).to(device=device, dtype=torch.bool)
    cell_mean_rgba_t = torch.from_numpy(model.cell_mean_rgba).to(device=device, dtype=torch.float32)
    cell_alpha_mean_t = torch.from_numpy(model.cell_alpha_mean).to(device=device, dtype=torch.float32)
    cell_edge_strength_t = torch.from_numpy(model.cell_edge_strength).to(device=device, dtype=torch.float32)
    choice_rgba_t = candidate_rgba_t[choice_indices_t]
    choice_edge_peak_t = candidate_edge_peak_t[choice_indices_t]
    color_error = (choice_rgba_t - cell_mean_rgba_t[..., None, :]).abs().mean(dim=-1)
    area_error = torch.log(candidate_area_t[choice_indices_t].clamp_min(1e-4) + 1e-4).abs()
    alpha_error = (choice_rgba_t[..., 3] - cell_alpha_mean_t[..., None]).abs()
    coverage_error = 1.0 - candidate_coverage_t[choice_indices_t].clamp(0.0, 1.0)
    edge_error = (choice_edge_peak_t - cell_edge_strength_t[..., None]).abs()
    unary_cost_t = (
        color_error
        + area_error * solver_params.tile_graph_area_weight
        + alpha_error * solver_params.tile_graph_alpha_weight
        + coverage_error * solver_params.tile_graph_coverage_weight
        + edge_error * solver_params.tile_graph_edge_peak_weight
    )
    unary_cost_t = unary_cost_t.masked_fill(~choice_mask_t, float("inf"))
    return unary_cost_t, candidate_rgba_t, candidate_neighbor_rgba_t, candidate_neighbor_mask_t, choice_rgba_t


def _pair_penalty_selected_torch(
    torch,
    candidate_rgba_t,
    candidate_neighbor_rgba_t,
    candidate_neighbor_mask_t,
    left_indices_t,
    right_indices_t,
    direction: int,
    weight: float,
):
    left_rgba = candidate_rgba_t[left_indices_t]
    right_rgba = candidate_rgba_t[right_indices_t]
    forward_mask = candidate_neighbor_mask_t[left_indices_t, direction]
    reverse_mask = candidate_neighbor_mask_t[right_indices_t, _OPPOSITE[direction]]
    forward_penalty = (right_rgba - candidate_neighbor_rgba_t[left_indices_t, direction]).abs().mean(dim=-1)
    reverse_penalty = (left_rgba - candidate_neighbor_rgba_t[right_indices_t, _OPPOSITE[direction]]).abs().mean(dim=-1)
    total_weight = forward_mask.to(dtype=torch.float32) + reverse_mask.to(dtype=torch.float32)
    combined = (
        forward_penalty * forward_mask.to(dtype=torch.float32)
        + reverse_penalty * reverse_mask.to(dtype=torch.float32)
    )
    return torch.where(total_weight > 0.0, combined / total_weight.clamp_min(1.0), torch.zeros_like(combined)) * weight


def _pair_penalty_option_right_torch(
    torch,
    fixed_left_indices_t,
    option_rgba_t,
    candidate_rgba_t,
    candidate_neighbor_rgba_t,
    candidate_neighbor_mask_t,
    option_neighbor_rgba_t,
    option_neighbor_mask_t,
    *,
    weight: float,
):
    fixed_left_rgba = candidate_rgba_t[fixed_left_indices_t]
    forward_mask = candidate_neighbor_mask_t[fixed_left_indices_t, _RIGHT][..., None]
    reverse_mask = option_neighbor_mask_t[..., _LEFT]
    forward_penalty = (option_rgba_t - candidate_neighbor_rgba_t[fixed_left_indices_t, _RIGHT][..., None, :]).abs().mean(dim=-1)
    reverse_penalty = (fixed_left_rgba[..., None, :] - option_neighbor_rgba_t[..., _LEFT, :]).abs().mean(dim=-1)
    total_weight = forward_mask.to(dtype=torch.float32) + reverse_mask.to(dtype=torch.float32)
    combined = (
        forward_penalty * forward_mask.to(dtype=torch.float32)
        + reverse_penalty * reverse_mask.to(dtype=torch.float32)
    )
    return torch.where(total_weight > 0.0, combined / total_weight.clamp_min(1.0), torch.zeros_like(combined)) * weight


def _pair_penalty_option_left_torch(
    torch,
    option_rgba_t,
    fixed_right_indices_t,
    candidate_rgba_t,
    candidate_neighbor_rgba_t,
    candidate_neighbor_mask_t,
    option_neighbor_rgba_t,
    option_neighbor_mask_t,
    *,
    weight: float,
):
    fixed_right_rgba = candidate_rgba_t[fixed_right_indices_t]
    forward_mask = option_neighbor_mask_t[..., _RIGHT]
    reverse_mask = candidate_neighbor_mask_t[fixed_right_indices_t, _LEFT][..., None]
    forward_penalty = (fixed_right_rgba[..., None, :] - option_neighbor_rgba_t[..., _RIGHT, :]).abs().mean(dim=-1)
    reverse_penalty = (option_rgba_t - candidate_neighbor_rgba_t[fixed_right_indices_t, _LEFT][..., None, :]).abs().mean(dim=-1)
    total_weight = forward_mask.to(dtype=torch.float32) + reverse_mask.to(dtype=torch.float32)
    combined = (
        forward_penalty * forward_mask.to(dtype=torch.float32)
        + reverse_penalty * reverse_mask.to(dtype=torch.float32)
    )
    return torch.where(total_weight > 0.0, combined / total_weight.clamp_min(1.0), torch.zeros_like(combined)) * weight


def _pair_penalty_option_down_torch(
    torch,
    fixed_up_indices_t,
    option_rgba_t,
    candidate_rgba_t,
    candidate_neighbor_rgba_t,
    candidate_neighbor_mask_t,
    option_neighbor_rgba_t,
    option_neighbor_mask_t,
    *,
    weight: float,
):
    fixed_up_rgba = candidate_rgba_t[fixed_up_indices_t]
    forward_mask = candidate_neighbor_mask_t[fixed_up_indices_t, _DOWN][..., None]
    reverse_mask = option_neighbor_mask_t[..., _UP]
    forward_penalty = (option_rgba_t - candidate_neighbor_rgba_t[fixed_up_indices_t, _DOWN][..., None, :]).abs().mean(dim=-1)
    reverse_penalty = (fixed_up_rgba[..., None, :] - option_neighbor_rgba_t[..., _UP, :]).abs().mean(dim=-1)
    total_weight = forward_mask.to(dtype=torch.float32) + reverse_mask.to(dtype=torch.float32)
    combined = (
        forward_penalty * forward_mask.to(dtype=torch.float32)
        + reverse_penalty * reverse_mask.to(dtype=torch.float32)
    )
    return torch.where(total_weight > 0.0, combined / total_weight.clamp_min(1.0), torch.zeros_like(combined)) * weight


def _pair_penalty_option_up_torch(
    torch,
    option_rgba_t,
    fixed_down_indices_t,
    candidate_rgba_t,
    candidate_neighbor_rgba_t,
    candidate_neighbor_mask_t,
    option_neighbor_rgba_t,
    option_neighbor_mask_t,
    *,
    weight: float,
):
    fixed_down_rgba = candidate_rgba_t[fixed_down_indices_t]
    forward_mask = option_neighbor_mask_t[..., _DOWN]
    reverse_mask = candidate_neighbor_mask_t[fixed_down_indices_t, _UP][..., None]
    forward_penalty = (fixed_down_rgba[..., None, :] - option_neighbor_rgba_t[..., _DOWN, :]).abs().mean(dim=-1)
    reverse_penalty = (option_rgba_t - candidate_neighbor_rgba_t[fixed_down_indices_t, _UP][..., None, :]).abs().mean(dim=-1)
    total_weight = forward_mask.to(dtype=torch.float32) + reverse_mask.to(dtype=torch.float32)
    combined = (
        forward_penalty * forward_mask.to(dtype=torch.float32)
        + reverse_penalty * reverse_mask.to(dtype=torch.float32)
    )
    return torch.where(total_weight > 0.0, combined / total_weight.clamp_min(1.0), torch.zeros_like(combined)) * weight


def _assignment_score_torch(
    torch,
    selected_t,
    unary_cost_t,
    choice_indices_t,
    candidate_rgba_t,
    candidate_neighbor_rgba_t,
    candidate_neighbor_mask_t,
    solver_params: SolverHyperParams,
) -> float:
    selected_choice_t = (choice_indices_t == selected_t[..., None]).to(dtype=torch.int64).argmax(dim=-1)
    gathered = torch.take_along_dim(unary_cost_t, selected_choice_t[..., None], dim=2)[..., 0]
    score = float(gathered.mean().item())
    if selected_t.shape[1] > 1:
        score += float(
            _pair_penalty_selected_torch(
                torch,
                candidate_rgba_t,
                candidate_neighbor_rgba_t,
                candidate_neighbor_mask_t,
                selected_t[:, :-1],
                selected_t[:, 1:],
                _RIGHT,
                solver_params.tile_graph_adjacency_weight,
            ).mean().item()
        )
    if selected_t.shape[0] > 1:
        score += float(
            _pair_penalty_selected_torch(
                torch,
                candidate_rgba_t,
                candidate_neighbor_rgba_t,
                candidate_neighbor_mask_t,
                selected_t[:-1, :],
                selected_t[1:, :],
                _DOWN,
                solver_params.tile_graph_adjacency_weight,
            ).mean().item()
        )
    return score


def _assignment_rgba(model: TileGraphModel, selected: np.ndarray) -> np.ndarray:
    return model.candidate_rgba[selected]


def optimize_tile_graph(
    rgba: np.ndarray,
    inference: InferenceResult,
    analysis: TileGraphSourceAnalysis,
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
    model, build_stats = build_tile_graph_model(
        rgba,
        inference=inference,
        analysis=analysis,
        solver_params=solver_params,
        device=resolved_device,
    )
    choice_indices_np, choice_mask_np = _build_choice_grid(model)
    choice_indices_t = torch.from_numpy(choice_indices_np).to(device=resolved_device, dtype=torch.long)
    choice_mask_t = torch.from_numpy(choice_mask_np).to(device=resolved_device, dtype=torch.bool)
    unary_cost_t, candidate_rgba_t, candidate_neighbor_rgba_t, candidate_neighbor_mask_t, choice_rgba_t = _tile_graph_unary_cost_torch(
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
    choice_neighbor_rgba_t = candidate_neighbor_rgba_t[choice_indices_t]
    choice_neighbor_mask_t = candidate_neighbor_mask_t[choice_indices_t]
    loss_history = [
        _assignment_score_torch(
            torch,
            selected_t,
            unary_cost_t,
            choice_indices_t,
            candidate_rgba_t,
            candidate_neighbor_rgba_t,
            candidate_neighbor_mask_t,
            solver_params,
        )
    ]
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
                    candidate_rgba_t,
                    candidate_neighbor_rgba_t,
                    candidate_neighbor_mask_t,
                    choice_neighbor_rgba_t[:, 1:, :, :, :],
                    choice_neighbor_mask_t[:, 1:, :, :],
                    weight=solver_params.tile_graph_adjacency_weight,
                )
                local_cost_t[:, :-1, :] += _pair_penalty_option_left_torch(
                    torch,
                    choice_rgba_t[:, :-1, :, :],
                    selected_t[:, 1:],
                    candidate_rgba_t,
                    candidate_neighbor_rgba_t,
                    candidate_neighbor_mask_t,
                    choice_neighbor_rgba_t[:, :-1, :, :, :],
                    choice_neighbor_mask_t[:, :-1, :, :],
                    weight=solver_params.tile_graph_adjacency_weight,
                )
            if selected_t.shape[0] > 1:
                local_cost_t[1:, :, :] += _pair_penalty_option_down_torch(
                    torch,
                    selected_t[:-1, :],
                    choice_rgba_t[1:, :, :, :],
                    candidate_rgba_t,
                    candidate_neighbor_rgba_t,
                    candidate_neighbor_mask_t,
                    choice_neighbor_rgba_t[1:, :, :, :, :],
                    choice_neighbor_mask_t[1:, :, :, :],
                    weight=solver_params.tile_graph_adjacency_weight,
                )
                local_cost_t[:-1, :, :] += _pair_penalty_option_up_torch(
                    torch,
                    choice_rgba_t[:-1, :, :, :],
                    selected_t[1:, :],
                    candidate_rgba_t,
                    candidate_neighbor_rgba_t,
                    candidate_neighbor_mask_t,
                    choice_neighbor_rgba_t[:-1, :, :, :, :],
                    choice_neighbor_mask_t[:-1, :, :, :],
                    weight=solver_params.tile_graph_adjacency_weight,
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
                candidate_neighbor_rgba_t,
                candidate_neighbor_mask_t,
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
        "tile_graph_model_device": build_stats.model_device,
        "tile_graph_solver_device": resolved_device,
        "tile_graph_model_cache_hit": build_stats.cache_hit,
        "tile_graph_proposal_mode": "atomic-components",
        "tile_graph_component_count": build_stats.component_count,
        "tile_graph_candidate_count": build_stats.candidate_count,
        "tile_graph_edge_density": build_stats.edge_density,
        "tile_graph_average_choices": build_stats.average_choices,
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
