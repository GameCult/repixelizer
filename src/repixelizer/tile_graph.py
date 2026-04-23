from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
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
    geometry_reference_rgba: np.ndarray | None = None
    geometry_strength: np.ndarray | None = None
    cache_hit: bool = False


_TILE_GRAPH_MODEL_CACHE_MAX_ENTRIES = 8
_TILE_GRAPH_MODEL_CACHE: OrderedDict[tuple[object, ...], TileGraphModel] = OrderedDict()


def clear_tile_graph_model_cache() -> None:
    _TILE_GRAPH_MODEL_CACHE.clear()


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
        "tile_graph_source_region_stride",
        "tile_graph_source_region_min_area_ratio",
        "tile_graph_source_region_window_coverage",
        "tile_graph_stroke_linearity_threshold",
        "tile_graph_stroke_step_scale",
        "tile_graph_stroke_minor_limit_scale",
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


def _get_cached_tile_graph_model(cache_key: tuple[object, ...]) -> TileGraphModel | None:
    cached = _TILE_GRAPH_MODEL_CACHE.get(cache_key)
    if cached is None:
        return None
    _TILE_GRAPH_MODEL_CACHE.move_to_end(cache_key)
    return cached


def _store_cached_tile_graph_model(cache_key: tuple[object, ...], model: TileGraphModel) -> None:
    _TILE_GRAPH_MODEL_CACHE[cache_key] = model
    _TILE_GRAPH_MODEL_CACHE.move_to_end(cache_key)
    while len(_TILE_GRAPH_MODEL_CACHE) > _TILE_GRAPH_MODEL_CACHE_MAX_ENTRIES:
        _TILE_GRAPH_MODEL_CACHE.popitem(last=False)


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
    sample_area: float,
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
    min_region_pixels = max(1, int(np.ceil(cell_area * min_region_area_ratio / max(sample_area, 1e-6))))
    ordered_components = sorted(
        components,
        key=lambda comp: (abs(float(comp["size"]) - cell_area), -float(comp["edge_peak"])),
    )

    def overlap_tiles_for_component(component: dict[str, float | int | np.ndarray]) -> list[dict[str, float | int | np.ndarray]]:
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
                    "area_ratio": float((cell_linear.shape[0] * sample_area) / cell_area),
                    "coverage": float(np.clip((cell_linear.shape[0] * sample_area) / cell_area, 0.0, 1.0)),
                    "edge_peak": edge_peak,
                    "source_center_x": center_x,
                    "source_center_y": center_y,
                    "coord_x": int(flat_index % target_width),
                    "coord_y": int(flat_index // target_width),
                    "flat_index": flat_index,
                    "guaranteed_fill": True,
                }
            )
        return tiles

    for component in ordered_components:
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
            area_ratio = float((footprint_count * sample_area) / cell_area)
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
                buckets[int(tile["flat_index"])].append(tile)
            continue

        component_area_ratio = float((component_size * sample_area) / cell_area)
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
            }
        )

    for component in ordered_components:
        for tile in overlap_tiles_for_component(component):
            if buckets[int(tile["flat_index"])]:
                continue
            buckets[int(tile["flat_index"])].append(tile)

    return buckets


def _select_source_region_candidates(
    *,
    candidates: list[dict[str, float | int | np.ndarray]],
    allowed_candidates: int,
    reference_rgba: np.ndarray,
    edge_reference_rgba: np.ndarray,
    prefer_edge: bool,
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
    by_reference = sorted(
        candidates,
        key=lambda cand: float(np.mean(np.abs(np.asarray(cand["rep_rgba"], dtype=np.float32) - reference_rgba))),
    )
    by_edge_reference = sorted(
        candidates,
        key=lambda cand: float(np.mean(np.abs(np.asarray(cand["rep_rgba"], dtype=np.float32) - edge_reference_rgba))),
    )
    by_edge = sorted(candidates, key=lambda cand: (-float(cand["edge_peak"]), abs(float(cand["area_ratio"]) - 1.0)))

    add(by_area[0])
    add(by_reference[0])
    if prefer_edge:
        add(by_edge_reference[0])
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
    analysis: SourceAnalysis,
    solver_params: SolverHyperParams | None = None,
    device: str = "cpu",
    geometry_reference_rgba: np.ndarray | None = None,
    geometry_guidance_strength: np.ndarray | None = None,
) -> TileGraphModel:
    torch = _require_torch()
    resolved_device = _resolve_device(torch, device)
    solver_params = solver_params or SolverHyperParams()
    geometry_reference_np = None
    if geometry_reference_rgba is not None:
        geometry_reference_np = np.asarray(geometry_reference_rgba, dtype=np.float32)
        if geometry_reference_np.shape[:2] != (inference.target_height, inference.target_width):
            raise ValueError(
                "geometry_reference_rgba must match the inferred output size when provided to tile-graph."
            )
    geometry_strength_np = None
    if geometry_guidance_strength is not None:
        geometry_strength_np = np.asarray(geometry_guidance_strength, dtype=np.float32)
        if geometry_strength_np.shape != (inference.target_height, inference.target_width):
            raise ValueError(
                "geometry_guidance_strength must match the inferred output size when provided to tile-graph."
            )
    cache_key = _tile_graph_model_cache_key(
        source_rgba,
        inference=inference,
        solver_params=solver_params,
        device=resolved_device,
    )
    cached_model = _get_cached_tile_graph_model(cache_key)
    if cached_model is not None:
        return replace(
            cached_model,
            geometry_reference_rgba=geometry_reference_np,
            geometry_strength=geometry_strength_np,
            cache_hit=True,
        )
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
    sharp_x_flat_t = torch.from_numpy(source_reference.sharp_x.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    sharp_y_flat_t = torch.from_numpy(source_reference.sharp_y.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    edge_peak_x_flat_t = torch.from_numpy(source_reference.edge_peak_x.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    edge_peak_y_flat_t = torch.from_numpy(source_reference.edge_peak_y.reshape(-1)).to(device=resolved_device, dtype=torch.long)
    sharp_linear_t = sharp_y_flat_t * width + sharp_x_flat_t
    edge_linear_t = edge_peak_y_flat_t * width + edge_peak_x_flat_t

    flat_rgba_np = source_rgba.reshape(-1, source_rgba.shape[-1]).astype(np.float32)
    flat_edge_np = analysis.edge_map.reshape(-1).astype(np.float32)
    source_region_stride = int(getattr(solver_params, "tile_graph_source_region_stride", 0))
    if source_region_stride <= 0:
        if max(height, width) <= 256 or output_area <= 1024:
            source_region_stride = 1
        else:
            source_region_stride = max(1, int(np.floor(min(cell_w, cell_h) / 4.0)))
    source_region_stride = max(1, min(source_region_stride, 3))
    sampled_rgba = source_rgba[::source_region_stride, ::source_region_stride].astype(np.float32)
    sampled_edge = analysis.edge_map[::source_region_stride, ::source_region_stride].astype(np.float32)
    sampled_y_coords = np.arange(0, height, source_region_stride, dtype=np.int32)
    sampled_x_coords = np.arange(0, width, source_region_stride, dtype=np.int32)
    sampled_y_grid, sampled_x_grid = np.meshgrid(sampled_y_coords, sampled_x_coords, indexing="ij")
    sampled_flat_rgba_np = sampled_rgba.reshape(-1, sampled_rgba.shape[-1]).astype(np.float32)
    sampled_flat_edge_np = sampled_edge.reshape(-1).astype(np.float32)
    sampled_flat_y_np = sampled_y_grid.reshape(-1).astype(np.int32)
    sampled_flat_x_np = sampled_x_grid.reshape(-1).astype(np.int32)
    flat_y_np, flat_x_np = np.divmod(np.arange(flat_rgba_np.shape[0], dtype=np.int64), width)
    flat_y_np = flat_y_np.astype(np.int32)
    flat_x_np = flat_x_np.astype(np.int32)
    sharp_linear_np = sharp_linear_t.detach().cpu().numpy().astype(np.int64)
    edge_linear_np = edge_linear_t.detach().cpu().numpy().astype(np.int64)
    flat_alpha_np = flat_rgba_np[:, 3].astype(np.float32)
    edge_rgba_np = flat_rgba_np[edge_linear_np]
    edge_strength_np = source_reference.edge_strength.reshape(-1).astype(np.float32)
    global_components = _segment_atomic_source_regions(
        source_rgba=sampled_rgba,
        edge_map=sampled_edge,
        alpha_floor=solver_params.alpha_transparent_threshold,
        color_threshold=float(getattr(solver_params, "tile_graph_component_color_threshold", 0.055)),
        alpha_threshold=float(getattr(solver_params, "tile_graph_component_alpha_threshold", 0.12)),
        device=resolved_device,
    )
    region_buckets = _extract_source_region_tiles(
        components=global_components,
        flat_rgba=sampled_flat_rgba_np,
        flat_edge=sampled_flat_edge_np,
        flat_x=sampled_flat_x_np,
        flat_y=sampled_flat_y_np,
        cell_w=cell_w,
        cell_h=cell_h,
        sample_area=float(source_region_stride * source_region_stride),
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

    candidate_linear: list[int] = []
    candidate_coords: list[tuple[int, int]] = []
    candidate_area_ratio: list[float] = []
    candidate_coverage: list[float] = []
    candidate_source_x: list[float] = []
    candidate_source_y: list[float] = []
    choice_counts_list: list[int] = []
    component_total = len(global_components)
    edge_candidate_cap = max(max_candidates_per_coord, int(getattr(solver_params, "tile_graph_edge_candidates_per_coord", max_candidates_per_coord)))
    for flat_index in range(output_area):
        coord_y = flat_index // inference.target_width
        coord_x = flat_index % inference.target_width
        seen_pixels: set[int] = set()
        cell_start = len(candidate_linear)
        edge_cell = edge_strength_np[flat_index] >= solver_params.source_edge_detail_threshold
        allowed_candidates = edge_candidate_cap if edge_cell else max_candidates_per_coord

        def add_candidate(pixel_index: int, *, area_ratio: float, coverage: float, source_x: float, source_y: float) -> None:
            if pixel_index in seen_pixels or len(seen_pixels) >= allowed_candidates:
                return
            seen_pixels.add(pixel_index)
            candidate_linear.append(pixel_index)
            candidate_coords.append((coord_y, coord_x))
            candidate_area_ratio.append(float(max(area_ratio, 1e-3)))
            candidate_coverage.append(float(np.clip(coverage, 0.0, 1.0)))
            candidate_source_x.append(float(source_x))
            candidate_source_y.append(float(source_y))

        selected_regions = _select_source_region_candidates(
            candidates=region_buckets[flat_index],
            allowed_candidates=allowed_candidates,
            reference_rgba=source_reference.sharp_rgba.reshape(-1, 4)[flat_index],
            edge_reference_rgba=edge_rgba_np[flat_index],
            prefer_edge=edge_cell,
        )
        selected_only_guaranteed_fill = bool(selected_regions) and all(
            bool(region.get("guaranteed_fill", False))
            for region in selected_regions
        )
        for region in selected_regions:
            pixel_index = int(region["rep_linear"])
            add_candidate(
                pixel_index,
                area_ratio=float(region["area_ratio"]),
                coverage=float(region["coverage"]),
                source_x=float(region["source_center_x"]),
                source_y=float(region["source_center_y"]),
            )
        if len(candidate_linear) == cell_start or selected_only_guaranteed_fill:
            sharp_pixel = int(sharp_linear_np[flat_index])
            add_candidate(
                sharp_pixel,
                area_ratio=1.0,
                coverage=max(0.05, float(flat_alpha_np[sharp_pixel])),
                source_x=float(flat_x_np[sharp_pixel]),
                source_y=float(flat_y_np[sharp_pixel]),
            )
            if edge_cell:
                edge_pixel = int(edge_linear_np[flat_index])
                add_candidate(
                    edge_pixel,
                    area_ratio=1.0,
                    coverage=max(0.05, float(flat_alpha_np[edge_pixel])),
                    source_x=float(flat_x_np[edge_pixel]),
                    source_y=float(flat_y_np[edge_pixel]),
                )
        choice_counts_list.append(len(candidate_linear) - cell_start)

    candidate_linear_t = torch.as_tensor(candidate_linear, device=resolved_device, dtype=torch.long)
    center_rgba_t = flat_rgba_t[candidate_linear_t]
    center_y_t = torch.as_tensor(candidate_source_y, device=resolved_device, dtype=torch.float32)
    center_x_t = torch.as_tensor(candidate_source_x, device=resolved_device, dtype=torch.float32)
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
    base_model = TileGraphModel(
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
        geometry_reference_rgba=None,
        geometry_strength=None,
        cache_hit=False,
    )
    _store_cached_tile_graph_model(cache_key, base_model)
    return replace(
        base_model,
        geometry_reference_rgba=geometry_reference_np,
        geometry_strength=geometry_strength_np,
        cache_hit=False,
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
            total_cost = (
                color_error
                + area_error * solver_params.tile_graph_area_weight
                + alpha_error * solver_params.tile_graph_alpha_weight
                + coverage_error * solver_params.tile_graph_coverage_weight
            )
            if model.geometry_reference_rgba is not None:
                geometry_reference = model.geometry_reference_rgba[y, x]
                geometry_error = np.mean(np.abs(geometry_reference[None, :] - candidates), axis=-1)
                geometry_strength = 1.0
                if model.geometry_strength is not None:
                    geometry_strength += (
                        float(np.clip(model.geometry_strength[y, x], 0.0, 1.0))
                        * solver_params.hybrid_geometry_edge_boost
                    )
                total_cost = total_cost + geometry_error * (
                    solver_params.hybrid_geometry_match_weight * geometry_strength
                )
            unary_cost[start:end] = total_cost.astype(np.float32)
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
    geometry_reference_t = (
        torch.from_numpy(model.geometry_reference_rgba).to(device=device, dtype=torch.float32)
        if model.geometry_reference_rgba is not None
        else None
    )
    geometry_strength_t = (
        torch.from_numpy(model.geometry_strength).to(device=device, dtype=torch.float32)
        if model.geometry_strength is not None
        else None
    )

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
    if geometry_reference_t is not None:
        geometry_error_t = (geometry_reference_t[..., None, :] - choice_rgba_t).abs().mean(dim=-1)
        geometry_scale_t = 1.0
        if geometry_strength_t is not None:
            geometry_scale_t = 1.0 + geometry_strength_t[..., None].clamp(0.0, 1.0) * solver_params.hybrid_geometry_edge_boost
        unary_cost_t = unary_cost_t + geometry_error_t * (
            solver_params.hybrid_geometry_match_weight * geometry_scale_t
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
    geometry_reference_rgba: np.ndarray | None = None,
    geometry_guidance_strength: np.ndarray | None = None,
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
        geometry_reference_rgba=geometry_reference_rgba,
        geometry_guidance_strength=geometry_guidance_strength,
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
        "tile_graph_model_cache_hit": model.cache_hit,
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
        "tile_graph_geometry_prior": model.geometry_reference_rgba is not None,
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
