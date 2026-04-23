from __future__ import annotations

import hashlib
from collections import OrderedDict
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


@dataclass(slots=True)
class AtomicRegionLabeling:
    pixel_linear: np.ndarray
    component_ids: np.ndarray
    component_sizes: np.ndarray
    component_count: int


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


def _project_pixels_to_output_cells(
    *,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    cell_w: float,
    cell_h: float,
    target_width: int,
    target_height: int,
    phase_x: float,
    phase_y: float,
) -> np.ndarray:
    coord_x = np.clip(
        np.floor((pixel_x.astype(np.float32) + 0.5) / max(cell_w, 1e-6) - phase_x).astype(np.int32),
        0,
        max(0, target_width - 1),
    )
    coord_y = np.clip(
        np.floor((pixel_y.astype(np.float32) + 0.5) / max(cell_h, 1e-6) - phase_y).astype(np.int32),
        0,
        max(0, target_height - 1),
    )
    return coord_y * target_width + coord_x


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
    pixel_linear_parts: list[np.ndarray] = []
    component_ids_parts: list[np.ndarray] = []
    component_sizes: list[int] = []
    component_id = 0

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
        pixel_linear_parts.append(member_array)
        component_ids_parts.append(np.full(member_array.shape[0], component_id, dtype=np.int32))
        component_sizes.append(int(member_array.size))
        component_id += 1
    if not pixel_linear_parts:
        return AtomicRegionLabeling(
            pixel_linear=np.zeros((0,), dtype=np.int32),
            component_ids=np.zeros((0,), dtype=np.int32),
            component_sizes=np.zeros((0,), dtype=np.int32),
            component_count=0,
        )
    return AtomicRegionLabeling(
        pixel_linear=np.concatenate(pixel_linear_parts).astype(np.int32, copy=False),
        component_ids=np.concatenate(component_ids_parts).astype(np.int32, copy=False),
        component_sizes=np.asarray(component_sizes, dtype=np.int32),
        component_count=component_id,
    )


def _segment_atomic_source_regions(
    *,
    source_rgba: np.ndarray,
    edge_map: np.ndarray,
    alpha_floor: float,
    color_threshold: float,
    alpha_threshold: float,
    device: str,
) -> AtomicRegionLabeling:
    torch = _require_torch()
    resolved_device = _resolve_device(torch, device)
    height, width = source_rgba.shape[:2]
    if height <= 0 or width <= 0:
        return AtomicRegionLabeling(
            pixel_linear=np.zeros((0,), dtype=np.int32),
            component_ids=np.zeros((0,), dtype=np.int32),
            component_sizes=np.zeros((0,), dtype=np.int32),
            component_count=0,
        )

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
        return AtomicRegionLabeling(
            pixel_linear=np.zeros((0,), dtype=np.int32),
            component_ids=np.zeros((0,), dtype=np.int32),
            component_sizes=np.zeros((0,), dtype=np.int32),
            component_count=0,
        )

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
        return AtomicRegionLabeling(
            pixel_linear=np.zeros((0,), dtype=np.int32),
            component_ids=np.zeros((0,), dtype=np.int32),
            component_sizes=np.zeros((0,), dtype=np.int32),
            component_count=0,
        )

    component_labels_t = labels.reshape(-1)[valid_linear_t]
    _, inverse_t = torch.unique(component_labels_t, sorted=True, return_inverse=True)
    valid_linear = valid_linear_t.detach().cpu().numpy().astype(np.int32)
    inverse = inverse_t.detach().cpu().numpy().astype(np.int32)
    component_count = int(inverse.max()) + 1 if inverse.size else 0
    component_sizes = np.bincount(inverse, minlength=component_count).astype(np.int32)
    return AtomicRegionLabeling(
        pixel_linear=valid_linear,
        component_ids=inverse,
        component_sizes=component_sizes,
        component_count=component_count,
    )


def _extract_source_region_tiles(
    *,
    labeling: AtomicRegionLabeling,
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
) -> list[list[dict[str, float | int | np.ndarray]]]:
    output_area = max(1, target_width * target_height)
    cell_area = max(cell_w * cell_h, 1e-6)
    buckets: list[list[dict[str, float | int | np.ndarray]]] = [[] for _ in range(output_area)]
    if labeling.pixel_linear.size <= 0 or labeling.component_count <= 0:
        return buckets

    pixel_linear = labeling.pixel_linear.astype(np.int32, copy=False)
    component_ids = labeling.component_ids.astype(np.int32, copy=False)
    component_sizes = labeling.component_sizes.astype(np.int32, copy=False)
    pixel_x = flat_x[pixel_linear].astype(np.int32, copy=False)
    pixel_y = flat_y[pixel_linear].astype(np.int32, copy=False)
    projected_flat_index = _project_pixels_to_output_cells(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        cell_w=cell_w,
        cell_h=cell_h,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
    ).astype(np.int32, copy=False)
    compound_key = component_ids.astype(np.int64) * output_area + projected_flat_index.astype(np.int64)
    order = np.argsort(compound_key, kind="stable")
    sorted_key = compound_key[order]
    sorted_linear = pixel_linear[order]
    sorted_component = component_ids[order]
    sorted_cell = projected_flat_index[order]
    sorted_x = pixel_x[order].astype(np.float32)
    sorted_y = pixel_y[order].astype(np.float32)
    sorted_edge = flat_edge[sorted_linear].astype(np.float32)
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_key[1:] != sorted_key[:-1]) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([sorted_key.shape[0]], dtype=np.int64)))
    counts = (ends - starts).astype(np.int32)
    area_ratio = counts.astype(np.float32) / float(cell_area)
    coverage = np.clip(area_ratio, 0.0, 1.0).astype(np.float32)
    sum_x = np.add.reduceat(sorted_x, starts).astype(np.float32)
    sum_y = np.add.reduceat(sorted_y, starts).astype(np.float32)
    center_x = (sum_x / np.maximum(1, counts)).astype(np.float32)
    center_y = (sum_y / np.maximum(1, counts)).astype(np.float32)
    edge_peak = np.maximum.reduceat(sorted_edge, starts).astype(np.float32)
    segment_component = sorted_component[starts].astype(np.int32, copy=False)
    segment_cell = sorted_cell[starts].astype(np.int32, copy=False)
    component_area_ratio = component_sizes[segment_component].astype(np.float32) / float(cell_area)
    keep = (component_area_ratio >= float(min_region_area_ratio)) & (area_ratio >= float(min_window_coverage))

    rep_linear = np.empty(starts.shape[0], dtype=np.int32)
    for segment_index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=False)):
        segment_linear = sorted_linear[start:end]
        if segment_linear.size <= 0:
            rep_linear[segment_index] = 0
            continue
        if edge_peak[segment_index] > 0.0:
            local_edge = sorted_edge[start:end]
            rep_linear[segment_index] = int(segment_linear[int(np.argmax(local_edge))])
        else:
            rep_linear[segment_index] = int(segment_linear[0])

    kept_by_cell: dict[int, list[int]] = {}
    all_by_cell: dict[int, list[int]] = {}
    for segment_index, flat_index in enumerate(segment_cell.tolist()):
        all_by_cell.setdefault(flat_index, []).append(segment_index)
        if keep[segment_index]:
            kept_by_cell.setdefault(flat_index, []).append(segment_index)

    for flat_index, segment_indices in all_by_cell.items():
        use_segments = kept_by_cell.get(flat_index)
        if not use_segments:
            best_segment = min(
                segment_indices,
                key=lambda idx: (
                    abs(float(area_ratio[idx]) - 1.0),
                    -float(coverage[idx]),
                    -float(edge_peak[idx]),
                ),
            )
            use_segments = [best_segment]
        for segment_index in use_segments:
            buckets[flat_index].append(
                {
                    "rep_linear": int(rep_linear[segment_index]),
                    "rep_rgba": flat_rgba[int(rep_linear[segment_index])].astype(np.float32),
                    "area_ratio": float(area_ratio[segment_index]),
                    "coverage": float(coverage[segment_index]),
                    "edge_peak": float(edge_peak[segment_index]),
                    "source_center_x": float(center_x[segment_index]),
                    "source_center_y": float(center_y[segment_index]),
                    "coord_x": int(flat_index % target_width),
                    "coord_y": int(flat_index // target_width),
                    "flat_index": int(flat_index),
                    "component_id": int(segment_component[segment_index]),
                }
            )

    segment_index_by_key = {
        int(segment_component[idx]) * output_area + int(segment_cell[idx]): idx
        for idx in range(segment_cell.shape[0])
    }
    rep_rgba = flat_rgba[rep_linear].astype(np.float32)
    neighbor_rgba = np.zeros((segment_cell.shape[0], 4, 4), dtype=np.float32)
    neighbor_mask = np.zeros((segment_cell.shape[0], 4), dtype=bool)
    for segment_index, flat_index in enumerate(segment_cell.tolist()):
        coord_y = int(flat_index // target_width)
        coord_x = int(flat_index % target_width)
        component_id = int(segment_component[segment_index])
        for direction, dy, dx in _DIRS:
            neighbor_x = coord_x + dx
            neighbor_y = coord_y + dy
            if neighbor_x < 0 or neighbor_x >= target_width or neighbor_y < 0 or neighbor_y >= target_height:
                continue
            neighbor_flat_index = neighbor_y * target_width + neighbor_x
            neighbor_segment = segment_index_by_key.get(component_id * output_area + neighbor_flat_index)
            if neighbor_segment is None:
                continue
            neighbor_rgba[segment_index, direction] = rep_rgba[neighbor_segment]
            neighbor_mask[segment_index, direction] = True

    for bucket in buckets:
        for tile in bucket:
            key = int(tile["component_id"]) * output_area + int(tile["flat_index"])
            segment_index = segment_index_by_key[key]
            tile["neighbor_rgba"] = neighbor_rgba[segment_index]
            tile["neighbor_mask"] = neighbor_mask[segment_index]

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
        labeling=global_components,
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
    )

    candidate_rgba: list[np.ndarray] = []
    candidate_area_ratio: list[float] = []
    candidate_coverage: list[float] = []
    candidate_edge_peak: list[float] = []
    candidate_neighbor_rgba: list[np.ndarray] = []
    candidate_neighbor_mask: list[np.ndarray] = []
    choice_counts_list: list[int] = []
    component_total = int(global_components.component_count)
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
