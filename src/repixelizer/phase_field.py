from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import premultiply
from .observe import PipelineObserver, check_observer_cancelled, emit_observer, observer_attribute
from .types import InferenceResult, SolverArtifacts
from .params import SolverHyperParams


@dataclass(slots=True)
class _PhaseFieldPrep:
    uv0_norm: np.ndarray
    signal: np.ndarray
    width: int
    height: int

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
    energy = np.zeros(full_shape, dtype=np.float32)
    level_field = luma
    levels = max(1, int(solver_params.phase_field_signal_pyramid_levels))
    level_weight = np.float32(1.0)
    level_decay = np.float32(solver_params.phase_field_signal_level_decay)
    for level in range(levels):
        gradient = _gradient_magnitude_np(level_field)
        curvature = _laplacian_abs_np(level_field)
        level_energy = (
            gradient * np.float32(solver_params.phase_field_signal_gradient_weight)
            + curvature * np.float32(solver_params.phase_field_signal_curvature_weight)
        ).astype(np.float32)
        energy += _resize_scalar_to_shape_np(level_energy, full_shape) * level_weight
        if min(level_field.shape[:2]) <= 8:
            break
        level_field = _downsample_half_np(level_field)
        level_weight *= level_decay

    energy = _normalize_percentile_np(
        np.power(np.maximum(energy, np.float32(0.0)), np.float32(solver_params.phase_field_signal_energy_power)),
        low_percentile=5.0,
        high_percentile=99.0,
    )
    signal = np.clip(1.0 - energy, 0.0, 1.0).astype(np.float32)
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
    steps: int,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
    observer: PipelineObserver | None = None,
) -> SolverArtifacts:
    del device
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
        solver_params,
        cell_x=cell_x,
        cell_y=cell_y,
    )
    signal = signal_map
    prep = _PhaseFieldPrep(
        uv0_norm=uv0_norm,
        signal=signal,
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
        prepared_payload["signal"] = signal.copy()
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
        signal_strength=prep.signal,
        initial_rgba=initial_rgba,
        loss_history=loss_history,
        stage_diagnostics=stage_diagnostics,
    )
