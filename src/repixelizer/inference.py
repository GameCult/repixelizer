from __future__ import annotations

import numpy as np

from .io import premultiply
from .metrics import source_lattice_evidence_breakdown
from .observe import PipelineObserver, emit_observer
from .types import InferenceCandidate, InferenceResult


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - exercised only when torch missing
        raise RuntimeError(
            "PyTorch is required for lattice inference. Install project dependencies first."
        ) from exc
    return torch, F


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for lattice inference, but this PyTorch build does not have a usable CUDA device. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return requested


def _candidate_dims(
    width: int,
    height: int,
    target_size: int | None,
    *,
    hinted_sizes: list[int] | None = None,
) -> list[tuple[int, int]]:
    if target_size is not None:
        if width >= height:
            return [(target_size, max(1, round(height * target_size / width)))]
        return [(max(1, round(width * target_size / height)), target_size)]
    max_dim = max(width, height)
    min_size = max(12, int(round(max_dim / 48)))
    max_size = min(256, max(24, int(round(max_dim / 2.5))))
    step = 2 if max_size <= 96 else 4
    size_values = set(range(min_size, max_size + 1, step))
    if hinted_sizes:
        for hinted_size in hinted_sizes:
            for delta in range(-2, 3):
                candidate = int(hinted_size) + delta
                if min_size <= candidate <= max_size:
                    size_values.add(candidate)
    dims: list[tuple[int, int]] = []
    for size in sorted(size_values):
        if width >= height:
            dims.append((size, max(1, round(height * size / width))))
        else:
            dims.append((max(1, round(width * size / height)), size))
    return dims


def _strong_spacing_size_window(
    hinted_sizes: list[int],
    *,
    spacing_x: tuple[float | None, float],
    spacing_y: tuple[float | None, float],
    prior_reliability: float,
) -> tuple[int, int] | None:
    if not hinted_sizes:
        return None
    axis_confidences = [float(confidence) for _spacing, confidence in (spacing_x, spacing_y) if confidence > 1e-6]
    if not axis_confidences:
        return None
    center = int(round(float(np.median(np.asarray(hinted_sizes, dtype=np.float32)))))
    spread = max(abs(int(size) - center) for size in hinted_sizes)
    mean_confidence = float(np.mean(axis_confidences))
    max_confidence = float(max(axis_confidences))
    if prior_reliability < 0.45 or max_confidence < 0.55 or spread > 2:
        return None
    radius = 0 if spread == 0 and mean_confidence >= 0.72 and prior_reliability >= 0.68 else 1
    return center, radius


def _resolve_candidate_dims_from_spacing(
    width: int,
    height: int,
    target_size: int | None,
    *,
    hinted_sizes: list[int],
    spacing_x: tuple[float | None, float],
    spacing_y: tuple[float | None, float],
    prior_reliability: float,
) -> list[tuple[int, int]]:
    dims = _candidate_dims(width, height, target_size, hinted_sizes=hinted_sizes)
    size_window = _strong_spacing_size_window(
        hinted_sizes,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        prior_reliability=prior_reliability,
    )
    if size_window is None:
        return dims
    center, radius = size_window
    narrowed = [dim for dim in dims if abs(max(dim) - center) <= radius]
    return narrowed or dims


def _estimate_cell_size(profile: np.ndarray) -> float:
    centered = profile.astype(np.float32) - float(np.mean(profile))
    max_lag = min(64, max(4, profile.shape[0] // 3))
    best_lag = 1
    best_score = -float("inf")
    for lag in range(2, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denom = float(np.linalg.norm(left) * np.linalg.norm(right)) + 1e-6
        score = float(np.dot(left, right) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    return float(best_lag)


def _estimate_spacing_cell_size(deltas: np.ndarray, axis_length: int) -> tuple[float | None, float]:
    intervals, weights = _collect_change_intervals(deltas)
    if intervals.size < 6:
        return None, 0.0
    max_candidate = min(64, max(4, axis_length // 2))
    candidates = np.arange(2, max_candidate + 1, dtype=np.float32)
    scores = np.asarray([_spacing_score(intervals, weights, cell) for cell in candidates], dtype=np.float32)
    best_index = int(np.argmax(scores))
    best_cell = float(candidates[best_index])
    best_score = float(scores[best_index])
    multiples = np.maximum(1.0, np.rint(intervals / best_cell))
    residual = np.abs(intervals / best_cell - multiples)
    keep = residual < 0.2
    if np.any(keep):
        refined_weights = weights[keep] / np.sqrt(multiples[keep])
        refined = float(np.average(intervals[keep] / multiples[keep], weights=refined_weights))
    else:
        refined = best_cell
    coverage = min(1.0, intervals.size / 24.0)
    confidence = np.clip((best_score - 0.35) / 0.45, 0.0, 1.0) * coverage
    return refined, float(confidence)


def _collect_change_intervals(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = deltas[deltas > 1e-4]
    if positive.size < 8:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    threshold = float(np.clip(np.quantile(positive, 0.72) * 0.55, 0.018, 0.16))
    intervals: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for line in deltas:
        centers, strengths = _boundary_runs(line, threshold)
        if centers.size < 2:
            continue
        line_intervals = np.diff(centers)
        if line_intervals.size == 0:
            continue
        keep = line_intervals >= 2.0
        if not np.any(keep):
            continue
        line_weights = ((strengths[:-1] + strengths[1:]) * 0.5)[keep]
        intervals.append(line_intervals[keep].astype(np.float32))
        weights.append(line_weights.astype(np.float32))
    if not intervals:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(intervals), np.concatenate(weights)


def _boundary_runs(line: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(line >= threshold)
    if indices.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    splits = np.where(np.diff(indices) > 1)[0]
    starts = np.concatenate([indices[:1], indices[splits + 1]])
    ends = np.concatenate([indices[splits], indices[-1:]])
    centers = (starts.astype(np.float32) + ends.astype(np.float32) + 1.0) * 0.5
    strengths = np.asarray([float(np.max(line[start : end + 1])) for start, end in zip(starts, ends)], dtype=np.float32)
    return centers, strengths


def _spacing_score(intervals: np.ndarray, weights: np.ndarray, cell_size: float) -> float:
    multiples = np.maximum(1.0, np.rint(intervals / cell_size))
    residual = np.abs(intervals / cell_size - multiples)
    closeness = np.exp(-((residual / 0.18) ** 2))
    interval_weights = weights / np.sqrt(multiples)
    return float(np.sum(interval_weights * closeness) / (np.sum(interval_weights) + 1e-6))


def _estimate_lattice_spacing(rgba: np.ndarray) -> tuple[tuple[float | None, float], tuple[float | None, float]]:
    premult_rgba = premultiply(rgba)
    delta_x = np.linalg.norm(premult_rgba[:, 1:] - premult_rgba[:, :-1], axis=-1)
    delta_y = np.linalg.norm(premult_rgba[1:, :] - premult_rgba[:-1, :], axis=-1)
    return _estimate_spacing_cell_size(delta_x, premult_rgba.shape[1]), _estimate_spacing_cell_size(delta_y, premult_rgba.shape[0])


def _weighted_geometric_mean(values: list[float], weights: list[float]) -> float:
    safe_values = [float(max(1e-6, value)) for value in values]
    safe_weights = [float(max(0.0, weight)) for weight in weights]
    total_weight = float(sum(safe_weights))
    if total_weight <= 1e-6:
        return float(np.mean(safe_values))
    logs = np.log(np.asarray(safe_values, dtype=np.float32))
    return float(np.exp(np.dot(logs, np.asarray(safe_weights, dtype=np.float32)) / total_weight))


def _axis_prior_from_estimates(spacing: float | None, spacing_confidence: float, autocorr: float) -> tuple[float, float]:
    if spacing is None:
        return float(autocorr), 0.12
    ratio = max(autocorr, spacing) / max(1e-6, min(autocorr, spacing))
    suspicious_multiple = ratio >= 1.9 and abs(ratio - round(ratio)) <= 0.2
    autocorr_weight = float(np.clip(1.0 - spacing_confidence, 0.05, 0.65))
    if suspicious_multiple and spacing_confidence >= 0.15:
        autocorr_weight *= 0.35
    prior = _weighted_geometric_mean([spacing, autocorr], [1.0, autocorr_weight])
    reliability = float(np.clip(0.15 + 0.75 * spacing_confidence, 0.1, 0.9))
    if suspicious_multiple:
        reliability *= 0.9
    return prior, float(reliability)


def _combine_axis_priors(axis_priors: list[tuple[float, float]]) -> tuple[float, float]:
    priors = [float(prior) for prior, weight in axis_priors if weight > 1e-6]
    weights = [float(weight) for _, weight in axis_priors if weight > 1e-6]
    if not priors:
        return 16.0, 0.0
    shared_prior = _weighted_geometric_mean(priors, weights)
    if len(priors) == 1:
        return shared_prior, float(np.clip(weights[0], 0.1, 0.9))
    log_priors = np.log(np.asarray(priors, dtype=np.float32))
    spread = float(np.mean(np.abs(log_priors - np.mean(log_priors))))
    consistency = float(np.exp(-spread * 1.25))
    reliability = float(np.clip((sum(weights) / len(weights)) * consistency, 0.08, 0.9))
    return shared_prior, reliability


def _estimate_lattice_prior_details(
    rgba: np.ndarray,
    *,
    spacing_x: tuple[float | None, float] | None = None,
    spacing_y: tuple[float | None, float] | None = None,
) -> tuple[float, float, float]:
    alpha = rgba[..., 3]
    luminance = rgba[..., 0] * 0.2126 + rgba[..., 1] * 0.7152 + rgba[..., 2] * 0.0722
    dx = np.zeros_like(luminance)
    dy = np.zeros_like(luminance)
    dx[:, 1:] = np.abs(luminance[:, 1:] - luminance[:, :-1]) + np.abs(alpha[:, 1:] - alpha[:, :-1])
    dy[1:, :] = np.abs(luminance[1:, :] - luminance[:-1, :]) + np.abs(alpha[1:, :] - alpha[:-1, :])
    profile_x = (dx.mean(axis=0) + alpha.mean(axis=0) * 0.1).astype(np.float32)
    profile_y = (dy.mean(axis=1) + alpha.mean(axis=1) * 0.1).astype(np.float32)
    if spacing_x is None or spacing_y is None:
        inferred_spacing_x, inferred_spacing_y = _estimate_lattice_spacing(rgba)
        spacing_x = inferred_spacing_x if spacing_x is None else spacing_x
        spacing_y = inferred_spacing_y if spacing_y is None else spacing_y
    autocorr_x = _estimate_cell_size(profile_x)
    autocorr_y = _estimate_cell_size(profile_y)
    axis_x = _axis_prior_from_estimates(spacing_x[0], spacing_x[1], autocorr_x)
    axis_y = _axis_prior_from_estimates(spacing_y[0], spacing_y[1], autocorr_y)
    shared_prior, reliability = _combine_axis_priors([axis_x, axis_y])
    return shared_prior, shared_prior, reliability


def _hint_target_sizes_from_spacing(
    width: int,
    height: int,
    spacing_x: tuple[float | None, float],
    spacing_y: tuple[float | None, float],
) -> list[int]:
    hinted_sizes: set[int] = set()
    if width >= height:
        if spacing_x[0] is not None and spacing_x[1] >= 0.15:
            hinted_sizes.add(max(1, int(round(width / max(1e-6, spacing_x[0])))))
        if spacing_y[0] is not None and spacing_y[1] >= 0.15:
            hinted_sizes.add(max(1, int(round(height / max(1e-6, spacing_y[0])))))
    else:
        if spacing_y[0] is not None and spacing_y[1] >= 0.15:
            hinted_sizes.add(max(1, int(round(height / max(1e-6, spacing_y[0])))))
        if spacing_x[0] is not None and spacing_x[1] >= 0.15:
            hinted_sizes.add(max(1, int(round(width / max(1e-6, spacing_x[0])))))
    return sorted(hinted_sizes)


def _top_candidates_by_size(candidates: list[InferenceCandidate], limit: int = 8) -> list[InferenceCandidate]:
    selected: list[InferenceCandidate] = []
    seen_sizes: set[tuple[int, int]] = set()
    for candidate in candidates:
        size_key = (candidate.target_width, candidate.target_height)
        if size_key in seen_sizes:
            continue
        selected.append(candidate)
        seen_sizes.add(size_key)
        if len(selected) >= limit:
            break
    return selected


def _normalize_candidate_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    if values.size == 1:
        return np.ones_like(values, dtype=np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span <= 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - minimum) / span).astype(np.float32)


def _rerank_size_candidates_with_source_evidence(
    rgba: np.ndarray,
    candidates: list[InferenceCandidate],
) -> list[InferenceCandidate]:
    if len(candidates) <= 1:
        return candidates

    base_scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float32)
    evidence_breakdowns = [
        source_lattice_evidence_breakdown(
            rgba,
            target_width=candidate.target_width,
            target_height=candidate.target_height,
            phase_x=candidate.phase_x,
            phase_y=candidate.phase_y,
        )
        for candidate in candidates
    ]
    evidence_scores = np.asarray([breakdown["score"] for breakdown in evidence_breakdowns], dtype=np.float32)
    base_norm = _normalize_candidate_scores(base_scores)
    evidence_norm = _normalize_candidate_scores(evidence_scores)

    reranked: list[InferenceCandidate] = []
    for candidate, evidence, base_component, evidence_component in zip(candidates, evidence_breakdowns, base_norm, evidence_norm):
        breakdown = dict(candidate.breakdown)
        breakdown["base_score"] = float(candidate.score)
        breakdown["source_cell_dispersion"] = float(evidence["cell_dispersion"])
        breakdown["source_adjacency_strength"] = float(evidence["adjacency_strength"])
        breakdown["source_lattice_evidence"] = float(evidence["score"])
        breakdown["base_score_norm"] = float(base_component)
        breakdown["source_lattice_evidence_norm"] = float(evidence_component)
        combined_score = float(base_component * 0.35 + evidence_component * 0.65)
        reranked.append(
            InferenceCandidate(
                target_width=candidate.target_width,
                target_height=candidate.target_height,
                phase_x=candidate.phase_x,
                phase_y=candidate.phase_y,
                score=combined_score,
                breakdown=breakdown,
            )
        )
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked


def _estimate_lattice_prior(rgba: np.ndarray) -> tuple[float, float]:
    prior_x, prior_y, _ = _estimate_lattice_prior_details(rgba)
    return prior_x, prior_y


def _build_phase_grid(
    torch,
    *,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
    phase_xs,
    phase_ys,
    device: str,
):
    batch = phase_xs.shape[0]
    cell_x = width / target_width
    cell_y = height / target_height
    base_x = (torch.arange(target_width, device=device, dtype=torch.float32) + 0.5) * cell_x - 0.5
    base_y = (torch.arange(target_height, device=device, dtype=torch.float32) + 0.5) * cell_y - 0.5
    xs = base_x[None, None, :].expand(batch, target_height, target_width) + phase_xs[:, None, None] * cell_x
    ys = base_y[None, :, None].expand(batch, target_height, target_width) + phase_ys[:, None, None] * cell_y
    xs = xs.clamp(0.0, max(0.0, width - 1))
    ys = ys.clamp(0.0, max(0.0, height - 1))
    grid_x = (xs / max(1.0, width - 1)) * 2.0 - 1.0
    grid_y = (ys / max(1.0, height - 1)) * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def _unpremultiply_batch(torch, premult_batch):
    alpha = premult_batch[:, 3:4].clamp_min(1e-6)
    rgb = torch.where(alpha > 1e-6, premult_batch[:, :3] / alpha, torch.zeros_like(premult_batch[:, :3]))
    return torch.cat([rgb, premult_batch[:, 3:4]], dim=1).clamp(0.0, 1.0)


def _coherence_breakdown_torch(torch, F, rgba_batch):
    batch = rgba_batch.shape[0]
    color_rgba = rgba_batch.permute(0, 2, 3, 1)
    alpha = rgba_batch[:, 3]

    cluster_sims = []
    for dy, dx in ((0, 1), (1, 0)):
        shifted = torch.roll(color_rgba, shifts=(dy, dx), dims=(1, 2))
        dist = torch.linalg.vector_norm(color_rgba - shifted, dim=-1)
        cluster_sims.append(torch.exp(-dist * 8.0))
    cluster_continuity = torch.stack(cluster_sims, dim=0).mean(dim=(0, 2, 3))

    alpha_crispness = 1.0 - (4.0 * alpha * (1.0 - alpha)).mean(dim=(1, 2))

    neighbors = []
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        shifted = torch.roll(color_rgba, shifts=(dy, dx), dims=(1, 2))
        dist = torch.linalg.vector_norm(color_rgba - shifted, dim=-1)
        neighbors.append(dist > 0.15)
    isolated_penalty = (torch.stack(neighbors, dim=0).sum(dim=0) >= 3).float().mean(dim=(1, 2))

    mask = alpha >= 0.5
    has_mask = mask.flatten(1).any(dim=1)
    if mask.shape[1] > 1 and mask.shape[2] > 1:
        a = mask[:, :-1, :-1].to(dtype=torch.int32)
        b = mask[:, :-1, 1:].to(dtype=torch.int32)
        c = mask[:, 1:, :-1].to(dtype=torch.int32)
        d = mask[:, 1:, 1:].to(dtype=torch.int32)
        block_sum = a + b + c + d
        corner_blocks = ((block_sum == 1) | (block_sum == 3)).sum(dim=(1, 2)).to(dtype=torch.float32)
    else:
        corner_blocks = torch.zeros(batch, device=rgba_batch.device, dtype=torch.float32)
    edge_pixels = (mask ^ torch.roll(mask, shifts=1, dims=1)) | (mask ^ torch.roll(mask, shifts=1, dims=2))
    edge_count = torch.count_nonzero(edge_pixels, dim=(1, 2)).to(dtype=torch.float32).clamp_min(1.0)
    outline_straightness = torch.where(
        has_mask,
        1.0 - corner_blocks / edge_count,
        torch.ones_like(edge_count),
    )

    color_rgb = rgba_batch[:, :3]
    padded = F.pad(color_rgb, (1, 1, 1, 1), mode="replicate")
    patches = padded.unfold(2, 3, 1).unfold(3, 3, 1).contiguous()
    patches = patches.view(batch, 3, rgba_batch.shape[2], rgba_batch.shape[3], 9)
    median = patches.median(dim=-1).values
    color_chatter = torch.linalg.vector_norm(color_rgb - median, dim=1).mean(dim=(1, 2))

    coherence_score = (
        0.25 * cluster_continuity
        + 0.20 * alpha_crispness
        + 0.20 * outline_straightness
        + 0.20 * (1.0 - isolated_penalty)
        + 0.15 * (1.0 - torch.minimum(torch.ones_like(color_chatter), color_chatter))
    )
    return {
        "cluster_continuity": cluster_continuity,
        "alpha_crispness": alpha_crispness,
        "outline_straightness": outline_straightness,
        "isolated_penalty": isolated_penalty,
        "color_chatter": color_chatter,
        "coherence_score": coherence_score,
    }


def _score_phase_group(
    rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    prior_cell_x: float,
    prior_cell_y: float,
    prior_reliability: float,
    phase_x_values: np.ndarray,
    phase_y_values: np.ndarray,
    device: str,
):
    torch, F = _require_torch()
    height, width = rgba.shape[:2]
    premult_rgba = premultiply(rgba)
    source_t = torch.from_numpy(premult_rgba.transpose(2, 0, 1)[None, ...]).to(device=device, dtype=torch.float32)
    phase_xs, phase_ys = np.meshgrid(phase_x_values, phase_y_values, indexing="xy")
    phase_x_t = torch.from_numpy(phase_xs.reshape(-1).astype(np.float32)).to(device=device)
    phase_y_t = torch.from_numpy(phase_ys.reshape(-1).astype(np.float32)).to(device=device)
    batch = phase_x_t.shape[0]
    grid = _build_phase_grid(
        torch,
        width=width,
        height=height,
        target_width=target_width,
        target_height=target_height,
        phase_xs=phase_x_t,
        phase_ys=phase_y_t,
        device=device,
    )
    source_batch = source_t.expand(batch, -1, -1, -1)
    sampled = F.grid_sample(source_batch, grid, align_corners=True, mode="bilinear", padding_mode="border")
    rgba_sample = _unpremultiply_batch(torch, sampled)
    breakdown = _coherence_breakdown_torch(torch, F, rgba_sample)
    cell_x = width / target_width
    cell_y = height / target_height
    prior_score_x = np.exp(-abs(np.log((cell_x + 1e-6) / (prior_cell_x + 1e-6))) * 0.8)
    prior_score_y = np.exp(-abs(np.log((cell_y + 1e-6) / (prior_cell_y + 1e-6))) * 0.8)
    size_prior = float((prior_score_x + prior_score_y) * 0.5)
    size_prior_weight = float(np.clip(0.10 + prior_reliability * 0.35, 0.10, 0.45))
    score = breakdown["coherence_score"] * (1.0 - size_prior_weight) + size_prior * size_prior_weight
    score_np = score.detach().cpu().numpy()

    candidates: list[InferenceCandidate] = []
    for index, phase_x in enumerate(phase_x_t.detach().cpu().numpy()):
        phase_y = float(phase_y_t[index].detach().cpu().item())
        candidate_breakdown = {
            "cluster_continuity": float(breakdown["cluster_continuity"][index].detach().cpu().item()),
            "alpha_crispness": float(breakdown["alpha_crispness"][index].detach().cpu().item()),
            "outline_straightness": float(breakdown["outline_straightness"][index].detach().cpu().item()),
            "isolated_penalty": float(breakdown["isolated_penalty"][index].detach().cpu().item()),
            "color_chatter": float(breakdown["color_chatter"][index].detach().cpu().item()),
            "coherence_score": float(breakdown["coherence_score"][index].detach().cpu().item()),
            "size_prior": size_prior,
            "size_prior_weight": size_prior_weight,
        }
        candidates.append(
            InferenceCandidate(
                target_width=target_width,
                target_height=target_height,
                phase_x=float(phase_x),
                phase_y=phase_y,
                score=float(score_np[index]),
                breakdown=candidate_breakdown,
            )
        )
    return candidates


def infer_lattice(
    rgba: np.ndarray,
    target_size: int | None = None,
    device: str = "auto",
    observer: PipelineObserver | None = None,
) -> InferenceResult:
    torch, _ = _require_torch()
    resolved_device = _resolve_device(torch, device)
    height, width = rgba.shape[:2]
    spacing_x, spacing_y = _estimate_lattice_spacing(rgba)
    hinted_sizes = _hint_target_sizes_from_spacing(width, height, spacing_x, spacing_y)
    prior_cell_x, prior_cell_y, prior_reliability = _estimate_lattice_prior_details(
        rgba,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
    )
    phase_values = np.linspace(-0.4, 0.4, num=5, dtype=np.float32)
    candidate_dims = _resolve_candidate_dims_from_spacing(
        width,
        height,
        target_size,
        hinted_sizes=hinted_sizes,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        prior_reliability=prior_reliability,
    )
    phase_sample_count = int(phase_values.size * phase_values.size)

    emit_observer(
        observer,
        "lattice_search_started",
        candidate_count=len(candidate_dims),
        phase_sample_count=phase_sample_count,
        device=resolved_device,
    )

    candidates: list[InferenceCandidate] = []
    for candidate_index, (target_width, target_height) in enumerate(candidate_dims, start=1):
        scored_group = _score_phase_group(
            rgba,
            target_width=target_width,
            target_height=target_height,
            prior_cell_x=prior_cell_x,
            prior_cell_y=prior_cell_y,
            prior_reliability=prior_reliability,
            phase_x_values=phase_values,
            phase_y_values=phase_values,
            device=resolved_device,
        )
        candidates.extend(scored_group)
        emit_observer(
            observer,
            "lattice_search_progress",
            completed_candidates=candidate_index,
            total_candidates=len(candidate_dims),
            target_width=int(target_width),
            target_height=int(target_height),
            phase_sample_count=phase_sample_count,
            best_score=None if not scored_group else float(max(candidate.score for candidate in scored_group)),
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    size_candidates = _top_candidates_by_size(candidates, limit=len(candidates))
    reranked_candidates = _rerank_size_candidates_with_source_evidence(rgba, size_candidates)
    best = reranked_candidates[0]
    second = reranked_candidates[1] if len(reranked_candidates) > 1 else best
    confidence = max(0.0, best.score - second.score)
    top_candidates = reranked_candidates[:8]
    return InferenceResult(
        target_width=best.target_width,
        target_height=best.target_height,
        phase_x=best.phase_x,
        phase_y=best.phase_y,
        confidence=confidence,
        top_candidates=top_candidates,
    )


def infer_fixed_lattice(
    rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    phase_x: float | None = None,
    phase_y: float | None = None,
    device: str = "auto",
) -> InferenceResult:
    torch, _ = _require_torch()
    resolved_device = _resolve_device(torch, device)
    prior_cell_x, prior_cell_y, prior_reliability = _estimate_lattice_prior_details(rgba)
    phase_values = np.linspace(-0.4, 0.4, num=5, dtype=np.float32)
    phase_x_values = np.asarray([phase_x], dtype=np.float32) if phase_x is not None else phase_values
    phase_y_values = np.asarray([phase_y], dtype=np.float32) if phase_y is not None else phase_values

    candidates = _score_phase_group(
        rgba,
        target_width=max(1, int(target_width)),
        target_height=max(1, int(target_height)),
        prior_cell_x=prior_cell_x,
        prior_cell_y=prior_cell_y,
        prior_reliability=prior_reliability,
        phase_x_values=phase_x_values,
        phase_y_values=phase_y_values,
        device=resolved_device,
    )
    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else best
    confidence = max(0.0, best.score - second.score)
    top_candidates = candidates[:8]
    return InferenceResult(
        target_width=best.target_width,
        target_height=best.target_height,
        phase_x=best.phase_x,
        phase_y=best.phase_y,
        confidence=confidence,
        top_candidates=top_candidates,
    )


def inference_to_json(result: InferenceResult) -> dict[str, object]:
    return {
        "target_width": result.target_width,
        "target_height": result.target_height,
        "phase_x": result.phase_x,
        "phase_y": result.phase_y,
        "confidence": result.confidence,
        "top_candidates": [
            {
                "target_width": candidate.target_width,
                "target_height": candidate.target_height,
                "phase_x": candidate.phase_x,
                "phase_y": candidate.phase_y,
                "score": candidate.score,
                "breakdown": candidate.breakdown,
            }
            for candidate in result.top_candidates
        ],
    }
