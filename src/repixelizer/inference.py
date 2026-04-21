from __future__ import annotations

import numpy as np
from PIL import Image

from .metrics import coherence_breakdown
from .types import InferenceCandidate, InferenceResult


def _to_pil(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8), mode="RGBA")


def _apply_phase(image: Image.Image, offset_x: float, offset_y: float) -> Image.Image:
    rgba = np.asarray(image, dtype=np.uint8)
    pad_x = int(np.ceil(abs(offset_x))) + 2
    pad_y = int(np.ceil(abs(offset_y))) + 2
    padded = np.pad(rgba, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
    start_x = pad_x + int(round(offset_x))
    start_y = pad_y + int(round(offset_y))
    cropped = padded[start_y : start_y + rgba.shape[0], start_x : start_x + rgba.shape[1]]
    return Image.fromarray(cropped, mode="RGBA")


def _candidate_dims(width: int, height: int, target_size: int | None) -> list[tuple[int, int]]:
    if target_size is not None:
        if width >= height:
            return [(target_size, max(1, round(height * target_size / width)))]
        return [(max(1, round(width * target_size / height)), target_size)]
    max_dim = max(width, height)
    min_size = max(12, int(round(max_dim / 48)))
    max_size = min(256, max(24, int(round(max_dim / 2.5))))
    step = 2 if max_size <= 96 else 4
    dims: list[tuple[int, int]] = []
    for size in range(min_size, max_size + 1, step):
        if width >= height:
            dims.append((size, max(1, round(height * size / width))))
        else:
            dims.append((max(1, round(width * size / height)), size))
    return dims


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


def _estimate_lattice_prior(rgba: np.ndarray) -> tuple[float, float]:
    alpha = rgba[..., 3]
    luminance = rgba[..., 0] * 0.2126 + rgba[..., 1] * 0.7152 + rgba[..., 2] * 0.0722
    dx = np.zeros_like(luminance)
    dy = np.zeros_like(luminance)
    dx[:, 1:] = np.abs(luminance[:, 1:] - luminance[:, :-1]) + np.abs(alpha[:, 1:] - alpha[:, :-1])
    dy[1:, :] = np.abs(luminance[1:, :] - luminance[:-1, :]) + np.abs(alpha[1:, :] - alpha[:-1, :])
    profile_x = (dx.mean(axis=0) + alpha.mean(axis=0) * 0.1).astype(np.float32)
    profile_y = (dy.mean(axis=1) + alpha.mean(axis=1) * 0.1).astype(np.float32)
    return _estimate_cell_size(profile_x), _estimate_cell_size(profile_y)


def infer_lattice(rgba: np.ndarray, target_size: int | None = None) -> InferenceResult:
    height, width = rgba.shape[:2]
    image = _to_pil(rgba)
    prior_cell_x, prior_cell_y = _estimate_lattice_prior(rgba)
    candidates: list[InferenceCandidate] = []
    for target_width, target_height in _candidate_dims(width, height, target_size):
        cell_x = width / target_width
        cell_y = height / target_height
        for phase_x in np.linspace(-0.4, 0.4, num=5):
            for phase_y in np.linspace(-0.4, 0.4, num=5):
                shifted = _apply_phase(image, phase_x * cell_x, phase_y * cell_y)
                sample = shifted.resize((target_width, target_height), resample=Image.Resampling.BOX)
                rgba_sample = np.asarray(sample, dtype=np.float32) / 255.0
                breakdown = coherence_breakdown(rgba_sample)
                prior_score_x = np.exp(-abs(np.log((cell_x + 1e-6) / (prior_cell_x + 1e-6))) * 0.8)
                prior_score_y = np.exp(-abs(np.log((cell_y + 1e-6) / (prior_cell_y + 1e-6))) * 0.8)
                size_prior = float((prior_score_x + prior_score_y) * 0.5)
                breakdown["size_prior"] = size_prior
                score = breakdown["coherence_score"] * 0.65 + size_prior * 0.35
                candidates.append(
                    InferenceCandidate(
                        target_width=target_width,
                        target_height=target_height,
                        phase_x=float(phase_x),
                        phase_y=float(phase_y),
                        score=float(score),
                        breakdown=breakdown,
                    )
                )
    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else best
    confidence = max(0.0, best.score - second.score)
    return InferenceResult(
        target_width=best.target_width,
        target_height=best.target_height,
        phase_x=best.phase_x,
        phase_y=best.phase_y,
        confidence=confidence,
        top_candidates=candidates[:8],
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
