from __future__ import annotations

import numpy as np
import pytest

from repixelizer.inference import (
    SpacingEstimate,
    _axis_prior_from_estimates,
    _candidate_dims,
    _combine_axis_priors,
    _hint_target_sizes_from_spacing,
    _resolve_candidate_dims_from_spacing,
    _top_candidates_by_size,
    infer_fixed_lattice,
    infer_lattice,
)
from repixelizer.observe import PipelineCancelled
from repixelizer.types import InferenceCandidate
from repixelizer.synthetic import fake_pixelize, make_emblem


def _make_spacing_estimate(
    *,
    best_cell: int,
    confidence: float,
    peaks: dict[int, float],
    start: int = 2,
    stop: int = 24,
) -> SpacingEstimate:
    candidate_cells = tuple(float(cell) for cell in range(start, stop + 1))
    candidate_scores: list[float] = []
    for cell in range(start, stop + 1):
        score = 0.16
        for peak_cell, peak_score in peaks.items():
            distance = abs(cell - peak_cell)
            if distance == 0:
                score = max(score, peak_score)
            elif distance == 1:
                score = max(score, peak_score - 0.14)
        candidate_scores.append(float(score))
    return SpacingEstimate(
        spacing=float(best_cell),
        confidence=confidence,
        best_cell=float(best_cell),
        best_score=float(peaks[best_cell]),
        candidate_cells=candidate_cells,
        candidate_scores=tuple(candidate_scores),
    )


def test_axis_prior_stays_close_to_spacing_when_autocorr_hits_large_multiple() -> None:
    prior, reliability = _axis_prior_from_estimates(8.0, 0.75, 32.0)
    assert prior < 10.0
    assert reliability > 0.6


def test_combined_axis_prior_prefers_consistent_shared_cell_size() -> None:
    shared_prior, reliability = _combine_axis_priors([(9.0, 0.7), (11.2, 0.2)])
    assert 8.5 < shared_prior < 10.5
    assert 0.2 < reliability < 0.6


def test_spacing_hints_can_add_dense_candidates_around_true_size() -> None:
    hinted_sizes = _hint_target_sizes_from_spacing(
        1024,
        1024,
        (8.05, 0.74),
        (4.0, 0.08),
    )
    dims = _candidate_dims(1024, 1024, None, hinted_sizes=hinted_sizes)
    sizes = {width for width, _ in dims}
    assert 128 in sizes


def test_strong_spacing_signal_can_collapse_candidate_search_to_single_size() -> None:
    dims = _resolve_candidate_dims_from_spacing(
        1024,
        1024,
        None,
        hinted_sizes=[128],
        spacing_x=(8.0, 0.82),
        spacing_y=(8.02, 0.78),
        prior_reliability=0.74,
    )
    sizes = {width for width, _ in dims}
    assert sizes == {128}


def test_weak_spacing_signal_without_spectrum_keeps_broad_candidate_search() -> None:
    dims = _resolve_candidate_dims_from_spacing(
        1024,
        1024,
        None,
        hinted_sizes=[128],
        spacing_x=(8.0, 0.18),
        spacing_y=(8.1, 0.12),
        prior_reliability=0.24,
    )
    sizes = {width for width, _ in dims}
    assert 128 in sizes
    assert len(sizes) > 10


def test_weak_spacing_signal_uses_spectrum_modes_to_keep_candidate_search_compact() -> None:
    spectrum = _make_spacing_estimate(
        best_cell=8,
        confidence=0.18,
        peaks={4: 0.78, 8: 0.74, 16: 0.71},
    )
    dims = _resolve_candidate_dims_from_spacing(
        1024,
        1024,
        None,
        hinted_sizes=[128],
        spacing_x=(8.0, 0.18),
        spacing_y=(8.1, 0.12),
        prior_reliability=0.24,
        spacing_x_estimate=spectrum,
        spacing_y_estimate=spectrum,
    )
    sizes = {width for width, _ in dims}
    assert sizes
    assert len(sizes) <= 12
    assert any(abs(size - 128) <= 1 for size in sizes)
    assert all(any(abs(size - target) <= 2 for target in (64, 128, 256)) for size in sizes)


def test_top_candidates_are_diversified_by_size() -> None:
    candidates = [
        InferenceCandidate(target_width=113, target_height=113, phase_x=0.0, phase_y=0.0, score=0.90, breakdown={}),
        InferenceCandidate(target_width=113, target_height=113, phase_x=0.2, phase_y=0.0, score=0.89, breakdown={}),
        InferenceCandidate(target_width=117, target_height=117, phase_x=0.0, phase_y=0.0, score=0.88, breakdown={}),
        InferenceCandidate(target_width=128, target_height=128, phase_x=0.0, phase_y=0.0, score=0.87, breakdown={}),
    ]
    selected = _top_candidates_by_size(candidates, limit=8)
    assert [(candidate.target_width, candidate.phase_x) for candidate in selected] == [
        (113, 0.0),
        (117, 0.0),
        (128, 0.0),
    ]


def test_infer_lattice_recovers_emblem_scale() -> None:
    source = make_emblem(32, 32)
    fake = fake_pixelize(
        source,
        upscale=12,
        phase_x=0.2,
        phase_y=0.35,
        blur_radius=0.75,
        warp_strength=0.28,
        warp_detail=6,
        seed=5,
    )
    result = infer_lattice(fake)
    assert result.target_width in range(28, 37)
    assert result.target_height in range(28, 37)
    assert result.confidence >= 0.0


def test_infer_fixed_lattice_honors_exact_size_and_phase() -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=10, phase_x=0.2, phase_y=-0.2, blur_radius=0.4, seed=9)
    result = infer_fixed_lattice(
        fake,
        target_width=16,
        target_height=16,
        phase_x=0.2,
        phase_y=-0.2,
        device="cpu",
    )
    assert result.target_width == 16
    assert result.target_height == 16
    assert abs(result.phase_x - 0.2) <= 1e-6
    assert abs(result.phase_y + 0.2) <= 1e-6
    assert len(result.top_candidates) == 1


def test_infer_fixed_lattice_searches_phase_within_pinned_size() -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, phase_x=0.2, phase_y=0.25, blur_radius=0.45, seed=3)
    result = infer_fixed_lattice(
        fake,
        target_width=16,
        target_height=16,
        device="cpu",
    )
    assert result.target_width == 16
    assert result.target_height == 16
    assert len(result.top_candidates) > 1
    assert all(candidate.target_width == 16 for candidate in result.top_candidates)
    assert all(candidate.target_height == 16 for candidate in result.top_candidates)


def test_infer_lattice_honors_cooperative_cancellation(monkeypatch) -> None:
    import repixelizer.inference as inference_module

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    class CancelObserver:
        def __call__(self, event: str, payload: dict[str, object]) -> None:
            del event, payload

        def check_cancelled(self) -> bool:
            return True

    monkeypatch.setattr(inference_module, "_require_torch", lambda: (_FakeTorch(), object()))
    monkeypatch.setattr(
        inference_module,
        "_estimate_lattice_spacing_details",
        lambda rgba: (inference_module.SpacingEstimate(None, 0.0, None, 0.0, (), ()),) * 2,
    )
    monkeypatch.setattr(inference_module, "_hint_target_sizes_from_spacing", lambda width, height, spacing_x, spacing_y: [])
    monkeypatch.setattr(inference_module, "_estimate_lattice_prior_details", lambda rgba, **kwargs: (4.0, 4.0, 0.5))
    monkeypatch.setattr(
        inference_module,
        "_resolve_candidate_dims_from_spacing",
        lambda *args, **kwargs: [(10, 8), (12, 10)],
    )
    monkeypatch.setattr(
        inference_module,
        "_score_phase_group",
        lambda *args, **kwargs: [
            inference_module.InferenceCandidate(
                target_width=10,
                target_height=8,
                phase_x=0.0,
                phase_y=0.0,
                score=0.5,
                breakdown={},
            )
        ],
    )

    with pytest.raises(PipelineCancelled):
        inference_module.infer_lattice(np.zeros((16, 16, 4), dtype=np.float32), observer=CancelObserver())
