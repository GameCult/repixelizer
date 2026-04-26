from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repixelizer.inference import (
    AutocorrEstimate,
    _consensus_autocorr_lag,
    _estimate_lattice_autocorr_details,
    _estimate_lattice_prior_details,
    _candidate_dims,
    _combine_axis_priors,
    _hint_target_sizes_from_autocorr,
    _resolve_candidate_dims_from_autocorr,
    _top_candidates_by_size,
    infer_fixed_lattice,
    infer_lattice,
)
from repixelizer.io import load_rgba
from repixelizer.observe import PipelineCancelled
from repixelizer.synthetic import fake_pixelize, make_emblem
from repixelizer.types import InferenceCandidate


def _make_autocorr_estimate(
    *,
    best_lag: int,
    peaks: dict[int, float],
    start: int = 2,
    stop: int = 24,
) -> AutocorrEstimate:
    candidate_lags = tuple(float(lag) for lag in range(start, stop + 1))
    candidate_scores = tuple(float(peaks.get(lag, 0.1)) for lag in range(start, stop + 1))
    return AutocorrEstimate(
        best_lag=float(best_lag),
        best_score=float(peaks[best_lag]),
        candidate_lags=candidate_lags,
        candidate_scores=candidate_scores,
    )


def test_combined_axis_prior_prefers_consistent_shared_cell_size() -> None:
    shared_prior, reliability = _combine_axis_priors([(9.0, 0.7), (11.2, 0.2)])
    assert 8.5 < shared_prior < 10.5
    assert 0.2 < reliability < 0.6


def test_autocorr_consensus_prefers_shared_dense_neighbor() -> None:
    estimate_x = _make_autocorr_estimate(best_lag=11, peaks={10: 0.724, 11: 0.732})
    estimate_y = _make_autocorr_estimate(best_lag=10, peaks={10: 0.878, 11: 0.859})
    consensus = _consensus_autocorr_lag(estimate_x, estimate_y)
    assert consensus is not None
    assert 9.9 <= consensus[0] <= 10.2


def test_autocorr_hints_can_add_dense_candidates_around_true_size() -> None:
    estimate = _make_autocorr_estimate(best_lag=8, peaks={8: 0.92})
    hinted_sizes = _hint_target_sizes_from_autocorr(
        1024,
        1024,
        autocorr_x_estimate=estimate,
        autocorr_y_estimate=estimate,
        shared_prior=8.0,
    )
    dims = _candidate_dims(1024, 1024, None, hinted_sizes=hinted_sizes)
    sizes = {width for width, _ in dims}
    assert 128 in sizes


def test_autocorr_hints_can_break_major_axis_cap_for_dense_candidates() -> None:
    dims = _candidate_dims(1254, 1254, None, hinted_sizes=[418])
    sizes = {width for width, _ in dims}
    assert 418 in sizes
    assert 416 in sizes
    assert 420 in sizes
    assert 512 not in sizes


def test_consistent_autocorr_signal_can_collapse_candidate_search_to_single_size() -> None:
    dims = _resolve_candidate_dims_from_autocorr(
        1024,
        1024,
        None,
        hinted_sizes=[128],
        prior_reliability=0.85,
    )
    sizes = {width for width, _ in dims}
    assert sizes == {128}


def test_divergent_autocorr_signal_keeps_candidate_search_wide_enough_for_badge() -> None:
    dims = _resolve_candidate_dims_from_autocorr(
        1024,
        1024,
        None,
        hinted_sizes=[114, 120, 125],
        prior_reliability=0.9,
    )
    sizes = {width for width, _ in dims}
    assert 126 in sizes
    assert any(abs(size - 114) <= 1 for size in sizes)
    assert any(abs(size - 125) <= 1 for size in sizes)


def test_dense_landscape_candidate_search_keeps_autocorr_size() -> None:
    rgba = load_rgba(Path("tests/fixtures/real/dense-landscape.png"))
    autocorr_x, autocorr_y = _estimate_lattice_autocorr_details(rgba)
    prior_x, _prior_y, prior_reliability = _estimate_lattice_prior_details(rgba)
    hinted_sizes = _hint_target_sizes_from_autocorr(
        rgba.shape[1],
        rgba.shape[0],
        autocorr_x_estimate=autocorr_x,
        autocorr_y_estimate=autocorr_y,
        shared_prior=prior_x,
    )
    dims = _resolve_candidate_dims_from_autocorr(
        rgba.shape[1],
        rgba.shape[0],
        None,
        hinted_sizes=hinted_sizes,
        prior_reliability=prior_reliability,
    )
    sizes = {width for width, _ in dims}
    assert any(size > 256 for size in sizes)
    assert any(abs(size - 418) <= 1 for size in sizes)


def test_badge_autocorr_consensus_stays_in_canonical_family() -> None:
    rgba = load_rgba(Path("tests/fixtures/real/ai-badge-cleaned.png"))
    result = infer_lattice(rgba, device="cpu")
    assert result.target_width in {125, 126}
    assert result.top_candidates
    assert result.top_candidates[0].target_width in {125, 126}


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

    fake_estimate = _make_autocorr_estimate(best_lag=4, peaks={4: 0.9})
    monkeypatch.setattr(inference_module, "_require_torch", lambda: (_FakeTorch(), object()))
    monkeypatch.setattr(inference_module, "_estimate_lattice_autocorr_details", lambda rgba: (fake_estimate, fake_estimate))
    monkeypatch.setattr(
        inference_module,
        "_hint_target_sizes_from_autocorr",
        lambda width, height, *, autocorr_x_estimate, autocorr_y_estimate, shared_prior: [],
    )
    monkeypatch.setattr(inference_module, "_estimate_lattice_prior_details", lambda rgba: (4.0, 4.0, 0.5))
    monkeypatch.setattr(
        inference_module,
        "_resolve_candidate_dims_from_autocorr",
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
