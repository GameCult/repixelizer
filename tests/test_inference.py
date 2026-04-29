from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repixelizer.inference import (
    AutocorrEstimate,
    _consensus_autocorr_lag,
    _estimate_lattice_autocorr_details,
    _estimate_lattice_prior_details,
    _candidate_dims_from_autocorr_hints,
    _combine_axis_priors,
    _hint_target_sizes_from_autocorr,
    _top_candidates_by_size,
    infer_autocorr_lattice,
    infer_fixed_lattice,
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
    dims = _candidate_dims_from_autocorr_hints(1024, 1024, hinted_sizes=hinted_sizes)
    sizes = {width for width, _ in dims}
    assert 128 in sizes


def test_autocorr_hints_can_break_major_axis_cap_for_dense_candidates() -> None:
    dims = _candidate_dims_from_autocorr_hints(1254, 1254, hinted_sizes=[418])
    sizes = {width for width, _ in dims}
    assert 418 in sizes
    assert 512 not in sizes








def test_badge_autocorr_candidates_stay_in_canonical_family() -> None:
    rgba = load_rgba(Path("tests/fixtures/real/ai-badge-cleaned.png"))
    result = infer_autocorr_lattice(rgba, device="cpu")
    assert result.target_width in {125, 126}
    assert result.top_candidates
    assert result.top_candidates[0].target_width in {125, 126}


def test_direct_badge_autocorr_consensus_stays_in_canonical_family() -> None:
    rgba = load_rgba(Path("tests/fixtures/real/ai-badge-cleaned.png"))
    result = infer_autocorr_lattice(rgba, device="cpu")
    assert result.target_width in {125, 126}
    assert result.top_candidates
    assert result.top_candidates[0].target_width == result.target_width


def test_top_candidates_are_diversified_by_size() -> None:
    candidates = [
        InferenceCandidate(target_width=113, target_height=113, score=0.90, breakdown={}),
        InferenceCandidate(target_width=113, target_height=113, score=0.89, breakdown={}),
        InferenceCandidate(target_width=117, target_height=117, score=0.88, breakdown={}),
        InferenceCandidate(target_width=128, target_height=128, score=0.87, breakdown={}),
    ]
    selected = _top_candidates_by_size(candidates, limit=8)
    assert [(candidate.target_width, candidate.target_height) for candidate in selected] == [
        (113, 113),
        (117, 117),
        (128, 128),
    ]


def test_infer_autocorr_lattice_recovers_emblem_scale() -> None:
    source = make_emblem(32, 32)
    fake = fake_pixelize(
        source,
        upscale=12,
        blur_radius=0.75,
        warp_strength=0.28,
        warp_detail=6,
        seed=5,
    )
    result = infer_autocorr_lattice(fake)
    assert result.target_width in range(28, 37)
    assert result.target_height in range(28, 37)
    assert result.confidence >= 0.0


def test_infer_fixed_lattice_uses_canonical_cell_centers_for_pinned_size() -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=10, blur_radius=0.4, seed=9)
    result = infer_fixed_lattice(
        fake,
        target_width=16,
        target_height=16,
        device="cpu",
    )
    assert result.target_width == 16
    assert result.target_height == 16
    assert len(result.top_candidates) == 1


def test_infer_fixed_lattice_uses_canonical_cell_centers_within_pinned_size() -> None:
    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, blur_radius=0.45, seed=3)
    result = infer_fixed_lattice(
        fake,
        target_width=16,
        target_height=16,
        device="cpu",
    )
    assert result.target_width == 16
    assert result.target_height == 16
    assert len(result.top_candidates) == 1
    assert all(candidate.target_width == 16 for candidate in result.top_candidates)
    assert all(candidate.target_height == 16 for candidate in result.top_candidates)


def test_infer_autocorr_lattice_honors_cooperative_cancellation(monkeypatch) -> None:
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
        "_candidate_dims_from_autocorr_hints",
        lambda *args, **kwargs: [(10, 8), (12, 10)],
    )
    monkeypatch.setattr(
        inference_module,
        "_score_size_candidate",
        lambda *args, **kwargs: [
            inference_module.InferenceCandidate(
                target_width=10,
                target_height=8,
                score=0.5,
                breakdown={},
            )
        ],
    )

    with pytest.raises(PipelineCancelled):
        inference_module.infer_autocorr_lattice(np.zeros((16, 16, 4), dtype=np.float32), observer=CancelObserver())
