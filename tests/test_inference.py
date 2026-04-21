from __future__ import annotations

from repixelizer.inference import (
    _axis_prior_from_estimates,
    _candidate_dims,
    _combine_axis_priors,
    _hint_target_sizes_from_spacing,
    _top_candidates_by_size,
    infer_lattice,
)
from repixelizer.types import InferenceCandidate
from repixelizer.synthetic import fake_pixelize, make_emblem


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
