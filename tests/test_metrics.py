from __future__ import annotations

import numpy as np

from repixelizer.metrics import (
    exact_match_ratio,
    foreground_adjacency_error,
    foreground_edge_concentration,
    foreground_edge_position_error,
    foreground_exact_match_ratio,
    foreground_motif_error,
    foreground_reconstruction_error,
    source_lattice_consistency_breakdown,
    reconstruction_error,
)


def test_reconstruction_error_ignores_hidden_rgb_in_transparent_pixels() -> None:
    transparent_a = np.zeros((4, 4, 4), dtype=np.float32)
    transparent_b = transparent_a.copy()
    transparent_b[..., 0] = 1.0

    assert reconstruction_error(transparent_a, transparent_b) == 0.0
    assert exact_match_ratio(transparent_a, transparent_b) == 1.0


def test_foreground_metrics_penalize_visible_sprite_errors_and_leaks() -> None:
    original = np.zeros((4, 4, 4), dtype=np.float32)
    original[1, 1] = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    hidden_rgb = original.copy()
    hidden_rgb[0, 0, :3] = np.asarray([0.2, 0.8, 0.4], dtype=np.float32)
    assert foreground_reconstruction_error(original, hidden_rgb) == 0.0
    assert foreground_exact_match_ratio(original, hidden_rgb) == 1.0

    wrong_foreground = original.copy()
    wrong_foreground[1, 1] = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    assert foreground_reconstruction_error(original, wrong_foreground) > 0.0
    assert foreground_exact_match_ratio(original, wrong_foreground) < 1.0

    leaked_foreground = original.copy()
    leaked_foreground[0, 0] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert foreground_reconstruction_error(original, leaked_foreground) > 0.0


def test_texture_metrics_track_adjacency_and_local_motifs() -> None:
    original = np.zeros((4, 4, 4), dtype=np.float32)
    original[1:3, 1:3] = np.asarray(
        [
            [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    same_colors_wrong_structure = original.copy()
    same_colors_wrong_structure[1:3, 1:3] = np.asarray(
        [
            [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    assert foreground_adjacency_error(original, original) == 0.0
    assert foreground_motif_error(original, original) == 0.0
    assert foreground_adjacency_error(original, same_colors_wrong_structure) > 0.0
    assert foreground_motif_error(original, same_colors_wrong_structure) > 0.0


def test_edge_position_metric_prefers_crisp_shift_to_blurry_smear() -> None:
    original = np.zeros((7, 7, 4), dtype=np.float32)
    original[1:6, 3] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    shifted = np.zeros_like(original)
    shifted[1:6, 4] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    blurred = np.zeros_like(original)
    blurred[1:6, 2] = np.asarray([0.4, 0.4, 0.4, 1.0], dtype=np.float32)
    blurred[1:6, 3] = np.asarray([0.7, 0.7, 0.7, 1.0], dtype=np.float32)
    blurred[1:6, 4] = np.asarray([0.4, 0.4, 0.4, 1.0], dtype=np.float32)

    assert foreground_edge_position_error(original, original) == 0.0
    assert foreground_edge_position_error(original, shifted) > 0.0
    assert foreground_edge_position_error(original, blurred) > foreground_edge_position_error(original, shifted)


def test_edge_concentration_prefers_crisp_edges_to_blurry_smear() -> None:
    shifted = np.zeros((7, 7, 4), dtype=np.float32)
    shifted[1:6, 4] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    blurred = np.zeros_like(shifted)
    blurred[1:6, 2] = np.asarray([0.4, 0.4, 0.4, 1.0], dtype=np.float32)
    blurred[1:6, 3] = np.asarray([0.7, 0.7, 0.7, 1.0], dtype=np.float32)
    blurred[1:6, 4] = np.asarray([0.4, 0.4, 0.4, 1.0], dtype=np.float32)

    assert foreground_edge_concentration(shifted) > foreground_edge_concentration(blurred)


def test_source_lattice_consistency_prefers_matching_cell_structure() -> None:
    lowres = np.asarray(
        [
            [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    source = np.repeat(np.repeat(lowres, 4, axis=0), 4, axis=1)
    good = lowres.copy()
    bad = lowres[:, ::-1].copy()

    good_score = source_lattice_consistency_breakdown(
        source,
        good,
        target_width=2,
        target_height=2,
        phase_x=0.0,
        phase_y=0.0,
    )["score"]
    bad_score = source_lattice_consistency_breakdown(
        source,
        bad,
        target_width=2,
        target_height=2,
        phase_x=0.0,
        phase_y=0.0,
    )["score"]

    assert good_score < bad_score
