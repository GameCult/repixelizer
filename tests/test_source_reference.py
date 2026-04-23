from __future__ import annotations

import numpy as np

from repixelizer.source_reference import build_source_lattice_reference, build_tile_graph_source_reference


def test_source_lattice_reference_recovers_stable_cell_colors() -> None:
    lowres = np.asarray(
        [
            [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    source = np.repeat(np.repeat(lowres, 4, axis=0), 4, axis=1)
    source[1, 1, :3] = np.asarray([0.6, 0.1, 0.1], dtype=np.float32)

    reference = build_source_lattice_reference(
        source,
        target_width=2,
        target_height=2,
        phase_x=0.0,
        phase_y=0.0,
    )

    assert np.allclose(reference.sharp_rgba, lowres, atol=1e-5)
    assert np.allclose(reference.mean_rgba[0, 1], lowres[0, 1], atol=1e-5)
    assert reference.dispersion >= 0.0


def test_source_lattice_reference_tracks_edge_peaks_and_gradients() -> None:
    source = np.zeros((6, 6, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[:, 2, :3] = 1.0

    edge_hint = np.zeros((6, 6), dtype=np.float32)
    edge_hint[:, 2] = 1.0
    edge_grad_x = np.zeros((6, 6), dtype=np.float32)
    edge_grad_x[:, 2] = 1.0
    edge_grad_y = np.zeros((6, 6), dtype=np.float32)

    reference = build_source_lattice_reference(
        source,
        target_width=2,
        target_height=2,
        phase_x=0.0,
        phase_y=0.0,
        edge_hint=edge_hint,
        edge_grad_x_hint=edge_grad_x,
        edge_grad_y_hint=edge_grad_y,
    )

    assert np.all(reference.edge_peak_x[:, 0] == 2)
    assert np.all(reference.edge_strength[:, 0] == 1.0)
    assert np.all(reference.edge_grad_x[:, 0] == 1.0)
    assert np.all(reference.edge_grad_y[:, 0] == 0.0)


def test_tile_graph_source_reference_matches_continuous_anchor_fields() -> None:
    source = np.zeros((6, 6, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[:, 2, :3] = 1.0

    full = build_source_lattice_reference(
        source,
        target_width=2,
        target_height=2,
        phase_x=0.0,
        phase_y=0.0,
    )
    tile = build_tile_graph_source_reference(
        source,
        target_width=2,
        target_height=2,
        phase_x=0.0,
        phase_y=0.0,
    )

    assert np.allclose(tile.sharp_rgba, full.sharp_rgba, atol=1e-5)
    assert np.array_equal(tile.sharp_x, full.sharp_x)
    assert np.array_equal(tile.sharp_y, full.sharp_y)
    assert np.array_equal(tile.edge_peak_x, full.edge_peak_x)
    assert np.array_equal(tile.edge_peak_y, full.edge_peak_y)
    assert np.allclose(tile.edge_strength, full.edge_strength, atol=1e-5)


def test_source_lattice_reference_cpu_device_matches_default_path() -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[2:6, 4, :3] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    default = build_source_lattice_reference(
        source,
        target_width=4,
        target_height=4,
        phase_x=0.0,
        phase_y=0.0,
    )
    accelerated = build_source_lattice_reference(
        source,
        target_width=4,
        target_height=4,
        phase_x=0.0,
        phase_y=0.0,
        device="cpu",
    )

    assert np.allclose(accelerated.mean_rgba, default.mean_rgba, atol=1e-5)
    assert np.allclose(accelerated.sharp_rgba, default.sharp_rgba, atol=1e-5)
    assert np.allclose(accelerated.edge_strength, default.edge_strength, atol=1e-5)
    assert np.array_equal(accelerated.sharp_x, default.sharp_x)
    assert np.array_equal(accelerated.sharp_y, default.sharp_y)
