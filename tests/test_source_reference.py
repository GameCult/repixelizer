from __future__ import annotations

import numpy as np

from repixelizer.io import premultiply
from repixelizer.source_reference import build_source_lattice_reference


def test_source_lattice_reference_recovers_stable_cell_colors_and_deltas() -> None:
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

    expected_premul = premultiply(lowres)
    assert np.allclose(reference.sharp_rgba, lowres, atol=1e-5)
    assert np.allclose(reference.mean_rgba[0, 1], lowres[0, 1], atol=1e-5)
    assert reference.delta_x is not None
    assert reference.delta_y is not None
    assert reference.delta_diag is not None
    assert reference.delta_anti is not None
    assert np.allclose(reference.delta_x, expected_premul[:, 1:, :] - expected_premul[:, :-1, :], atol=1e-5)
    assert np.allclose(reference.delta_y, expected_premul[1:, :, :] - expected_premul[:-1, :, :], atol=1e-5)
    assert reference.dispersion >= 0.0
