from __future__ import annotations

import numpy as np

from repixelizer.preprocess import strip_edge_background


def test_strip_edge_background_removes_light_checkerboard_and_preserves_subject() -> None:
    rgba = np.ones((80, 80, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    for y in range(80):
        for x in range(80):
            tile = 0.97 if ((x // 4) + (y // 4)) % 2 == 0 else 0.91
            rgba[y, x, :3] = tile
    rgba[12:68, 10:70] = np.array([0.1, 0.7, 0.2, 1.0], dtype=np.float32)
    for y in range(24, 56):
        for x in range(26, 54):
            tile = 0.97 if ((x // 4) + (y // 4)) % 2 == 0 else 0.91
            rgba[y, x, :3] = tile
    rgba[20:60, 16:20] = np.array([0.98, 0.97, 0.96, 1.0], dtype=np.float32)

    stripped = strip_edge_background(rgba)

    assert stripped[0, 0, 3] == 0.0
    assert stripped[4, 4, 3] == 0.0
    assert stripped[40, 40, 3] == 0.0
    assert stripped[30, 18, 3] == 1.0
    assert np.allclose(stripped[30, 18, :3], rgba[30, 18, :3])
    assert stripped[16, 16, 3] == 1.0


def test_strip_edge_background_handles_trivial_monochrome_background() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.float32)
    rgba[..., :3] = np.array([0.2, 0.3, 0.8], dtype=np.float32)
    rgba[..., 3] = 1.0
    rgba[8:24, 8:24] = np.array([1.0, 0.6, 0.1, 1.0], dtype=np.float32)

    stripped = strip_edge_background(rgba)

    assert stripped[0, 0, 3] == 0.0
    assert stripped[16, 16, 3] == 1.0
    assert np.allclose(stripped[16, 16, :3], rgba[16, 16, :3])
