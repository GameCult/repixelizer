from __future__ import annotations

import numpy as np

from repixelizer.analysis import analyze_source


def test_analyze_source_cpu_device_matches_default_path() -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[2:6, 3:5, :3] = np.asarray([1.0, 0.8, 0.2], dtype=np.float32)

    default = analyze_source(source, seed=7)
    accelerated = analyze_source(source, seed=7, device="cpu")

    assert np.allclose(accelerated.edge_map, default.edge_map, atol=1e-5)
    assert accelerated.cluster_centers.shape == default.cluster_centers.shape
    assert np.allclose(accelerated.alpha_map, default.alpha_map, atol=1e-5)
    assert accelerated.cluster_map.shape == default.cluster_map.shape
    assert accelerated.cluster_preview.shape == default.cluster_preview.shape
