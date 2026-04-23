from __future__ import annotations

import numpy as np

from repixelizer.analysis import analyze_continuous_source, analyze_tile_graph_source


def test_continuous_source_analysis_cpu_device_matches_default_path() -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[2:6, 3:5, :3] = np.asarray([1.0, 0.8, 0.2], dtype=np.float32)

    default = analyze_continuous_source(source, seed=7)
    accelerated = analyze_continuous_source(source, seed=7, device="cpu")

    assert np.allclose(accelerated.edge_map, default.edge_map, atol=1e-5)
    assert accelerated.cluster_map.shape == default.cluster_map.shape
    assert np.array_equal(accelerated.cluster_map, default.cluster_map)


def test_tile_graph_source_analysis_is_edge_only() -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[2:6, 3:5, :3] = np.asarray([1.0, 0.8, 0.2], dtype=np.float32)

    analysis = analyze_tile_graph_source(source, device="cpu")

    assert analysis.edge_map.shape == source.shape[:2]
    assert not hasattr(analysis, "cluster_map")
