from __future__ import annotations

import numpy as np

from repixelizer.analysis import analyze_phase_field_source


def test_phase_field_source_analysis_cpu_device_matches_default_path() -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[2:6, 3:5, :3] = np.asarray([1.0, 0.8, 0.2], dtype=np.float32)

    default = analyze_phase_field_source(source, seed=7)
    accelerated = analyze_phase_field_source(source, seed=7, device="cpu")

    assert np.allclose(accelerated.edge_map, default.edge_map, atol=1e-5)
    assert not hasattr(accelerated, "cluster_map")
