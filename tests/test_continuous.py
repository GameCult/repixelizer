from __future__ import annotations

import numpy as np
import torch

from repixelizer.analysis import analyze_source
from repixelizer.continuous import _exemplar_colors, _line_pattern_loss, optimize_uv_field
from repixelizer.metrics import source_lattice_consistency_breakdown
from repixelizer.synthetic import fake_pixelize
from repixelizer.types import InferenceResult


def test_line_pattern_loss_penalizes_wobbling_line() -> None:
    reference = torch.zeros((1, 5, 5, 4), dtype=torch.float32)
    reference[:, :, 2, :] = 1.0

    good = reference.clone()
    bad = reference.clone()
    bad[:, 2, 2, :] = 0.0
    bad[:, 2, 3, :] = 1.0

    good_loss = float(_line_pattern_loss(torch, good, reference).item())
    bad_loss = float(_line_pattern_loss(torch, bad, reference).item())

    assert good_loss <= 1e-6
    assert bad_loss > good_loss + 0.01


def test_exemplar_colors_selects_an_actual_patch_sample() -> None:
    patches = torch.tensor(
        [
            [
                [
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.8, 0.7, 0.1, 1.0],
                        [0.2, 0.3, 0.9, 1.0],
                    ]
                ]
            ]
        ],
        dtype=torch.float32,
    )

    exemplar = _exemplar_colors(patches)[0, 0, 0]
    options = patches[0, 0, 0]

    assert any(torch.allclose(exemplar, option) for option in options)


def test_optimize_uv_field_does_not_regress_from_snap_on_thin_feature_case() -> None:
    source = np.zeros((16, 16, 4), dtype=np.float32)
    source[4:12, 8] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    source[4, 7:10] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    source[11, 7] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    source[10, 6] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    source[9, 5] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)

    fake = fake_pixelize(
        source,
        upscale=9,
        phase_x=0.18,
        phase_y=-0.12,
        blur_radius=0.45,
        warp_strength=0.2,
        warp_detail=5,
        seed=11,
    )
    inference = InferenceResult(
        target_width=16,
        target_height=16,
        phase_x=0.18,
        phase_y=-0.12,
        confidence=1.0,
        top_candidates=[],
    )
    artifacts = optimize_uv_field(
        fake,
        inference=inference,
        analysis=analyze_source(fake, seed=7),
        steps=4,
        seed=7,
        device="cpu",
    )

    snap_score = source_lattice_consistency_breakdown(
        fake,
        artifacts.initial_rgba,
        target_width=16,
        target_height=16,
        phase_x=0.18,
        phase_y=-0.12,
    )["score"]
    final_score = source_lattice_consistency_breakdown(
        fake,
        artifacts.target_rgba,
        target_width=16,
        target_height=16,
        phase_x=0.18,
        phase_y=-0.12,
    )["score"]

    assert final_score <= snap_score + 1e-6
