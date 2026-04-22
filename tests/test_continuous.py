from __future__ import annotations

import torch

from repixelizer.continuous import _exemplar_colors, _line_pattern_loss


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
