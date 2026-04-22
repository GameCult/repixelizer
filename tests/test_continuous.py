from __future__ import annotations

import torch

from repixelizer.continuous import _line_pattern_loss


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
