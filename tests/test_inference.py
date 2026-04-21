from __future__ import annotations

from repixelizer.inference import infer_lattice
from repixelizer.synthetic import fake_pixelize, make_emblem


def test_infer_lattice_recovers_emblem_scale() -> None:
    source = make_emblem(32, 32)
    fake = fake_pixelize(
        source,
        upscale=12,
        phase_x=0.2,
        phase_y=0.35,
        blur_radius=0.75,
        warp_strength=0.28,
        warp_detail=6,
        seed=5,
    )
    result = infer_lattice(fake)
    assert result.target_width in range(28, 37)
    assert result.target_height in range(28, 37)
    assert result.confidence >= 0.0
