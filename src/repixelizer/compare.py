from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .baselines import error_diffusion_baseline, naive_resize_baseline
from .diagnostics import write_compare_csv, write_json
from .io import nearest_resize, save_rgba
from .metrics import (
    coherence_breakdown,
    foreground_adjacency_error,
    foreground_edge_concentration,
    foreground_edge_position_error,
    foreground_motif_error,
    foreground_reconstruction_error,
    foreground_stroke_wobble_error,
    reconstruction_error,
)
from .palette import load_palette
from .pipeline import run_pipeline


def _to_uint8(rgba: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)


def _contact_sheet(
    source: np.ndarray,
    optimized: np.ndarray,
    naive: np.ndarray,
    diffusion: np.ndarray,
    path: str | Path,
) -> None:
    height, width = source.shape[:2]
    panels = [
        ("Source", source),
        ("Optimized", nearest_resize(optimized, width=width, height=height)),
        ("Naive", nearest_resize(naive, width=width, height=height)),
        ("Diffusion", nearest_resize(diffusion, width=width, height=height)),
    ]
    canvas = Image.new("RGBA", (width * len(panels), height + 24), (20, 20, 20, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (label, rgba) in enumerate(panels):
        image = Image.fromarray(_to_uint8(rgba), mode="RGBA")
        canvas.alpha_composite(image, (index * width, 24))
        draw.text((index * width + 8, 4), label, fill=(255, 255, 255, 255))
    canvas.save(path)


def run_compare(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_size: int | None = None,
    palette_path: str | Path | None = None,
    palette_mode: str = "off",
    diagnostics_dir: str | Path | None = None,
    seed: int = 7,
    steps: int = 200,
    device: str = "auto",
    strip_background: bool = False,
) -> dict[str, Any]:
    diagnostics_path = Path(diagnostics_dir) if diagnostics_dir else Path(output_path).with_suffix("")
    diagnostics_path.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(
        input_path,
        output_path,
        target_size=target_size,
        palette_path=palette_path,
        palette_mode=palette_mode,
        diagnostics_dir=diagnostics_path,
        seed=seed,
        steps=steps,
        device=device,
        strip_background=strip_background,
    )
    source = result.source_rgba
    palette = load_palette(palette_path) if palette_path else None
    naive = naive_resize_baseline(source, width=result.inference.target_width, height=result.inference.target_height)
    diffusion = error_diffusion_baseline(
        source,
        width=result.inference.target_width,
        height=result.inference.target_height,
        palette=palette,
    )
    _contact_sheet(source, result.output_rgba, naive, diffusion, diagnostics_path / "compare-sheet.png")
    rows = []
    for name, image in (
        ("optimized", result.output_rgba),
        ("naive", naive),
        ("diffusion", diffusion),
    ):
        preview = nearest_resize(image, width=source.shape[1], height=source.shape[0])
        coherence = coherence_breakdown(image)
        rows.append(
            {
                "name": name,
                "coherence_score": coherence["coherence_score"],
                "cluster_continuity": coherence["cluster_continuity"],
                "alpha_crispness": coherence["alpha_crispness"],
                "outline_straightness": coherence["outline_straightness"],
                "isolated_penalty": coherence["isolated_penalty"],
                "color_chatter": coherence["color_chatter"],
                "reconstruction_error": reconstruction_error(preview, source),
                "foreground_reconstruction_error": foreground_reconstruction_error(preview, source),
                "foreground_edge_concentration": foreground_edge_concentration(image),
                "foreground_edge_position_error": foreground_edge_position_error(preview, source),
                "foreground_stroke_wobble_error": foreground_stroke_wobble_error(preview, source),
                "foreground_adjacency_error": foreground_adjacency_error(preview, source),
                "foreground_motif_error": foreground_motif_error(preview, source),
            }
        )
    write_compare_csv(diagnostics_path / "compare.csv", rows)
    payload = {"rows": rows}
    write_json(diagnostics_path / "compare.json", payload)
    return payload
