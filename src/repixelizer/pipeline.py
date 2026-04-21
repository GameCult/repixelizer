from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze_source
from .continuous import optimize_uv_field
from .diagnostics import (
    summarize_run,
    write_alpha_preview,
    write_comparison,
    write_heatmap,
    write_json,
    write_lattice_overlay,
)
from .discrete import cleanup_pixels
from .inference import infer_lattice, inference_to_json
from .io import load_rgba, save_rgba
from .palette import load_palette, quantize_rgba, save_palette_report
from .types import RunResult


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_size: int | None = None,
    palette_path: str | Path | None = None,
    palette_mode: str = "off",
    diagnostics_dir: str | Path | None = None,
    seed: int = 7,
    steps: int = 200,
    device: str = "cpu",
) -> RunResult:
    started = time.perf_counter()
    source = load_rgba(input_path)
    inference = infer_lattice(source, target_size=target_size)
    analysis = analyze_source(source, seed=seed)
    solver = optimize_uv_field(
        source,
        inference=inference,
        analysis=analysis,
        steps=steps,
        seed=seed,
        device=device,
    )
    cleanup = cleanup_pixels(solver.target_rgba, source_guidance=solver.guidance_strength)
    palette = load_palette(palette_path) if palette_path else None
    palette_result = quantize_rgba(cleanup.cleaned_rgba, mode=palette_mode, palette=palette)
    output_rgba = palette_result.rgba if palette_result else cleanup.cleaned_rgba
    save_rgba(output_path, output_rgba)

    diagnostics: dict[str, Any] = {"elapsed_seconds": time.perf_counter() - started}
    result = RunResult(
        source_rgba=source,
        output_rgba=output_rgba,
        inference=inference,
        analysis=analysis,
        solver=solver,
        cleanup=cleanup,
        palette_result=palette_result,
        diagnostics=diagnostics,
    )
    if diagnostics_dir:
        diagnostics_path = Path(diagnostics_dir)
        diagnostics_path.mkdir(parents=True, exist_ok=True)
        write_lattice_overlay(diagnostics_path / "lattice-overlay.png", source, inference)
        write_comparison(diagnostics_path / "comparison.png", source, output_rgba)
        write_alpha_preview(diagnostics_path / "alpha-preview.png", source, output_rgba)
        write_heatmap(diagnostics_path / "noise-heatmap.png", cleanup.isolated_heatmap)
        save_rgba(diagnostics_path / "cluster-preview.png", analysis.cluster_preview)
        run_json = summarize_run(result)
        run_json["inference"] = inference_to_json(inference)
        run_json["settings"] = {
            "target_size": target_size,
            "palette_mode": palette_mode,
            "seed": seed,
            "steps": steps,
            "device": device,
        }
        write_json(diagnostics_path / "run.json", run_json)
        if palette_result is not None:
            save_palette_report(diagnostics_path / "palette-report.json", palette_result.palette)
    return result
