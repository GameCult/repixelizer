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
from .io import load_rgba, nearest_resize, save_rgba
from .metrics import source_lattice_consistency_breakdown
from .params import SolverHyperParams
from .palette import load_palette, quantize_rgba, save_palette_report
from .preprocess import strip_edge_background
from .types import InferenceResult, RunResult


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
    device: str = "auto",
    solver_params: SolverHyperParams | None = None,
    strip_background: bool = False,
) -> RunResult:
    started = time.perf_counter()
    solver_params = solver_params or SolverHyperParams()
    source = load_rgba(input_path)
    if strip_background:
        source = strip_edge_background(source)
    inference = infer_lattice(source, target_size=target_size, device=device)
    analysis = analyze_source(source, seed=seed)
    inference = _select_phase_candidate(
        source,
        inference,
        analysis=analysis,
        seed=seed,
        device=device,
        solver_params=solver_params,
    )
    solver = optimize_uv_field(
        source,
        inference=inference,
        analysis=analysis,
        steps=steps,
        seed=seed,
        device=device,
        solver_params=solver_params,
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
        if strip_background:
            save_rgba(diagnostics_path / "preprocessed-source.png", source)
        write_lattice_overlay(diagnostics_path / "lattice-overlay.png", source, inference)
        write_comparison(diagnostics_path / "comparison.png", source, output_rgba)
        save_rgba(
            diagnostics_path / "output-preview.png",
            nearest_resize(output_rgba, width=output_rgba.shape[1] * 8, height=output_rgba.shape[0] * 8),
        )
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
            "strip_background": strip_background,
            "solver_params": solver_params.to_dict(),
        }
        write_json(diagnostics_path / "run.json", run_json)
        if palette_result is not None:
            save_palette_report(diagnostics_path / "palette-report.json", palette_result.palette)
    return result


def _select_phase_candidate(
    source: np.ndarray,
    inference: InferenceResult,
    *,
    analysis,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
) -> InferenceResult:
    solver_params = solver_params or SolverHyperParams()
    if len(inference.top_candidates) <= 1 or inference.confidence >= 0.02:
        return inference

    top_score = float(inference.top_candidates[0].score)
    baseline_rank: float | None = None
    best_rank = float("inf")
    best_candidate = inference
    for candidate in inference.top_candidates[:8]:
        candidate_inference = InferenceResult(
            target_width=candidate.target_width,
            target_height=candidate.target_height,
            phase_x=candidate.phase_x,
            phase_y=candidate.phase_y,
            confidence=inference.confidence,
            top_candidates=inference.top_candidates,
        )
        candidate_artifacts = optimize_uv_field(
            source,
            inference=candidate_inference,
            analysis=analysis,
            steps=0,
            seed=seed,
            device=device,
            solver_params=solver_params,
        )
        support = source_lattice_consistency_breakdown(
            source,
            candidate_artifacts.target_rgba,
            target_width=candidate.target_width,
            target_height=candidate.target_height,
            phase_x=candidate.phase_x,
            phase_y=candidate.phase_y,
        )
        coherence_penalty = top_score - float(candidate.score)
        rank = support["score"] + coherence_penalty * 0.5
        if baseline_rank is None:
            baseline_rank = rank
        if rank < best_rank:
            best_rank = rank
            best_candidate = candidate_inference
    if baseline_rank is None or best_rank > baseline_rank - solver_params.phase_rerank_margin:
        return inference
    return best_candidate
