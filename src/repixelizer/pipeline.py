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
from .metrics import (
    foreground_edge_concentration,
    foreground_edge_position_error,
    foreground_stroke_wobble_error,
    source_lattice_consistency_breakdown,
)
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
    if len(inference.top_candidates) <= 1 or inference.confidence >= solver_params.phase_rerank_confidence_threshold:
        return inference

    top_score = float(inference.top_candidates[0].score)
    candidate_records: list[dict[str, float | InferenceResult]] = []
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
        preview = nearest_resize(candidate_artifacts.target_rgba, width=source.shape[1], height=source.shape[0])
        candidate_records.append(
            {
                "inference": candidate_inference,
                "support_score": support["score"],
                "edge_position_error": foreground_edge_position_error(preview, source),
                "stroke_wobble_error": foreground_stroke_wobble_error(preview, source),
                "edge_concentration": foreground_edge_concentration(candidate_artifacts.target_rgba),
                "inference_penalty": top_score - float(candidate.score),
            }
        )

    if not candidate_records:
        return inference

    support_penalty = _normalize_penalty(record["support_score"] for record in candidate_records)
    edge_position_penalty = _normalize_penalty(record["edge_position_error"] for record in candidate_records)
    wobble_penalty = _normalize_penalty(record["stroke_wobble_error"] for record in candidate_records)
    edge_concentration_penalty = _normalize_penalty(
        (record["edge_concentration"] for record in candidate_records),
        higher_is_better=True,
    )
    inference_penalty = _normalize_penalty(record["inference_penalty"] for record in candidate_records)

    baseline_rank: float | None = None
    best_rank = float("inf")
    best_candidate = inference
    for index, record in enumerate(candidate_records):
        rank = (
            solver_params.phase_rerank_support_weight * support_penalty[index]
            + solver_params.phase_rerank_edge_position_weight * edge_position_penalty[index]
            + solver_params.phase_rerank_wobble_weight * wobble_penalty[index]
            + solver_params.phase_rerank_edge_concentration_weight * edge_concentration_penalty[index]
            + solver_params.phase_rerank_inference_penalty_weight * inference_penalty[index]
        )
        if baseline_rank is None:
            baseline_rank = rank
        if rank < best_rank:
            best_rank = rank
            best_candidate = record["inference"]
    if baseline_rank is None or best_rank > baseline_rank - solver_params.phase_rerank_margin:
        return inference
    return best_candidate


def _normalize_penalty(values, *, higher_is_better: bool = False) -> list[float]:
    raw = np.asarray(list(values), dtype=np.float32)
    if raw.size == 0:
        return []
    lo = float(np.min(raw))
    hi = float(np.max(raw))
    if hi - lo <= 1e-6:
        return [0.0] * int(raw.size)
    normalized = (raw - lo) / (hi - lo)
    if higher_is_better:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32).tolist()
