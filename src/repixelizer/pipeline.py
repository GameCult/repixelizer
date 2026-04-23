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
from .tile_graph import optimize_tile_graph
from .types import InferenceResult, RunResult


def _reuse_phase_probe_reconstruction(reconstruction_mode: str) -> bool:
    return reconstruction_mode in {"tile-graph", "hybrid"}


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
    reconstruction_mode: str = "continuous",
) -> RunResult:
    started = time.perf_counter()
    solver_params = solver_params or SolverHyperParams()
    source = load_rgba(input_path)
    if strip_background:
        source = strip_edge_background(source)
    inference = infer_lattice(source, target_size=target_size, device=device)
    analysis = analyze_source(source, seed=seed, device=device if reconstruction_mode in {"tile-graph", "hybrid"} else None)
    inference, cached_reconstruction = _select_phase_candidate_with_reconstruction(
        source,
        inference,
        analysis=analysis,
        seed=seed,
        device=device,
        solver_params=solver_params,
        reconstruction_mode=reconstruction_mode,
    )
    if cached_reconstruction is not None:
        solver, reconstruction_diagnostics = cached_reconstruction
        reconstruction_diagnostics = dict(reconstruction_diagnostics)
        reconstruction_diagnostics["reused_phase_probe_reconstruction"] = True
    else:
        solver, reconstruction_diagnostics = _run_reconstruction(
            source,
            inference=inference,
            analysis=analysis,
            steps=steps,
            seed=seed,
            device=device,
            solver_params=solver_params,
            reconstruction_mode=reconstruction_mode,
        )
    cleanup = cleanup_pixels(solver.target_rgba, source_guidance=solver.guidance_strength)
    palette = load_palette(palette_path) if palette_path else None
    palette_result = quantize_rgba(cleanup.cleaned_rgba, mode=palette_mode, palette=palette)
    output_rgba = palette_result.rgba if palette_result else cleanup.cleaned_rgba
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_rgba(output_path, output_rgba)

    diagnostics: dict[str, Any] = {
        "elapsed_seconds": time.perf_counter() - started,
        "reconstruction": reconstruction_diagnostics,
    }
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
            "reconstruction_mode": reconstruction_mode,
            "solver_params": solver_params.to_dict(),
        }
        if diagnostics.get("reconstruction"):
            run_json["reconstruction"] = diagnostics["reconstruction"]
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
    reconstruction_mode: str = "continuous",
) -> InferenceResult:
    selected, _ = _select_phase_candidate_with_reconstruction(
        source,
        inference,
        analysis=analysis,
        seed=seed,
        device=device,
        solver_params=solver_params,
        reconstruction_mode=reconstruction_mode,
    )
    return selected


def _select_phase_candidate_with_reconstruction(
    source: np.ndarray,
    inference: InferenceResult,
    *,
    analysis,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
    reconstruction_mode: str = "continuous",
):
    solver_params = solver_params or SolverHyperParams()
    if len(inference.top_candidates) <= 1 or inference.confidence >= solver_params.phase_rerank_confidence_threshold:
        return inference, None

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
        candidate_artifacts, candidate_diagnostics = _run_reconstruction(
            source,
            inference=candidate_inference,
            analysis=analysis,
            steps=0,
            seed=seed,
            device=device,
            solver_params=solver_params,
            reconstruction_mode=reconstruction_mode,
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
                "size_delta_ratio": _size_delta_ratio(inference, candidate_inference),
                "cached_reconstruction": (candidate_artifacts, candidate_diagnostics),
            }
        )

    if not candidate_records:
        return inference, None

    support_penalty = _normalize_penalty(record["support_score"] for record in candidate_records)
    edge_position_penalty = _normalize_penalty(record["edge_position_error"] for record in candidate_records)
    wobble_penalty = _normalize_penalty(record["stroke_wobble_error"] for record in candidate_records)
    edge_concentration_penalty = _normalize_penalty(
        (record["edge_concentration"] for record in candidate_records),
        higher_is_better=True,
    )
    size_penalty = [
        min(1.0, float(record["size_delta_ratio"]) / max(solver_params.phase_rerank_max_size_delta_ratio, 1e-6))
        for record in candidate_records
    ]
    inference_penalty = _normalize_penalty(record["inference_penalty"] for record in candidate_records)

    baseline_rank: float | None = None
    best_rank = float("inf")
    best_candidate = inference
    best_cached_reconstruction = None
    baseline_cached_reconstruction = None
    annotated_candidates = []
    for index, record in enumerate(candidate_records):
        candidate_inference = record["inference"]
        if float(record["size_delta_ratio"]) > solver_params.phase_rerank_max_size_delta_ratio:
            continue
        rank = (
            solver_params.phase_rerank_support_weight * support_penalty[index]
            + solver_params.phase_rerank_edge_position_weight * edge_position_penalty[index]
            + solver_params.phase_rerank_wobble_weight * wobble_penalty[index]
            + solver_params.phase_rerank_edge_concentration_weight * edge_concentration_penalty[index]
            + solver_params.phase_rerank_size_penalty_weight * size_penalty[index]
            + solver_params.phase_rerank_inference_penalty_weight * inference_penalty[index]
        )
        source_candidate = inference.top_candidates[index]
        breakdown = dict(source_candidate.breakdown)
        breakdown["phase_rerank_support_score"] = float(record["support_score"])
        breakdown["phase_rerank_edge_position_error"] = float(record["edge_position_error"])
        breakdown["phase_rerank_stroke_wobble_error"] = float(record["stroke_wobble_error"])
        breakdown["phase_rerank_edge_concentration"] = float(record["edge_concentration"])
        breakdown["phase_rerank_size_delta_ratio"] = float(record["size_delta_ratio"])
        breakdown["phase_rerank_size_penalty"] = float(size_penalty[index])
        breakdown["phase_rerank_inference_penalty"] = float(record["inference_penalty"])
        breakdown["phase_rerank_score"] = float(rank)
        annotated_candidates.append(
            source_candidate.__class__(
                target_width=source_candidate.target_width,
                target_height=source_candidate.target_height,
                phase_x=source_candidate.phase_x,
                phase_y=source_candidate.phase_y,
                score=source_candidate.score,
                breakdown=breakdown,
            )
        )
        if baseline_rank is None:
            baseline_rank = rank
            baseline_cached_reconstruction = record.get("cached_reconstruction")
        if rank < best_rank:
            best_rank = rank
            best_candidate = record["inference"]
            best_cached_reconstruction = record.get("cached_reconstruction")
    annotated_candidates.sort(key=lambda candidate: float(candidate.breakdown.get("phase_rerank_score", float("inf"))))
    for rank, candidate in enumerate(annotated_candidates, start=1):
        candidate.breakdown["phase_rerank_rank"] = float(rank)
    if annotated_candidates:
        best_candidate = InferenceResult(
            target_width=best_candidate.target_width,
            target_height=best_candidate.target_height,
            phase_x=best_candidate.phase_x,
            phase_y=best_candidate.phase_y,
            confidence=inference.confidence,
            top_candidates=annotated_candidates,
        )
    if baseline_rank is None or best_rank > baseline_rank - solver_params.phase_rerank_margin:
        if annotated_candidates:
            selected = InferenceResult(
                target_width=inference.target_width,
                target_height=inference.target_height,
                phase_x=inference.phase_x,
                phase_y=inference.phase_y,
                confidence=inference.confidence,
                top_candidates=annotated_candidates,
            )
            cached = baseline_cached_reconstruction if _reuse_phase_probe_reconstruction(reconstruction_mode) else None
            return selected, cached
        return inference, None
    cached = best_cached_reconstruction if _reuse_phase_probe_reconstruction(reconstruction_mode) else None
    return best_candidate, cached


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


def _size_delta_ratio(a: InferenceResult, b: InferenceResult) -> float:
    width_ratio = abs(b.target_width - a.target_width) / max(1.0, float(a.target_width))
    height_ratio = abs(b.target_height - a.target_height) / max(1.0, float(a.target_height))
    return max(width_ratio, height_ratio)


def _run_reconstruction(
    source: np.ndarray,
    *,
    inference: InferenceResult,
    analysis,
    steps: int,
    seed: int,
    device: str,
    solver_params: SolverHyperParams,
    reconstruction_mode: str,
):
    if reconstruction_mode == "tile-graph":
        return optimize_tile_graph(
            source,
            inference=inference,
            analysis=analysis,
            steps=steps,
            seed=seed,
            device=device,
            solver_params=solver_params,
        )
    if reconstruction_mode == "hybrid":
        geometry_solver = optimize_uv_field(
            source,
            inference=inference,
            analysis=analysis,
            steps=0,
            seed=seed,
            device=device,
            solver_params=solver_params,
        )
        hybrid_solver, hybrid_diagnostics = optimize_tile_graph(
            source,
            inference=inference,
            analysis=analysis,
            steps=steps,
            seed=seed,
            device=device,
            solver_params=solver_params,
            geometry_reference_rgba=geometry_solver.target_rgba,
            geometry_guidance_strength=geometry_solver.guidance_strength,
        )
        geometry_score = source_lattice_consistency_breakdown(
            source,
            geometry_solver.target_rgba,
            target_width=inference.target_width,
            target_height=inference.target_height,
            phase_x=inference.phase_x,
            phase_y=inference.phase_y,
        )["score"]
        diagnostics = dict(hybrid_diagnostics)
        diagnostics["mode"] = "hybrid"
        diagnostics["hybrid_geometry_prepass_mode"] = "continuous"
        diagnostics["hybrid_geometry_prepass_steps"] = 0.0
        diagnostics["hybrid_geometry_prepass_source_fidelity"] = float(geometry_score)
        diagnostics["hybrid_geometry_match_weight"] = float(solver_params.hybrid_geometry_match_weight)
        diagnostics["hybrid_geometry_edge_boost"] = float(solver_params.hybrid_geometry_edge_boost)
        return hybrid_solver, diagnostics
    solver = optimize_uv_field(
        source,
        inference=inference,
        analysis=analysis,
        steps=steps,
        seed=seed,
        device=device,
        solver_params=solver_params,
    )
    return solver, {"mode": "continuous"}
