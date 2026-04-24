from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze_phase_field_source, analyze_tile_graph_source
from .diagnostics import (
    summarize_run,
    write_alpha_preview,
    write_comparison,
    write_displacement_preview,
    write_heatmap,
    write_json,
    write_lattice_overlay,
)
from .discrete import cleanup_pixels
from .inference import infer_fixed_lattice, infer_lattice, inference_to_json
from .io import load_rgba, nearest_resize, save_rgba
from .metrics import (
    foreground_edge_concentration,
    foreground_edge_position_error,
    foreground_stroke_wobble_error,
    source_lattice_consistency_breakdown,
)
from .params import SolverHyperParams
from .phase_field import optimize_phase_field
from .palette import load_palette, quantize_rgba, save_palette_report
from .preprocess import strip_edge_background
from .tile_graph import optimize_tile_graph
from .types import InferenceResult, PhaseFieldSourceAnalysis, RunResult, TileGraphSourceAnalysis


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_size: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    phase_x: float | None = None,
    phase_y: float | None = None,
    palette_path: str | Path | None = None,
    palette_mode: str = "off",
    diagnostics_dir: str | Path | None = None,
    seed: int = 7,
    steps: int = 200,
    device: str = "auto",
    solver_params: SolverHyperParams | None = None,
    strip_background: bool = False,
    reconstruction_mode: str = "phase-field",
    enable_phase_rerank: bool = True,
) -> RunResult:
    started = time.perf_counter()
    solver_params = solver_params or SolverHyperParams()
    source = load_rgba(input_path)
    if strip_background:
        source = strip_edge_background(source)
    fixed_dims = _resolve_requested_target_dims(
        source_width=source.shape[1],
        source_height=source.shape[0],
        target_size=target_size,
        target_width=target_width,
        target_height=target_height,
        phase_x=phase_x,
        phase_y=phase_y,
    )
    if fixed_dims is None:
        inference = infer_lattice(source, target_size=target_size, device=device)
        inference_mode = "searched"
    else:
        inference = infer_fixed_lattice(
            source,
            target_width=fixed_dims[0],
            target_height=fixed_dims[1],
            phase_x=phase_x,
            phase_y=phase_y,
            device=device,
        )
        inference_mode = "fixed"
    analysis = _prepare_analysis(
        source,
        seed=seed,
        device=device,
        reconstruction_mode=reconstruction_mode,
    )
    inference = _select_phase_candidate(
        source,
        inference,
        analysis=analysis,
        steps=steps,
        seed=seed,
        device=device,
        solver_params=solver_params,
        reconstruction_mode=reconstruction_mode,
        enable_phase_rerank=enable_phase_rerank,
    )
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
        stage_displacements = result.solver.stage_diagnostics.get("displacements", {})
        for stage_name, payload in stage_displacements.items():
            displacement_x = payload.get("displacement_x")
            displacement_y = payload.get("displacement_y")
            if isinstance(displacement_x, np.ndarray) and isinstance(displacement_y, np.ndarray):
                displacement_rgba = write_displacement_preview(
                    diagnostics_path / f"displacement-{stage_name}.png",
                    displacement_x,
                    displacement_y,
                )
                save_rgba(
                    diagnostics_path / f"displacement-{stage_name}-preview.png",
                    nearest_resize(
                        displacement_rgba,
                        width=output_rgba.shape[1] * 8,
                        height=output_rgba.shape[0] * 8,
                    ),
                )
        run_json = summarize_run(result)
        run_json["inference"] = inference_to_json(inference)
        run_json["settings"] = {
            "target_size": target_size,
            "target_width": target_width,
            "target_height": target_height,
            "phase_x": phase_x,
            "phase_y": phase_y,
            "palette_mode": palette_mode,
            "seed": seed,
            "steps": steps,
            "device": device,
            "strip_background": strip_background,
            "reconstruction_mode": reconstruction_mode,
            "enable_phase_rerank": enable_phase_rerank,
            "inference_mode": inference_mode,
            "solver_params": solver_params.to_dict(),
        }
        if diagnostics.get("reconstruction"):
            run_json["reconstruction"] = diagnostics["reconstruction"]
        write_json(diagnostics_path / "run.json", run_json)
        if palette_result is not None:
            save_palette_report(diagnostics_path / "palette-report.json", palette_result.palette)
    return result


def _prepare_analysis(
    source: np.ndarray,
    *,
    seed: int,
    device: str,
    reconstruction_mode: str,
) -> PhaseFieldSourceAnalysis | TileGraphSourceAnalysis:
    if reconstruction_mode == "tile-graph":
        return analyze_tile_graph_source(source, device=device)
    return analyze_phase_field_source(source, seed=seed, device=device)


def _select_phase_candidate(
    source: np.ndarray,
    inference: InferenceResult,
    *,
    analysis,
    steps: int = 0,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
    reconstruction_mode: str = "phase-field",
    enable_phase_rerank: bool = True,
) -> InferenceResult:
    return _select_phase_candidate_with_reconstruction(
        source,
        inference,
        analysis=analysis,
        steps=steps,
        seed=seed,
        device=device,
        solver_params=solver_params,
        reconstruction_mode=reconstruction_mode,
        enable_phase_rerank=enable_phase_rerank,
    )


def _select_phase_candidate_with_reconstruction(
    source: np.ndarray,
    inference: InferenceResult,
    *,
    analysis,
    steps: int = 0,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
    reconstruction_mode: str = "phase-field",
    enable_phase_rerank: bool = True,
) -> InferenceResult:
    solver_params = solver_params or SolverHyperParams()
    if not enable_phase_rerank:
        return inference
    if reconstruction_mode != "phase-field":
        return inference
    if len(inference.top_candidates) <= 1 or inference.confidence >= solver_params.phase_rerank_confidence_threshold:
        return inference

    preview_steps = min(max(0, int(steps)), max(0, int(solver_params.phase_rerank_preview_steps)))
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
        candidate_artifacts, _candidate_diagnostics = _run_reconstruction(
            source,
            inference=candidate_inference,
            analysis=analysis,
            steps=preview_steps,
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
    size_penalty = [
        min(1.0, float(record["size_delta_ratio"]) / max(solver_params.phase_rerank_max_size_delta_ratio, 1e-6))
        for record in candidate_records
    ]
    inference_penalty = _normalize_penalty(record["inference_penalty"] for record in candidate_records)

    baseline_rank: float | None = None
    best_rank = float("inf")
    best_candidate = inference
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
        if rank < best_rank:
            best_rank = rank
            best_candidate = record["inference"]
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
            return selected
        return inference
    return best_candidate


def _resolve_requested_target_dims(
    *,
    source_width: int,
    source_height: int,
    target_size: int | None,
    target_width: int | None,
    target_height: int | None,
    phase_x: float | None,
    phase_y: float | None,
) -> tuple[int, int] | None:
    explicit_size = target_width is not None or target_height is not None
    explicit_phase = phase_x is not None or phase_y is not None
    if not explicit_size and target_size is None:
        if explicit_phase:
            raise ValueError("phase_x/phase_y require target_size or target_width/target_height so the lattice can be fixed explicitly.")
        return None
    if target_size is not None and explicit_size:
        raise ValueError("target_size cannot be combined with target_width or target_height.")
    if target_size is not None:
        if source_width >= source_height:
            return int(target_size), max(1, round(source_height * int(target_size) / max(1, source_width)))
        return max(1, round(source_width * int(target_size) / max(1, source_height))), int(target_size)
    if target_width is None and target_height is None:
        return None
    if target_width is None:
        resolved_height = int(target_height)
        resolved_width = max(1, round(source_width * resolved_height / max(1, source_height)))
        return resolved_width, resolved_height
    if target_height is None:
        resolved_width = int(target_width)
        resolved_height = max(1, round(source_height * resolved_width / max(1, source_width)))
        return resolved_width, resolved_height
    return int(target_width), int(target_height)


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
    if reconstruction_mode == "phase-field":
        solver = optimize_phase_field(
            source,
            inference=inference,
            analysis=analysis,
            steps=steps,
            seed=seed,
            device=device,
            solver_params=solver_params,
        )
        phase_field_metrics = getattr(solver, "stage_diagnostics", {}).get("phase_field", {})
        return solver, {
            "mode": "phase-field",
            **{
                f"phase_field_{key}": value
                for key, value in phase_field_metrics.items()
            },
        }
    raise ValueError(f"Unknown reconstruction mode: {reconstruction_mode}")
