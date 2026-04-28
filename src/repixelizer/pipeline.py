from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .observe import PipelineObserver, check_observer_cancelled, emit_observer
from .analysis import analyze_phase_field_source
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
from .inference import infer_autocorr_lattice, infer_fixed_lattice, infer_lattice, inference_to_json
from .io import load_rgba, nearest_resize, save_rgba
from .metrics import (
    foreground_edge_concentration,
    foreground_edge_position_error,
    foreground_stroke_wobble_error,
    source_lattice_consistency_breakdown,
)
from .params import SolverHyperParams, load_solver_params
from .phase_field import optimize_phase_field
from .palette import load_palette, quantize_rgba, save_palette_report
from .preprocess import strip_edge_background
from .types import InferenceResult, PhaseFieldSourceAnalysis, RunResult


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_size: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    palette_path: str | Path | None = None,
    palette_mode: str = "off",
    diagnostics_dir: str | Path | None = None,
    seed: int = 7,
    steps: int = 200,
    device: str = "auto",
    solver_params: SolverHyperParams | None = None,
    strip_background: bool = False,
    enable_candidate_rerank: bool = True,
    lattice_inference_mode: str = "search",
    max_inferred_target_size: int | None = None,
    observer: PipelineObserver | None = None,
) -> RunResult:
    source = load_rgba(input_path)
    return run_pipeline_rgba(
        source,
        output_path=output_path,
        target_size=target_size,
        target_width=target_width,
        target_height=target_height,
        palette_path=palette_path,
        palette_mode=palette_mode,
        diagnostics_dir=diagnostics_dir,
        seed=seed,
        steps=steps,
        device=device,
        solver_params=solver_params,
        strip_background=strip_background,
        enable_candidate_rerank=enable_candidate_rerank,
        lattice_inference_mode=lattice_inference_mode,
        max_inferred_target_size=max_inferred_target_size,
        observer=observer,
    )


def run_pipeline_rgba(
    source: np.ndarray,
    *,
    output_path: str | Path | None = None,
    target_size: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    palette_path: str | Path | None = None,
    palette_mode: str = "off",
    diagnostics_dir: str | Path | None = None,
    seed: int = 7,
    steps: int = 200,
    device: str = "auto",
    solver_params: SolverHyperParams | None = None,
    strip_background: bool = False,
    enable_candidate_rerank: bool = True,
    lattice_inference_mode: str = "search",
    max_inferred_target_size: int | None = None,
    observer: PipelineObserver | None = None,
) -> RunResult:
    started = time.perf_counter()
    solver_params = solver_params or load_solver_params()
    check_observer_cancelled(observer)
    emit_observer(observer, "source_loaded", source_rgba=source.copy())
    if strip_background:
        check_observer_cancelled(observer)
        emit_observer(
            observer,
            "stage_started",
            stage="preprocess",
            label="Background cleanup",
            detail="Stripping edge-connected neutral junk before lattice search.",
        )
        source = strip_edge_background(source)
        emit_observer(observer, "preprocess_completed", source_rgba=source.copy(), operation="strip_background")
    check_observer_cancelled(observer)
    fixed_dims = _resolve_requested_target_dims(
        source_width=source.shape[1],
        source_height=source.shape[0],
        target_size=target_size,
        target_width=target_width,
        target_height=target_height,
    )
    if fixed_dims is None:
        check_observer_cancelled(observer)
        if lattice_inference_mode == "autocorr":
            inference_label = "Autocorr lattice"
            inference_detail = "Scoring autocorr candidate sizes from canonical cell centers; the phase field is free to drift."
        elif lattice_inference_mode == "search":
            inference_label = "Lattice search"
            inference_detail = "Searching for output size from canonical cell centers; the phase field handles local drift."
        else:
            raise ValueError(f"Unknown lattice_inference_mode '{lattice_inference_mode}'.")
        emit_observer(
            observer,
            "stage_started",
            stage="inference",
            label=inference_label,
            detail=inference_detail,
        )
        if lattice_inference_mode == "autocorr":
            inference = infer_autocorr_lattice(
                source,
                max_target_size=max_inferred_target_size,
                device=device,
                observer=observer,
            )
            inference_mode = "autocorr"
        else:
            inference = infer_lattice(source, target_size=target_size, device=device, observer=observer)
            inference_mode = "searched"
    else:
        check_observer_cancelled(observer)
        emit_observer(
            observer,
            "stage_started",
            stage="inference",
            label="Pinned lattice",
            detail="Locking in the requested size from canonical cell centers; the phase field handles local drift.",
        )
        inference = infer_fixed_lattice(
            source,
            target_width=fixed_dims[0],
            target_height=fixed_dims[1],
            device=device,
        )
        inference_mode = "fixed"
    emit_observer(observer, "inference_candidates_ready", inference=inference, inference_mode=inference_mode)
    check_observer_cancelled(observer)
    emit_observer(
        observer,
        "stage_started",
        stage="analysis",
        label="Input analysis",
        detail="Mapping sharp cells, edges, and guidance before the solver starts.",
    )
    analysis = _prepare_analysis(
        source,
        seed=seed,
        device=device,
    )
    emit_observer(observer, "analysis_completed", edge_map=analysis.edge_map.copy())
    check_observer_cancelled(observer)
    emit_observer(
        observer,
        "stage_started",
        stage="selection",
        label="Candidate selection",
        detail="Choosing the lattice candidate that deserves the full solve.",
    )
    inference = _select_candidate(
        source,
        inference,
        analysis=analysis,
        steps=steps,
        seed=seed,
        device=device,
        solver_params=solver_params,
        enable_candidate_rerank=enable_candidate_rerank,
        observer=observer,
    )
    emit_observer(observer, "candidate_selection_completed", inference=inference, inference_mode=inference_mode)
    check_observer_cancelled(observer)
    emit_observer(
        observer,
        "stage_started",
        stage="solver",
        label="Phase-field solve",
        detail=(
            "Running the solver over the pinned lattice."
            if steps > 0
            else "Skipping solver drift and keeping the initial placement."
        ),
    )
    solver, reconstruction_diagnostics = _run_reconstruction(
        source,
        inference=inference,
        analysis=analysis,
        steps=steps,
        seed=seed,
        device=device,
        solver_params=solver_params,
        observer=observer,
    )
    check_observer_cancelled(observer)
    emit_observer(
        observer,
        "stage_started",
        stage="cleanup",
        label="Cleanup",
        detail="Sweeping isolated noise and stubborn single-pixel junk.",
    )
    cleanup = cleanup_pixels(solver.target_rgba, source_guidance=solver.guidance_strength)
    emit_observer(
        observer,
        "cleanup_completed",
        cleaned_rgba=cleanup.cleaned_rgba.copy(),
        isolated_heatmap=cleanup.isolated_heatmap.copy(),
    )
    palette = load_palette(palette_path) if palette_path else None
    check_observer_cancelled(observer)
    emit_observer(
        observer,
        "stage_started",
        stage="output",
        label="Final output",
        detail="Writing the finished output and packaging the summary.",
    )
    palette_result = quantize_rgba(cleanup.cleaned_rgba, mode=palette_mode, palette=palette)
    output_rgba = palette_result.rgba if palette_result else cleanup.cleaned_rgba
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_rgba(output_path, output_rgba)
    emit_observer(
        observer,
        "palette_completed",
        output_rgba=output_rgba.copy(),
        palette_mode=palette_mode,
        palette_result=palette_result,
    )

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
    run_summary: dict[str, Any] | None = None
    if diagnostics_dir or observer is not None:
        run_summary = summarize_run(result)
        run_summary["inference"] = inference_to_json(inference)
        run_summary["settings"] = {
            "target_size": target_size,
            "target_width": target_width,
            "target_height": target_height,
            "palette_mode": palette_mode,
            "seed": seed,
            "steps": steps,
            "device": device,
            "strip_background": strip_background,
            "enable_candidate_rerank": enable_candidate_rerank,
            "lattice_inference_mode": lattice_inference_mode,
            "max_inferred_target_size": max_inferred_target_size,
            "inference_mode": inference_mode,
            "solver_params": solver_params.to_dict(),
        }
        if diagnostics.get("reconstruction"):
            run_summary["reconstruction"] = diagnostics["reconstruction"]
    emit_observer(
        observer,
        "pipeline_completed",
        output_rgba=output_rgba.copy(),
        diagnostics=diagnostics.copy(),
        run_summary=run_summary,
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
        write_json(diagnostics_path / "run.json", run_summary or {})
        if palette_result is not None:
            save_palette_report(diagnostics_path / "palette-report.json", palette_result.palette)
    return result


def _prepare_analysis(
    source: np.ndarray,
    *,
    seed: int,
    device: str,
) -> PhaseFieldSourceAnalysis:
    return analyze_phase_field_source(source, seed=seed, device=device)


def _select_candidate(
    source: np.ndarray,
    inference: InferenceResult,
    *,
    analysis,
    steps: int = 0,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
    enable_candidate_rerank: bool = True,
    observer: PipelineObserver | None = None,
) -> InferenceResult:
    return _select_candidate_with_reconstruction(
        source,
        inference,
        analysis=analysis,
        steps=steps,
        seed=seed,
        device=device,
        solver_params=solver_params,
        enable_candidate_rerank=enable_candidate_rerank,
        observer=observer,
    )


def _select_candidate_with_reconstruction(
    source: np.ndarray,
    inference: InferenceResult,
    *,
    analysis,
    steps: int = 0,
    seed: int,
    device: str,
    solver_params: SolverHyperParams | None = None,
    enable_candidate_rerank: bool = True,
    observer: PipelineObserver | None = None,
) -> InferenceResult:
    solver_params = solver_params or load_solver_params()
    if not enable_candidate_rerank:
        return inference
    if len(inference.top_candidates) <= 1 or inference.confidence >= solver_params.candidate_rerank_confidence_threshold:
        return inference

    preview_steps = min(max(0, int(steps)), max(0, int(solver_params.candidate_rerank_preview_steps)))
    top_score = float(inference.top_candidates[0].score)
    emit_observer(
        observer,
        "candidate_rerank_started",
        preview_steps=preview_steps,
        candidate_count=min(8, len(inference.top_candidates)),
        confidence=float(inference.confidence),
    )
    total_candidates = min(8, len(inference.top_candidates))
    candidate_records: list[dict[str, float | InferenceResult]] = []
    for candidate_index, candidate in enumerate(inference.top_candidates[:total_candidates], start=1):
        check_observer_cancelled(observer)
        candidate_inference = InferenceResult(
            target_width=candidate.target_width,
            target_height=candidate.target_height,
            confidence=inference.confidence,
            top_candidates=inference.top_candidates,
        )
        emit_observer(
            observer,
            "candidate_rerank_candidate_started",
            candidate_index=candidate_index,
            total_candidates=total_candidates,
            target_width=int(candidate.target_width),
            target_height=int(candidate.target_height),
            preview_steps=preview_steps,
        )

        def rerank_observer(event: str, payload: dict[str, Any]) -> None:
            if observer is None:
                return
            common = {
                "candidate_index": candidate_index,
                "total_candidates": total_candidates,
                "target_width": int(candidate.target_width),
                "target_height": int(candidate.target_height),
            }
            if event in {"phase_field_initial", "phase_field_step"}:
                emit_observer(
                    observer,
                    "candidate_rerank_candidate_step",
                    step=int(payload["step"]),
                    total_steps=int(payload["total_steps"]),
                    loss=None if payload.get("loss") is None else float(payload["loss"]),
                    **common,
                )

        rerank_observer.phase_field_include_snapshot = False  # type: ignore[attr-defined]

        candidate_artifacts, _candidate_diagnostics = _run_reconstruction(
            source,
            inference=candidate_inference,
            analysis=analysis,
            steps=preview_steps,
            seed=seed,
            device=device,
            solver_params=solver_params,
            observer=rerank_observer if observer is not None else None,
        )
        check_observer_cancelled(observer)
        emit_observer(
            observer,
            "candidate_rerank_candidate_completed",
            candidate_index=candidate_index,
            total_candidates=total_candidates,
            completed_candidates=candidate_index,
            target_width=int(candidate.target_width),
            target_height=int(candidate.target_height),
            total_steps=preview_steps,
            final_loss=(
                None
                if not getattr(candidate_artifacts, "loss_history", None)
                else float(candidate_artifacts.loss_history[-1])
            ),
        )
        support = source_lattice_consistency_breakdown(
            source,
            candidate_artifacts.target_rgba,
            target_width=candidate.target_width,
            target_height=candidate.target_height,
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
        min(1.0, float(record["size_delta_ratio"]) / max(solver_params.candidate_rerank_max_size_delta_ratio, 1e-6))
        for record in candidate_records
    ]
    inference_penalty = _normalize_penalty(record["inference_penalty"] for record in candidate_records)

    baseline_rank: float | None = None
    best_rank = float("inf")
    best_candidate = inference
    annotated_candidates = []
    for index, record in enumerate(candidate_records):
        candidate_inference = record["inference"]
        if float(record["size_delta_ratio"]) > solver_params.candidate_rerank_max_size_delta_ratio:
            continue
        rank = (
            solver_params.candidate_rerank_support_weight * support_penalty[index]
            + solver_params.candidate_rerank_edge_position_weight * edge_position_penalty[index]
            + solver_params.candidate_rerank_wobble_weight * wobble_penalty[index]
            + solver_params.candidate_rerank_edge_concentration_weight * edge_concentration_penalty[index]
            + solver_params.candidate_rerank_size_penalty_weight * size_penalty[index]
            + solver_params.candidate_rerank_inference_penalty_weight * inference_penalty[index]
        )
        source_candidate = inference.top_candidates[index]
        breakdown = dict(source_candidate.breakdown)
        breakdown["candidate_rerank_support_score"] = float(record["support_score"])
        breakdown["candidate_rerank_edge_position_error"] = float(record["edge_position_error"])
        breakdown["candidate_rerank_stroke_wobble_error"] = float(record["stroke_wobble_error"])
        breakdown["candidate_rerank_edge_concentration"] = float(record["edge_concentration"])
        breakdown["candidate_rerank_size_delta_ratio"] = float(record["size_delta_ratio"])
        breakdown["candidate_rerank_size_penalty"] = float(size_penalty[index])
        breakdown["candidate_rerank_inference_penalty"] = float(record["inference_penalty"])
        breakdown["candidate_rerank_score"] = float(rank)
        annotated_candidates.append(
            source_candidate.__class__(
                target_width=source_candidate.target_width,
                target_height=source_candidate.target_height,
                score=source_candidate.score,
                breakdown=breakdown,
            )
        )
        if baseline_rank is None:
            baseline_rank = rank
        if rank < best_rank:
            best_rank = rank
            best_candidate = record["inference"]
    annotated_candidates.sort(key=lambda candidate: float(candidate.breakdown.get("candidate_rerank_score", float("inf"))))
    for rank, candidate in enumerate(annotated_candidates, start=1):
        candidate.breakdown["candidate_rerank_rank"] = float(rank)
    if annotated_candidates:
        best_candidate = InferenceResult(
            target_width=best_candidate.target_width,
            target_height=best_candidate.target_height,
            confidence=inference.confidence,
            top_candidates=annotated_candidates,
        )
    if baseline_rank is None or best_rank > baseline_rank - solver_params.candidate_rerank_margin:
        if annotated_candidates:
            selected = InferenceResult(
                target_width=inference.target_width,
                target_height=inference.target_height,
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
) -> tuple[int, int] | None:
    explicit_size = target_width is not None or target_height is not None
    if not explicit_size and target_size is None:
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
    observer: PipelineObserver | None = None,
):
    if observer is None:
        solver = optimize_phase_field(
            source,
            inference=inference,
            analysis=analysis,
            steps=steps,
            seed=seed,
            device=device,
            solver_params=solver_params,
        )
    else:
        solver = optimize_phase_field(
            source,
            inference=inference,
            analysis=analysis,
            steps=steps,
            seed=seed,
            device=device,
            solver_params=solver_params,
            observer=observer,
        )
    phase_field_metrics = getattr(solver, "stage_diagnostics", {}).get("phase_field", {})
    return solver, {
        "mode": "phase-field",
        **{
            f"phase_field_{key}": value
            for key, value in phase_field_metrics.items()
        },
    }
