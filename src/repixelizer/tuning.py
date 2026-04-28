from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import run_roundtrip_benchmark
from .diagnostics import write_json
from .params import SolverHyperParams, load_solver_params


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-6))


def _row_score(row: dict[str, Any]) -> float:
    error_ratio = _safe_ratio(float(row["optimized_error_to_original"]), float(row["naive_error_to_original"]))
    adjacency_ratio = _safe_ratio(
        float(row["optimized_adjacency_error_to_original"]),
        float(row["naive_adjacency_error_to_original"]),
    )
    motif_ratio = _safe_ratio(float(row["optimized_motif_error_to_original"]), float(row["naive_motif_error_to_original"]))
    return error_ratio * 0.20 + adjacency_ratio * 0.45 + motif_ratio * 0.35


def _score_summary(summary: dict[str, Any]) -> dict[str, float]:
    row_scores = np.asarray([_row_score(row) for row in summary["rows"]], dtype=np.float64)
    mean_row_score = float(np.mean(row_scores))
    worst_row_score = float(np.max(row_scores))
    score = mean_row_score * 0.70 + worst_row_score * 0.30
    return {
        "score": score,
        "mean_row_score": mean_row_score,
        "worst_row_score": worst_row_score,
    }


def _mutate_positive(
    rng: np.random.Generator,
    value: float,
    *,
    scale: float,
    low: float,
    high: float,
) -> float:
    mutated = value * float(np.exp(rng.normal(0.0, 0.22 * scale)))
    return float(np.clip(mutated, low, high))


def _mutate_linear(
    rng: np.random.Generator,
    value: float,
    *,
    scale: float,
    sigma: float,
    low: float,
    high: float,
) -> float:
    mutated = value + float(rng.normal(0.0, sigma * scale))
    return float(np.clip(mutated, low, high))


def _mutate_params(base: SolverHyperParams, rng: np.random.Generator, scale: float) -> SolverHyperParams:
    return replace(
        base,
        phase_field_grid_alignment_weight=_mutate_positive(
            rng,
            base.phase_field_grid_alignment_weight,
            scale=scale,
            low=1.0,
            high=10.0,
        ),
        phase_field_local_search_radius_ratio=_mutate_linear(
            rng,
            base.phase_field_local_search_radius_ratio,
            scale=scale,
            sigma=0.12,
            low=0.25,
            high=1.20,
        ),
        phase_field_local_search_blend=_mutate_linear(
            rng,
            base.phase_field_local_search_blend,
            scale=scale,
            sigma=0.055,
            low=0.06,
            high=0.45,
        ),
        phase_field_local_search_move_weight=_mutate_positive(
            rng,
            base.phase_field_local_search_move_weight,
            scale=scale,
            low=0.03,
            high=0.50,
        ),
        phase_field_signal_weight=_mutate_positive(
            rng,
            base.phase_field_signal_weight,
            scale=scale,
            low=0.05,
            high=0.80,
        ),
        phase_field_signal_boundary_weight=_mutate_positive(
            rng,
            base.phase_field_signal_boundary_weight,
            scale=scale,
            low=0.20,
            high=2.50,
        ),
        phase_field_signal_curvature_weight=_mutate_positive(
            rng,
            base.phase_field_signal_curvature_weight,
            scale=scale,
            low=0.10,
            high=2.00,
        ),
        phase_field_signal_self_penalty=_mutate_positive(
            rng,
            base.phase_field_signal_self_penalty,
            scale=scale,
            low=1.00,
            high=8.00,
        ),
        phase_field_signal_context_power=_mutate_positive(
            rng,
            base.phase_field_signal_context_power,
            scale=scale,
            low=0.25,
            high=2.00,
        ),
        phase_field_signal_peak_power=_mutate_positive(
            rng,
            base.phase_field_signal_peak_power,
            scale=scale,
            low=1.00,
            high=5.00,
        ),
        phase_field_signal_pyramid_levels=int(
            round(
                _mutate_linear(
                    rng,
                    float(base.phase_field_signal_pyramid_levels),
                    scale=scale,
                    sigma=0.65,
                    low=3.0,
                    high=7.0,
                )
            )
        ),
        phase_field_local_search_grid_weight=_mutate_linear(
            rng,
            base.phase_field_local_search_grid_weight,
            scale=scale,
            sigma=0.08,
            low=0.0,
            high=0.60,
        ),
    )


def tune_solver_hyperparams(
    corpus_dir: str | Path,
    out_dir: str | Path,
    *,
    trials: int = 8,
    variants: int = 1,
    profiles: list[str] | None = None,
    seed: int = 7,
    steps: int = 48,
    device: str = "auto",
    infer_size: bool = False,
    include_cases: list[str] | None = None,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be at least 1")

    selected_profiles = profiles or ["soft"]
    effective_limit = limit_cases

    out_path = Path(out_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_path / "_scratch"
    best_dir = out_path / "best-benchmark"
    rng = np.random.default_rng(seed)

    best_params = load_solver_params()
    best_score = float("inf")
    best_summary: dict[str, Any] | None = None
    trials_payload: list[dict[str, Any]] = []

    for trial_index in range(trials):
        if trial_index == 0:
            candidate = best_params
            origin = "default"
        else:
            anneal = max(0.35, 1.0 - (trial_index / max(2, trials)))
            candidate = _mutate_params(best_params, rng, scale=anneal)
            origin = "mutated-best"

        summary = run_roundtrip_benchmark(
            corpus_dir,
            scratch_dir,
            variants=variants,
            profiles=selected_profiles,
            seed=seed,
            steps=steps,
            device=device,
            infer_size=infer_size,
            include_cases=include_cases,
            limit_cases=effective_limit,
            keep_existing=False,
            solver_params=candidate,
        )
        score_info = _score_summary(summary)
        improved = score_info["score"] < best_score
        trials_payload.append(
            {
                "trial": trial_index + 1,
                "origin": origin,
                "score": score_info["score"],
                "mean_row_score": score_info["mean_row_score"],
                "worst_row_score": score_info["worst_row_score"],
                "params": candidate.to_dict(),
                "improved_best": improved,
            }
        )
        if improved:
            best_score = score_info["score"]
            best_params = candidate
            best_summary = summary

    final_summary = run_roundtrip_benchmark(
        corpus_dir,
        best_dir,
        variants=variants,
        profiles=selected_profiles,
        seed=seed,
        steps=steps,
        device=device,
        infer_size=infer_size,
        include_cases=include_cases,
        limit_cases=effective_limit,
        keep_existing=False,
        solver_params=best_params,
    )
    final_score = _score_summary(final_summary)

    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)

    payload = {
        "trials_requested": trials,
        "trials_completed": len(trials_payload),
        "variants": variants,
        "profiles": selected_profiles,
        "seed": seed,
        "steps": steps,
        "device": device,
        "infer_size": infer_size,
        "include_cases": include_cases or [],
        "limit_cases_effective": effective_limit,
        "objective": "0.20 * error_ratio_to_naive + 0.45 * adjacency_ratio_to_naive + 0.35 * motif_ratio_to_naive; lower is better",
        "mutation_scope": "explicit solver only: signal, local search, and lattice spring parameters",
        "trial_results": trials_payload,
        "best_score": final_score["score"],
        "best_mean_row_score": final_score["mean_row_score"],
        "best_worst_row_score": final_score["worst_row_score"],
        "best_params": best_params.to_dict(),
        "best_benchmark_dir": str(best_dir),
        "best_benchmark": final_summary,
        "initial_benchmark": best_summary,
    }
    write_json(out_path / "tuning-results.json", payload)
    write_json(out_path / "best-solver-params.json", best_params.to_dict())
    return payload
