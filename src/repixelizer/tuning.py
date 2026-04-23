from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import run_roundtrip_benchmark
from .diagnostics import write_json
from .params import SolverHyperParams


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


def _normalize_group(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), 1e-4, None)
    return values / np.sum(values)


def _mutate_group(rng: np.random.Generator, values: list[float], scale: float) -> np.ndarray:
    base = np.asarray(values, dtype=np.float64)
    noise = np.exp(rng.normal(0.0, 0.28 * scale, size=base.shape))
    return _normalize_group(base * noise)


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
    refine_group = _mutate_group(
        rng,
        [
            base.refine_anchor_weight,
            base.refine_source_weight,
            base.refine_alpha_weight,
            base.refine_distance_weight,
        ],
        scale,
    )
    structure_group = _mutate_group(
        rng,
        [
            base.structure_boundary_weight,
            base.structure_anchor_adjacency_weight,
            base.structure_anchor_motif_weight,
            base.structure_anchor_line_weight,
        ],
        scale,
    )
    return replace(
        base,
        representative_softmax_scale=_mutate_positive(
            rng,
            base.representative_softmax_scale,
            scale=scale,
            low=8.0,
            high=36.0,
        ),
        boundary_probe_scale=_mutate_linear(
            rng,
            base.boundary_probe_scale,
            scale=scale,
            sigma=0.03,
            low=0.14,
            high=0.32,
        ),
        refine_anchor_weight=float(refine_group[0]),
        refine_source_weight=float(refine_group[1]),
        refine_alpha_weight=float(refine_group[2]),
        refine_distance_weight=float(refine_group[3]),
        refine_source_delta_weight=_mutate_linear(
            rng,
            base.refine_source_delta_weight,
            scale=scale,
            sigma=0.08,
            low=0.05,
            high=0.45,
        ),
        refine_orthogonal_weight=_mutate_linear(
            rng,
            base.refine_orthogonal_weight,
            scale=scale,
            sigma=0.05,
            low=0.10,
            high=0.40,
        ),
        refine_diagonal_weight=_mutate_linear(
            rng,
            base.refine_diagonal_weight,
            scale=scale,
            sigma=0.025,
            low=0.02,
            high=0.18,
        ),
        refine_motif_weight=_mutate_linear(
            rng,
            base.refine_motif_weight,
            scale=scale,
            sigma=0.05,
            low=0.02,
            high=0.40,
        ),
        refine_line_weight=_mutate_linear(
            rng,
            base.refine_line_weight,
            scale=scale,
            sigma=0.04,
            low=0.02,
            high=0.30,
        ),
        relax_anchor_scale=_mutate_linear(
            rng,
            base.relax_anchor_scale,
            scale=scale,
            sigma=0.16,
            low=0.15,
            high=1.0,
        ),
        relax_orthogonal_weight=_mutate_linear(
            rng,
            base.relax_orthogonal_weight,
            scale=scale,
            sigma=0.08,
            low=0.10,
            high=0.70,
        ),
        relax_diagonal_weight=_mutate_linear(
            rng,
            base.relax_diagonal_weight,
            scale=scale,
            sigma=0.05,
            low=0.02,
            high=0.35,
        ),
        relax_source_adjacency_weight=_mutate_linear(
            rng,
            base.relax_source_adjacency_weight,
            scale=scale,
            sigma=0.10,
            low=0.05,
            high=0.80,
        ),
        relax_source_motif_weight=_mutate_linear(
            rng,
            base.relax_source_motif_weight,
            scale=scale,
            sigma=0.10,
            low=0.05,
            high=0.80,
        ),
        relax_source_line_weight=_mutate_linear(
            rng,
            base.relax_source_line_weight,
            scale=scale,
            sigma=0.08,
            low=0.02,
            high=0.45,
        ),
        structure_boundary_weight=float(structure_group[0]),
        structure_anchor_adjacency_weight=float(structure_group[1]),
        structure_anchor_motif_weight=float(structure_group[2]),
        structure_anchor_line_weight=float(structure_group[3]),
        structure_source_adjacency_weight=_mutate_linear(
            rng,
            base.structure_source_adjacency_weight,
            scale=scale,
            sigma=0.12,
            low=0.05,
            high=0.90,
        ),
        structure_source_motif_weight=_mutate_linear(
            rng,
            base.structure_source_motif_weight,
            scale=scale,
            sigma=0.12,
            low=0.05,
            high=0.90,
        ),
        structure_source_line_weight=_mutate_linear(
            rng,
            base.structure_source_line_weight,
            scale=scale,
            sigma=0.08,
            low=0.02,
            high=0.50,
        ),
        phase_rerank_margin=_mutate_linear(
            rng,
            base.phase_rerank_margin,
            scale=scale,
            sigma=0.0015,
            low=0.0005,
            high=0.010,
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
    if effective_limit is None and not include_cases:
        effective_limit = 8

    out_path = Path(out_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_path / "_scratch"
    best_dir = out_path / "best-benchmark"
    rng = np.random.default_rng(seed)

    best_params = SolverHyperParams()
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
    return payload
