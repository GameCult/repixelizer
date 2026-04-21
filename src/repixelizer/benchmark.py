from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
import json

import numpy as np

from .baselines import error_diffusion_baseline, naive_resize_baseline
from .diagnostics import write_compare_csv, write_json
from .io import load_rgba, nearest_resize, save_rgba
from .metrics import (
    coherence_breakdown,
    exact_match_ratio,
    foreground_adjacency_error,
    foreground_coverage,
    foreground_exact_match_ratio,
    foreground_motif_error,
    foreground_reconstruction_error,
    reconstruction_error,
)
from .params import SolverHyperParams
from .pipeline import run_pipeline
from .synthetic import fake_pixelize

SUPPORTED_EXTENSIONS = {".png", ".bmp", ".gif", ".webp"}


def run_roundtrip_benchmark(
    corpus_dir: str | Path,
    out_dir: str | Path,
    *,
    variants: int = 3,
    profiles: list[str] | None = None,
    seed: int = 7,
    steps: int = 200,
    device: str = "auto",
    infer_size: bool = False,
    include_cases: list[str] | None = None,
    limit_cases: int | None = None,
    keep_existing: bool = False,
    solver_params: SolverHyperParams | None = None,
) -> dict[str, Any]:
    corpus_path = Path(corpus_dir)
    originals_dir = corpus_path / "originals"
    all_original_paths = sorted(
        path for path in originals_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    original_paths = [
        path
        for path in all_original_paths
        if _case_matches(path.relative_to(originals_dir).with_suffix("").as_posix(), path, include_cases or [])
    ]
    if limit_cases is not None:
        original_paths = original_paths[: max(0, limit_cases)]
    if not original_paths:
        raise RuntimeError(f"No source pixel-art originals found in {originals_dir}.")
    selected_profiles = profiles or ["soft", "crisp"]

    out_path = Path(out_dir)
    if out_path.exists() and not keep_existing:
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case_index, original_path in enumerate(original_paths):
        original = load_rgba(original_path)
        metadata = _load_metadata(original_path)
        case_id = original_path.relative_to(originals_dir).with_suffix("").as_posix()
        case_dir = out_path / "cases" / Path(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        save_rgba(case_dir / "original.png", original)
        for profile_index, profile in enumerate(selected_profiles):
            for variant_index in range(variants):
                variant_seed = seed + case_index * 1000 + profile_index * 100 + variant_index
                settings = _variant_settings(variant_seed, profile=profile)
                variant_dir = case_dir / f"profile-{profile}" / f"variant-{variant_index + 1:02d}"
                diagnostics_dir = variant_dir / "diagnostics"
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                fake = fake_pixelize(original, seed=variant_seed, **settings)
                input_path = variant_dir / "input.png"
                output_path = variant_dir / "optimized.png"
                save_rgba(input_path, fake)

                locked_target_size = max(original.shape[0], original.shape[1]) if not infer_size else None
                result = run_pipeline(
                    input_path,
                    output_path,
                    target_size=locked_target_size,
                    diagnostics_dir=diagnostics_dir,
                    seed=variant_seed,
                    steps=steps,
                    device=device,
                    solver_params=solver_params,
                )

                naive = naive_resize_baseline(fake, width=original.shape[1], height=original.shape[0])
                diffusion = error_diffusion_baseline(fake, width=original.shape[1], height=original.shape[0])
                save_rgba(variant_dir / "naive.png", naive)
                save_rgba(variant_dir / "diffusion.png", diffusion)
                _save_preview(variant_dir / "optimized-preview.png", result.output_rgba, original)
                _save_preview(variant_dir / "naive-preview.png", naive, original)
                _save_preview(variant_dir / "diffusion-preview.png", diffusion, original)

                optimized_preview = nearest_resize(result.output_rgba, width=original.shape[1], height=original.shape[0])
                optimized_error = foreground_reconstruction_error(optimized_preview, original)
                naive_error = foreground_reconstruction_error(naive, original)
                diffusion_error = foreground_reconstruction_error(diffusion, original)
                optimized_adjacency = foreground_adjacency_error(optimized_preview, original)
                naive_adjacency = foreground_adjacency_error(naive, original)
                diffusion_adjacency = foreground_adjacency_error(diffusion, original)
                optimized_motif = foreground_motif_error(optimized_preview, original)
                naive_motif = foreground_motif_error(naive, original)
                diffusion_motif = foreground_motif_error(diffusion, original)
                row = {
                    "case_id": case_id,
                    "profile": profile,
                    "variant": variant_index + 1,
                    "source_file": str(original_path),
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "license": metadata.get("license", ""),
                    "source_url": metadata.get("source_url", ""),
                    "original_width": int(original.shape[1]),
                    "original_height": int(original.shape[0]),
                    "input_width": int(fake.shape[1]),
                    "input_height": int(fake.shape[0]),
                    "upscale": settings["upscale"],
                    "phase_x": settings["phase_x"],
                    "phase_y": settings["phase_y"],
                    "blur_radius": settings["blur_radius"],
                    "warp_strength": settings["warp_strength"],
                    "warp_detail": settings["warp_detail"],
                    "warp_sample_mode": settings["warp_sample_mode"],
                    "primary_metric": "foreground_premultiplied_mae",
                    "reference_foreground_coverage": foreground_coverage(original, original),
                    "target_size_locked": not infer_size,
                    "inferred_width": result.inference.target_width,
                    "inferred_height": result.inference.target_height,
                    "inference_confidence": result.inference.confidence,
                    "optimized_error_to_original": optimized_error,
                    "naive_error_to_original": naive_error,
                    "diffusion_error_to_original": diffusion_error,
                    "optimized_adjacency_error_to_original": optimized_adjacency,
                    "naive_adjacency_error_to_original": naive_adjacency,
                    "diffusion_adjacency_error_to_original": diffusion_adjacency,
                    "optimized_motif_error_to_original": optimized_motif,
                    "naive_motif_error_to_original": naive_motif,
                    "diffusion_motif_error_to_original": diffusion_motif,
                    "optimized_canvas_error_to_original": reconstruction_error(optimized_preview, original),
                    "naive_canvas_error_to_original": reconstruction_error(naive, original),
                    "diffusion_canvas_error_to_original": reconstruction_error(diffusion, original),
                    "optimized_exact_match": foreground_exact_match_ratio(optimized_preview, original),
                    "naive_exact_match": foreground_exact_match_ratio(naive, original),
                    "diffusion_exact_match": foreground_exact_match_ratio(diffusion, original),
                    "optimized_canvas_exact_match": exact_match_ratio(optimized_preview, original),
                    "naive_canvas_exact_match": exact_match_ratio(naive, original),
                    "diffusion_canvas_exact_match": exact_match_ratio(diffusion, original),
                    "optimized_coherence": coherence_breakdown(result.output_rgba)["coherence_score"],
                    "naive_coherence": coherence_breakdown(naive)["coherence_score"],
                    "diffusion_coherence": coherence_breakdown(diffusion)["coherence_score"],
                    "optimized_beats_naive_error": optimized_error <= naive_error,
                    "optimized_beats_diffusion_error": optimized_error <= diffusion_error,
                    "optimized_beats_naive_adjacency": optimized_adjacency <= naive_adjacency,
                    "optimized_beats_diffusion_adjacency": optimized_adjacency <= diffusion_adjacency,
                    "optimized_beats_naive_motif": optimized_motif <= naive_motif,
                    "optimized_beats_diffusion_motif": optimized_motif <= diffusion_motif,
                }
                rows.append(row)

    summary = {
        "case_count": len(original_paths),
        "row_count": len(rows),
        "variants_per_case": variants,
        "profiles": selected_profiles,
        "target_size_locked": not infer_size,
        "primary_metric": "foreground_premultiplied_mae",
        "solver_params": solver_params.to_dict() if solver_params is not None else SolverHyperParams().to_dict(),
        "cases": _summarize_cases(rows),
        "rows": rows,
    }
    write_compare_csv(out_path / "benchmark.csv", rows)
    write_json(out_path / "benchmark.json", summary)
    return summary


def _load_metadata(original_path: Path) -> dict[str, Any]:
    metadata_path = original_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _case_matches(case_id: str, original_path: Path, include_cases: list[str]) -> bool:
    if not include_cases:
        return True
    normalized = {token.lower() for token in include_cases}
    aliases = {
        case_id,
        case_id.lower(),
        original_path.stem,
        original_path.stem.lower(),
        original_path.name,
        original_path.name.lower(),
    }
    return any(token in aliases for token in normalized)


def _variant_settings(seed: int, profile: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if profile == "crisp":
        return {
            "upscale": int(rng.integers(8, 15)),
            "phase_x": float(rng.uniform(-0.45, 0.45)),
            "phase_y": float(rng.uniform(-0.45, 0.45)),
            "blur_radius": 0.0,
            "warp_strength": float(rng.uniform(0.14, 0.46)),
            "warp_detail": int(rng.integers(5, 9)),
            "warp_sample_mode": "nearest",
        }
    if profile == "soft":
        return {
            "upscale": int(rng.integers(8, 15)),
            "phase_x": float(rng.uniform(-0.45, 0.45)),
            "phase_y": float(rng.uniform(-0.45, 0.45)),
            "blur_radius": float(rng.uniform(0.45, 1.15)),
            "warp_strength": float(rng.uniform(0.18, 0.55)),
            "warp_detail": int(rng.integers(4, 8)),
            "warp_sample_mode": "bilinear",
        }
    raise ValueError(f"Unsupported benchmark profile: {profile}")


def _save_preview(path: Path, rgba: np.ndarray, original: np.ndarray) -> None:
    preview = nearest_resize(rgba, width=original.shape[1] * 8, height=original.shape[0] * 8)
    save_rgba(path, preview)


def _summarize_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case_id"]), str(row["profile"])), []).append(row)
    summaries = []
    for (case_id, profile), case_rows in sorted(grouped.items()):
        optimized = [float(row["optimized_error_to_original"]) for row in case_rows]
        naive = [float(row["naive_error_to_original"]) for row in case_rows]
        diffusion = [float(row["diffusion_error_to_original"]) for row in case_rows]
        optimized_adjacency = [float(row["optimized_adjacency_error_to_original"]) for row in case_rows]
        naive_adjacency = [float(row["naive_adjacency_error_to_original"]) for row in case_rows]
        diffusion_adjacency = [float(row["diffusion_adjacency_error_to_original"]) for row in case_rows]
        optimized_motif = [float(row["optimized_motif_error_to_original"]) for row in case_rows]
        naive_motif = [float(row["naive_motif_error_to_original"]) for row in case_rows]
        diffusion_motif = [float(row["diffusion_motif_error_to_original"]) for row in case_rows]
        summaries.append(
            {
                "case_id": case_id,
                "profile": profile,
                "variants": len(case_rows),
                "optimized_error_mean": float(np.mean(optimized)),
                "naive_error_mean": float(np.mean(naive)),
                "diffusion_error_mean": float(np.mean(diffusion)),
                "optimized_adjacency_error_mean": float(np.mean(optimized_adjacency)),
                "naive_adjacency_error_mean": float(np.mean(naive_adjacency)),
                "diffusion_adjacency_error_mean": float(np.mean(diffusion_adjacency)),
                "optimized_motif_error_mean": float(np.mean(optimized_motif)),
                "naive_motif_error_mean": float(np.mean(naive_motif)),
                "diffusion_motif_error_mean": float(np.mean(diffusion_motif)),
                "optimized_beats_naive_rate": float(np.mean([row["optimized_beats_naive_error"] for row in case_rows])),
                "optimized_beats_diffusion_rate": float(
                    np.mean([row["optimized_beats_diffusion_error"] for row in case_rows])
                ),
                "optimized_beats_naive_adjacency_rate": float(
                    np.mean([row["optimized_beats_naive_adjacency"] for row in case_rows])
                ),
                "optimized_beats_diffusion_adjacency_rate": float(
                    np.mean([row["optimized_beats_diffusion_adjacency"] for row in case_rows])
                ),
                "optimized_beats_naive_motif_rate": float(
                    np.mean([row["optimized_beats_naive_motif"] for row in case_rows])
                ),
                "optimized_beats_diffusion_motif_rate": float(
                    np.mean([row["optimized_beats_diffusion_motif"] for row in case_rows])
                ),
            }
        )
    return summaries
