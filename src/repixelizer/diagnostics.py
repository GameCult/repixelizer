from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .io import nearest_resize, save_rgba
from .metrics import (
    coherence_breakdown,
    foreground_reconstruction_error,
    reconstruction_error,
    source_lattice_consistency_breakdown,
)
from .types import InferenceResult, RunResult


def _to_uint8(rgba: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)


def write_lattice_overlay(path: str | Path, source_rgba: np.ndarray, inference: InferenceResult) -> None:
    image = Image.fromarray(_to_uint8(source_rgba), mode="RGBA").convert("RGBA")
    draw = ImageDraw.Draw(image)
    height, width = source_rgba.shape[:2]
    cell_x = width / inference.target_width
    cell_y = height / inference.target_height
    phase_x = inference.phase_x * cell_x
    phase_y = inference.phase_y * cell_y
    for ix in range(inference.target_width + 1):
        x = ix * cell_x + phase_x
        draw.line((x, 0, x, height), fill=(255, 255, 255, 96), width=1)
    for iy in range(inference.target_height + 1):
        y = iy * cell_y + phase_y
        draw.line((0, y, width, y), fill=(255, 255, 255, 96), width=1)
    image.save(path)


def _make_checker(size: tuple[int, int], tile: int = 8) -> Image.Image:
    width, height = size
    checker = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(checker)
    a = (245, 245, 245, 255)
    b = (220, 220, 220, 255)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            color = a if ((x // tile) + (y // tile)) % 2 == 0 else b
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=color)
    return checker


def write_comparison(path: str | Path, source_rgba: np.ndarray, output_rgba: np.ndarray) -> None:
    height, width = source_rgba.shape[:2]
    preview = nearest_resize(output_rgba, width=width, height=height)
    tiles = [
        Image.fromarray(_to_uint8(source_rgba), mode="RGBA"),
        Image.fromarray(_to_uint8(preview), mode="RGBA"),
        Image.fromarray(_to_uint8(output_rgba), mode="RGBA").resize((width, height), resample=Image.Resampling.NEAREST),
    ]
    labels = ["Source Facsimile", "Output @ Source Size", "Output Grid"]
    canvas = Image.new("RGBA", (width * len(tiles), height + 24), (24, 24, 24, 255))
    for index, tile in enumerate(tiles):
        panel = _make_checker((width, height))
        panel.alpha_composite(tile, (0, 0))
        canvas.alpha_composite(panel, (index * width, 24))
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        draw.text((index * width + 8, 4), label, fill=(255, 255, 255, 255))
    canvas.save(path)


def write_alpha_preview(path: str | Path, source_rgba: np.ndarray, output_rgba: np.ndarray) -> None:
    source_alpha = np.repeat(source_rgba[..., 3:4], 4, axis=-1)
    source_alpha[..., 3] = 1.0
    output_alpha = np.repeat(output_rgba[..., 3:4], 4, axis=-1)
    output_alpha[..., 3] = 1.0
    write_comparison(path, source_alpha, output_alpha)


def write_heatmap(path: str | Path, heatmap: np.ndarray) -> None:
    max_value = float(np.max(heatmap))
    scaled = heatmap / max_value if max_value > 0 else heatmap
    rgba = np.zeros((*scaled.shape, 4), dtype=np.float32)
    rgba[..., 0] = scaled
    rgba[..., 1] = np.sqrt(scaled) * 0.5
    rgba[..., 3] = np.clip(scaled * 1.5, 0.0, 1.0)
    save_rgba(path, rgba)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_compare_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(result: RunResult) -> dict[str, Any]:
    source_preview = nearest_resize(
        result.output_rgba,
        width=result.source_rgba.shape[1],
        height=result.source_rgba.shape[0],
    )
    source_fidelity = {
        "snap_initial": source_lattice_consistency_breakdown(
            result.source_rgba,
            result.solver.initial_rgba,
            target_width=result.inference.target_width,
            target_height=result.inference.target_height,
            phase_x=result.inference.phase_x,
            phase_y=result.inference.phase_y,
        ),
        "solver_target": source_lattice_consistency_breakdown(
            result.source_rgba,
            result.solver.target_rgba,
            target_width=result.inference.target_width,
            target_height=result.inference.target_height,
            phase_x=result.inference.phase_x,
            phase_y=result.inference.phase_y,
        ),
        "final_output": source_lattice_consistency_breakdown(
            result.source_rgba,
            result.output_rgba,
            target_width=result.inference.target_width,
            target_height=result.inference.target_height,
            phase_x=result.inference.phase_x,
            phase_y=result.inference.phase_y,
        ),
    }
    rerank_candidates = [
        {
            "target_width": candidate.target_width,
            "target_height": candidate.target_height,
            "phase_x": candidate.phase_x,
            "phase_y": candidate.phase_y,
            "score": candidate.score,
            "phase_rerank_score": candidate.breakdown.get("phase_rerank_score"),
            "phase_rerank_rank": candidate.breakdown.get("phase_rerank_rank"),
            "phase_rerank_support_score": candidate.breakdown.get("phase_rerank_support_score"),
            "phase_rerank_size_delta_ratio": candidate.breakdown.get("phase_rerank_size_delta_ratio"),
            "phase_rerank_size_penalty": candidate.breakdown.get("phase_rerank_size_penalty"),
        }
        for candidate in result.inference.top_candidates
        if "phase_rerank_score" in candidate.breakdown
    ]
    return {
        "target_width": result.inference.target_width,
        "target_height": result.inference.target_height,
        "phase_x": result.inference.phase_x,
        "phase_y": result.inference.phase_y,
        "confidence": result.inference.confidence,
        "coherence": coherence_breakdown(result.output_rgba),
        "source_preview_reconstruction_error": reconstruction_error(source_preview, result.source_rgba),
        "source_preview_foreground_error": foreground_reconstruction_error(source_preview, result.source_rgba),
        "output_colors_from_source_ratio": _source_color_ratio(result.source_rgba, result.output_rgba),
        "source_fidelity": source_fidelity,
        "phase_rerank_candidates": rerank_candidates,
        "loss_history": result.solver.loss_history,
    }


def _source_color_ratio(source_rgba: np.ndarray, output_rgba: np.ndarray) -> float:
    source8 = _to_uint8(source_rgba).reshape(-1, 4)
    output8 = _to_uint8(output_rgba).reshape(-1, 4)
    source_colors = {tuple(px.tolist()) for px in source8}
    output_colors = {tuple(px.tolist()) for px in output8}
    if not output_colors:
        return 1.0
    return float(sum(color in source_colors for color in output_colors) / len(output_colors))
