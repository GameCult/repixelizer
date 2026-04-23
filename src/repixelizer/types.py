from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class InferenceCandidate:
    target_width: int
    target_height: int
    phase_x: float
    phase_y: float
    score: float
    breakdown: dict[str, float]


@dataclass(slots=True)
class InferenceResult:
    target_width: int
    target_height: int
    phase_x: float
    phase_y: float
    confidence: float
    top_candidates: list[InferenceCandidate] = field(default_factory=list)


@dataclass(slots=True)
class ContinuousSourceAnalysis:
    edge_map: np.ndarray
    cluster_map: np.ndarray


@dataclass(slots=True)
class TileGraphSourceAnalysis:
    edge_map: np.ndarray


SourceAnalysis = ContinuousSourceAnalysis | TileGraphSourceAnalysis


@dataclass(slots=True)
class SourceLatticeReference:
    mean_rgba: np.ndarray
    sharp_rgba: np.ndarray
    dispersion: float
    cell_dispersion: np.ndarray
    cell_support: np.ndarray
    cell_alpha_max: np.ndarray
    sharp_x: np.ndarray
    sharp_y: np.ndarray
    edge_peak_x: np.ndarray
    edge_peak_y: np.ndarray
    edge_strength: np.ndarray
    edge_grad_x: np.ndarray
    edge_grad_y: np.ndarray


@dataclass(slots=True)
class SolverArtifacts:
    target_rgba: np.ndarray
    uv_field: np.ndarray
    guidance_strength: np.ndarray
    initial_rgba: np.ndarray
    loss_history: list[float]


@dataclass(slots=True)
class CleanupArtifacts:
    cleaned_rgba: np.ndarray
    isolated_heatmap: np.ndarray


@dataclass(slots=True)
class PaletteResult:
    rgba: np.ndarray
    palette: list[tuple[int, int, int]]
    indexed_rgba: np.ndarray | None = None
    indexed_png_path: Path | None = None


@dataclass(slots=True)
class RunResult:
    source_rgba: np.ndarray
    output_rgba: np.ndarray
    inference: InferenceResult
    analysis: ContinuousSourceAnalysis | TileGraphSourceAnalysis
    solver: SolverArtifacts
    cleanup: CleanupArtifacts
    palette_result: PaletteResult | None
    diagnostics: dict[str, Any]
