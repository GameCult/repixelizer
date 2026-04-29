from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(slots=True)
class SolverHyperParams:
    phase_field_grid_alignment_weight: float = 0.90
    phase_field_local_search_radius_ratio: float = 0.80
    phase_field_local_search_blend: float = 0.36
    phase_field_local_search_move_weight: float = 0.04
    phase_field_signal_weight: float = 1.00
    phase_field_signal_pyramid_levels: int = 5
    phase_field_signal_level_decay: float = 0.50
    phase_field_signal_gradient_weight: float = 0.25
    phase_field_signal_curvature_weight: float = 0.65
    phase_field_signal_energy_power: float = 1.00
    phase_field_signal_peak_power: float = 2.00
    phase_field_local_search_grid_weight: float = 0.0
    phase_field_local_search_interval: int = 1
    candidate_rerank_preview_steps: int = 8
    candidate_rerank_support_weight: float = 0.45
    candidate_rerank_edge_position_weight: float = 0.20
    candidate_rerank_wobble_weight: float = 0.20
    candidate_rerank_edge_concentration_weight: float = 0.10
    candidate_rerank_size_penalty_weight: float = 0.18
    candidate_rerank_inference_penalty_weight: float = 0.05
    candidate_rerank_confidence_threshold: float = 0.12
    candidate_rerank_max_size_delta_ratio: float = 0.40
    candidate_rerank_margin: float = 0.004

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def default_solver_params_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "solver_params.json"


def solver_params_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get("REPIXELIZER_SOLVER_PARAMS")
    if env_path:
        return Path(env_path)
    return default_solver_params_config_path()


def load_solver_params(path: str | Path | None = None) -> SolverHyperParams:
    config_path = solver_params_config_path(path)
    params = SolverHyperParams()
    if not config_path.exists():
        return params

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Solver params config must contain a JSON object: {config_path}")

    allowed = {field.name: field for field in fields(SolverHyperParams)}
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown solver params in {config_path}: {names}")

    values = params.to_dict()
    for key, value in raw.items():
        if isinstance(values[key], int):
            values[key] = int(value)
        else:
            values[key] = float(value)
    return SolverHyperParams(**values)
