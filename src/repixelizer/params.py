from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SolverHyperParams:
    phase_field_patch_extent: float = 0.18
    phase_field_data_coherence_weight: float = 1.00
    phase_field_data_edge_weight: float = 0.35
    phase_field_smoothness_weight: float = 0.28
    phase_field_edge_gate_strength: float = 6.0
    phase_field_collapse_weight: float = 1.20
    phase_field_min_spacing_ratio: float = 0.18
    phase_field_magnitude_weight: float = 0.08
    phase_field_learning_rate: float = 0.10
    phase_field_max_displacement_ratio: float = 0.48
    phase_rerank_preview_steps: int = 8
    source_edge_detail_threshold: float = 0.25
    alpha_opaque_threshold: float = 0.95
    alpha_transparent_threshold: float = 0.05
    phase_rerank_support_weight: float = 0.45
    phase_rerank_edge_position_weight: float = 0.20
    phase_rerank_wobble_weight: float = 0.20
    phase_rerank_edge_concentration_weight: float = 0.10
    phase_rerank_size_penalty_weight: float = 0.18
    phase_rerank_inference_penalty_weight: float = 0.05
    phase_rerank_confidence_threshold: float = 0.12
    phase_rerank_max_size_delta_ratio: float = 0.40
    phase_rerank_margin: float = 0.004

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
