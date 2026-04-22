from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SolverHyperParams:
    representative_softmax_scale: float = 18.0
    boundary_probe_scale: float = 0.22
    boundary_signed_weight: float = 0.55
    boundary_direction_weight: float = 0.30
    boundary_magnitude_weight: float = 0.15
    snap_base_match_weight: float = 0.65
    snap_neighbor_weight: float = 0.35
    snap_diagonal_weight: float = 0.14
    refine_anchor_weight: float = 0.60
    refine_representative_weight: float = 0.20
    refine_alpha_weight: float = 0.10
    refine_distance_weight: float = 0.10
    refine_source_delta_weight: float = 0.25
    refine_orthogonal_weight: float = 0.24
    refine_diagonal_weight: float = 0.08
    refine_motif_weight: float = 0.12
    refine_line_weight: float = 0.10
    relax_iterations: int = 8
    relax_start_temperature: float = 0.55
    relax_end_temperature: float = 0.10
    relax_damping: float = 0.35
    structure_boundary_weight: float = 0.60
    structure_anchor_adjacency_weight: float = 0.15
    structure_anchor_motif_weight: float = 0.20
    structure_anchor_line_weight: float = 0.10
    structure_representative_weight: float = 0.05
    alpha_foreground_threshold: float = 0.50
    alpha_representative_foreground_threshold: float = 0.60
    alpha_opaque_threshold: float = 0.95
    alpha_transparent_threshold: float = 0.05
    phase_rerank_support_weight: float = 0.45
    phase_rerank_edge_position_weight: float = 0.20
    phase_rerank_wobble_weight: float = 0.20
    phase_rerank_edge_concentration_weight: float = 0.10
    phase_rerank_inference_penalty_weight: float = 0.05
    phase_rerank_confidence_threshold: float = 0.12
    phase_rerank_max_size_delta_ratio: float = 0.15
    phase_rerank_margin: float = 0.004

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
