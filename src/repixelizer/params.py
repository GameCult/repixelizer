from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class SolverHyperParams:
    representative_softmax_scale: float = 18.0
    source_edge_reliability_gain: float = 1.6
    source_edge_reliability_floor: float = 0.25
    source_edge_alpha_floor: float = 0.45
    source_edge_detail_mix: float = 0.65
    source_edge_detail_threshold: float = 0.25
    boundary_probe_scale: float = 0.22
    boundary_signed_weight: float = 0.55
    boundary_direction_weight: float = 0.30
    boundary_magnitude_weight: float = 0.15
    snap_base_match_weight: float = 0.65
    snap_representative_match_weight: float = 0.15
    snap_source_match_weight: float = 0.85
    snap_neighbor_weight: float = 0.35
    snap_diagonal_weight: float = 0.14
    snap_representative_delta_weight: float = 0.20
    snap_source_delta_weight: float = 0.80
    refine_anchor_weight: float = 0.60
    refine_representative_weight: float = 0.20
    refine_representative_match_weight: float = 0.15
    refine_source_match_weight: float = 0.85
    refine_alpha_weight: float = 0.10
    refine_distance_weight: float = 0.10
    refine_source_delta_weight: float = 0.25
    refine_orthogonal_weight: float = 0.24
    refine_diagonal_weight: float = 0.08
    refine_motif_weight: float = 0.12
    refine_line_weight: float = 0.10
    refine_relaxed_mode_weight: float = 0.14
    refine_candidate_extent: float = 0.70
    refine_candidate_levels: int = 7
    guided_candidate_edge_threshold: float = 0.06
    guided_candidate_inner_scale: float = 0.32
    guided_candidate_outer_scale: float = 0.60
    relax_iterations: int = 24
    relax_start_temperature: float = 0.95
    relax_end_temperature: float = 0.18
    relax_damping: float = 0.10
    relax_anchor_scale: float = 0.50
    relax_orthogonal_weight: float = 0.34
    relax_diagonal_weight: float = 0.12
    relax_source_adjacency_weight: float = 0.35
    relax_motif_weight: float = 0.22
    relax_line_weight: float = 0.06
    relax_source_motif_weight: float = 0.30
    relax_source_line_weight: float = 0.12
    relax_handoff_weight: float = 0.18
    structure_boundary_weight: float = 0.60
    structure_anchor_adjacency_weight: float = 0.15
    structure_anchor_motif_weight: float = 0.20
    structure_anchor_line_weight: float = 0.10
    structure_source_adjacency_weight: float = 0.35
    structure_source_motif_weight: float = 0.40
    structure_source_line_weight: float = 0.20
    structure_representative_weight: float = 0.02
    alpha_foreground_threshold: float = 0.50
    alpha_representative_foreground_threshold: float = 0.60
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
    tile_graph_max_candidates: int = 768
    tile_graph_max_candidates_per_coord: int = 2
    tile_graph_edge_candidates_per_coord: int = 6
    tile_graph_component_color_threshold: float = 0.055
    tile_graph_component_alpha_threshold: float = 0.12
    tile_graph_component_min_coverage: float = 0.02
    tile_graph_nonedge_sharp_weight: float = 0.85
    tile_graph_nonedge_mean_weight: float = 0.15
    tile_graph_edge_mean_weight: float = 0.03
    tile_graph_area_weight: float = 0.03
    tile_graph_alpha_weight: float = 0.25
    tile_graph_coverage_weight: float = 0.05
    tile_graph_delta_weight: float = 0.45
    tile_graph_iterations: int = 10

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
