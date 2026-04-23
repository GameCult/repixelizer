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
    tile_graph_max_candidates: int = 768
    tile_graph_max_candidates_per_coord: int = 2
    tile_graph_edge_candidates_per_coord: int = 6
    tile_graph_component_color_threshold: float = 0.055
    tile_graph_component_alpha_threshold: float = 0.12
    tile_graph_source_region_min_area_ratio: float = 0.06
    tile_graph_source_region_window_coverage: float = 0.12
    tile_graph_area_weight: float = 0.03
    tile_graph_alpha_weight: float = 0.25
    tile_graph_coverage_weight: float = 0.05
    tile_graph_edge_peak_weight: float = 0.10
    tile_graph_adjacency_weight: float = 0.45
    tile_graph_iterations: int = 10

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
