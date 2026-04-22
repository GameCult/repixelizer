from __future__ import annotations

from pathlib import Path

import numpy as np

from repixelizer.analysis import analyze_source
from repixelizer.baselines import naive_resize_baseline
from repixelizer.io import save_rgba
from repixelizer.metrics import source_lattice_consistency_breakdown
from repixelizer.pipeline import run_pipeline
from repixelizer.synthetic import fake_pixelize, make_emblem
from repixelizer.tile_graph import build_tile_graph_model, optimize_tile_graph
from repixelizer.types import InferenceResult


def test_tile_graph_model_extracts_candidates_and_adjacency_from_component_walk() -> None:
    lowres = np.zeros((4, 4, 4), dtype=np.float32)
    lowres[..., 3] = 1.0
    lowres[:, 1] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    source = np.repeat(np.repeat(lowres, 8, axis=0), 8, axis=1)
    inference = InferenceResult(
        target_width=4,
        target_height=4,
        phase_x=0.0,
        phase_y=0.0,
        confidence=1.0,
        top_candidates=[],
    )

    model = build_tile_graph_model(
        source,
        inference=inference,
        analysis=analyze_source(source, seed=7),
    )

    assert model.candidate_rgba.shape[0] > 4
    assert model.pair_penalty.shape == (4, model.candidate_rgba.shape[0], model.candidate_rgba.shape[0])
    assert model.edge_density > 0.0


def test_tile_graph_reconstructs_synthetic_thin_feature_better_than_naive() -> None:
    source = np.zeros((16, 16, 4), dtype=np.float32)
    source[4:12, 8] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    source[4, 7:10] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    source[11, 7] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    source[10, 6] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
    source[9, 5] = np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)

    fake = fake_pixelize(
        source,
        upscale=9,
        phase_x=0.18,
        phase_y=-0.12,
        blur_radius=0.45,
        warp_strength=0.2,
        warp_detail=5,
        seed=11,
    )
    inference = InferenceResult(
        target_width=16,
        target_height=16,
        phase_x=0.18,
        phase_y=-0.12,
        confidence=1.0,
        top_candidates=[],
    )
    artifacts, _ = optimize_tile_graph(
        fake,
        inference=inference,
        analysis=analyze_source(fake, seed=7),
        steps=0,
        seed=7,
        device="cpu",
    )
    naive = naive_resize_baseline(fake, width=16, height=16)

    tile_score = source_lattice_consistency_breakdown(
        fake,
        artifacts.target_rgba,
        target_width=16,
        target_height=16,
        phase_x=0.18,
        phase_y=-0.12,
    )["score"]
    naive_score = source_lattice_consistency_breakdown(
        fake,
        naive,
        target_width=16,
        target_height=16,
        phase_x=0.18,
        phase_y=-0.12,
    )["score"]

    assert tile_score <= naive_score


def test_pipeline_tile_graph_mode_writes_reconstruction_diagnostics(tmp_path: Path) -> None:
    source = make_emblem(24, 24)
    fake = fake_pixelize(source, upscale=10, phase_x=0.15, phase_y=0.2, blur_radius=0.6, warp_strength=0.2, warp_detail=5)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "nested" / "output.png"
    diagnostics_dir = tmp_path / "diagnostics"

    save_rgba(input_path, fake)
    result = run_pipeline(
        input_path,
        output_path,
        diagnostics_dir=diagnostics_dir,
        steps=24,
        reconstruction_mode="tile-graph",
    )

    import json

    run_json = json.loads((diagnostics_dir / "run.json").read_text(encoding="utf-8"))
    assert output_path.exists()
    assert run_json["settings"]["reconstruction_mode"] == "tile-graph"
    assert run_json["reconstruction"]["mode"] == "tile-graph"
    assert run_json["reconstruction"]["tile_graph_candidate_count"] > 0
    assert set(run_json["source_fidelity"].keys()) == {"snap_initial", "solver_target", "final_output"}
    assert result.diagnostics["reconstruction"]["mode"] == "tile-graph"
