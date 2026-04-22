from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repixelizer.analysis import analyze_source
from repixelizer.baselines import naive_resize_baseline
from repixelizer.io import save_rgba
from repixelizer.metrics import source_lattice_consistency_breakdown
from repixelizer.params import SolverHyperParams
from repixelizer.pipeline import run_pipeline
from repixelizer.synthetic import fake_pixelize, make_emblem
from repixelizer.tile_graph import build_tile_graph_model, optimize_tile_graph
from repixelizer.types import InferenceResult


def test_tile_graph_model_extracts_candidates_and_adjacency_from_lattice_proposals() -> None:
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

    model = build_tile_graph_model(source, inference=inference, analysis=analyze_source(source, seed=7), device="cpu")

    assert model.candidate_rgba.shape[0] > 4
    choice_counts = np.diff(model.cell_candidate_offsets)
    assert choice_counts.shape[0] == inference.target_width * inference.target_height
    assert np.all(choice_counts >= 1)
    assert model.average_choices >= 1.0
    assert model.edge_density > 0.0
    assert model.model_device == "cpu"


def test_tile_graph_candidates_use_literal_source_pixel_colors() -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    source[..., 3] = 1.0
    palette = (
        np.asarray([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
        np.asarray([0.1, 1.0, 0.1, 1.0], dtype=np.float32),
        np.asarray([0.1, 0.1, 1.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 0.1, 1.0], dtype=np.float32),
    )
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            source[y, x] = palette[(2 * (y % 2)) + (x % 2)]

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
        analysis=analyze_source(source, seed=13),
        solver_params=SolverHyperParams(tile_graph_max_candidates=32, tile_graph_max_candidates_per_coord=2),
    )

    source_colors = {tuple(np.round(color, 4)) for color in source.reshape(-1, 4)}
    candidate_colors = {tuple(np.round(color, 4)) for color in model.candidate_rgba}

    assert candidate_colors.issubset(source_colors | {tuple(np.zeros(4, dtype=np.float32))})


def test_tile_graph_candidates_are_scoped_to_their_output_coord() -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    source[..., 3] = 1.0
    source[:, 2:4] = np.asarray([1.0, 0.9, 0.2, 1.0], dtype=np.float32)
    source[2:6, 5:7] = np.asarray([0.2, 0.8, 1.0, 1.0], dtype=np.float32)
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
        analysis=analyze_source(source, seed=11),
    )

    for y in range(inference.target_height):
        for x in range(inference.target_width):
            flat = y * inference.target_width + x
            start = int(model.cell_candidate_offsets[flat])
            end = int(model.cell_candidate_offsets[flat + 1])
            coords = model.candidate_coords[model.cell_candidate_indices[start:end]]
            assert np.all(coords[:, 0] == y)
            assert np.all(coords[:, 1] == x)


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


def test_tile_graph_preserves_transparent_background_on_sparse_detail() -> None:
    source = np.zeros((12, 12, 4), dtype=np.float32)
    source[5:7, 5:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    fake = fake_pixelize(
        source,
        upscale=8,
        phase_x=0.12,
        phase_y=-0.08,
        blur_radius=0.35,
        warp_strength=0.1,
        warp_detail=4,
        seed=5,
    )
    artifacts, _ = optimize_tile_graph(
        fake,
        inference=InferenceResult(
            target_width=12,
            target_height=12,
            phase_x=0.12,
            phase_y=-0.08,
            confidence=1.0,
            top_candidates=[],
        ),
        analysis=analyze_source(fake, seed=7),
        steps=0,
        seed=7,
        device="cpu",
    )

    alpha_zero_ratio = float((artifacts.target_rgba[..., 3] <= 0.05).mean())

    assert alpha_zero_ratio >= 0.75


def test_tile_graph_cuda_matches_cpu_on_small_case() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    source = make_emblem(16, 16)
    fake = fake_pixelize(source, upscale=8, phase_x=0.1, phase_y=-0.05, blur_radius=0.4, warp_strength=0.15, warp_detail=4)
    inference = InferenceResult(
        target_width=16,
        target_height=16,
        phase_x=0.1,
        phase_y=-0.05,
        confidence=1.0,
        top_candidates=[],
    )
    cpu_analysis = analyze_source(fake, seed=7, device="cpu")
    cuda_analysis = analyze_source(fake, seed=7, device="cuda")

    cpu_artifacts, cpu_diag = optimize_tile_graph(
        fake,
        inference=inference,
        analysis=cpu_analysis,
        steps=0,
        seed=7,
        device="cpu",
    )
    cuda_artifacts, cuda_diag = optimize_tile_graph(
        fake,
        inference=inference,
        analysis=cuda_analysis,
        steps=0,
        seed=7,
        device="cuda",
    )

    assert cpu_diag["tile_graph_model_device"] == "cpu"
    assert cpu_diag["tile_graph_solver_device"] == "cpu"
    assert cuda_diag["tile_graph_model_device"] == "cuda"
    assert cuda_diag["tile_graph_solver_device"] == "cuda"
    assert np.allclose(cpu_artifacts.target_rgba, cuda_artifacts.target_rgba, atol=1e-5)


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
        device="cpu",
    )

    import json

    run_json = json.loads((diagnostics_dir / "run.json").read_text(encoding="utf-8"))
    assert output_path.exists()
    assert run_json["settings"]["reconstruction_mode"] == "tile-graph"
    assert run_json["reconstruction"]["mode"] == "tile-graph"
    assert run_json["reconstruction"]["tile_graph_model_device"] == "cpu"
    assert run_json["reconstruction"]["tile_graph_candidate_count"] > 0
    assert set(run_json["source_fidelity"].keys()) == {"snap_initial", "solver_target", "final_output"}
    assert result.diagnostics["reconstruction"]["mode"] == "tile-graph"
