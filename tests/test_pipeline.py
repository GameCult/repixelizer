from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from repixelizer.analysis import analyze_continuous_source
from repixelizer.baselines import naive_resize_baseline
from repixelizer.continuous import optimize_uv_field
from repixelizer.metrics import coherence_breakdown, foreground_reconstruction_error, source_lattice_consistency_breakdown
from repixelizer.params import SolverHyperParams
from repixelizer.pipeline import _run_reconstruction, _select_phase_candidate, run_pipeline
from repixelizer.io import load_rgba, nearest_resize
from repixelizer.synthetic import fake_pixelize, make_emblem, make_sprite
from repixelizer.types import InferenceCandidate, InferenceResult, SolverArtifacts


def test_pipeline_writes_output_and_diagnostics(tmp_path: Path) -> None:
    source = make_emblem(32, 32)
    fake = fake_pixelize(source, upscale=10, phase_x=0.15, phase_y=0.25, blur_radius=0.5)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    diagnostics_dir = tmp_path / "diagnostics"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(input_path, output_path, diagnostics_dir=diagnostics_dir, steps=24)
    assert output_path.exists()
    assert (diagnostics_dir / "run.json").exists()
    assert (diagnostics_dir / "output-preview.png").exists()
    import json

    run_json = json.loads((diagnostics_dir / "run.json").read_text(encoding="utf-8"))
    assert set(run_json["source_fidelity"].keys()) == {"snap_initial", "solver_target", "final_output"}
    assert "phase_rerank_candidates" in run_json
    assert result.output_rgba.shape[0] == result.inference.target_height
    assert result.output_rgba.shape[1] == result.inference.target_width


def test_pipeline_beats_naive_on_roundtrip_fidelity_for_synthetic_emblem(tmp_path: Path) -> None:
    source = make_emblem(32, 32)
    fake = fake_pixelize(
        source,
        upscale=12,
        phase_x=0.2,
        phase_y=0.35,
        blur_radius=0.65,
        warp_strength=0.28,
        warp_detail=6,
        seed=5,
    )
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(input_path, output_path, steps=32)
    naive = naive_resize_baseline(fake, width=result.inference.target_width, height=result.inference.target_height)
    optimized_preview = nearest_resize(result.output_rgba, width=source.shape[1], height=source.shape[0])
    naive_preview = nearest_resize(naive, width=source.shape[1], height=source.shape[0])
    optimized_error = foreground_reconstruction_error(optimized_preview, source)
    naive_error = foreground_reconstruction_error(naive_preview, source)
    assert optimized_error <= naive_error


def test_pipeline_preserves_transparency_for_sprite(tmp_path: Path) -> None:
    source = make_sprite(24, 24)
    fake = fake_pixelize(source, upscale=9, phase_x=0.1, phase_y=0.22, blur_radius=0.45)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(input_path, output_path, steps=20)
    alpha = result.output_rgba[..., 3]
    assert alpha.min() < 0.05
    assert alpha.max() > 0.9
    partial = alpha[(alpha > 0.05) & (alpha < 0.95)]
    assert partial.size == 0


def test_pipeline_can_strip_checkerboard_background(tmp_path: Path) -> None:
    source = make_sprite(24, 24)
    checker = np.ones_like(source)
    checker[..., 3] = 1.0
    for y in range(checker.shape[0]):
        for x in range(checker.shape[1]):
            tile = 0.97 if ((x // 4) + (y // 4)) % 2 == 0 else 0.91
            checker[y, x, :3] = tile
    alpha = source[..., 3:4]
    opaque_input = checker.copy()
    opaque_input[..., :3] = source[..., :3] * alpha + checker[..., :3] * (1.0 - alpha)

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, opaque_input)
    result = run_pipeline(input_path, output_path, target_size=24, steps=0, strip_background=True)
    assert result.source_rgba[..., 3].min() < 0.05
    assert result.output_rgba[..., 3].min() < 0.05
    assert result.output_rgba[..., 3].max() > 0.9


def test_pipeline_can_emit_initialized_output_without_optimizer_steps(tmp_path: Path) -> None:
    source = make_emblem(24, 24)
    fake = fake_pixelize(source, upscale=10, phase_x=0.15, phase_y=0.2, blur_radius=0.6, warp_strength=0.2, warp_detail=5)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(input_path, output_path, steps=0)
    assert output_path.exists()
    assert result.output_rgba.shape[0] == result.inference.target_height
    assert result.output_rgba.shape[1] == result.inference.target_width


def test_fixed_tile_graph_pipeline_output_matches_refactor_baseline(tmp_path: Path) -> None:
    source = make_emblem(24, 24)
    fake = fake_pixelize(
        source,
        upscale=8,
        phase_x=0.1,
        phase_y=-0.05,
        blur_radius=0.4,
        warp_strength=0.15,
        warp_detail=4,
        seed=9,
    )
    input_path = tmp_path / "tile-baseline-input.png"
    output_path = tmp_path / "tile-baseline-output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(
        input_path,
        output_path,
        reconstruction_mode="tile-graph",
        target_width=24,
        target_height=24,
        phase_x=0.1,
        phase_y=-0.05,
        enable_phase_rerank=False,
        steps=0,
        device="cpu",
    )

    output_hash = hashlib.sha256(np.ascontiguousarray(result.output_rgba).view(np.uint8)).hexdigest()
    assert output_hash == "1f6d497a65953ab60a37fce98e69b101955e5f0e5a06a1e32167c6b8582a23f6"


def test_fixed_continuous_pipeline_output_matches_refactor_baseline(tmp_path: Path) -> None:
    source = make_emblem(24, 24)
    fake = fake_pixelize(
        source,
        upscale=8,
        phase_x=0.1,
        phase_y=-0.05,
        blur_radius=0.4,
        warp_strength=0.15,
        warp_detail=4,
        seed=9,
    )
    input_path = tmp_path / "continuous-baseline-input.png"
    output_path = tmp_path / "continuous-baseline-output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, fake)
    result = run_pipeline(
        input_path,
        output_path,
        reconstruction_mode="continuous",
        target_width=24,
        target_height=24,
        phase_x=0.1,
        phase_y=-0.05,
        enable_phase_rerank=False,
        steps=0,
        device="cpu",
    )

    output_hash = hashlib.sha256(np.ascontiguousarray(result.output_rgba).view(np.uint8)).hexdigest()
    assert output_hash == "404748af3c7241a6602cb91fdfed1fd7e116dd14098827e33e78c7da5e3d7903"


def test_phase_rerank_can_override_low_confidence_inference_pick(monkeypatch) -> None:
    source = np.zeros((2, 2, 4), dtype=np.float32)
    source[0, 0] = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    candidate_a = InferenceCandidate(target_width=2, target_height=2, phase_x=-0.4, phase_y=0.2, score=0.91, breakdown={})
    candidate_b = InferenceCandidate(target_width=2, target_height=2, phase_x=0.4, phase_y=0.2, score=0.909, breakdown={})
    inference = InferenceResult(
        target_width=2,
        target_height=2,
        phase_x=candidate_a.phase_x,
        phase_y=candidate_a.phase_y,
        confidence=0.1,
        top_candidates=[candidate_a, candidate_b],
    )
    outputs = {
        (candidate_a.phase_x, candidate_a.phase_y): np.ones_like(source),
        (candidate_b.phase_x, candidate_b.phase_y): source.copy(),
    }

    class DummyArtifacts:
        def __init__(self, rgba: np.ndarray) -> None:
            self.target_rgba = rgba

    def fake_optimize_uv_field(source_rgba, inference, analysis, steps, seed, device, solver_params=None):
        assert steps == 0
        return DummyArtifacts(outputs[(inference.phase_x, inference.phase_y)])

    monkeypatch.setattr("repixelizer.pipeline.optimize_uv_field", fake_optimize_uv_field)

    selected = _select_phase_candidate(source, inference, analysis=object(), seed=7, device="cpu")
    assert selected.phase_x == candidate_b.phase_x
    assert selected.phase_y == candidate_b.phase_y


def test_phase_rerank_can_override_to_better_size_candidate(monkeypatch) -> None:
    source = np.zeros((16, 16, 4), dtype=np.float32)
    candidate_a = InferenceCandidate(target_width=16, target_height=16, phase_x=0.0, phase_y=0.0, score=0.91, breakdown={})
    candidate_b = InferenceCandidate(target_width=18, target_height=18, phase_x=0.0, phase_y=0.0, score=0.88, breakdown={})
    inference = InferenceResult(
        target_width=16,
        target_height=16,
        phase_x=0.0,
        phase_y=0.0,
        confidence=0.0,
        top_candidates=[candidate_a, candidate_b],
    )

    class DummyArtifacts:
        def __init__(self, rgba: np.ndarray) -> None:
            self.target_rgba = rgba

    def fake_optimize_uv_field(source_rgba, inference, analysis, steps, seed, device, solver_params=None):
        rgba = np.zeros((inference.target_height, inference.target_width, 4), dtype=np.float32)
        return DummyArtifacts(rgba)

    def fake_support(source_rgba, output_rgba, *, target_width, target_height, phase_x, phase_y):
        if target_width == 16:
            return {"score": 0.10}
        return {"score": 0.01}

    monkeypatch.setattr("repixelizer.pipeline.optimize_uv_field", fake_optimize_uv_field)
    monkeypatch.setattr("repixelizer.pipeline.source_lattice_consistency_breakdown", fake_support)

    selected = _select_phase_candidate(source, inference, analysis=object(), seed=7, device="cpu")
    assert selected.target_width == candidate_b.target_width
    assert selected.target_height == candidate_b.target_height


def test_phase_rerank_rejects_large_size_jump(monkeypatch) -> None:
    source = np.zeros((16, 16, 4), dtype=np.float32)
    candidate_a = InferenceCandidate(target_width=16, target_height=16, phase_x=0.0, phase_y=0.0, score=0.91, breakdown={})
    candidate_b = InferenceCandidate(target_width=24, target_height=24, phase_x=0.0, phase_y=0.0, score=0.88, breakdown={})
    inference = InferenceResult(
        target_width=16,
        target_height=16,
        phase_x=0.0,
        phase_y=0.0,
        confidence=0.0,
        top_candidates=[candidate_a, candidate_b],
    )

    class DummyArtifacts:
        def __init__(self, rgba: np.ndarray) -> None:
            self.target_rgba = rgba

    def fake_optimize_uv_field(source_rgba, inference, analysis, steps, seed, device, solver_params=None):
        rgba = np.zeros((inference.target_height, inference.target_width, 4), dtype=np.float32)
        return DummyArtifacts(rgba)

    def fake_support(source_rgba, output_rgba, *, target_width, target_height, phase_x, phase_y):
        if target_width == 16:
            return {"score": 0.10}
        return {"score": 0.01}

    monkeypatch.setattr("repixelizer.pipeline.optimize_uv_field", fake_optimize_uv_field)
    monkeypatch.setattr("repixelizer.pipeline.source_lattice_consistency_breakdown", fake_support)

    selected = _select_phase_candidate(source, inference, analysis=object(), seed=7, device="cpu")
    assert selected.target_width == candidate_a.target_width
    assert selected.target_height == candidate_a.target_height


def test_select_phase_candidate_can_skip_phase_rerank(monkeypatch) -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    candidate_a = InferenceCandidate(target_width=8, target_height=8, phase_x=0.0, phase_y=0.0, score=0.9, breakdown={})
    candidate_b = InferenceCandidate(target_width=10, target_height=10, phase_x=0.2, phase_y=0.2, score=0.89, breakdown={})
    inference = InferenceResult(
        target_width=8,
        target_height=8,
        phase_x=0.0,
        phase_y=0.0,
        confidence=0.0,
        top_candidates=[candidate_a, candidate_b],
    )

    def fail(*args, **kwargs):
        raise AssertionError("phase rerank probe should not run when disabled")

    monkeypatch.setattr("repixelizer.pipeline.optimize_uv_field", fail)
    selected = _select_phase_candidate(
        source,
        inference,
        analysis=object(),
        seed=7,
        device="cpu",
        enable_phase_rerank=False,
    )
    assert selected.target_width == candidate_a.target_width
    assert selected.target_height == candidate_a.target_height


def test_select_phase_candidate_skips_phase_rerank_for_tile_graph(monkeypatch) -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    candidate_a = InferenceCandidate(target_width=8, target_height=8, phase_x=0.0, phase_y=0.0, score=0.9, breakdown={})
    candidate_b = InferenceCandidate(target_width=10, target_height=10, phase_x=0.2, phase_y=0.2, score=0.89, breakdown={})
    inference = InferenceResult(
        target_width=8,
        target_height=8,
        phase_x=0.0,
        phase_y=0.0,
        confidence=0.0,
        top_candidates=[candidate_a, candidate_b],
    )

    def fail(*args, **kwargs):
        raise AssertionError("tile-graph phase rerank probe should not run")

    monkeypatch.setattr("repixelizer.pipeline.optimize_tile_graph", fail)
    selected = _select_phase_candidate(
        source,
        inference,
        analysis=object(),
        seed=7,
        device="cpu",
        reconstruction_mode="tile-graph",
    )
    assert selected.target_width == candidate_a.target_width
    assert selected.target_height == candidate_a.target_height


def test_pipeline_runs_tile_graph_once_without_phase_probe_reuse(tmp_path: Path, monkeypatch) -> None:
    source = np.zeros((6, 6, 4), dtype=np.float32)
    source[..., 3] = 1.0
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, source)

    candidate_a = InferenceCandidate(target_width=3, target_height=3, phase_x=0.0, phase_y=0.0, score=0.91, breakdown={})
    candidate_b = InferenceCandidate(target_width=4, target_height=4, phase_x=0.2, phase_y=-0.2, score=0.90, breakdown={})
    inference = InferenceResult(
        target_width=3,
        target_height=3,
        phase_x=0.0,
        phase_y=0.0,
        confidence=0.0,
        top_candidates=[candidate_a, candidate_b],
    )

    class DummyAnalysis:
        edge_map = np.zeros((6, 6), dtype=np.float32)

    call_count = {"tile_graph": 0}

    def fake_optimize_tile_graph(
        source_rgba,
        inference,
        analysis,
        steps,
        seed,
        device,
        solver_params=None,
    ):
        del source_rgba, analysis, steps, seed, device, solver_params
        call_count["tile_graph"] += 1
        rgba = np.zeros((inference.target_height, inference.target_width, 4), dtype=np.float32)
        rgba[..., 3] = 1.0
        rgba[..., 0] = 0.2 if inference.target_width == 3 else 0.8
        return (
            SolverArtifacts(
                target_rgba=rgba,
                uv_field=np.zeros((inference.target_height, inference.target_width, 2), dtype=np.float32),
                guidance_strength=np.zeros((inference.target_height, inference.target_width), dtype=np.float32),
                initial_rgba=rgba.copy(),
                loss_history=[],
            ),
            {"mode": "tile-graph", "tile_graph_candidate_count": 1},
        )

    def fake_support(source_rgba, output_rgba, *, target_width, target_height, phase_x, phase_y):
        del source_rgba, output_rgba, target_height, phase_x, phase_y
        return {"score": 0.05 if target_width == 3 else 0.15}

    monkeypatch.setattr("repixelizer.pipeline.infer_lattice", lambda *args, **kwargs: inference)
    monkeypatch.setattr("repixelizer.pipeline.analyze_tile_graph_source", lambda *args, **kwargs: DummyAnalysis())
    monkeypatch.setattr("repixelizer.pipeline.optimize_tile_graph", fake_optimize_tile_graph)
    monkeypatch.setattr("repixelizer.pipeline.source_lattice_consistency_breakdown", fake_support)
    monkeypatch.setattr("repixelizer.pipeline.foreground_edge_position_error", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr("repixelizer.pipeline.foreground_stroke_wobble_error", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr("repixelizer.pipeline.foreground_edge_concentration", lambda *args, **kwargs: 0.2)

    result = run_pipeline(
        input_path,
        output_path,
        reconstruction_mode="tile-graph",
        steps=200,
        device="cpu",
    )

    assert output_path.exists()
    assert call_count["tile_graph"] == 1
    assert "reused_phase_probe_reconstruction" not in result.diagnostics["reconstruction"]
    assert result.output_rgba.shape[:2] == (3, 3)


def test_pipeline_fixed_target_and_phase_skip_search_inference(tmp_path: Path, monkeypatch) -> None:
    source = np.zeros((6, 6, 4), dtype=np.float32)
    source[..., 3] = 1.0
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"

    from repixelizer.io import save_rgba

    save_rgba(input_path, source)

    inference = InferenceResult(
        target_width=3,
        target_height=3,
        phase_x=0.2,
        phase_y=-0.2,
        confidence=1.0,
        top_candidates=[],
    )

    class DummyAnalysis:
        edge_map = np.zeros((6, 6), dtype=np.float32)
        cluster_map = np.zeros((6, 6), dtype=np.int32)

    def fail_infer(*args, **kwargs):
        raise AssertionError("full lattice search should not run for fixed target/phase")

    def fake_infer_fixed(*args, **kwargs):
        return inference

    def fake_optimize_uv_field(source_rgba, inference, analysis, steps, seed, device, solver_params=None):
        rgba = np.zeros((inference.target_height, inference.target_width, 4), dtype=np.float32)
        rgba[..., 3] = 1.0
        return SolverArtifacts(
            target_rgba=rgba,
            uv_field=np.zeros((inference.target_height, inference.target_width, 2), dtype=np.float32),
            guidance_strength=np.zeros((inference.target_height, inference.target_width), dtype=np.float32),
            initial_rgba=rgba.copy(),
            loss_history=[],
        )

    monkeypatch.setattr("repixelizer.pipeline.infer_lattice", fail_infer)
    monkeypatch.setattr("repixelizer.pipeline.infer_fixed_lattice", fake_infer_fixed)
    monkeypatch.setattr("repixelizer.pipeline.analyze_continuous_source", lambda *args, **kwargs: DummyAnalysis())
    monkeypatch.setattr("repixelizer.pipeline.optimize_uv_field", fake_optimize_uv_field)

    result = run_pipeline(
        input_path,
        output_path,
        target_width=3,
        target_height=3,
        phase_x=0.2,
        phase_y=-0.2,
        enable_phase_rerank=False,
        steps=0,
        device="cpu",
    )

    assert output_path.exists()
    assert result.inference.target_width == 3
    assert result.inference.target_height == 3
    assert result.inference.phase_x == 0.2
    assert result.inference.phase_y == -0.2


def test_phase_rerank_can_accept_low_confidence_size_jump_with_strong_support(monkeypatch) -> None:
    source = np.zeros((16, 16, 4), dtype=np.float32)
    candidate_a = InferenceCandidate(target_width=16, target_height=16, phase_x=0.0, phase_y=0.0, score=0.91, breakdown={})
    candidate_b = InferenceCandidate(target_width=21, target_height=21, phase_x=0.2, phase_y=-0.2, score=0.89, breakdown={})
    inference = InferenceResult(
        target_width=16,
        target_height=16,
        phase_x=0.0,
        phase_y=0.0,
        confidence=0.0,
        top_candidates=[candidate_a, candidate_b],
    )

    class DummyArtifacts:
        def __init__(self, rgba: np.ndarray) -> None:
            self.target_rgba = rgba

    outputs = {
        candidate_a.target_width: np.zeros((candidate_a.target_height, candidate_a.target_width, 4), dtype=np.float32),
        candidate_b.target_width: np.ones((candidate_b.target_height, candidate_b.target_width, 4), dtype=np.float32),
    }

    def fake_optimize_uv_field(source_rgba, inference, analysis, steps, seed, device, solver_params=None):
        return DummyArtifacts(outputs[inference.target_width])

    def fake_support(source_rgba, output_rgba, *, target_width, target_height, phase_x, phase_y):
        return {"score": 0.22 if target_width == 16 else 0.03}

    def fake_edge_position(preview, source_rgba):
        return 0.07 if np.mean(preview) < 0.1 else 0.02

    def fake_wobble(preview, source_rgba):
        return 1.25 if np.mean(preview) < 0.1 else 0.55

    def fake_concentration(rgba):
        return 0.18 if np.mean(rgba) < 0.1 else 0.31

    monkeypatch.setattr("repixelizer.pipeline.optimize_uv_field", fake_optimize_uv_field)
    monkeypatch.setattr("repixelizer.pipeline.source_lattice_consistency_breakdown", fake_support)
    monkeypatch.setattr("repixelizer.pipeline.foreground_edge_position_error", fake_edge_position)
    monkeypatch.setattr("repixelizer.pipeline.foreground_stroke_wobble_error", fake_wobble)
    monkeypatch.setattr("repixelizer.pipeline.foreground_edge_concentration", fake_concentration)

    selected = _select_phase_candidate(source, inference, analysis=object(), seed=7, device="cpu")
    assert selected.target_width == candidate_b.target_width
    assert selected.target_height == candidate_b.target_height


def test_phase_rerank_keeps_high_confidence_candidate_without_probe(monkeypatch) -> None:
    source = np.zeros((8, 8, 4), dtype=np.float32)
    candidate_a = InferenceCandidate(target_width=8, target_height=8, phase_x=0.0, phase_y=0.0, score=0.9, breakdown={})
    candidate_b = InferenceCandidate(target_width=10, target_height=10, phase_x=0.2, phase_y=0.2, score=0.89, breakdown={})
    inference = InferenceResult(
        target_width=8,
        target_height=8,
        phase_x=0.0,
        phase_y=0.0,
        confidence=0.5,
        top_candidates=[candidate_a, candidate_b],
    )

    def fail(*args, **kwargs):
        raise AssertionError("rerank probe should not run for high-confidence inference")

    monkeypatch.setattr("repixelizer.pipeline.optimize_uv_field", fail)
    selected = _select_phase_candidate(source, inference, analysis=object(), seed=7, device="cpu")
    assert selected.target_width == candidate_a.target_width
    assert selected.target_height == candidate_a.target_height


def test_phase_rerank_can_prefer_better_line_metrics(monkeypatch) -> None:
    source = np.zeros((4, 4, 4), dtype=np.float32)
    candidate_a = InferenceCandidate(target_width=2, target_height=2, phase_x=-0.2, phase_y=0.0, score=0.91, breakdown={})
    candidate_b = InferenceCandidate(target_width=2, target_height=2, phase_x=0.2, phase_y=0.0, score=0.909, breakdown={})
    inference = InferenceResult(
        target_width=2,
        target_height=2,
        phase_x=candidate_a.phase_x,
        phase_y=candidate_a.phase_y,
        confidence=0.0,
        top_candidates=[candidate_a, candidate_b],
    )

    class DummyArtifacts:
        def __init__(self, rgba: np.ndarray) -> None:
            self.target_rgba = rgba

    def fake_support(source_rgba, output_rgba, *, target_width, target_height, phase_x, phase_y):
        return {"score": 0.02}

    def fake_edge_position(preview, source_rgba):
        return 0.05

    def fake_wobble(preview, source_rgba):
        return 1.2 if np.mean(preview) < 0.01 else 0.8

    def fake_concentration(rgba):
        return 0.15 if np.mean(rgba) < 0.01 else 0.28

    outputs = {
        candidate_a.phase_x: np.zeros((2, 2, 4), dtype=np.float32),
        candidate_b.phase_x: np.ones((2, 2, 4), dtype=np.float32),
    }

    def fake_optimize_uv_field(source_rgba, inference, analysis, steps, seed, device, solver_params=None):
        return DummyArtifacts(outputs[inference.phase_x])

    monkeypatch.setattr("repixelizer.pipeline.optimize_uv_field", fake_optimize_uv_field)
    monkeypatch.setattr("repixelizer.pipeline.source_lattice_consistency_breakdown", fake_support)
    monkeypatch.setattr("repixelizer.pipeline.foreground_edge_position_error", fake_edge_position)
    monkeypatch.setattr("repixelizer.pipeline.foreground_stroke_wobble_error", fake_wobble)
    monkeypatch.setattr("repixelizer.pipeline.foreground_edge_concentration", fake_concentration)

    selected = _select_phase_candidate(source, inference, analysis=object(), seed=7, device="cpu")
    assert selected.phase_x == candidate_b.phase_x
    assert selected.phase_y == candidate_b.phase_y


def test_badge_fixture_candidate_beats_previous_source_fidelity_baselines() -> None:
    source = load_rgba("tests/fixtures/real/ai-badge-cleaned.png")
    inference = InferenceResult(
        target_width=170,
        target_height=170,
        phase_x=-0.2,
        phase_y=0.2,
        confidence=0.0,
        top_candidates=[],
    )
    artifacts = optimize_uv_field(
        source,
        inference=inference,
        analysis=analyze_continuous_source(source, seed=7),
        steps=2,
        seed=7,
        device="cpu",
    )

    snap_consistency = source_lattice_consistency_breakdown(
        source,
        artifacts.initial_rgba,
        target_width=170,
        target_height=170,
        phase_x=-0.2,
        phase_y=0.2,
    )["score"]
    final_consistency = source_lattice_consistency_breakdown(
        source,
        artifacts.target_rgba,
        target_width=170,
        target_height=170,
        phase_x=-0.2,
        phase_y=0.2,
    )["score"]

    assert final_consistency <= 0.1494
    assert final_consistency <= 0.1369
    assert final_consistency <= snap_consistency + 1e-6
