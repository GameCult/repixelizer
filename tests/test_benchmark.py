from __future__ import annotations

import json
from pathlib import Path

from repixelizer.benchmark import run_roundtrip_benchmark
from repixelizer.io import save_rgba
from repixelizer.synthetic import make_emblem


def test_roundtrip_benchmark_writes_summary_and_case_outputs(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    originals_dir = corpus_dir / "originals"
    originals_dir.mkdir(parents=True)
    save_rgba(originals_dir / "crest.png", make_emblem(16, 16))
    (originals_dir / "crest.json").write_text(
        json.dumps(
            {
                "title": "Crest",
                "author": "Test Artist",
                "license": "CC-BY 4.0",
                "source_url": "https://example.com/crest",
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "benchmark-out"
    payload = run_roundtrip_benchmark(corpus_dir, out_dir, variants=1, steps=4, seed=3)

    assert payload["case_count"] == 1
    assert payload["row_count"] == 2
    assert payload["profiles"] == ["soft", "crisp"]
    assert payload["primary_metric"] == "foreground_premultiplied_mae"
    assert (out_dir / "benchmark.csv").exists()
    assert (out_dir / "benchmark.json").exists()
    assert (out_dir / "cases" / "crest" / "original.png").exists()
    assert (out_dir / "cases" / "crest" / "profile-soft" / "variant-01" / "input.png").exists()
    assert (out_dir / "cases" / "crest" / "profile-crisp" / "variant-01" / "optimized.png").exists()
    assert "reference_foreground_coverage" in payload["rows"][0]
    assert "optimized_canvas_error_to_original" in payload["rows"][0]
    assert "optimized_adjacency_error_to_original" in payload["rows"][0]
    assert "optimized_motif_error_to_original" in payload["rows"][0]
    assert "profile" in payload["rows"][0]
    assert "warp_sample_mode" in payload["rows"][0]


def test_roundtrip_benchmark_can_filter_cases(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    originals_dir = corpus_dir / "originals"
    originals_dir.mkdir(parents=True)
    save_rgba(originals_dir / "crest-a.png", make_emblem(16, 16))
    save_rgba(originals_dir / "crest-b.png", make_emblem(18, 18))

    out_dir = tmp_path / "benchmark-out"
    payload = run_roundtrip_benchmark(
        corpus_dir,
        out_dir,
        variants=1,
        profiles=["crisp"],
        steps=2,
        seed=3,
        include_cases=["crest-b"],
    )

    assert payload["case_count"] == 1
    assert payload["row_count"] == 1
    assert payload["rows"][0]["case_id"] == "crest-b"
    assert payload["rows"][0]["profile"] == "crisp"


def test_roundtrip_benchmark_supports_ai_profile(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    originals_dir = corpus_dir / "originals"
    originals_dir.mkdir(parents=True)
    save_rgba(originals_dir / "crest.png", make_emblem(16, 16))

    out_dir = tmp_path / "benchmark-out"
    payload = run_roundtrip_benchmark(
        corpus_dir,
        out_dir,
        variants=1,
        profiles=["ai"],
        steps=2,
        seed=5,
    )

    assert payload["profiles"] == ["ai"]
    assert payload["row_count"] == 1
    row = payload["rows"][0]
    assert row["profile"] == "ai"
    assert row["artifact_density"] > 0.0
    assert row["artifact_strength"] > 0.0


def test_roundtrip_benchmark_clears_existing_output_dir_by_default(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    originals_dir = corpus_dir / "originals"
    originals_dir.mkdir(parents=True)
    save_rgba(originals_dir / "crest.png", make_emblem(16, 16))

    out_dir = tmp_path / "benchmark-out"
    stale_file = out_dir / "stale.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old", encoding="utf-8")

    run_roundtrip_benchmark(corpus_dir, out_dir, variants=1, profiles=["soft"], steps=2, seed=3)

    assert not stale_file.exists()
    assert (out_dir / "benchmark.json").exists()
