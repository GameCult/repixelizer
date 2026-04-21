from __future__ import annotations

from pathlib import Path

from repixelizer.io import save_rgba
from repixelizer.synthetic import make_emblem
from repixelizer.tuning import tune_solver_hyperparams


def test_tuning_writes_results_and_best_benchmark(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    originals_dir = corpus_dir / "originals"
    originals_dir.mkdir(parents=True)
    save_rgba(originals_dir / "crest.png", make_emblem(16, 16))

    out_dir = tmp_path / "tuning-out"
    payload = tune_solver_hyperparams(
        corpus_dir,
        out_dir,
        trials=2,
        variants=1,
        profiles=["soft"],
        steps=2,
        seed=3,
        device="cpu",
    )

    assert payload["trials_completed"] == 2
    assert "best_params" in payload
    assert (out_dir / "tuning-results.json").exists()
    assert (out_dir / "best-benchmark" / "benchmark.json").exists()
    assert not (out_dir / "_scratch").exists()
