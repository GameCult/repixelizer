from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from repixelizer.corpus import prepare_corpus
from repixelizer.io import save_rgba


def test_prepare_corpus_extracts_frames_splits_sheets_and_archives_sources(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    originals_dir = corpus_dir / "originals"
    originals_dir.mkdir(parents=True)

    mage = np.zeros((188, 340, 4), dtype=np.float32)
    mage[..., 1] = 1.0
    mage[20:60, 20:50] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    mage[110:160, 270:320] = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    save_rgba(originals_dir / "mage-1-85x94.png", mage)

    shadow = np.zeros((350, 320, 4), dtype=np.float32)
    shadow[..., :3] = np.array([0.0, 1.0 / 255.0, 1.0 / 255.0], dtype=np.float32)
    shadow[..., 3] = 1.0
    shadow[290:340, 250:300] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    save_rgba(originals_dir / "shadow-80x70.png", shadow)

    multi = np.ones((110, 317, 4), dtype=np.float32)
    multi[..., 3] = 1.0
    multi[8:102, 0:79] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    multi[0:103, 88:210] = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    multi[0:110, 231:317] = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    save_rgba(originals_dir / "threeformsPJ2.png", multi)

    payload = prepare_corpus(corpus_dir)

    assert payload["output_count"] == 5
    assert (originals_dir / "mage-1.png").exists()
    assert (originals_dir / "shadow.png").exists()
    assert (originals_dir / "threeformspj2-01.png").exists()
    assert (originals_dir / "threeformspj2-02.png").exists()
    assert (originals_dir / "threeformspj2-03.png").exists()
    assert (corpus_dir / "source-sheets" / "mage-1-85x94.png").exists()
    assert (corpus_dir / "source-sheets" / "shadow-80x70.png").exists()
    assert (corpus_dir / "source-sheets" / "threeformsPJ2.png").exists()
    attribution_text = (corpus_dir / "ATTRIBUTION.md").read_text(encoding="utf-8")
    mage_sidecar = json.loads((originals_dir / "mage-1.json").read_text(encoding="utf-8"))
    shadow_sidecar = json.loads((originals_dir / "shadow.json").read_text(encoding="utf-8"))
    boss_sidecar = json.loads((originals_dir / "threeformspj2-01.json").read_text(encoding="utf-8"))
    assert mage_sidecar["source_title"] == "Bosses and monsters spritesheets (Ars Notoria)"
    assert mage_sidecar["attribution_confidence"] == "inferred"
    assert shadow_sidecar["licenses"] == ["CC-BY-SA 3.0"]
    assert shadow_sidecar["original_source_title"] == "First Person Dungeon Crawl Enemies Remixed"
    assert boss_sidecar["source_title"] == '3-form RPG boss: "Harlequin Epicycle."'
    assert "## Bosses and monsters spritesheets (Ars Notoria)" in attribution_text
    assert "`mage-1.png`" in attribution_text
    assert "`shadow.png`" in attribution_text
    assert "## 3-form RPG boss: \"Harlequin Epicycle.\"" in attribution_text
