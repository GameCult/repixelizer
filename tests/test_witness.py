from __future__ import annotations

import json
from pathlib import Path

from repixelizer.cli import main as cli_main
from repixelizer.witness import build_provider_advertisement


def test_provider_advertisement_names_witness_schemas(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_QUEUE_CAPACITY", "7")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "off")

    payload = build_provider_advertisement(updated_at="2026-06-03T00:00:00Z")

    assert payload["schema"] == "gamecult.eve.provider_advertisement.v1"
    assert payload["providerId"] == "repixelizer"
    assert payload["runtime"]["hostedDemo"] is True
    assert payload["runtime"]["queueCapacity"] == 7
    assert payload["runtime"]["auth"]["mode"] == "off"

    schema_names = {entry["schema"] for entry in payload["schemas"]}
    assert schema_names == {
        "repixelizer.auth_projection.v0",
        "repixelizer.job.v0",
        "repixelizer.job_event.v0",
        "repixelizer.queue_snapshot.v0",
        "repixelizer.runtime_config.v0",
    }
    witness = payload["witnesses"][0]
    assert witness["kind"] == "cc-export-path-reserved"
    assert witness["path"] == "state/repixelizer.witness.cc"
    assert witness["freshness"]["state"] == "planned"
    surface = payload["surfaces"][0]
    assert surface["status"] == "available"
    assert surface["transport"] == "http-json"
    assert surface["url"] == "https://repixelizer.gamecult.org/eve/surface"
    assert surface["styleProfile"] == "repixelizer.retro.pixel"


def test_witness_advertisement_cli_writes_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "off")
    out = tmp_path / "repixelizer.provider-advertisement.json"

    exit_code = cli_main(
        [
            "witness-advertisement",
            "--out",
            str(out),
            "--updated-at",
            "2026-06-03T00:00:00Z",
            "--verse-id",
            "gamecult.test",
            "--cc-witness-path",
            "state/test-repixelizer.witness.cc",
        ]
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verseId"] == "gamecult.test"
    assert payload["witnesses"][0]["path"] == "state/test-repixelizer.witness.cc"
    assert payload["commands"][0]["route"] == "/api/jobs"
