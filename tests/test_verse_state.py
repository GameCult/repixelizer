from __future__ import annotations

from pathlib import Path

from repixelizer.access import AccessController
from repixelizer.gui import GuiJobManager, HostedDemoConfig
from repixelizer.verse_state import RepixelizerVerseRuntime, RepixelizerVerseRuntimeConfig


def test_gui_module_imports_cleanly() -> None:
    import repixelizer.gui as gui

    assert callable(gui.create_app)


def test_verse_runtime_publish_once_writes_witness(tmp_path: Path, monkeypatch) -> None:
    spool_dir = tmp_path / "spool"
    witness_path = tmp_path / "cultcache" / "repixelizer.service.cc"

    monkeypatch.setenv("REPIXELIZER_VERSE_RUNTIME", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(spool_dir))
    monkeypatch.setenv("GC_ACCESS_MODE", "off")
    monkeypatch.setenv("GC_ACCESS_REQUIRED", "0")
    monkeypatch.setenv("GC_ACCESS_CULTCACHE_PATH", str(witness_path))
    monkeypatch.setenv("GC_ACCESS_IDUNN_DAEMON", "repixelizer-test")
    monkeypatch.setenv("GC_ACCESS_IDUNN_HEALTH_CONTRACT", "repixelizer.cultnet-rudp-service-health")
    monkeypatch.delenv("GC_ACCESS_IDUNN_RUDP_HEALTH", raising=False)

    config = HostedDemoConfig.from_env()
    manager = GuiJobManager(config)
    access_controller = AccessController.from_env(hosted_demo=config.hosted_demo)
    runtime = RepixelizerVerseRuntime(
        RepixelizerVerseRuntimeConfig.from_env(config),
        config,
        access_controller,
        manager,
    )

    runtime.publish_once()
    health = runtime.build_health_payload()

    assert runtime.enabled is True
    assert witness_path.exists()
    assert witness_path.stat().st_size > 0
    assert health["status"] == "ok"
    assert health["verseRuntime"] == "enabled"
    assert health["cultCachePath"] == str(witness_path)
    assert health["idunnDaemon"] == "repixelizer-test"
    assert health["idunnRudpHealth"] is None
