from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CultLibBindings:
    CultCache: Any
    CultMesh: Any
    SingleFileMessagePackBackingStore: Any
    define_database_entry_type: Any
    document_put_raw: Any


def _candidate_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root.parent / "CultLib" / "packages" / "cultcache-py" / "src",
        repo_root.parent / "cultcache-py" / "src",
    ]


def _ensure_cultlib_on_path() -> None:
    for candidate in _candidate_roots():
        if not candidate.exists():
            continue
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
        return
    raise RuntimeError(
        "Repixelizer verse runtime requires cultcache-py/cultnet_py. "
        "Install E:\\Projects\\CultLib\\packages\\cultcache-py or add it to PYTHONPATH."
    )


def _import_bindings() -> CultLibBindings:
    cultcache_py = importlib.import_module("cultcache_py")
    cultmesh_py = importlib.import_module("cultmesh_py")
    cultnet_py = importlib.import_module("cultnet_py")
    return CultLibBindings(
        CultCache=cultcache_py.CultCache,
        CultMesh=cultmesh_py.CultMesh,
        SingleFileMessagePackBackingStore=cultcache_py.SingleFileMessagePackBackingStore,
        define_database_entry_type=cultcache_py.define_database_entry_type,
        document_put_raw=cultnet_py.document_put_raw,
    )


def load_cultlib() -> CultLibBindings:
    try:
        return _import_bindings()
    except ModuleNotFoundError:
        _ensure_cultlib_on_path()
        return _import_bindings()


__all__ = ["CultLibBindings", "load_cultlib"]
