from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    src = _repo_root() / "src"
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Repixelizer GUI from the repo checkout.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload while developing the GUI server")
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    from repixelizer.gui import main as gui_main

    args = build_parser().parse_args(argv)
    return gui_main(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    raise SystemExit(main())
