from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    src = _repo_root() / "src"
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def _socket_family_for_host(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(_socket_family_for_host(host), socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _powershell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )


def _find_listener_pid(port: int) -> int | None:
    if sys.platform == "win32":
        result = _powershell(
            f"$conn = Get-NetTCPConnection -State Listen -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -First 1; "
            "if ($conn) { Write-Output $conn.OwningProcess }"
        )
        text = result.stdout.strip()
        return int(text) if text.isdigit() else None
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip().splitlines()
    return int(text[0]) if text and text[0].isdigit() else None


def _process_commandline(pid: int) -> str:
    if sys.platform == "win32":
        result = _powershell(
            f'$proc = Get-CimInstance Win32_Process -Filter "ProcessId = {pid}" -ErrorAction SilentlyContinue; '
            "if ($proc) { Write-Output $proc.CommandLine }"
        )
        return result.stdout.strip()
    result = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _normalize_pathish(text: str) -> str:
    return text.replace("\\", "/").lower()


def _is_repixelizer_gui_process(commandline: str, repo_root: Path) -> bool:
    if not commandline:
        return False
    normalized = _normalize_pathish(commandline)
    repo_text = _normalize_pathish(str(repo_root))
    markers = (
        "scripts/run_gui.py",
        "repixelizer.gui:create_app",
        "repixelizer.gui",
    )
    return repo_text in normalized and any(marker in normalized for marker in markers)


def _terminate_process_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=True, capture_output=True, text=True)
        return
    os.kill(pid, 15)


def _wait_for_port_release(host: str, port: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_is_available(host, port):
            return True
        time.sleep(0.1)
    return _port_is_available(host, port)


def _reclaim_stale_gui_port(host: str, port: int, repo_root: Path) -> str | None:
    if _port_is_available(host, port):
        return None
    pid = _find_listener_pid(port)
    if pid is None:
        raise RuntimeError(f"Port {port} is already in use, but the owning process could not be identified.")
    commandline = _process_commandline(pid)
    if not _is_repixelizer_gui_process(commandline, repo_root):
        detail = commandline or "<unknown command line>"
        raise RuntimeError(
            f"Port {port} is already in use by PID {pid}. Refusing to kill an unrelated process: {detail}"
        )
    _terminate_process_tree(pid)
    if not _wait_for_port_release(host, port):
        raise RuntimeError(f"Stopped stale Repixelizer GUI process {pid}, but port {port} did not become available.")
    return f"Reclaimed stale Repixelizer GUI process {pid} on port {port}."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Repixelizer GUI from the repo checkout.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload while developing the GUI server")
    return parser


def main(argv: list[str] | None = None) -> int:
    repo_root = _repo_root()
    _ensure_src_on_path()
    from repixelizer.gui import main as gui_main

    args = build_parser().parse_args(argv)
    try:
        message = _reclaim_stale_gui_port(args.host, args.port, repo_root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if message:
        print(message)
    return gui_main(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    raise SystemExit(main())
