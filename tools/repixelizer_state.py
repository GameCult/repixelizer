from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
MAP_PATH = STATE_DIR / "map.yaml"
BRANCHES_PATH = STATE_DIR / "branches.json"
EVIDENCE_PATH = STATE_DIR / "evidence.jsonl"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def load_branches() -> dict[str, Any]:
    return json.loads(read_text(BRANCHES_PATH))


def save_branches(data: dict[str, Any]) -> None:
    write_text(BRANCHES_PATH, json.dumps(data, indent=2) + "\n")


def append_evidence(record: dict[str, Any]) -> None:
    with EVIDENCE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def extract_map_field(name: str) -> str | None:
    prefix = f"  {name}:"
    for line in read_text(MAP_PATH).splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def extract_active_subgoals() -> list[str]:
    lines = read_text(MAP_PATH).splitlines()
    results: list[str] = []
    in_section = False
    for line in lines:
        if not in_section:
            if line.strip() == "active_subgoals:":
                in_section = True
            continue
        if line.startswith("  - "):
            results.append(line[4:].strip())
            continue
        if line and not line.startswith(" "):
            break
    return results


def cmd_status(_: argparse.Namespace) -> int:
    branches = load_branches().get("branches", [])
    active = [branch for branch in branches if branch.get("status") == "active"]
    summary = extract_map_field("summary") or "(missing)"
    next_action = extract_map_field("next_action") or "(missing)"
    subgoals = extract_active_subgoals()

    print(f"Workspace: {ROOT}")
    print(f"Summary: {summary}")
    print(f"Next action: {next_action}")
    print(f"Active branches: {len(active)} / {len(branches)}")
    if subgoals:
        print("Active subgoals:")
        for item in subgoals:
            print(f"- {item}")
    return 0


def cmd_add_evidence(args: argparse.Namespace) -> int:
    record: dict[str, Any] = {
        "ts": utc_stamp(),
        "type": args.type,
        "status": args.status,
        "note": args.note,
    }
    if args.branch:
        record["branch"] = args.branch
    append_evidence(record)
    print("Appended evidence record.")
    return 0


def cmd_add_branch(args: argparse.Namespace) -> int:
    data = load_branches()
    branches = data.setdefault("branches", [])
    if any(branch.get("id") == args.id for branch in branches):
        raise SystemExit(f"Branch '{args.id}' already exists.")
    branches.append(
        {
            "id": args.id,
            "hypothesis": args.hypothesis,
            "status": "active",
            "artifacts": args.artifact or [],
            "notes": args.note or "",
        }
    )
    save_branches(data)
    print(f"Added branch '{args.id}'.")
    return 0


def cmd_close_branch(args: argparse.Namespace) -> int:
    data = load_branches()
    for branch in data.get("branches", []):
        if branch.get("id") != args.id:
            continue
        branch["status"] = args.status
        if args.note:
            branch["notes"] = args.note
        save_branches(data)
        print(f"Updated branch '{args.id}' to status '{args.status}'.")
        return 0
    raise SystemExit(f"Branch '{args.id}' was not found.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and update Repixelizer persistent state files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show a compact state summary.")
    status_parser.set_defaults(func=cmd_status)

    evidence_parser = subparsers.add_parser(
        "add-evidence", help="Append one distilled JSONL evidence record."
    )
    evidence_parser.add_argument("--type", required=True)
    evidence_parser.add_argument("--status", required=True)
    evidence_parser.add_argument("--note", required=True)
    evidence_parser.add_argument("--branch")
    evidence_parser.set_defaults(func=cmd_add_evidence)

    add_branch_parser = subparsers.add_parser(
        "add-branch", help="Create a new active branch entry."
    )
    add_branch_parser.add_argument("--id", required=True)
    add_branch_parser.add_argument("--hypothesis", required=True)
    add_branch_parser.add_argument("--artifact", action="append")
    add_branch_parser.add_argument("--note")
    add_branch_parser.set_defaults(func=cmd_add_branch)

    close_branch_parser = subparsers.add_parser(
        "close-branch", help="Close an existing branch."
    )
    close_branch_parser.add_argument("--id", required=True)
    close_branch_parser.add_argument(
        "--status", required=True, choices=["accepted", "rejected", "archived"]
    )
    close_branch_parser.add_argument("--note")
    close_branch_parser.set_defaults(func=cmd_close_branch)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
