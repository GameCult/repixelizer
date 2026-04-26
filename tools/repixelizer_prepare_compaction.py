from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
NOTES_DIR = ROOT / "notes"
DOCS_DIR = ROOT / "docs"

MAP_PATH = STATE_DIR / "map.yaml"
SCRATCH_PATH = STATE_DIR / "scratch.md"
BRANCHES_PATH = STATE_DIR / "branches.json"
EVIDENCE_PATH = STATE_DIR / "evidence.jsonl"
HANDOFF_PATH = NOTES_DIR / "fresh-workspace-handoff.md"
PLAN_PATH = DOCS_DIR / "implementation-plan.md"
ALGO_MAP_PATH = DOCS_DIR / "lean-optimizer-algorithm-map.md"
AGENTS_PATH = ROOT / "AGENTS.md"


@dataclass
class Finding:
    level: str
    message: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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


def current_scratch_subgoal() -> str | None:
    lines = read_text(SCRATCH_PATH).splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "## Current Subgoal":
            continue
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if stripped:
                return stripped
    return None


def load_evidence() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_text(EVIDENCE_PATH).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"evidence line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"evidence line {line_number} is not a JSON object")
        records.append(record)
    return records


def load_branches() -> dict[str, Any]:
    data = json.loads(read_text(BRANCHES_PATH))
    if not isinstance(data, dict):
        raise ValueError("branches.json must be a JSON object")
    branches = data.get("branches")
    if not isinstance(branches, list):
        raise ValueError("branches.json must contain a branches list")
    return data


def add_required_file_checks(findings: list[Finding]) -> None:
    for path in [
        MAP_PATH,
        SCRATCH_PATH,
        BRANCHES_PATH,
        EVIDENCE_PATH,
        HANDOFF_PATH,
        PLAN_PATH,
        ALGO_MAP_PATH,
        AGENTS_PATH,
    ]:
        if path.exists():
            findings.append(Finding("ok", f"found {path.relative_to(ROOT)}"))
        else:
            findings.append(Finding("error", f"missing {path.relative_to(ROOT)}"))


def add_content_checks(findings: list[Finding]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    branches: dict[str, Any] = {}

    summary = extract_map_field("summary")
    next_action = extract_map_field("next_action")
    if summary:
        findings.append(Finding("ok", "state/map.yaml has current_status.summary"))
    else:
        findings.append(Finding("error", "state/map.yaml is missing current_status.summary"))
    if next_action:
        findings.append(Finding("ok", "state/map.yaml has current_status.next_action"))
    else:
        findings.append(Finding("error", "state/map.yaml is missing current_status.next_action"))

    subgoals = extract_active_subgoals()
    if subgoals:
        findings.append(Finding("ok", f"state/map.yaml has {len(subgoals)} active subgoal(s)"))
    else:
        findings.append(Finding("warn", "state/map.yaml has no active_subgoals entries"))

    scratch_subgoal = current_scratch_subgoal()
    if scratch_subgoal == "No active scratch subgoal.":
        findings.append(Finding("ok", "state/scratch.md has no stale active scratch subgoal"))
    elif scratch_subgoal:
        findings.append(Finding("warn", f"state/scratch.md has active scratch subgoal: {scratch_subgoal}"))
    else:
        findings.append(Finding("warn", "state/scratch.md has no Current Subgoal value"))

    try:
        evidence = load_evidence()
        findings.append(Finding("ok", f"state/evidence.jsonl parses ({len(evidence)} record(s))"))
    except ValueError as exc:
        findings.append(Finding("error", str(exc)))

    if EVIDENCE_PATH.exists() and EVIDENCE_PATH.stat().st_size > 25_000:
        findings.append(Finding("warn", "state/evidence.jsonl is larger than 25 KB; consider distillation"))

    try:
        branches = load_branches()
        active = [
            branch
            for branch in branches.get("branches", [])
            if isinstance(branch, dict) and branch.get("status") == "active"
        ]
        findings.append(Finding("ok", f"state/branches.json parses ({len(active)} active branch(es))"))
    except (json.JSONDecodeError, ValueError) as exc:
        findings.append(Finding("error", f"branches.json parse failed: {exc}"))

    return evidence, branches


def add_handoff_checks(findings: list[Finding]) -> None:
    text = read_text(HANDOFF_PATH)
    stale_branch = re.search(r"Current branch before .*ahead \d+", text)
    stale_head = re.search(r"Current HEAD before .*\b[0-9a-f]{7,40}\b", text)
    if stale_branch or stale_head:
        findings.append(Finding("error", "handoff embeds an exact branch or HEAD snapshot; use git commands instead"))
    else:
        findings.append(Finding("ok", "handoff avoids exact branch/HEAD snapshots"))

    required_phrases = [
        "Do not continue implementation automatically from a rehydrate-only request.",
        "Do not trust this file for the exact live HEAD.",
        "Immediate Re-entry Instruction",
    ]
    for phrase in required_phrases:
        if phrase in text:
            findings.append(Finding("ok", f"handoff contains: {phrase}"))
        else:
            findings.append(Finding("warn", f"handoff missing: {phrase}"))


def add_agents_checks(findings: list[Finding]) -> None:
    text = read_text(AGENTS_PATH)
    if "repixelizer_prepare_compaction.py" in text:
        findings.append(Finding("ok", "AGENTS.md tells agents to use the compaction helper"))
    else:
        findings.append(Finding("error", "AGENTS.md does not mention tools/repixelizer_prepare_compaction.py"))
    if "prepare for imminent compaction" in text.lower():
        findings.append(Finding("ok", "AGENTS.md names the imminent-compaction trigger"))
    else:
        findings.append(Finding("warn", "AGENTS.md does not name the imminent-compaction trigger phrase"))


def add_git_checks(findings: list[Finding]) -> tuple[str, str]:
    status = run_git(["status", "--short", "--branch"])
    log = run_git(["log", "--oneline", "-5"])
    dirty_lines = [line for line in status.splitlines()[1:] if line.strip()]
    if dirty_lines:
        findings.append(Finding("warn", "git worktree has uncommitted changes; commit or explain before compaction"))
    else:
        findings.append(Finding("ok", "git worktree is clean"))
    return status, log


def render_report(findings: list[Finding], status: str, log: str, evidence: list[dict[str, Any]]) -> str:
    summary = extract_map_field("summary") or "(missing)"
    next_action = extract_map_field("next_action") or "(missing)"
    subgoals = extract_active_subgoals()

    counts = {
        "ok": sum(1 for finding in findings if finding.level == "ok"),
        "warn": sum(1 for finding in findings if finding.level == "warn"),
        "error": sum(1 for finding in findings if finding.level == "error"),
    }

    lines = [
        "Repixelizer pre-compaction persistence check",
        f"Workspace: {ROOT}",
        f"Findings: {counts['ok']} ok, {counts['warn']} warn, {counts['error']} error",
        "",
        "Git status:",
        status,
        "",
        "Recent commits:",
        log,
        "",
        f"Summary: {summary}",
        f"Next action: {next_action}",
    ]
    if subgoals:
        lines.append("Active subgoals:")
        lines.extend(f"- {item}" for item in subgoals)
    if evidence:
        latest = evidence[-1]
        lines.extend(
            [
                "",
                "Latest distilled evidence:",
                f"- {latest.get('ts', '(missing ts)')} {latest.get('type', '(missing type)')}/{latest.get('status', '(missing status)')}: {latest.get('note', '(missing note)')}",
            ]
        )

    lines.append("")
    lines.append("Findings:")
    for finding in findings:
        lines.append(f"[{finding.level.upper()}] {finding.message}")

    lines.extend(
        [
            "",
            "Pre-compaction checklist:",
            "- Update state/map.yaml only if current understanding changed.",
            "- Refresh notes/fresh-workspace-handoff.md if re-entry instructions changed.",
            "- Add distilled evidence only for a belief-changing lesson, verification, rejected path, or scar.",
            "- Keep exact branch/HEAD out of handoff prose; git commands own volatile truth.",
            "- Commit completed persistence changes, or state why the worktree must stay dirty.",
            "- Re-run this helper after edits before yielding to compaction.",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Repixelizer persistent state before imminent compaction."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero on warnings as well as errors.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    findings: list[Finding] = []
    evidence: list[dict[str, Any]] = []

    add_required_file_checks(findings)
    try:
        evidence, _branches = add_content_checks(findings)
    except OSError as exc:
        findings.append(Finding("error", f"state content check failed: {exc}"))
    try:
        add_handoff_checks(findings)
    except OSError as exc:
        findings.append(Finding("error", f"handoff check failed: {exc}"))
    try:
        add_agents_checks(findings)
    except OSError as exc:
        findings.append(Finding("error", f"AGENTS check failed: {exc}"))
    try:
        status, log = add_git_checks(findings)
    except (OSError, subprocess.CalledProcessError) as exc:
        findings.append(Finding("error", f"git check failed: {exc}"))
        status = "(git status unavailable)"
        log = "(git log unavailable)"

    print(render_report(findings, status, log, evidence))

    has_error = any(finding.level == "error" for finding in findings)
    has_warning = any(finding.level == "warn" for finding in findings)
    if has_error or (args.strict and has_warning):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
