from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from .access import AccessController
from .gui import HostedDemoConfig


PROVIDER_ADVERTISEMENT_SCHEMA = "gamecult.eve.provider_advertisement.v1"
PROVIDER_ID = "repixelizer"
SERVICE_ID = "repixelizer.service"
DEFAULT_VERSE_ID = "gamecult.local"
DEFAULT_PUBLIC_BASE_URL = "https://repixelizer.gamecult.org"
DEFAULT_CC_WITNESS_PATH = "state/repixelizer.witness.cc"
DEFAULT_EVE_SURFACE_KEY = "cultmesh://repixelizer/surfaces/operator"

WITNESS_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "schema": "repixelizer.job.v0",
        "owner": "repixelizer.gui",
        "authority": "accepted",
        "storage": "memory-dev-only-with-cc-witness-reserved",
        "portable": True,
        "description": "Job id, owner binding, status, timestamps, input/output artifact refs, error summary, and retention state.",
    },
    {
        "schema": "repixelizer.queue_snapshot.v0",
        "owner": "repixelizer.gui",
        "authority": "accepted",
        "storage": "memory-dev-only-with-cc-witness-reserved",
        "portable": True,
        "description": "Queue capacity, waiting/running counts, active job ids, oldest waiting age, and hosted mode.",
    },
    {
        "schema": "repixelizer.job_event.v0",
        "owner": "repixelizer.gui",
        "authority": "redacted-projection",
        "storage": "memory-dev-only-with-cc-witness-reserved",
        "portable": True,
        "description": "Redacted progress event, stage, index, timestamp, and job id.",
    },
    {
        "schema": "repixelizer.runtime_config.v0",
        "owner": "repixelizer.gui",
        "authority": "accepted",
        "storage": "memory-dev-only-with-cc-witness-reserved",
        "portable": True,
        "description": "Hosted flags, queue limits, visible UI flags, solver config hash placeholder, deployment id, and spool path.",
    },
    {
        "schema": "repixelizer.auth_projection.v0",
        "owner": "repixelizer.access",
        "authority": "external-authority-projection",
        "storage": "heimdall-projection-with-cc-witness-reserved",
        "portable": True,
        "description": "Auth mode, provider availability, subject capability summary, and Heimdall claim freshness without raw tokens.",
    },
)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_provider_advertisement(
    *,
    updated_at: str | None = None,
    verse_id: str = DEFAULT_VERSE_ID,
    public_base_url: str = DEFAULT_PUBLIC_BASE_URL,
    cc_witness_path: str = DEFAULT_CC_WITNESS_PATH,
    eve_surface_key: str = DEFAULT_EVE_SURFACE_KEY,
    config: HostedDemoConfig | None = None,
    access_controller: AccessController | None = None,
) -> dict[str, Any]:
    config = config or HostedDemoConfig.from_env()
    access_controller = access_controller or AccessController.from_env(hosted_demo=config.hosted_demo)
    updated_at = updated_at or utc_now_iso()
    auth_payload = access_controller.public_payload()

    return {
        "schema": PROVIDER_ADVERTISEMENT_SCHEMA,
        "providerId": PROVIDER_ID,
        "serviceId": SERVICE_ID,
        "verseId": verse_id,
        "title": "Repixelizer",
        "kind": "service.product",
        "updatedAt": updated_at,
        "freshness": {
            "state": "fresh",
            "lastSeenAt": updated_at,
            "maxAgeMs": 15000,
        },
        "schemas": list(WITNESS_SCHEMAS),
        "witnesses": [
            {
                "id": "repixelizer.hosted.witness",
                "kind": "cc-export-path-reserved",
                "path": cc_witness_path,
                "schemas": [entry["schema"] for entry in WITNESS_SCHEMAS],
                "redaction": "bulk-images-raw-tokens-and-private-claims-removed",
                "freshness": {
                    "state": "planned",
                    "updatedAt": updated_at,
                },
            }
        ],
        "surfaces": [
            {
                "surfaceId": "repixelizer.operator.surface",
                "schema": "gamecult.eve.surface.v1",
                "transport": "http-json",
                "key": eve_surface_key,
                "url": f"{public_base_url.rstrip('/')}/eve/surface",
                "audience": "operator",
                "mode": "read-only-first-cut",
                "commands": ["repixelizer.web.open", "repixelizer.job.submit", "repixelizer.job.cancel_own"],
                "status": "available",
                "styleProfile": "repixelizer.retro.pixel",
            },
            {
                "surfaceId": "repixelizer.web.app",
                "schema": "gamecult.eve.surface.v1",
                "transport": "browser-lowering",
                "url": f"{public_base_url.rstrip('/')}/app/",
                "canonical": False,
                "canonicalSurfaceId": "repixelizer.operator.surface",
            },
        ],
        "commands": [
            {
                "command": "repixelizer.job.submit",
                "surfaceId": "repixelizer.web.app",
                "transport": "http",
                "route": "/api/jobs",
                "method": "POST",
                "authority": "repixelizer-route-policy",
                "result": "accepted-denied-or-queued",
            },
            {
                "command": "repixelizer.job.cancel_own",
                "surfaceId": "repixelizer.web.app",
                "transport": "http",
                "route": "/api/jobs/{job_id}",
                "method": "DELETE",
                "authority": "repixelizer-job-owner",
                "result": "accepted-denied-or-not-found",
            },
        ],
        "nestedVerses": [
            {
                "verseId": "repixelizer.session.{sessionId}",
                "parentVerseId": verse_id,
                "kind": "session-space",
                "authorityBoundary": "repixelizer-local-session-or-heimdall-account",
                "surfaceIds": ["repixelizer.web.app"],
                "stateSchemas": [
                    "repixelizer.job.v0",
                    "repixelizer.job_event.v0",
                    "repixelizer.auth_projection.v0",
                ],
                "carryRules": {
                    "identity": "heimdall-claims-projected-when-enabled",
                    "commands": "repixelizer-route-policy-only",
                },
            },
            {
                "verseId": "repixelizer.operator",
                "parentVerseId": verse_id,
                "kind": "hosted-operator-space",
                "authorityBoundary": "read-only-first-cut",
                "surfaceIds": ["repixelizer.operator.surface"],
                "stateSchemas": [entry["schema"] for entry in WITNESS_SCHEMAS],
            },
        ],
        "styleCapabilities": [
            {
                "styleProfile": "repixelizer.product",
                "tokenGroups": [
                    "repixelizer.retro.pixel",
                    "repixelizer.comparisonCanvas",
                    "repixelizer.cleanupTool",
                    "repixelizer.queuePanel",
                    "repixelizer.operatorStatus",
                ],
                "preferredLowerings": ["css", "android-native", "native-eve", "tui"],
                "lossiness": {
                    "tui": "no-pixel-canvas-editing; status, queue, artifact refs, and command intents only",
                    "read-only-first-cut": "operator surface names state and routes but does not own queue truth",
                },
            }
        ],
        "contacts": [
            {
                "kind": "repo",
                "path": "E:\\Projects\\repixelizer",
            }
        ],
        "runtime": {
            "hostedDemo": config.hosted_demo,
            "queueCapacity": config.queue_capacity,
            "showQueuePanel": config.show_queue_panel,
            "spoolPath": str(config.spool_dir),
            "auth": {
                "enabled": bool(auth_payload.get("enabled")),
                "mode": auth_payload.get("mode"),
                "required": bool(auth_payload.get("required")),
                "protectQueue": bool(auth_payload.get("protectQueue")),
                "providers": auth_payload.get("providers", []),
            },
        },
    }


def write_provider_advertisement(
    output: Path | None,
    *,
    updated_at: str | None = None,
    verse_id: str = DEFAULT_VERSE_ID,
    public_base_url: str = DEFAULT_PUBLIC_BASE_URL,
    cc_witness_path: str = DEFAULT_CC_WITNESS_PATH,
    eve_surface_key: str = DEFAULT_EVE_SURFACE_KEY,
    stream: TextIO = sys.stdout,
) -> None:
    payload = build_provider_advertisement(
        updated_at=updated_at,
        verse_id=verse_id,
        public_base_url=public_base_url,
        cc_witness_path=cc_witness_path,
        eve_surface_key=eve_surface_key,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        stream.write(f"{encoded}\n")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{encoded}\n", encoding="utf-8")


def build_witness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repixelize witness-advertisement",
        description="Emit Repixelizer's read-only Eve provider advertisement fixture.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write JSON to this path instead of stdout.")
    parser.add_argument("--updated-at", default=None, help="Override the advertisement timestamp.")
    parser.add_argument("--verse-id", default=DEFAULT_VERSE_ID, help="Authoritative Verse id.")
    parser.add_argument("--public-base-url", default=DEFAULT_PUBLIC_BASE_URL, help="Hosted browser lowering base URL.")
    parser.add_argument("--cc-witness-path", default=DEFAULT_CC_WITNESS_PATH, help="Reserved CultCache .cc witness path.")
    parser.add_argument("--eve-surface-key", default=DEFAULT_EVE_SURFACE_KEY, help="Planned CultMesh Eve surface key.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_witness_parser()
    args = parser.parse_args(argv)
    write_provider_advertisement(
        args.out,
        updated_at=args.updated_at,
        verse_id=args.verse_id,
        public_base_url=args.public_base_url,
        cc_witness_path=args.cc_witness_path,
        eve_surface_key=args.eve_surface_key,
    )
    return 0
