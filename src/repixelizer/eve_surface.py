from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from .access import AccessController
from .gui import HostedDemoConfig


EVE_SURFACE_SCHEMA = "gamecult.eve.surface.v1"
PROVIDER_ID = "repixelizer"
PROVIDER_KIND = "repixelizer.product"
DEFAULT_PUBLIC_BASE_URL = "https://repixelizer.gamecult.org"


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _text(element_id: str, text: str, *, role: str = "body", **props: Any) -> dict[str, Any]:
    return {
        "id": element_id,
        "kind": "text",
        "role": role,
        "text": text,
        "props": {"text": text, "role": role, **props},
        "children": [],
    }


def _card(element_id: str, title: str, children: list[dict[str, Any]], *, tone: str = "panel") -> dict[str, Any]:
    return {
        "id": element_id,
        "kind": "card",
        "text": title,
        "role": "group",
        "style": {"tone": tone, "variant": "repixelizer-retro-card"},
        "props": {"title": title, "styleProfile": "repixelizer.retro.pixel"},
        "children": children,
    }


def _pane(element_id: str, title: str, children: list[dict[str, Any]], *, direction: str = "vertical") -> dict[str, Any]:
    return {
        "id": element_id,
        "kind": "pane",
        "text": title,
        "role": "section",
        "layout": {"direction": direction, "padding": 10},
        "style": {"tone": "warm", "variant": "repixelizer-retro-pane"},
        "props": {"title": title, "styleProfile": "repixelizer.retro.pixel"},
        "children": children,
    }


def _command_card(
    element_id: str,
    title: str,
    body: str,
    *,
    command_id: str,
    route: str,
    method: str,
) -> dict[str, Any]:
    element = _card(
        element_id,
        title,
        [
            _text(f"{element_id}.body", body),
            _text(f"{element_id}.route", f"{method} {route}", role="mono", route=route, method=method),
        ],
        tone="cool",
    )
    element["commandId"] = command_id
    element["props"]["commandId"] = command_id
    element["props"]["transport"] = {"kind": "http", "route": route, "method": method}
    return element


def _styles(public_base_url: str) -> dict[str, Any]:
    base = public_base_url.rstrip("/")
    return {
        "profile": "repixelizer.retro.pixel",
        "tokens": {
            "fontTitle": "\"Press Start 2P\", \"VT323\", monospace",
            "fontBody": "\"VT323\", \"Courier New\", monospace",
            "fontMono": "\"VT323\", \"Courier New\", monospace",
            "colorBackgroundTop": "#01040b",
            "colorBackgroundMid": "#020813",
            "colorBackgroundBottom": "#01050d",
            "colorShellSurfaceTop": "#061424",
            "colorShellSurfaceBottom": "#030b16",
            "colorPanel": "#0a2239",
            "colorPanelAlt": "#061627",
            "colorTextBright": "#f7efd8",
            "colorText": "#efe5c9",
            "colorMuted": "#d3bb7f",
            "colorAccent": "#ffd84a",
            "colorAccentStrong": "#ffae1a",
            "colorPanelBorder": "#0a2740",
            "colorPanelBorderDeep": "#06192a",
            "colorCornerAccent": "#ffb72a",
            "colorShadow": "rgba(0, 0, 0, 0.55)",
            "colorPixelGrid": "rgba(0, 0, 0, 0.3)",
            "imageRendering": "pixelated",
            "borderWidthPx": 4,
            "borderBottomWidthPx": 5,
            "cornerAccentPx": 12,
            "scanlineOverlay": True,
            "dashedInnerFrame": True,
            "buttonGlyph": ">",
        },
        "assets": {
            "fontCss": "https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap",
            "landingInput": f"{base}/app/landing-assets/character-input.png",
            "landingOutput": f"{base}/app/landing-assets/character-repixelized.png",
            "appUrl": f"{base}/app/",
        },
        "loweringHints": {
            "css": "Use the tokens as CSS custom properties and retain scanline/pixel-grid overlays.",
            "android-native": "Load packaged or downloadable Press Start 2P and VT323 fonts, map style tokens before hard-coded tones, and render asset images instead of asset labels.",
            "tui": "Preserve title/body font distinction as role metadata; canvas editing lowers to artifact refs and command intents only.",
        },
    }


def _runtime_cards(
    *,
    config: HostedDemoConfig,
    auth_payload: dict[str, Any],
    queue_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    provider_labels = []
    for provider in auth_payload.get("providers", []):
        if isinstance(provider, dict):
            label = provider.get("label") or provider.get("slug")
            if isinstance(label, str) and label:
                provider_labels.append(label)
        elif isinstance(provider, str):
            provider_labels.append(provider)
    queue_copy = (
        f"{queue_summary.get('queueDepth', 0)} active or waiting / "
        f"{queue_summary.get('queueCapacity', config.queue_capacity)} capacity"
    )
    return [
        _card(
            "repixelizer.runtime.config",
            "Runtime",
            [
                _text("repixelizer.runtime.hosted", f"hostedDemo: {config.hosted_demo}", role="mono"),
                _text("repixelizer.runtime.steps", f"defaultSteps: {config.default_steps} maxSteps: {config.max_steps}", role="mono"),
                _text("repixelizer.runtime.dimensions", f"maxOutputDimension: {config.max_output_dimension}px", role="mono"),
            ],
        ),
        _card(
            "repixelizer.runtime.queue",
            "Queue",
            [
                _text("repixelizer.queue.depth", queue_copy, role="strong"),
                _text("repixelizer.queue.waiting", f"waitingCount: {queue_summary.get('waitingCount', 0)}", role="mono"),
                _text("repixelizer.queue.active", f"hasActiveJob: {queue_summary.get('hasActiveJob', False)}", role="mono"),
            ],
        ),
        _card(
            "repixelizer.runtime.auth",
            "Access",
            [
                _text("repixelizer.auth.mode", f"mode: {auth_payload.get('mode', 'off')}", role="mono"),
                _text("repixelizer.auth.required", f"required: {bool(auth_payload.get('required'))}", role="mono"),
                _text("repixelizer.auth.providers", f"providers: {', '.join(provider_labels) or 'none'}"),
            ],
        ),
    ]


def _lowering_comparison() -> dict[str, Any]:
    return {
        "periwinkleAndroidKotlin": {
            "source": "E:\\Projects\\Eve\\android\\app\\src\\main\\java\\org\\gamecult\\eve\\MainActivity.kt",
            "status": "surface-tree-renderer-present-style-token-gap",
            "keeps": [
                "surface root traversal",
                "pane/card/text hierarchy",
                "TextView body content",
                "basic ProgressBar metric rendering",
                "node-bound command taps",
            ],
            "losesUntilRendererParity": [
                "Press Start 2P and VT323 font selection",
                "Repixelizer shell frame, scanlines, pixel grid, corner accents, and chunky border widths",
                "landing/app image assets; current Android lowerer prints asset refs when images are not explicitly implemented",
                "upload picker, comparison canvas, pan/zoom inspection, eyedropper, and paint controls",
                "provider style tokens; current Kotlin toneColor/textColorFor/textSizeFor are hard-coded",
            ],
            "minimumParityCuts": [
                "map surface.styles.tokens before toneColor/textColorFor/textSizeFor",
                "add packaged pixel fonts or font download cache",
                "render image assetUri/assetRef as ImageView with pixelated sampling",
                "lower command controls to Android upload/document picker and job route calls",
            ],
        }
    }


def build_repixelizer_eve_surface(
    *,
    updated_at: str | None = None,
    public_base_url: str = DEFAULT_PUBLIC_BASE_URL,
    config: HostedDemoConfig | None = None,
    access_controller: AccessController | None = None,
    queue_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or HostedDemoConfig.from_env()
    access_controller = access_controller or AccessController.from_env(hosted_demo=config.hosted_demo)
    auth_payload = access_controller.public_payload()
    queue_summary = queue_summary or {
        "queueDepth": 0,
        "waitingCount": 0,
        "queueCapacity": config.queue_capacity,
        "hasActiveJob": False,
        "activeStatus": None,
    }
    updated_at = updated_at or utc_now_iso()
    base = public_base_url.rstrip("/")
    root = {
        "id": "repixelizer.surface.root",
        "kind": "surface",
        "text": "Repixelizer",
        "role": "root",
        "layout": {"direction": "vertical", "padding": 14},
        "style": {"tone": "warm", "variant": "repixelizer-retro-shell"},
        "props": {
            "title": "Repixelizer",
            "subtitle": "Fake pixel art goes in. Real grid-aligned sprite art comes out.",
            "styleProfile": "repixelizer.retro.pixel",
            "browserLowering": f"{base}/app/",
            "frontPage": f"{base}/",
        },
        "children": [
            _pane(
                "repixelizer.hero",
                "Repixelizer hosted demo",
                [
                    _text(
                        "repixelizer.hero.copy",
                        "Force fake pixel art back onto a real grid, then inspect each solver step without sanding off the crunchy old-machine charm.",
                    ),
                    _command_card(
                        "repixelizer.hero.open",
                        "Open demo",
                        "Browser lowering remains the full interactive app until native Eve gains upload and canvas editing parity.",
                        command_id="repixelizer.web.open",
                        route="/app/",
                        method="GET",
                    ),
                ],
            ),
            _pane(
                "repixelizer.workflow",
                "Single page app",
                [
                    _command_card(
                        "repixelizer.workflow.submit",
                        "Upload image",
                        "Submit a PNG/JPEG/WebP/GIF into the GUI job queue; queue ownership stays in repixelizer.gui.",
                        command_id="repixelizer.job.submit",
                        route="/api/jobs",
                        method="POST",
                    ),
                    _card(
                        "repixelizer.workflow.inspect",
                        "Inspect solver",
                        [
                            _text("repixelizer.inspect.source", "source, edge map, lattice overlay, phase field, cleaned output"),
                            _text("repixelizer.inspect.canvas", "comparison canvas and editor lower to native only after Periwinkle gets image/control parity"),
                        ],
                    ),
                    _command_card(
                        "repixelizer.workflow.cancel",
                        "Cancel own job",
                        "The command intent delegates to the existing owner-bound DELETE route.",
                        command_id="repixelizer.job.cancel_own",
                        route="/api/jobs/{job_id}",
                        method="DELETE",
                    ),
                ],
            ),
            _pane("repixelizer.runtime", "Live projection", _runtime_cards(config=config, auth_payload=auth_payload, queue_summary=queue_summary)),
        ],
    }
    return {
        "type": "surface-state",
        "schema": EVE_SURFACE_SCHEMA,
        "providerId": PROVIDER_ID,
        "providerKind": PROVIDER_KIND,
        "title": "Repixelizer",
        "version": 1,
        "updatedAt": updated_at,
        "surface": {
            "id": "repixelizer.operator.surface",
            "schema": EVE_SURFACE_SCHEMA,
            "title": "Repixelizer",
            "root": root,
            "styles": _styles(public_base_url),
        },
        "commands": [
            {
                "command": "repixelizer.web.open",
                "surfaceId": "repixelizer.operator.surface",
                "transport": "http",
                "route": "/app/",
                "method": "GET",
                "authority": "browser-lowering",
            },
            {
                "command": "repixelizer.job.submit",
                "surfaceId": "repixelizer.operator.surface",
                "transport": "http",
                "route": "/api/jobs",
                "method": "POST",
                "authority": "repixelizer-route-policy",
            },
            {
                "command": "repixelizer.job.cancel_own",
                "surfaceId": "repixelizer.operator.surface",
                "transport": "http",
                "route": "/api/jobs/{job_id}",
                "method": "DELETE",
                "authority": "repixelizer-job-owner",
            },
        ],
        "runtime": {
            "hostedDemo": config.hosted_demo,
            "queue": queue_summary,
            "auth": auth_payload,
            "limits": config.public_payload()["limits"],
        },
        "loweringComparison": _lowering_comparison(),
    }


def write_repixelizer_eve_surface(
    output: Path | None,
    *,
    updated_at: str | None = None,
    public_base_url: str = DEFAULT_PUBLIC_BASE_URL,
    stream: TextIO = sys.stdout,
) -> None:
    payload = build_repixelizer_eve_surface(updated_at=updated_at, public_base_url=public_base_url)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        stream.write(f"{encoded}\n")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{encoded}\n", encoding="utf-8")


def build_eve_surface_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repixelize eve-surface",
        description="Emit Repixelizer's read-only Eve surface projection.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write JSON to this path instead of stdout.")
    parser.add_argument("--updated-at", default=None, help="Override the surface timestamp.")
    parser.add_argument("--public-base-url", default=DEFAULT_PUBLIC_BASE_URL, help="Hosted browser lowering base URL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_eve_surface_parser()
    args = parser.parse_args(argv)
    write_repixelizer_eve_surface(args.out, updated_at=args.updated_at, public_base_url=args.public_base_url)
    return 0
