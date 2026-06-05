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


def _leaf(element_id: str, kind: str, *, text: str | None = None, role: str = "body", **props: Any) -> dict[str, Any]:
    payload = {"text": text, "role": role, **props}
    if text is None:
        payload.pop("text")
    return {
        "id": element_id,
        "kind": kind,
        "role": role,
        "text": text or props.get("label") or props.get("title") or element_id,
        "props": payload,
        "children": [],
    }


def _button(element_id: str, label: str, *, command_id: str, **props: Any) -> dict[str, Any]:
    return _leaf(element_id, "control.button", role="command", label=label, commandId=command_id, **props)


def _field(element_id: str, label: str, *, value: str, field_kind: str = "input.number", **props: Any) -> dict[str, Any]:
    return _leaf(element_id, field_kind, role="field", label=label, value=value, **props)


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


def _landing_gallery(base: str) -> dict[str, Any]:
    return _pane(
        "repixelizer.product.landing",
        "Hosted demo",
        [
            _text(
                "repixelizer.product.pitch",
                "You need pixel art for a project, but pixel art is a craft and your favorite AI is mostly vibes in a trench coat.",
                role="strong",
            ),
            {
                "id": "repixelizer.product.gallery",
                "kind": "partition",
                "role": "gallery",
                "layout": {"direction": "horizontal", "gap": 12},
                "props": {"split": "x", "gap": 12, "role": "product.gallery"},
                "children": [
                    _card(
                        "repixelizer.product.input",
                        "The AI gives you this",
                        [
                            _leaf(
                                "repixelizer.product.input.image",
                                "image.preview",
                                label="AI-generated hazmat character before repixelizing",
                                src=f"{base}/app/landing-assets/character-input.png",
                                imageRendering="pixelated",
                                aspectRatio="1/1",
                            ),
                            _text("repixelizer.product.input.copy", "Looks charming from orbit. Suspiciously charming."),
                        ],
                        tone="cool",
                    ),
                    _card(
                        "repixelizer.product.closeup",
                        "Then you zoom in",
                        [
                            _leaf(
                                "repixelizer.product.closeup.image",
                                "image.preview",
                                label="Zoomed close-up showing fake pixel-art artifacts",
                                src=f"{base}/app/landing-assets/character-input.png",
                                crop="42% 63%",
                                zoom=3.45,
                                imageRendering="pixelated",
                                aspectRatio="1/1",
                            ),
                            _text("repixelizer.product.closeup.copy", "Smears, half-cells, anti-grid nonsense. The crime scene has opinions."),
                        ],
                        tone="cool",
                    ),
                    _card(
                        "repixelizer.product.output",
                        "Repixelizer spits out this",
                        [
                            _leaf(
                                "repixelizer.product.output.image",
                                "image.preview",
                                label="Repixelized hazmat character output",
                                src=f"{base}/app/landing-assets/character-repixelized.png",
                                imageRendering="pixelated",
                                aspectRatio="1/1",
                            ),
                            _text("repixelizer.product.output.copy", "Real cells on a real lattice, ready for cleanup or export."),
                        ],
                        tone="cool",
                    ),
                ],
            },
        ],
    )


def _workspace_surface(config: HostedDemoConfig, queue_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "repixelizer.workspace",
        "kind": "partition",
        "role": "workspace",
        "layout": {"direction": "horizontal", "gap": 12},
        "props": {"split": "x", "gap": 12, "role": "repixelizer.workspace"},
        "children": [
            _pane(
                "repixelizer.workspace.viewer",
                "Pipeline Stages",
                [
                    _leaf(
                        "repixelizer.workspace.status",
                        "status.stage",
                        label="Idle",
                        stage="Waiting for input",
                        detail="Choose a file, then run the machine.",
                    ),
                    {
                        "id": "repixelizer.workspace.compare",
                        "kind": "partition",
                        "role": "comparison",
                        "layout": {"direction": "horizontal", "gap": 10},
                        "props": {"split": "x", "gap": 10, "role": "comparison"},
                        "children": [
                            _leaf("repixelizer.workspace.input.canvas", "canvas.preview", label="Input", state="empty"),
                            _leaf("repixelizer.workspace.output.canvas", "canvas.preview", label="Output", state="empty"),
                        ],
                    },
                    _card(
                        "repixelizer.workspace.inspect",
                        "Zoom",
                        [
                            _text("repixelizer.workspace.inspect.copy", "Hold either preview to inspect both panels together."),
                            _field("repixelizer.workspace.inspect.zoom", "Inspect Zoom", value="8x", field_kind="input.select"),
                        ],
                    ),
                ],
            ),
            _pane(
                "repixelizer.workspace.editor",
                "Pixel Fixes",
                [
                    {
                        "id": "repixelizer.workspace.editor.actions",
                        "kind": "partition",
                        "role": "toolbar",
                        "layout": {"direction": "horizontal", "gap": 8},
                        "props": {"split": "x", "gap": 8, "role": "toolbar"},
                        "children": [
                            _button("repixelizer.workspace.reset", "Reset Output", command_id="repixelizer.editor.reset"),
                            _button("repixelizer.workspace.export", "Export PNG", command_id="repixelizer.editor.export"),
                        ],
                    },
                    {
                        "id": "repixelizer.workspace.paint",
                        "kind": "partition",
                        "role": "paint-controls",
                        "layout": {"direction": "horizontal", "gap": 6},
                        "props": {"split": "x", "gap": 6, "role": "paint-controls"},
                        "children": [
                            _leaf("repixelizer.workspace.swatch", "color.swatch", label="Paint", value="#ffffffff"),
                            _field("repixelizer.workspace.paint.r", "R", value="255"),
                            _field("repixelizer.workspace.paint.g", "G", value="255"),
                            _field("repixelizer.workspace.paint.b", "B", value="255"),
                            _field("repixelizer.workspace.paint.a", "A", value="255"),
                            _field("repixelizer.workspace.zoom", "Zoom", value="12x", field_kind="control.range", min=4, max=32),
                            _leaf("repixelizer.workspace.grid", "control.toggle", label="Grid", value=True),
                        ],
                    },
                    _leaf("repixelizer.workspace.editor.canvas", "canvas.editor", label="Editable output canvas", state="empty"),
                ],
            ),
            _pane(
                "repixelizer.workspace.sidebar",
                "Run Controls",
                [
                    _card(
                        "repixelizer.workspace.input",
                        "Input",
                        [
                            _text("repixelizer.workspace.input.copy", "Start with the ugly input."),
                            _leaf("repixelizer.workspace.file", "input.file", label="Choose File", accept="image/png,image/*"),
                            _leaf("repixelizer.workspace.dropzone", "dropzone", label="PNG, sprite scrap, cursed logo. Whatever started this."),
                        ],
                    ),
                    _card(
                        "repixelizer.workspace.controls",
                        "Controls",
                        [
                            _field("repixelizer.workspace.target.size", "Target Size", value="auto"),
                            _field("repixelizer.workspace.target.width", "Target Width", value="auto"),
                            _field("repixelizer.workspace.target.height", "Target Height", value="auto"),
                            _field("repixelizer.workspace.steps", "Steps", value=str(config.default_steps)),
                            _button("repixelizer.workspace.run", "Run The Machine", command_id="repixelizer.job.submit", route="/api/jobs", method="POST"),
                        ],
                    ),
                    _card(
                        "repixelizer.workspace.queue",
                        "Queue",
                        [
                            _leaf("repixelizer.workspace.queue.depth", "metric", label="Queue", value=queue_summary.get("queueDepth", 0), max=queue_summary.get("queueCapacity", config.queue_capacity)),
                            _leaf("repixelizer.workspace.queue.worker", "metric", label="Worker", value=1 if queue_summary.get("hasActiveJob") else 0, max=1),
                        ],
                    ),
                    _card(
                        "repixelizer.workspace.lattice",
                        "Lattice",
                        [_text("repixelizer.workspace.lattice.copy", "No lattice picked yet.")],
                    ),
                ],
            ),
        ],
    }


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
            _landing_gallery(base),
            _workspace_surface(config, queue_summary),
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
