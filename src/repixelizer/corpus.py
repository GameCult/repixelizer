from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from .io import load_rgba, save_rgba

FRAME_SHEETS: dict[str, int] = {
    "mage-1-85x94.png": 0,
    "mage-2-122x110.png": 0,
    "mage-3-87x110.png": 0,
    "minion-45x66.png": 0,
    "andromalius-57x88.png": 0,
    "disciple-45x51.png": 0,
    "gnu-120x100.png": 0,
    "shadow-80x70.png": -1,
}

MULTI_CHARACTER_SHEETS: tuple[str, ...] = (
    "9RPGenemies.PNG",
    "DAGRONS5.png",
    "more rpg enemies.PNG",
    "threeformsPJ2.png",
)

MIN_COMPONENT_PIXELS = 100

SOURCE_ATTRIBUTION: dict[str, dict[str, Any]] = {
    "9RPGenemies.PNG": {
        "source_title": "More RPG enemies!",
        "source_authors": ["Stephen Challener (Redshrike)"],
        "source_url": "https://opengameart.org/content/more-rpg-enemies",
        "licenses": ["CC-BY 3.0", "CC-BY-SA 3.0", "OGA-BY 3.0"],
        "attribution_text": "Stephen Challener (Redshrike), hosted by OpenGameArt.org",
        "attribution_instructions": "Credit Stephen Challener (Redshrike) and include a link back to OpenGameArt.org.",
        "attribution_confidence": "exact",
    },
    "DAGRONS5.png": {
        "source_title": "RPG Enemies: 11 Dragons",
        "source_authors": ["Stephen Challener (Redshrike)", "MrBeast", "Surt", "Blarumyrran", "Sharm", "Zabin"],
        "source_url": "https://opengameart.org/content/rpg-enemies-11-dragons",
        "licenses": ["CC-BY 3.0"],
        "attribution_text": "Credit to: Stephen \"Redshrike\" Challener, MrBeast, Surt, Blarumyrran, Sharm, Zabin",
        "attribution_instructions": "Credit the listed contributors and include a link to the source page.",
        "attribution_confidence": "exact",
    },
    "more rpg enemies.PNG": {
        "source_title": "6 More RPG Enemies",
        "source_authors": ["Stephen Challener (Redshrike)", "Blarumyrran", "LordNeo"],
        "source_url": "https://opengameart.org/content/6-more-rpg-enemies",
        "licenses": ["CC-BY 3.0", "OGA-BY 3.0"],
        "attribution_text": "Stephen Challener (Redshrike), Blarumyrran and LordNeo, hosted by OpenGameArt.org",
        "attribution_instructions": "Credit Stephen Challener (Redshrike), Blarumyrran, and LordNeo, and include a link back to OpenGameArt.org.",
        "attribution_confidence": "exact",
    },
    "threeformsPJ2.png": {
        "source_title": "3-form RPG boss: \"Harlequin Epicycle.\"",
        "source_authors": ["Stephen Challener (Redshrike)"],
        "source_url": "https://opengameart.org/content/3-form-rpg-boss-harlequin-epicycle",
        "licenses": ["CC-BY 3.0", "GPL 3.0", "GPL 2.0", "OGA-BY 3.0"],
        "attribution_text": "Stephen Challener (Redshrike), hosted by OpenGameArt.org",
        "attribution_instructions": "Credit Stephen Challener (Redshrike) and include a link back to OpenGameArt.org.",
        "attribution_confidence": "exact",
    },
    "mage-1-85x94.png": {
        "source_title": "Bosses and monsters spritesheets (Ars Notoria)",
        "source_authors": ["Balmer", "Stephen Challener (Redshrike)"],
        "source_url": "https://opengameart.org/content/bosses-and-monsters-spritesheets-ars-notoria",
        "licenses": ["CC-BY 3.0"],
        "attribution_text": "Balmer; attribute Stephen Challener (Redshrike) for the original sprites and link back to OpenGameArt.org",
        "attribution_instructions": "Best-effort recovery: credit Balmer and Stephen Challener (Redshrike), and include a link to the OpenGameArt source page.",
        "attribution_confidence": "inferred",
    },
    "mage-2-122x110.png": {
        "source_title": "Bosses and monsters spritesheets (Ars Notoria)",
        "source_authors": ["Balmer", "Stephen Challener (Redshrike)"],
        "source_url": "https://opengameart.org/content/bosses-and-monsters-spritesheets-ars-notoria",
        "licenses": ["CC-BY 3.0"],
        "attribution_text": "Balmer; attribute Stephen Challener (Redshrike) for the original sprites and link back to OpenGameArt.org",
        "attribution_instructions": "Best-effort recovery: credit Balmer and Stephen Challener (Redshrike), and include a link to the OpenGameArt source page.",
        "attribution_confidence": "inferred",
    },
    "mage-3-87x110.png": {
        "source_title": "Bosses and monsters spritesheets (Ars Notoria)",
        "source_authors": ["Balmer", "Stephen Challener (Redshrike)"],
        "source_url": "https://opengameart.org/content/bosses-and-monsters-spritesheets-ars-notoria",
        "licenses": ["CC-BY 3.0"],
        "attribution_text": "Balmer; attribute Stephen Challener (Redshrike) for the original sprites and link back to OpenGameArt.org",
        "attribution_instructions": "Best-effort recovery: credit Balmer and Stephen Challener (Redshrike), and include a link to the OpenGameArt source page.",
        "attribution_confidence": "inferred",
    },
    "minion-45x66.png": {
        "source_title": "Bosses and monsters spritesheets (Ars Notoria)",
        "source_authors": ["Balmer", "Stephen Challener (Redshrike)"],
        "source_url": "https://opengameart.org/content/bosses-and-monsters-spritesheets-ars-notoria",
        "licenses": ["CC-BY 3.0"],
        "attribution_text": "Balmer; attribute Stephen Challener (Redshrike) for the original sprites and link back to OpenGameArt.org",
        "attribution_instructions": "Best-effort recovery: credit Balmer and Stephen Challener (Redshrike), and include a link to the OpenGameArt source page.",
        "attribution_confidence": "inferred",
    },
    "andromalius-57x88.png": {
        "source_title": "Bosses and monsters spritesheets (Ars Notoria)",
        "source_authors": ["Balmer", "Stephen Challener (Redshrike)"],
        "source_url": "https://opengameart.org/content/bosses-and-monsters-spritesheets-ars-notoria",
        "licenses": ["CC-BY 3.0"],
        "attribution_text": "Balmer; attribute Stephen Challener (Redshrike) for the original sprites and link back to OpenGameArt.org",
        "attribution_instructions": "Best-effort recovery: credit Balmer and Stephen Challener (Redshrike), and include a link to the OpenGameArt source page.",
        "attribution_confidence": "inferred",
    },
    "disciple-45x51.png": {
        "source_title": "Bosses and monsters spritesheets (Ars Notoria)",
        "source_authors": ["Balmer", "Stephen Challener (Redshrike)"],
        "source_url": "https://opengameart.org/content/bosses-and-monsters-spritesheets-ars-notoria",
        "licenses": ["CC-BY 3.0"],
        "attribution_text": "Balmer; attribute Stephen Challener (Redshrike) for the original sprites and link back to OpenGameArt.org",
        "attribution_instructions": "Best-effort recovery: credit Balmer and Stephen Challener (Redshrike), and include a link to the OpenGameArt source page.",
        "attribution_confidence": "inferred",
    },
    "gnu-120x100.png": {
        "source_title": "Bosses and monsters spritesheets (Ars Notoria)",
        "source_authors": ["Balmer", "Stephen Challener (Redshrike)", "Ben \"Cookiez\" Potter"],
        "source_url": "https://opengameart.org/content/bosses-and-monsters-spritesheets-ars-notoria",
        "licenses": ["CC-BY-SA 3.0", "GPL 3.0"],
        "attribution_text": "Balmer; also credit Stephen Challener (Redshrike) and Ben \"Cookiez\" Potter, with links back to OpenGameArt.org",
        "attribution_instructions": "Recovered from the Ars Notoria sheet plus the original Gnu Mage provenance. Use CC-BY-SA 3.0 or GPL 3.0 terms.",
        "attribution_confidence": "inferred",
        "original_source_title": "LPC in battle RPG sprites",
        "original_source_url": "https://opengameart.org/content/lpc-in-battle-rpg-sprites",
        "license_note": "Redshrike later clarified in the Ars Notoria page comments that the Gnu Man should inherit CC-BY-SA terms from its original source.",
    },
    "shadow-80x70.png": {
        "source_title": "Bosses and monsters spritesheets (Ars Notoria)",
        "source_authors": ["Balmer", "Stephen Challener (Redshrike)", "Clint Bellanger"],
        "source_url": "https://opengameart.org/content/bosses-and-monsters-spritesheets-ars-notoria",
        "licenses": ["CC-BY-SA 3.0"],
        "attribution_text": "Balmer; also credit Stephen Challener (Redshrike) and Clint Bellanger, with links back to OpenGameArt.org",
        "attribution_instructions": "Recovered from the Ars Notoria sheet plus the original shadow source provenance. Use CC-BY-SA 3.0 terms.",
        "attribution_confidence": "inferred",
        "original_source_title": "First Person Dungeon Crawl Enemies Remixed",
        "original_source_url": "https://opengameart.org/content/first-person-dungeon-crawl-enemies-remixed",
        "license_note": "Redshrike later clarified in the Ars Notoria page comments that the Shadow Soul should inherit CC-BY-SA terms from its original source.",
    },
}


def prepare_corpus(
    corpus_dir: str | Path,
    *,
    archive_dir: str | Path | None = None,
) -> dict[str, Any]:
    corpus_path = Path(corpus_dir)
    originals_dir = corpus_path / "originals"
    source_sheets_dir = Path(archive_dir) if archive_dir is not None else corpus_path / "source-sheets"
    originals_dir.mkdir(parents=True, exist_ok=True)
    source_sheets_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []
    for source_name, frame_index in FRAME_SHEETS.items():
        source_path = _resolve_source_path(source_name, originals_dir, source_sheets_dir)
        if source_path is None:
            continue
        rgba = load_rgba(source_path)
        frame_width, frame_height = _parse_frame_size(source_path.name)
        frame = _extract_frame(rgba, frame_width=frame_width, frame_height=frame_height, frame_index=frame_index)
        output_name = f"{_strip_size_suffix(source_path.stem)}.png"
        output_path = originals_dir / output_name
        save_rgba(output_path, frame)
        sidecar_path = output_path.with_suffix(".json")
        sidecar_payload = _merge_sidecar(
            sidecar_path,
            {
                "source_sheet": source_path.name,
                "kind": "animation-frame",
                "frame_index": _normalize_frame_index(frame_index, rgba.shape[1] // frame_width, rgba.shape[0] // frame_height),
                "frame_width": frame_width,
                "frame_height": frame_height,
                **_source_attribution(source_path.name),
            },
        )
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2),
            encoding="utf-8",
        )
        _archive_source_sheet(source_path, originals_dir, source_sheets_dir)
        outputs.append({"source": source_path.name, "output": output_name, "kind": "animation-frame"})

    for source_name in MULTI_CHARACTER_SHEETS:
        source_path = _resolve_source_path(source_name, originals_dir, source_sheets_dir)
        if source_path is None:
            continue
        rgba = load_rgba(source_path)
        sprites = _split_multi_character_sheet(rgba)
        stem = _sanitize_name(source_path.stem)
        for index, sprite in enumerate(sprites, start=1):
            output_name = f"{stem}-{index:02d}.png"
            output_path = originals_dir / output_name
            save_rgba(output_path, sprite["rgba"])
            sidecar_path = output_path.with_suffix(".json")
            sidecar_payload = _merge_sidecar(
                sidecar_path,
                {
                    "source_sheet": source_path.name,
                    "kind": "multi-character-crop",
                    "component_index": index,
                    "bbox": {
                        "left": sprite["bbox"][0],
                        "top": sprite["bbox"][1],
                        "right": sprite["bbox"][2],
                        "bottom": sprite["bbox"][3],
                    },
                    **_source_attribution(source_path.name),
                },
            )
            sidecar_path.write_text(
                json.dumps(sidecar_payload, indent=2),
                encoding="utf-8",
            )
            outputs.append({"source": source_path.name, "output": output_name, "kind": "multi-character-crop"})
        _archive_source_sheet(source_path, originals_dir, source_sheets_dir)

    summary = {
        "corpus_dir": str(corpus_path),
        "archive_dir": str(source_sheets_dir),
        "outputs": outputs,
        "output_count": len(outputs),
    }
    (corpus_path / "prepare-corpus.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_attribution_markdown(corpus_path)
    return summary


def write_attribution_markdown(corpus_dir: str | Path) -> Path:
    corpus_path = Path(corpus_dir)
    originals_dir = corpus_path / "originals"
    entries = _load_processed_entries(originals_dir)
    attribution_path = corpus_path / "ATTRIBUTION.md"
    attribution_path.write_text(_render_attribution_markdown(entries), encoding="utf-8")
    return attribution_path


def _resolve_source_path(source_name: str, originals_dir: Path, source_sheets_dir: Path) -> Path | None:
    original_path = originals_dir / source_name
    if original_path.exists():
        return original_path
    archived_path = source_sheets_dir / source_name
    if archived_path.exists():
        return archived_path
    return None


def _source_attribution(source_name: str) -> dict[str, Any]:
    return dict(SOURCE_ATTRIBUTION.get(source_name, {}))


def _merge_sidecar(sidecar_path: Path, generated: dict[str, Any]) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    if sidecar_path.exists():
        existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
    merged = dict(existing)
    merged.update(generated)
    return merged


def _load_processed_entries(originals_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for sidecar_path in sorted(originals_dir.glob("*.json")):
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if "source_sheet" not in payload:
            continue
        payload["derived_file"] = f"{sidecar_path.stem}.png"
        entries.append(payload)
    return entries


def _render_attribution_markdown(entries: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in entries:
        key = (
            entry.get("source_title", ""),
            entry.get("source_url", ""),
            tuple(entry.get("source_authors", [])),
            tuple(entry.get("licenses", [])),
            entry.get("attribution_confidence", ""),
            entry.get("attribution_text", ""),
            entry.get("attribution_instructions", ""),
            entry.get("original_source_title", ""),
            entry.get("original_source_url", ""),
            entry.get("license_note", ""),
        )
        grouped.setdefault(key, []).append(entry)

    lines = [
        "# Corpus Attribution",
        "",
        "Generated by `repixelize prepare-corpus`.",
        "",
        "This file summarizes attribution metadata for processed sprites in `examples/corpus/originals`.",
        "Entries marked `inferred` are best-effort recoveries and should be reviewed before distribution.",
        "",
    ]
    sort_keys = sorted(
        grouped,
        key=lambda key: (
            str(key[0]),
            str(key[1]),
            min(str(entry["source_sheet"]) for entry in grouped[key]),
        ),
    )
    for key in sort_keys:
        source_entries = sorted(grouped[key], key=lambda item: (str(item["source_sheet"]), str(item["derived_file"])))
        sample = source_entries[0]
        title = sample.get("source_title", str(sample["source_sheet"]))
        lines.append(f"## {title}")
        lines.append("")
        source_sheets = sorted({str(entry["source_sheet"]) for entry in source_entries})
        if len(source_sheets) == 1:
            lines.append(f"- Source sheet: `{source_sheets[0]}`")
        else:
            lines.append(f"- Source sheets: {', '.join(f'`{sheet}`' for sheet in source_sheets)}")
        if sample.get("source_url"):
            lines.append(f"- Source URL: {sample['source_url']}")
        if sample.get("source_authors"):
            lines.append(f"- Authors: {', '.join(sample['source_authors'])}")
        if sample.get("licenses"):
            lines.append(f"- Licenses: {', '.join(sample['licenses'])}")
        if sample.get("attribution_confidence"):
            lines.append(f"- Attribution confidence: {sample['attribution_confidence']}")
        if sample.get("attribution_text"):
            lines.append(f"- Attribution text: {sample['attribution_text']}")
        if sample.get("attribution_instructions"):
            lines.append(f"- Notes: {sample['attribution_instructions']}")
        if sample.get("original_source_title"):
            original = sample["original_source_title"]
            if sample.get("original_source_url"):
                original = f"{original} ({sample['original_source_url']})"
            lines.append(f"- Original upstream source: {original}")
        if sample.get("license_note"):
            lines.append(f"- License note: {sample['license_note']}")
        lines.append("- Derived files:")
        for entry in source_entries:
            lines.append(f"  - `{entry['derived_file']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _archive_source_sheet(source_path: Path, originals_dir: Path, source_sheets_dir: Path) -> None:
    if source_path.parent != originals_dir:
        return
    destination = source_sheets_dir / source_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.move(str(source_path), str(destination))


def _parse_frame_size(name: str) -> tuple[int, int]:
    match = re.search(r"-(\d+)x(\d+)\.[^.]+$", name, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"Could not parse frame size from {name}.")
    return int(match.group(1)), int(match.group(2))


def _strip_size_suffix(stem: str) -> str:
    return re.sub(r"-\d+x\d+$", "", stem)


def _sanitize_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", text)


def _normalize_frame_index(frame_index: int, columns: int, rows: int) -> int:
    total = columns * rows
    return total + frame_index if frame_index < 0 else frame_index


def _extract_frame(rgba: np.ndarray, *, frame_width: int, frame_height: int, frame_index: int) -> np.ndarray:
    height, width = rgba.shape[:2]
    if width % frame_width != 0 or height % frame_height != 0:
        raise RuntimeError("Sheet dimensions do not divide evenly into animation frames.")
    columns = width // frame_width
    rows = height // frame_height
    index = _normalize_frame_index(frame_index, columns, rows)
    if index < 0 or index >= columns * rows:
        raise RuntimeError("Frame index is out of range for the sprite sheet.")
    row = index // columns
    column = index % columns
    cell = rgba[row * frame_height : (row + 1) * frame_height, column * frame_width : (column + 1) * frame_width].copy()
    return _remove_edge_connected_background(cell)


def _split_multi_character_sheet(rgba: np.ndarray) -> list[dict[str, Any]]:
    transparent = _remove_edge_connected_background(rgba)
    alpha = transparent[..., 3] > 0.0
    components = _connected_components(alpha)
    sprites: list[dict[str, Any]] = []
    for component in components:
        if component["pixels"] < MIN_COMPONENT_PIXELS:
            continue
        left, top, right, bottom = component["bbox"]
        crop = _crop_with_padding(transparent, left, top, right, bottom, padding=1)
        sprites.append({"rgba": crop, "bbox": [left, top, right, bottom]})
    sprites.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return sprites


def _crop_with_padding(rgba: np.ndarray, left: int, top: int, right: int, bottom: int, *, padding: int) -> np.ndarray:
    height, width = rgba.shape[:2]
    x0 = max(0, left - padding)
    y0 = max(0, top - padding)
    x1 = min(width, right + padding)
    y1 = min(height, bottom + padding)
    return rgba[y0:y1, x0:x1].copy()


def _remove_edge_connected_background(rgba: np.ndarray) -> np.ndarray:
    image = rgba.copy()
    bg_rgb = _background_color(image)
    bg_match = np.all(np.abs(image[..., :3] - bg_rgb[None, None, :]) < 0.5 / 255.0, axis=-1)
    edge_connected = _edge_connected_mask(bg_match)
    image[edge_connected, 3] = 0.0
    image[edge_connected, :3] = 0.0
    return image


def _background_color(rgba: np.ndarray) -> np.ndarray:
    height, width = rgba.shape[:2]
    corners = [
        tuple(np.rint(rgba[0, 0, :3] * 255.0).astype(int)),
        tuple(np.rint(rgba[0, width - 1, :3] * 255.0).astype(int)),
        tuple(np.rint(rgba[height - 1, 0, :3] * 255.0).astype(int)),
        tuple(np.rint(rgba[height - 1, width - 1, :3] * 255.0).astype(int)),
    ]
    bg = Counter(corners).most_common(1)[0][0]
    return np.asarray(bg, dtype=np.float32) / 255.0


def _edge_connected_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if mask[0, x]:
            queue.append((0, x))
            visited[0, x] = True
        if mask[height - 1, x] and not visited[height - 1, x]:
            queue.append((height - 1, x))
            visited[height - 1, x] = True
    for y in range(height):
        if mask[y, 0] and not visited[y, 0]:
            queue.append((y, 0))
            visited[y, 0] = True
        if mask[y, width - 1] and not visited[y, width - 1]:
            queue.append((y, width - 1))
            visited[y, width - 1] = True
    while queue:
        y, x = queue.popleft()
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    return visited


def _connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            pixels = 0
            left = x
            right = x + 1
            top = y
            bottom = y + 1
            while queue:
                cy, cx = queue.popleft()
                pixels += 1
                left = min(left, cx)
                right = max(right, cx + 1)
                top = min(top, cy)
                bottom = max(bottom, cy + 1)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            components.append({"pixels": pixels, "bbox": [left, top, right, bottom]})
    return components
