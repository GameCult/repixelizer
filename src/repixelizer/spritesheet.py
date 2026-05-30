from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import write_json
from .io import load_rgba, save_rgba
from .pipeline import run_pipeline_rgba
from .preprocess import strip_edge_background
from .types import RunResult


@dataclass(frozen=True, slots=True)
class SpriteRegion:
    index: int
    left: int
    top: int
    right: int
    bottom: int
    foreground_pixels: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(slots=True)
class SpriteRun:
    region: SpriteRegion
    result: RunResult


@dataclass(slots=True)
class SpritesheetResult:
    output_rgba: np.ndarray
    sprites: list[SpriteRun]
    diagnostics: dict[str, Any]


def run_spritesheet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sprite_count: int | None = None,
    target_size: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    palette_path: str | Path | None = None,
    palette_mode: str = "off",
    diagnostics_dir: str | Path | None = None,
    seed: int = 7,
    steps: int = 200,
    device: str = "auto",
    strip_background: bool = False,
    enable_candidate_rerank: bool = True,
) -> SpritesheetResult:
    source = load_rgba(input_path)
    detection_source = strip_edge_background(source) if strip_background else source.copy()
    regions = detect_sprite_regions(detection_source, sprite_count=sprite_count)
    if not regions:
        raise ValueError("No sprite regions were detected in the spritesheet.")

    diagnostics_path = Path(diagnostics_dir) if diagnostics_dir else None
    if diagnostics_path is not None:
        diagnostics_path.mkdir(parents=True, exist_ok=True)
        save_rgba(diagnostics_path / "detected-source.png", detection_source)

    sprite_runs: list[SpriteRun] = []
    for region in regions:
        crop = detection_source[region.top : region.bottom, region.left : region.right].copy()
        sprite_diagnostics = None
        if diagnostics_path is not None:
            sprite_diagnostics = diagnostics_path / f"sprite-{region.index:03d}"
        result = run_pipeline_rgba(
            crop,
            target_size=target_size,
            target_width=target_width,
            target_height=target_height,
            palette_path=palette_path,
            palette_mode=palette_mode,
            diagnostics_dir=sprite_diagnostics,
            seed=seed + region.index,
            steps=steps,
            device=device,
            strip_background=False,
            enable_candidate_rerank=enable_candidate_rerank,
        )
        sprite_runs.append(SpriteRun(region=region, result=result))

    output = _pack_sprite_outputs(sprite_runs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_rgba(output_path, output)

    diagnostics = _spritesheet_summary(
        source_shape=source.shape,
        sprite_count_request=sprite_count,
        sprite_runs=sprite_runs,
        output_shape=output.shape,
    )
    if diagnostics_path is not None:
        write_json(diagnostics_path / "spritesheet.json", diagnostics)
    return SpritesheetResult(output_rgba=output, sprites=sprite_runs, diagnostics=diagnostics)


def detect_sprite_regions(
    rgba: np.ndarray,
    *,
    sprite_count: int | None = None,
    padding: int = 2,
    min_area_ratio: float = 0.0025,
    max_auto_aspect: float = 5.0,
) -> list[SpriteRegion]:
    mask = _foreground_mask(rgba)
    components = _connected_components(mask)
    if not components:
        return []

    image_area = max(1, rgba.shape[0] * rgba.shape[1])
    min_area = max(8, int(round(image_area * min_area_ratio)))
    if sprite_count is None:
        selected = [
            component
            for component in components
            if component.foreground_pixels >= min_area and _region_aspect(component) <= max_auto_aspect
        ]
    else:
        if sprite_count <= 0:
            raise ValueError("sprite_count must be positive.")
        selected = sorted(components, key=lambda component: component.foreground_pixels, reverse=True)[:sprite_count]
        if len(selected) < sprite_count:
            raise ValueError(f"Detected only {len(selected)} sprite regions, but {sprite_count} were requested.")

    padded = [_pad_region(component, rgba.shape[1], rgba.shape[0], padding=padding) for component in selected]
    return _reading_order(padded)


def _foreground_mask(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[..., 3]
    if np.any(alpha < 0.95):
        return alpha > 0.05
    rgb = rgba[..., :3]
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background[None, None, :], axis=-1)
    simple_mask = distance > (24.0 / 255.0)
    foreground_fraction = float(np.mean(simple_mask))
    if 0.001 <= foreground_fraction <= 0.8:
        return simple_mask
    stripped = strip_edge_background(rgba)
    stripped_alpha = stripped[..., 3] > 0.05
    if np.any(stripped_alpha):
        return stripped_alpha
    return simple_mask


def _connected_components(mask: np.ndarray) -> list[SpriteRegion]:
    height, width = mask.shape
    parent: list[int] = []
    rank: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    previous_row: list[tuple[int, int, int]] = []

    def make_label() -> int:
        label = len(parent)
        parent.append(label)
        rank.append(0)
        return label

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for y in range(height):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            previous_row = []
            continue
        breaks = np.flatnonzero(np.diff(xs) > 1) + 1
        starts = np.concatenate([xs[:1], xs[breaks]])
        ends = np.concatenate([xs[breaks - 1], xs[-1:]])
        current_row: list[tuple[int, int, int]] = []
        prev_index = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            label = make_label()
            while prev_index < len(previous_row) and previous_row[prev_index][1] < start - 1:
                prev_index += 1
            check_index = prev_index
            while check_index < len(previous_row) and previous_row[check_index][0] <= end + 1:
                union(label, previous_row[check_index][2])
                check_index += 1
            current_row.append((int(start), int(end), label))
            runs.append((y, int(start), int(end), label))
        previous_row = current_row

    aggregates: dict[int, list[int]] = {}
    for y, start, end, label in runs:
        root = find(label)
        width = end - start + 1
        if root not in aggregates:
            aggregates[root] = [start, y, end, y, width]
            continue
        aggregate = aggregates[root]
        aggregate[0] = min(aggregate[0], start)
        aggregate[1] = min(aggregate[1], y)
        aggregate[2] = max(aggregate[2], end)
        aggregate[3] = max(aggregate[3], y)
        aggregate[4] += width

    return [
        SpriteRegion(
            index=0,
            left=left,
            top=top,
            right=right + 1,
            bottom=bottom + 1,
            foreground_pixels=count,
        )
        for left, top, right, bottom, count in aggregates.values()
    ]


def _pad_region(region: SpriteRegion, image_width: int, image_height: int, *, padding: int) -> SpriteRegion:
    return SpriteRegion(
        index=region.index,
        left=max(0, region.left - padding),
        top=max(0, region.top - padding),
        right=min(image_width, region.right + padding),
        bottom=min(image_height, region.bottom + padding),
        foreground_pixels=region.foreground_pixels,
    )


def _region_aspect(region: SpriteRegion) -> float:
    short_span = max(1, min(region.width, region.height))
    long_span = max(region.width, region.height)
    return long_span / short_span


def _reading_order(regions: list[SpriteRegion]) -> list[SpriteRegion]:
    if not regions:
        return []
    median_height = float(np.median([region.height for region in regions]))
    row_threshold = max(4.0, median_height * 0.5)
    rows: list[list[SpriteRegion]] = []
    for region in sorted(regions, key=lambda item: (item.top + item.bottom) * 0.5):
        center_y = (region.top + region.bottom) * 0.5
        for row in rows:
            row_center = np.mean([(item.top + item.bottom) * 0.5 for item in row])
            if abs(center_y - row_center) <= row_threshold:
                row.append(region)
                break
        else:
            rows.append([region])

    ordered: list[SpriteRegion] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item.left))
    return [
        SpriteRegion(
            index=index,
            left=region.left,
            top=region.top,
            right=region.right,
            bottom=region.bottom,
            foreground_pixels=region.foreground_pixels,
        )
        for index, region in enumerate(ordered)
    ]


def _pack_sprite_outputs(sprite_runs: list[SpriteRun], *, gutter: int = 1) -> np.ndarray:
    rows = _group_sprite_runs_by_source_row(sprite_runs)
    row_heights = [max(run.result.output_rgba.shape[0] for run in row) for row in rows]
    row_widths = [sum(run.result.output_rgba.shape[1] for run in row) + gutter * max(0, len(row) - 1) for row in rows]
    width = max(row_widths)
    height = sum(row_heights) + gutter * max(0, len(rows) - 1)
    output = np.zeros((height, width, 4), dtype=np.float32)

    y = 0
    for row, row_height in zip(rows, row_heights):
        x = 0
        for run in sorted(row, key=lambda item: item.region.left):
            sprite = run.result.output_rgba
            output[y : y + sprite.shape[0], x : x + sprite.shape[1]] = sprite
            x += sprite.shape[1] + gutter
        y += row_height + gutter
    return output


def _group_sprite_runs_by_source_row(sprite_runs: list[SpriteRun]) -> list[list[SpriteRun]]:
    if not sprite_runs:
        return []
    median_height = float(np.median([run.region.height for run in sprite_runs]))
    row_threshold = max(4.0, median_height * 0.5)
    rows: list[list[SpriteRun]] = []
    for run in sorted(sprite_runs, key=lambda item: (item.region.top + item.region.bottom) * 0.5):
        center_y = (run.region.top + run.region.bottom) * 0.5
        for row in rows:
            row_center = np.mean([(item.region.top + item.region.bottom) * 0.5 for item in row])
            if abs(center_y - row_center) <= row_threshold:
                row.append(run)
                break
        else:
            rows.append([run])
    return rows


def _spritesheet_summary(
    *,
    source_shape: tuple[int, ...],
    sprite_count_request: int | None,
    sprite_runs: list[SpriteRun],
    output_shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "mode": "spritesheet",
        "source_width": int(source_shape[1]),
        "source_height": int(source_shape[0]),
        "output_width": int(output_shape[1]),
        "output_height": int(output_shape[0]),
        "sprite_count_request": sprite_count_request,
        "sprite_count_detected": len(sprite_runs),
        "sprites": [
            {
                "index": run.region.index,
                "source_box": [run.region.left, run.region.top, run.region.right, run.region.bottom],
                "source_width": run.region.width,
                "source_height": run.region.height,
                "foreground_pixels": run.region.foreground_pixels,
                "target_width": run.result.inference.target_width,
                "target_height": run.result.inference.target_height,
                "confidence": run.result.inference.confidence,
            }
            for run in sprite_runs
        ],
    }
