from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from repixelizer.baselines import lanczos_resize_baseline
from repixelizer.io import load_rgba, nearest_resize, save_rgba
from repixelizer.pipeline import run_pipeline


@dataclass(slots=True)
class RenderedPanel:
    image: Image.Image
    content_box: tuple[int, int, int, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the README comparison sheet and guard closeup from repo fixtures.")
    parser.add_argument(
        "--vector-input",
        default="tests/fixtures/real/ai-badge-vector.png",
        help="High-resolution non-pixel badge source.",
    )
    parser.add_argument(
        "--ai-input",
        default="tests/fixtures/real/ai-badge-cleaned.png",
        help="Cleaned AI fake-pixel badge source.",
    )
    parser.add_argument(
        "--out-sheet",
        default="docs/readme-assets/badge-example-sheet.png",
        help="Output path for the full README comparison sheet.",
    )
    parser.add_argument(
        "--out-guard-crop",
        default="docs/readme-assets/guard-right-crop-comparison.png",
        help="Output path for the standalone sword-guard closeup strip.",
    )
    parser.add_argument(
        "--scratch-dir",
        default="artifacts/readme-build",
        help="Directory for intermediate low-res outputs and diagnostics.",
    )
    parser.add_argument(
        "--vector-target-size",
        type=int,
        default=128,
        help="Target size for the non-pixel source repixelization run.",
    )
    parser.add_argument(
        "--guard-cell-bbox",
        nargs=4,
        type=int,
        default=(73, 20, 107, 44),
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Sword-guard crop bounds in output-grid cell coordinates for the AI row.",
    )
    parser.add_argument("--steps", type=int, default=48, help="Optimizer step budget for both runs.")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"), help="Torch device for generation.")
    return parser.parse_args()


def _to_image(rgba: np.ndarray) -> Image.Image:
    rgba8 = np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba8, mode="RGBA")


def _checkerboard(width: int, height: int, tile: int = 12) -> Image.Image:
    image = Image.new("RGBA", (width, height), (236, 236, 236, 255))
    draw = ImageDraw.Draw(image)
    accent = (218, 218, 218, 255)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if ((x // tile) + (y // tile)) % 2:
                draw.rectangle((x, y, min(width, x + tile), min(height, y + tile)), fill=accent)
    return image


def _fit_rgba(
    rgba: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    pixelated: bool,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = _to_image(rgba)
    scale = min(target_width / max(1, image.width), target_height / max(1, image.height))
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resample = Image.Resampling.NEAREST if pixelated else Image.Resampling.LANCZOS
    resized = image.resize((resized_width, resized_height), resample=resample)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    return resized, (offset_x, offset_y, resized_width, resized_height)


def _render_panel(
    label: str,
    *,
    rgba: np.ndarray,
    pixelated: bool,
    panel_width: int,
    panel_height: int,
    subtitle: str | None = None,
) -> RenderedPanel:
    header_height = 38 if subtitle else 28
    gutter = 10
    canvas = Image.new("RGBA", (panel_width, panel_height), (18, 18, 20, 255))
    checker = _checkerboard(panel_width - gutter * 2, panel_height - header_height - gutter * 2)
    canvas.alpha_composite(checker, (gutter, header_height + gutter))

    fitted, (inner_x, inner_y, inner_w, inner_h) = _fit_rgba(
        rgba,
        target_width=checker.width,
        target_height=checker.height,
        pixelated=pixelated,
    )
    content_x = gutter + inner_x
    content_y = header_height + gutter + inner_y
    canvas.alpha_composite(fitted, (content_x, content_y))

    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 6), label, fill=(255, 255, 255, 255))
    if subtitle:
        draw.text((gutter, 22), subtitle, fill=(200, 200, 200, 255))
    draw.rounded_rectangle((0, 0, panel_width - 1, panel_height - 1), radius=10, outline=(72, 72, 80, 255), width=1)
    return RenderedPanel(canvas, (content_x, content_y, inner_w, inner_h))


def _draw_source_bbox(render: RenderedPanel, source_bbox: tuple[int, int, int, int], *, source_width: int, source_height: int) -> None:
    x0, y0, x1, y1 = source_bbox
    inner_x, inner_y, inner_w, inner_h = render.content_box
    left = inner_x + int(round(inner_w * (x0 / max(1, source_width))))
    right = inner_x + int(round(inner_w * (x1 / max(1, source_width))))
    top = inner_y + int(round(inner_h * (y0 / max(1, source_height))))
    bottom = inner_y + int(round(inner_h * (y1 / max(1, source_height))))
    draw = ImageDraw.Draw(render.image)
    for delta, color in ((0, (255, 245, 120, 255)), (1, (20, 20, 20, 255))):
        draw.rectangle((left - delta, top - delta, right + delta, bottom + delta), outline=color, width=1)


def _build_guard_strip(
    source_crop: np.ndarray,
    lanczos_crop: np.ndarray,
    repixelized_crop: np.ndarray,
    *,
    include_lanczos: bool = True,
    panel_width: int = 220,
    panel_height: int = 170,
    title: str | None = "Sword guard close-up",
) -> Image.Image:
    panels = [
        _render_panel("Source", rgba=source_crop, pixelated=True, panel_width=panel_width, panel_height=panel_height),
        _render_panel("Repixelized", rgba=repixelized_crop, pixelated=True, panel_width=panel_width, panel_height=panel_height),
    ]
    if include_lanczos:
        panels.insert(
            1,
            _render_panel("Lanczos", rgba=lanczos_crop, pixelated=True, panel_width=panel_width, panel_height=panel_height),
        )
    title_height = 24 if title else 0
    gap = 10
    width = sum(panel.image.width for panel in panels) + gap * (len(panels) + 1)
    height = title_height + max(panel.image.height for panel in panels) + gap * 2
    canvas = Image.new("RGBA", (width, height), (14, 14, 16, 242))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=12, fill=(14, 14, 16, 242), outline=(92, 92, 104, 255), width=1)
    if title:
        draw.text((gap, 4), title, fill=(255, 255, 255, 255))
    x = gap
    for panel in panels:
        canvas.alpha_composite(panel.image, (x, title_height + gap))
        x += panel.image.width + gap
    return canvas


def _build_pip_inset(
    rgba_crop: np.ndarray,
    *,
    width: int = 126,
    height: int = 94,
) -> Image.Image:
    fitted, (inner_x, inner_y, inner_w, inner_h) = _fit_rgba(
        rgba_crop,
        target_width=width - 12,
        target_height=height - 12,
        pixelated=True,
    )
    canvas = Image.new("RGBA", (width, height), (14, 14, 16, 236))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=10, fill=(14, 14, 16, 236), outline=(92, 92, 104, 255), width=1)
    checker = _checkerboard(width - 12, height - 12, tile=8)
    canvas.alpha_composite(checker, (6, 6))
    canvas.alpha_composite(fitted, (6 + inner_x, 6 + inner_y))
    return canvas


def _add_pip_inset(render: RenderedPanel, inset: Image.Image, *, margin_x: int = 12, margin_y: int = 12) -> None:
    inner_x, inner_y, inner_w, inner_h = render.content_box
    placement_x = inner_x + max(0, inner_w - inset.width - margin_x)
    placement_y = inner_y + max(0, inner_h - inset.height - margin_y)
    render.image.alpha_composite(inset, (placement_x, placement_y))


def _cell_bbox_to_source_bbox(
    cell_bbox: tuple[int, int, int, int],
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = cell_bbox
    return (
        int(np.floor(x0 * source_width / max(1, target_width))),
        int(np.floor(y0 * source_height / max(1, target_height))),
        int(np.ceil(x1 * source_width / max(1, target_width))),
        int(np.ceil(y1 * source_height / max(1, target_height))),
    )


def _crop_rgba(rgba: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return rgba[y0:y1, x0:x1]


def _save_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    vector_input = Path(args.vector_input)
    ai_input = Path(args.ai_input)
    out_sheet = Path(args.out_sheet)
    out_guard_crop = Path(args.out_guard_crop)
    out_sheet.parent.mkdir(parents=True, exist_ok=True)
    out_guard_crop.parent.mkdir(parents=True, exist_ok=True)

    vector_result = run_pipeline(
        vector_input,
        scratch_dir / "badge-vector-repixelized.png",
        target_size=args.vector_target_size,
        diagnostics_dir=scratch_dir / "vector-diag",
        steps=args.steps,
        device=args.device,
    )
    ai_result = run_pipeline(
        ai_input,
        scratch_dir / "badge-ai-repixelized.png",
        diagnostics_dir=scratch_dir / "ai-diag",
        steps=args.steps,
        device=args.device,
    )

    vector_source = load_rgba(vector_input)
    ai_source = load_rgba(ai_input)
    vector_lanczos = lanczos_resize_baseline(vector_source, width=args.vector_target_size, height=args.vector_target_size)
    ai_lanczos = lanczos_resize_baseline(
        ai_source,
        width=ai_result.inference.target_width,
        height=ai_result.inference.target_height,
    )

    save_rgba(scratch_dir / "badge-vector-lanczos.png", vector_lanczos)
    save_rgba(scratch_dir / "badge-ai-lanczos.png", ai_lanczos)

    panel_width = 320
    panel_height = 360
    row_gap = 18
    col_gap = 18
    margin = 24
    row_title_height = 22

    vector_panels = [
        _render_panel(
            "Source",
            rgba=vector_source,
            pixelated=False,
            panel_width=panel_width,
            panel_height=panel_height,
            subtitle="Glossy badge art",
        ),
        _render_panel(
            "Lanczos",
            rgba=vector_lanczos,
            pixelated=True,
            panel_width=panel_width,
            panel_height=panel_height,
            subtitle=f"{args.vector_target_size}x{args.vector_target_size}",
        ),
        _render_panel(
            "Repixelized",
            rgba=vector_result.output_rgba,
            pixelated=True,
            panel_width=panel_width,
            panel_height=panel_height,
            subtitle=f"{vector_result.inference.target_width}x{vector_result.inference.target_height}",
        ),
    ]
    ai_panels = [
        _render_panel(
            "Source",
            rgba=ai_source,
            pixelated=False,
            panel_width=panel_width,
            panel_height=panel_height,
            subtitle="Cleaned AI fake pixel art",
        ),
        _render_panel(
            "Lanczos",
            rgba=ai_lanczos,
            pixelated=True,
            panel_width=panel_width,
            panel_height=panel_height,
            subtitle=f"{ai_result.inference.target_width}x{ai_result.inference.target_height}",
        ),
        _render_panel(
            "Repixelized",
            rgba=ai_result.output_rgba,
            pixelated=True,
            panel_width=panel_width,
            panel_height=panel_height,
            subtitle=f"{ai_result.inference.target_width}x{ai_result.inference.target_height} auto",
        ),
    ]

    guard_bbox = _cell_bbox_to_source_bbox(
        tuple(args.guard_cell_bbox),
        source_width=ai_source.shape[1],
        source_height=ai_source.shape[0],
        target_width=ai_result.inference.target_width,
        target_height=ai_result.inference.target_height,
    )
    vector_guard_bbox = _cell_bbox_to_source_bbox(
        tuple(args.guard_cell_bbox),
        source_width=vector_source.shape[1],
        source_height=vector_source.shape[0],
        target_width=args.vector_target_size,
        target_height=args.vector_target_size,
    )
    vector_lanczos_preview = nearest_resize(vector_lanczos, width=vector_source.shape[1], height=vector_source.shape[0])
    vector_repixelized_preview = nearest_resize(
        vector_result.output_rgba,
        width=vector_source.shape[1],
        height=vector_source.shape[0],
    )
    for panel in vector_panels:
        _draw_source_bbox(
            panel,
            vector_guard_bbox,
            source_width=vector_source.shape[1],
            source_height=vector_source.shape[0],
        )
    vector_crops = [
        _crop_rgba(vector_source, vector_guard_bbox),
        _crop_rgba(vector_lanczos_preview, vector_guard_bbox),
        _crop_rgba(vector_repixelized_preview, vector_guard_bbox),
    ]
    for panel, crop in zip(vector_panels, vector_crops, strict=True):
        _add_pip_inset(panel, _build_pip_inset(crop))

    ai_lanczos_preview = nearest_resize(ai_lanczos, width=ai_source.shape[1], height=ai_source.shape[0])
    ai_repixelized_preview = nearest_resize(ai_result.output_rgba, width=ai_source.shape[1], height=ai_source.shape[0])
    for panel in ai_panels:
        _draw_source_bbox(
            panel,
            guard_bbox,
            source_width=ai_source.shape[1],
            source_height=ai_source.shape[0],
        )

    guard_strip = _build_guard_strip(
        _crop_rgba(ai_source, guard_bbox),
        _crop_rgba(ai_lanczos_preview, guard_bbox),
        _crop_rgba(ai_repixelized_preview, guard_bbox),
        title=None,
    )
    guard_strip.save(out_guard_crop)
    ai_crops = [
        _crop_rgba(ai_source, guard_bbox),
        _crop_rgba(ai_lanczos_preview, guard_bbox),
        _crop_rgba(ai_repixelized_preview, guard_bbox),
    ]
    for panel, crop in zip(ai_panels, ai_crops, strict=True):
        _add_pip_inset(panel, _build_pip_inset(crop))

    width = margin * 2 + panel_width * 3 + col_gap * 2
    height = margin * 2 + row_title_height * 2 + panel_height * 2 + row_gap
    canvas = Image.new("RGBA", (width, height), (11, 11, 13, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin - 2), "Non-pixel source -> 128x128", fill=(255, 255, 255, 255))
    draw.text((margin, margin + row_title_height + panel_height + row_gap - 2), "AI fake pixel art -> auto lattice", fill=(255, 255, 255, 255))

    row1_y = margin + row_title_height
    row2_y = row1_y + panel_height + row_gap + row_title_height
    x = margin
    for panel in vector_panels:
        canvas.alpha_composite(panel.image, (x, row1_y))
        x += panel_width + col_gap
    x = margin
    for panel in ai_panels:
        canvas.alpha_composite(panel.image, (x, row2_y))
        x += panel_width + col_gap

    out_sheet.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_sheet)

    _save_summary(
        scratch_dir / "readme-preview-summary.json",
        {
            "vector_input": str(vector_input),
            "ai_input": str(ai_input),
            "vector_target_size": args.vector_target_size,
            "ai_inferred_width": ai_result.inference.target_width,
            "ai_inferred_height": ai_result.inference.target_height,
            "guard_cell_bbox": list(args.guard_cell_bbox),
            "vector_guard_source_bbox": list(vector_guard_bbox),
            "guard_source_bbox": list(guard_bbox),
            "out_sheet": str(out_sheet),
            "out_guard_crop": str(out_guard_crop),
            "scratch_dir": str(scratch_dir),
        },
    )

    print(str(out_sheet))
    print(str(out_guard_crop))


if __name__ == "__main__":
    main()
