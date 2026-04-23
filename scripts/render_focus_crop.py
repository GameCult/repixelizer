from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from repixelizer.analysis import analyze_phase_field_source
from repixelizer.inference import infer_lattice
from repixelizer.io import load_rgba
from repixelizer.metrics import source_lattice_consistency_breakdown
from repixelizer.params import SolverHyperParams
from repixelizer.phase_field import optimize_phase_field


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a source/initial/final crop sheet for a fixed output-grid region."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument(
        "--cell-bbox",
        required=True,
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Crop bounds in output-grid cell coordinates.",
    )
    parser.add_argument(
        "--panels",
        default="source,initial,final",
        help="Comma-separated panel list. Supported: source,initial,final. Legacy aliases: snap=initial, relaxed=final.",
    )
    parser.add_argument("--scale", type=int, default=16, help="Nearest-neighbor scale factor for each output cell.")
    parser.add_argument("--steps", type=int, default=48, help="Phase-field iteration budget.")
    parser.add_argument("--device", default="auto", help="Torch device, usually auto/cpu/cuda.")
    return parser.parse_args()


def _to_image(rgba: np.ndarray) -> Image.Image:
    rgba8 = np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgba8, mode="RGBA")


def _grid_panel(rgba: np.ndarray, label: str, *, x0: int, y0: int, x1: int, y1: int, scale: int) -> Image.Image:
    crop = rgba[y0:y1, x0:x1]
    base = _to_image(crop).resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (base.width, base.height + 20), (16, 16, 16, 255))
    canvas.alpha_composite(base, (0, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), label, fill=(255, 255, 255, 255))
    for gx in range(x1 - x0 + 1):
        px = gx * scale
        draw.line((px, 20, px, 20 + base.height), fill=(255, 255, 255, 55), width=1)
    for gy in range(y1 - y0 + 1):
        py = 20 + gy * scale
        draw.line((0, py, base.width, py), fill=(255, 255, 255, 55), width=1)
    return canvas


def _source_panel(source: np.ndarray, *, source_bbox: tuple[int, int, int, int], target_size: tuple[int, int], scale: int) -> Image.Image:
    sx0, sy0, sx1, sy1 = source_bbox
    source_crop = source[sy0:sy1, sx0:sx1]
    base = _to_image(source_crop).resize((target_size[0] * scale, target_size[1] * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (base.width, base.height + 20), (16, 16, 16, 255))
    canvas.alpha_composite(base, (0, 20))
    ImageDraw.Draw(canvas).text((4, 2), "Source crop", fill=(255, 255, 255, 255))
    return canvas


def _build_focus_states(
    input_path: str | Path,
    *,
    steps: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], dict[str, dict[str, float]]]:
    source = load_rgba(input_path)
    inference = infer_lattice(source, device=device)
    analysis = analyze_phase_field_source(source, seed=7, device=device)
    artifacts = optimize_phase_field(
        source,
        inference=inference,
        analysis=analysis,
        steps=steps,
        seed=7,
        device=device,
        solver_params=SolverHyperParams(),
    )

    height, width = source.shape[:2]
    cell_x = width / max(1, inference.target_width)
    cell_y = height / max(1, inference.target_height)
    fidelity = {
        "initial": source_lattice_consistency_breakdown(
            source,
            artifacts.initial_rgba,
            target_width=inference.target_width,
            target_height=inference.target_height,
            phase_x=inference.phase_x,
            phase_y=inference.phase_y,
        ),
        "final": source_lattice_consistency_breakdown(
            source,
            artifacts.target_rgba,
            target_width=inference.target_width,
            target_height=inference.target_height,
            phase_x=inference.phase_x,
            phase_y=inference.phase_y,
        ),
    }
    return source, artifacts.initial_rgba, artifacts.target_rgba, (cell_x, cell_y), fidelity


def main() -> None:
    args = _parse_args()
    x0, y0, x1, y1 = args.cell_bbox
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("Invalid --cell-bbox: expected X1 > X0 and Y1 > Y0.")

    source, initial_rgba, final_rgba, (cell_x, cell_y), fidelity = _build_focus_states(
        args.input,
        steps=args.steps,
        device=args.device,
    )

    normalized_panel_names = []
    for name in (token.strip().lower() for token in args.panels.split(",") if token.strip()):
        if name == "snap":
            normalized_panel_names.append("initial")
        elif name == "relaxed":
            normalized_panel_names.append("final")
        else:
            normalized_panel_names.append(name)

    panels: list[Image.Image] = []
    source_bbox = (
        int(np.floor(x0 * cell_x)),
        int(np.floor(y0 * cell_y)),
        int(np.ceil(x1 * cell_x)),
        int(np.ceil(y1 * cell_y)),
    )
    target_size = (x1 - x0, y1 - y0)
    panel_map = {
        "source": _source_panel(source, source_bbox=source_bbox, target_size=target_size, scale=args.scale),
        "initial": _grid_panel(initial_rgba, "Initial", x0=x0, y0=y0, x1=x1, y1=y1, scale=args.scale),
        "final": _grid_panel(final_rgba, "Final", x0=x0, y0=y0, x1=x1, y1=y1, scale=args.scale),
    }
    for name in normalized_panel_names:
        if name not in panel_map:
            raise SystemExit(f"Unsupported panel name: {name}")
        panels.append(panel_map[name])

    canvas = Image.new("RGBA", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), (16, 16, 16, 255))
    x = 0
    for panel in panels:
        canvas.alpha_composite(panel, (x, 0))
        x += panel.width

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(str(output_path))
    print(
        {
            "cell_bbox": [x0, y0, x1, y1],
            "source_bbox": list(source_bbox),
            "panels": normalized_panel_names,
            "scale": args.scale,
            "source_fidelity": fidelity,
        }
    )


if __name__ == "__main__":
    main()
