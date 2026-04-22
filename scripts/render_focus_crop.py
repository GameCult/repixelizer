from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from repixelizer.analysis import analyze_source
from repixelizer.continuous import (
    _build_candidate_positions,
    _discrete_refine_output,
    _build_source_reliability,
    _cluster_boundary_map,
    _edge_gradient_maps,
    _make_patch_offsets,
    _make_regular_uv,
    _relax_candidate_selection,
    _reference_match_energy,
    _representative_colors,
    _require_torch,
    _resolve_device,
    _sample_cell_patches,
    _select_colors,
    _snap_output_to_source_pixels,
    premultiply,
)
from repixelizer.inference import infer_lattice
from repixelizer.io import load_rgba
from repixelizer.metrics import source_lattice_consistency_breakdown
from repixelizer.params import SolverHyperParams
from repixelizer.source_reference import build_source_lattice_reference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a source/snap/relaxed/final crop sheet for a fixed output-grid region."
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
        default="source,snap,relaxed,final",
        help="Comma-separated panel list. Supported: source,snap,relaxed,final",
    )
    parser.add_argument("--scale", type=int, default=16, help="Nearest-neighbor scale factor for each output cell.")
    parser.add_argument("--steps", type=int, default=64, help="Discrete refine iteration budget.")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float], dict[str, dict[str, float]]]:
    source = load_rgba(input_path)
    solver_params = SolverHyperParams()
    inference = infer_lattice(source, None, device=device)
    analysis = analyze_source(source, seed=7)

    torch, F = _require_torch()
    resolved_device = _resolve_device(torch, device)

    premul = premultiply(source)
    height, width = premul.shape[:2]
    cell_x = width / max(1, inference.target_width)
    cell_y = height / max(1, inference.target_height)
    source_t = torch.from_numpy(premul.transpose(2, 0, 1)[None, ...]).to(device=resolved_device, dtype=torch.float32)
    uv0 = _make_regular_uv(
        height=height,
        width=width,
        target_height=inference.target_height,
        target_width=inference.target_width,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
    )
    uv0_t = torch.from_numpy(uv0[None, ...]).to(device=resolved_device, dtype=torch.float32)
    offsets_t = torch.from_numpy(
        _make_patch_offsets(height=height, width=width, target_height=inference.target_height, target_width=inference.target_width)
    ).to(device=resolved_device, dtype=torch.float32)
    edge = np.maximum(analysis.edge_map, _cluster_boundary_map(analysis.cluster_map))
    edge_grad_x, edge_grad_y = _edge_gradient_maps(edge)
    source_lattice_reference = build_source_lattice_reference(
        source,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
        alpha_threshold=solver_params.alpha_transparent_threshold,
        edge_hint=edge,
        edge_grad_x_hint=edge_grad_x,
        edge_grad_y_hint=edge_grad_y,
    )
    initial_patches = _sample_cell_patches(F, source_t, uv0_t, offsets_t)
    representative_t, _ = _representative_colors(initial_patches, solver_params)
    representative_t = representative_t.detach()
    source_reference_t = torch.from_numpy(premultiply(source_lattice_reference.sharp_rgba)[None, ...]).to(
        device=resolved_device,
        dtype=torch.float32,
    )
    source_reliability_t = torch.from_numpy(_build_source_reliability(source_lattice_reference, solver_params)[None, ...]).to(
        device=resolved_device,
        dtype=torch.float32,
    )
    source_delta_x_t = (
        torch.from_numpy(source_lattice_reference.delta_x[None, ...]).to(device=resolved_device, dtype=torch.float32)
        if source_lattice_reference.delta_x is not None
        else None
    )
    source_delta_y_t = (
        torch.from_numpy(source_lattice_reference.delta_y[None, ...]).to(device=resolved_device, dtype=torch.float32)
        if source_lattice_reference.delta_y is not None
        else None
    )
    source_delta_diag_t = (
        torch.from_numpy(source_lattice_reference.delta_diag[None, ...]).to(device=resolved_device, dtype=torch.float32)
        if source_lattice_reference.delta_diag is not None
        else None
    )
    source_delta_anti_t = (
        torch.from_numpy(source_lattice_reference.delta_anti[None, ...]).to(device=resolved_device, dtype=torch.float32)
        if source_lattice_reference.delta_anti is not None
        else None
    )

    snap_rgba = _snap_output_to_source_pixels(
        torch,
        source_t,
        uv0_t,
        representative_t,
        source_reference_t,
        source_reliability_t,
        source_lattice_reference,
        solver_params,
        cell_x=cell_x,
        cell_y=cell_y,
    )
    snap_t = torch.from_numpy(premultiply(snap_rgba).transpose(2, 0, 1)[None, ...]).to(
        device=resolved_device,
        dtype=torch.float32,
    )

    source_hw = source_t[0].permute(1, 2, 0)
    uv = uv0_t[0]
    representative = representative_t[0]
    anchor = snap_t[0].permute(1, 2, 0)
    output_height = representative.shape[0]
    output_width = representative.shape[1]

    candidate_levels = max(3, int(solver_params.refine_candidate_levels))
    if candidate_levels % 2 == 0:
        candidate_levels += 1
    candidate_extent = max(0.05, float(solver_params.refine_candidate_extent))
    fraction_values = np.linspace(-candidate_extent, candidate_extent, candidate_levels, dtype=np.float32)
    candidate_x, candidate_y, candidate_offset_x, candidate_offset_y = _build_candidate_positions(
        torch,
        uv,
        source_lattice_reference,
        solver_params,
        width=width,
        height=height,
        cell_x=cell_x,
        cell_y=cell_y,
        base_fraction_values=fraction_values,
    )
    candidate_colors = source_hw[candidate_y, candidate_x]
    anchor_energy = (candidate_colors - anchor[..., None, :]).abs().mean(dim=-1)
    rep_energy = (candidate_colors - representative[..., None, :]).abs().mean(dim=-1)
    source_reference_energy = (candidate_colors - source_reference_t[0][..., None, :]).abs().mean(dim=-1)
    alpha_energy = (candidate_colors[..., 3] - anchor[..., None, 3]).abs()
    distance_energy = (candidate_offset_x / max(cell_x, 1e-4)).square() + (candidate_offset_y / max(cell_y, 1e-4)).square()
    match_energy = _reference_match_energy(
        rep_energy,
        source_reference_energy,
        source_reliability_t[0],
        representative_weight=solver_params.refine_representative_match_weight,
        source_weight=solver_params.refine_source_match_weight,
    )
    relax_base_energy = (
        anchor_energy * solver_params.refine_anchor_weight * solver_params.relax_anchor_scale
        + match_energy * solver_params.refine_representative_weight
        + alpha_energy * solver_params.refine_alpha_weight
        + distance_energy * solver_params.refine_distance_weight
    )

    anchor_delta_x = anchor[:, 1:, :] - anchor[:, :-1, :] if output_width > 1 else None
    anchor_delta_y = anchor[1:, :, :] - anchor[:-1, :, :] if output_height > 1 else None
    anchor_delta_diag = anchor[1:, 1:, :] - anchor[:-1, :-1, :] if output_height > 1 and output_width > 1 else None
    anchor_delta_anti = anchor[1:, :-1, :] - anchor[:-1, 1:, :] if output_height > 1 and output_width > 1 else None
    source_weight = solver_params.refine_source_delta_weight
    anchor_weight = 1.0 - source_weight

    def blend(anchor_delta: torch.Tensor | None, source_delta_t: torch.Tensor | None) -> torch.Tensor | None:
        if anchor_delta is None:
            return None
        if source_delta_t is None:
            return anchor_delta
        return anchor_delta * anchor_weight + source_delta_t[0] * source_weight

    selected, _, _, _ = _relax_candidate_selection(
        torch,
        candidate_colors,
        relax_base_energy,
        anchor,
        source_reference_t[0],
        blend(anchor_delta_x, source_delta_x_t),
        blend(anchor_delta_y, source_delta_y_t),
        blend(anchor_delta_diag, source_delta_diag_t),
        blend(anchor_delta_anti, source_delta_anti_t),
        solver_params,
        iterations=solver_params.relax_iterations,
    )
    relaxed_rgba = _select_colors(candidate_colors, selected).detach().cpu().numpy()

    final_rgba, _ = _discrete_refine_output(
        torch,
        F,
        source_t,
        uv0_t,
        representative_t,
        source_reference_t,
        source_reliability_t,
        source_lattice_reference,
        snap_t,
        source_delta_x_t,
        source_delta_y_t,
        source_delta_diag_t,
        source_delta_anti_t,
        solver_params,
        cell_x=cell_x,
        cell_y=cell_y,
        iterations=steps,
    )
    fidelity = {}
    fidelity["snap"] = source_lattice_consistency_breakdown(
        source,
        snap_rgba,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
    )
    fidelity["relaxed"] = source_lattice_consistency_breakdown(
        source,
        relaxed_rgba,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
    )
    fidelity["final"] = source_lattice_consistency_breakdown(
        source,
        final_rgba,
        target_width=inference.target_width,
        target_height=inference.target_height,
        phase_x=inference.phase_x,
        phase_y=inference.phase_y,
    )
    if fidelity["final"]["score"] > fidelity["snap"]["score"] + 1e-6:
        final_rgba = snap_rgba
        fidelity["final"] = fidelity["snap"]
    return source, snap_rgba, relaxed_rgba, final_rgba, (cell_x, cell_y), fidelity


def main() -> None:
    args = _parse_args()
    x0, y0, x1, y1 = args.cell_bbox
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("Invalid --cell-bbox: expected X1 > X0 and Y1 > Y0.")

    source, snap_rgba, relaxed_rgba, final_rgba, (cell_x, cell_y), fidelity = _build_focus_states(
        args.input,
        steps=args.steps,
        device=args.device,
    )

    panel_names = [name.strip().lower() for name in args.panels.split(",") if name.strip()]
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
        "snap": _grid_panel(snap_rgba, "Snap", x0=x0, y0=y0, x1=x1, y1=y1, scale=args.scale),
        "relaxed": _grid_panel(relaxed_rgba, "Relaxed", x0=x0, y0=y0, x1=x1, y1=y1, scale=args.scale),
        "final": _grid_panel(final_rgba, "Final", x0=x0, y0=y0, x1=x1, y1=y1, scale=args.scale),
    }
    for name in panel_names:
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
            "panels": panel_names,
            "scale": args.scale,
            "source_fidelity": fidelity,
        }
    )


if __name__ == "__main__":
    main()
