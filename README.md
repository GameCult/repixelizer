# Repixelizer

Repixelizer is a standalone Python CLI that turns "fake pixel art" into true grid-aligned pixel art for icons and sprites.

It does not rely on a single resize pass. Instead it:

1. infers the source image's implied pixel lattice,
2. initializes a target grid at that resolution,
3. runs a continuous UV optimization stage in premultiplied RGBA space,
4. projects the result onto a real pixel grid with discrete cleanup,
5. optionally constrains the output to a palette, and
6. writes diagnostics and baseline comparisons.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

## Usage

```powershell
repixelize input.png --out output.png
repixelize input.png --out output.png --target-size 120 --diagnostics-dir diagnostics
repixelize compare input.png --out output.png --diagnostics-dir diagnostics
```

### Palette modes

- `off`: keep the full-color RGBA output.
- `fit`: derive or adapt to a useful palette and write palette diagnostics.
- `strict`: require a supplied palette file and stay inside it exactly.

Palette files can be `.gpl`, `.txt`, or `.json`.

## Diagnostics

When `--diagnostics-dir` is supplied, Repixelizer writes:

- `run.json`: chosen size, phase, score breakdown, timings, and settings
- `lattice-overlay.png`: inferred lattice drawn over the source
- `comparison.png`: source, optimized output, and nearest-neighbor preview
- `alpha-preview.png`: source/result alpha
- `noise-heatmap.png`: local noise and isolated-pixel hotspots
- `cluster-preview.png`: coarse material/color clustering
- compare mode extras:
  - `compare.json`
  - `compare.csv`
  - `compare-sheet.png`

## Architecture

- `repixelizer.inference`: lattice and target-size inference
- `repixelizer.continuous`: PyTorch UV-field optimization
- `repixelizer.discrete`: local-search cleanup on the final grid
- `repixelizer.baselines`: naive and error-diffusion baselines
- `repixelizer.diagnostics`: machine-readable and visual debug outputs

## Current scope

Version `0.1.0` is single-image only. It is aimed at icons, emblems, and simple sprites first, with automatic resolution inference and no manual masks.
