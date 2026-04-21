# Repixelizer

Repixelizer is a standalone Python CLI for turning "fake pixel art" into true grid-aligned pixel art.

The core idea is simple: generated images often look like pixel art without actually obeying a single clean pixel grid. Repixelizer treats that as an optimization problem instead of a resize problem. It estimates the implied lattice, builds a real target grid, optimizes how output pixels sample the source image, then cleans the result up on the discrete grid.

This project is meant to stand on its own. It should be understandable without any StreamPixels-specific context.

## What it does

Repixelizer currently focuses on single-image inputs such as:
- icons
- emblems
- logos
- simple sprites

The current pipeline is:

1. infer the source image's implied fake-pixel lattice
2. choose a target output resolution, unless the user overrides it
3. initialize a UV/sample field over the target grid
4. optimize that field in premultiplied RGBA space with PyTorch
5. project onto a true pixel grid
6. run discrete cleanup to reduce noise and isolated-pixel garbage
7. optionally quantize into a palette-constrained result
8. write diagnostics and baseline comparisons

## Project docs

- Product and technical spec: [docs/spec.md](</E:/Projects/repixelizer/docs/spec.md>)
- Living implementation plan: [docs/implementation-plan.md](</E:/Projects/repixelizer/docs/implementation-plan.md>)

If you are moving this project into a fresh workspace, those two docs are the main handoff.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

Run the optimizer:

```powershell
repixelize input.png --out output.png
repixelize input.png --out output.png --target-size 120 --diagnostics-dir diagnostics
```

Run the optimizer plus baselines:

```powershell
repixelize compare input.png --out output.png --diagnostics-dir diagnostics
```

## CLI contract

Primary command shape:

```powershell
repixelize input.png --out output.png
repixelize compare input.png --out output.png
```

Important flags:

- `--target-size <n>`: override inferred target max dimension
- `--palette <file>`: palette file in `.gpl`, `.txt`, or `.json`
- `--palette-mode off|fit|strict`: palette behavior
- `--diagnostics-dir <path>`: write visual and JSON diagnostics
- `--seed <n>`: deterministic optimizer seed
- `--steps <n>`: number of optimization steps
- `--device cpu|cuda`: PyTorch device

### Palette modes

- `off`: keep the output unconstrained
- `fit`: adapt the result to a derived or supplied palette
- `strict`: stay strictly inside the supplied palette

Palette constraints are optional by design. Many fake-pixel-art inputs are not born from a coherent palette, so palette enforcement is useful but not assumed.

## Outputs

Main output:
- final RGBA PNG

Optional diagnostics:
- `run.json`: chosen size, phase, settings, timings, and score breakdown
- `lattice-overlay.png`: inferred lattice preview over the source image
- `comparison.png`: source, optimized output, and nearest-neighbor preview
- `alpha-preview.png`: alpha behavior before and after
- `noise-heatmap.png`: isolated-pixel and local-noise hotspots
- `cluster-preview.png`: coarse material/color clustering

Additional compare-mode outputs:
- `compare.json`
- `compare.csv`
- `compare-sheet.png`

## Repo layout

- `src/repixelizer`: core package
- `tests`: synthetic and pipeline-focused tests
- `examples`: reserved for future real example inputs
- `artifacts`: local smoke-test outputs, not core source

Core modules:
- `inference.py`: target-size and phase inference
- `analysis.py`: edge, alpha, and coarse clustering analysis
- `continuous.py`: UV-field optimization
- `discrete.py`: local cleanup after projection
- `palette.py`: palette loading and optional quantization
- `compare.py`: baseline comparisons and contact-sheet output
- `synthetic.py`: benchmark generators for tests

## Current status

The project is bootstrapped and runnable:
- editable install works
- PyTorch-based optimization is wired in
- synthetic tests are in place
- compare mode writes diagnostics and baseline metrics

What exists today is a serious experimental baseline, not a finished research result. It is strong enough to iterate on algorithmically, but it is not yet proven across a broad real-world image corpus.

## Current limitations

- single-image only
- no multi-frame sprite-sheet consistency yet
- no GUI yet
- no user-painted masks or region hints yet
- lattice inference is heuristic, not learned
- continuous optimization is real, but still early and intentionally conservative

## Validation

The current test suite checks:
- resolution inference on synthetic emblem inputs
- palette loading and strict palette enforcement
- pipeline output and diagnostics writing
- compare-mode artifact generation
- a basic default CLI execution path

Run it with:

```powershell
.venv\Scripts\python -m pytest -q
```

## Near-term direction

The next useful steps are:
- improve inference on real-world generated inputs
- make the optimizer outperform resize-and-dither baselines more consistently
- expand the curated benchmark corpus
- add smarter discrete cleanup rules for outlines, highlights, and material bands
- optionally add manual guidance in a later version, once the automatic baseline is stable
