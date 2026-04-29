# Repixelizer Spec

## Summary

Repixelizer is a Python tool with a CLI and web GUI for converting "fake pixel art" into true grid-aligned pixel art. The target use cases are single-image icons, emblems, logos, and simple sprites, especially images that already resemble pixel art but fail to obey a consistent pixel lattice.

The project exists because generated imagery often creates appealing pixel-like local patterns while violating the actual constraints of pixel art:
- inconsistent subpixel alignment
- locally plausible but globally incoherent clusters
- noisy highlights and outlines
- fake micro-detail that collapses when downsampled

Repixelizer treats this as lattice inference plus reconstruction rather than a resize problem.

## Goals

- Recover an implied target pixel grid from a fake-pixel-art source image.
- Produce a true discrete-grid output image that looks authored, not merely resized.
- Preserve alpha, silhouette clarity, and strong interior structure.
- Support palette-free output by default, with optional palette-constrained modes.
- Provide diagnostics and baseline comparisons so results are inspectable and measurable.

## Non-goals for v1

- full sprite-sheet or animation consistency
- manual painting or mask authoring as a required workflow
- a native desktop app
- generalized photo-to-pixel-art stylization
- a learned model trained on a large dataset

## Primary users

- artists or developers trying to salvage generated fake pixel art
- tool builders experimenting with constrained repixelization
- game/UI developers creating icons, emblems, badges, and simple sprites

## Input and output contract

### Input

- one RGBA image file
- optional explicit target size override
- optional palette file
- optional diagnostics directory
- optional solver/device settings

### Output

- one final RGBA PNG
- optional palette-constrained output behavior
- optional diagnostics bundle containing JSON and visual artifacts

## Product behavior

### Canonical mode

The canonical workflow should be fully automatic:

1. load the source image
2. infer lattice size from autocorrelation candidates, unless the caller pins it
3. render a diagnostic edge preview
4. build the solver's hierarchical flatness signal from the source
5. run the `phase-field` reconstruction over the inferred lattice
6. sample real source pixels from the final source-position field
7. snap alpha to the final discrete mask
8. optionally quantize to a palette
9. write output and diagnostics

No manual masks or user-authored region hints are assumed in v1.

This is the canonical machine in the repo. Comparison mode just runs that same
`phase-field` result next to the baselines.

### CLI shape

Primary commands:

```powershell
repixelize input.png --out output.png
repixelize compare input.png --out output.png
```

Supported flags:
- `--target-size`
- `--target-width`
- `--target-height`
- `--palette`
- `--palette-mode off|fit|strict`
- `--diagnostics-dir`
- `--seed`
- `--steps`
- `--device cpu|cuda`

### Palette behavior

- `off`: produce unconstrained RGBA output
- `fit`: adapt to a useful palette derived from or compatible with the result
- `strict`: stay strictly inside a supplied palette

Palette constraints are optional because many generated fake-pixel-art images do not originate from a coherent palette.

## Technical design

### 1. Lattice inference

The tool must estimate the resolution of the fake lattice being mimicked.

Requirements:
- infer candidate target sizes from autocorrelation hints
- use canonical cell centers for each candidate
- score candidates using coherence-oriented metrics rather than reconstruction alone
- expose the top-ranked alternatives in diagnostics
- if `--target-size`, `--target-width`, or `--target-height` is provided, skip automatic inference

Scoring goals:
- low isolated-pixel rate
- high cluster continuity
- strong alpha-edge crispness
- straighter outlines
- lower local color chatter
- reasonable agreement with source-derived periodicity priors

### 2. Diagnostic analysis and solver signal

The pipeline keeps two source-derived views separate:

- a diagnostic edge preview for GUI and run inspection
- a hierarchical flatness signal that actually steers the solver

The live solver signal is built from multiscale luminance gradient and curvature energy. It rewards calm local source regions while leaving edge preview as a gauge, not a hidden steering wheel.

### 3. Phase-field reconstruction

The solver represents each output pixel as one live source-space sample position initialized at a fixed lattice center.

Requirements:
- operate in premultiplied RGBA space
- initialize from canonical fixed lattice centers
- let each sample choose from a small local candidate window
- blend toward the locally preferred flatter source position
- relax row and column springs so the sample field stays coherent
- nearest-sample the original source pixels for final output

The current implementation is explicit NumPy local search plus spring relaxation. It is not the old PyTorch autograd / projected-Adam displacement solver.

### 4. Discrete projection and cleanup

After the phase-field stage, the result must be treated as a true pixel grid.

The discrete stage should:
- preserve source-sampled RGBA values
- snap alpha to an opaque/transparent mask
- leave structural repair to the solver rather than hiding confusion with a cleanup pass

Cleanup is intentionally small in v1. It should not become a second solver wearing a trench coat.

### 5. Baselines and comparison mode

The project must ship comparison baselines:
- Lanczos downscale
- naive resize
- resize plus error diffusion

`compare` mode should run:
- the same `phase-field` pipeline
- all baselines
- metric collection
- a visual contact sheet

The project should treat these baselines as the minimum bar for usefulness.

## Diagnostics

Diagnostics should be machine-readable and human-readable.

Required artifacts:
- `run.json`
- `lattice-overlay.png`
- `comparison.png`
- `alpha-preview.png`
- `noise-heatmap.png`

Compare mode should additionally write:
- `compare.json`
- `compare.csv`
- `compare-sheet.png`

## Acceptance criteria

The tool is successful for v1 if it can:
- infer sensible target sizes on curated synthetic cases
- preserve transparency and silhouette quality
- produce outputs that are visibly more grid-coherent than simple resize baselines
- avoid obvious isolated-pixel explosions
- generate diagnostics that explain how and why a run behaved the way it did

## Future expansion

The design should leave room for:
- manual region hints or lock zones
- multi-frame consistency for sprite sheets
- more advanced warping or lattice-field models
- learned priors or ranking models for candidate outputs
