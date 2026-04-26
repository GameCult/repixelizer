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
2. infer lattice size and phase
3. analyze source structure
4. run the `phase-field` reconstruction over the inferred lattice
5. project that result to a real pixel grid
6. clean up obvious discrete-grid artifacts
7. optionally quantize to a palette
8. write output and diagnostics

No manual masks or user-authored region hints are assumed in v1.

This is the only canonical reconstruction pipeline in the repo. Comparison mode
adds baselines, not alternate reconstruction engines.

### CLI shape

Primary commands:

```powershell
repixelize input.png --out output.png
repixelize compare input.png --out output.png
```

Supported flags:
- `--target-size`
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
- search over candidate target sizes
- search over subpixel phase offsets for each candidate
- score candidates using coherence-oriented metrics rather than reconstruction alone
- expose the top-ranked alternatives in diagnostics
- if `--target-size` is provided, skip size search but still estimate phase

Scoring goals:
- low isolated-pixel rate
- high cluster continuity
- strong alpha-edge crispness
- straighter outlines
- lower local color chatter
- reasonable agreement with source-derived periodicity priors

### 2. Source analysis

The solver should derive automatic guidance from the image itself:
- edge map
- alpha-aware source structure

This guidance is used to preserve important boundaries while allowing smoother regions to settle into cleaner pixel clusters.

### 3. Phase-field optimization

The optimizer represents each output pixel by a base lattice center plus one learned displacement vector.

Requirements:
- operate in premultiplied RGBA space
- use PyTorch autograd
- optimize a displacement field initialized at zero from the inferred lattice
- preserve local adjacency instead of letting the sample field collapse or fold

Loss components should include:
- local coherence / solidity so samples settle inside fake-pixel interiors
- edge-aware smoothing so hard boundaries stay crisp
- anti-collapse spacing so neighboring cells do not stack onto the same source pixel
- bounded displacement so the field does not wander off into the weeds

### 4. Discrete projection and cleanup

After the phase-field stage, the result must be treated as a true pixel grid.

The discrete stage should:
- evaluate local neighborhoods
- remove orphan pixels where possible
- merge obvious micro-clusters
- reduce accidental highlight static
- avoid damaging strong guided boundaries

This stage is intentionally heuristic and local in v1.

### 5. Baselines and comparison mode

The project must ship comparison baselines:
- naive resize
- resize plus error diffusion

`compare` mode should run:
- the same `phase-field` pipeline
- both baselines
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
