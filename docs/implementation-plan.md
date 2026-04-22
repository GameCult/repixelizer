# Repixelizer Implementation Plan

## Current state

The project is bootstrapped and runnable as a standalone Python repo.

Implemented:
- package metadata and editable install
- CLI entrypoint with `run` and `compare` modes
- lattice/phase inference with coherence scoring plus source-period prior
- source-image structural analysis
- PyTorch-based continuous UV-field optimization
- discrete cleanup on the output grid
- optional palette loading and quantization modes
- compare mode with naive and diffusion baselines
- diagnostics writing
- synthetic test fixtures and automated tests

Verified:
- local editable install in a dedicated venv
- full test suite currently passing
- compare-mode smoke run against a real emblem image

## What is still provisional

The current implementation should be treated as a strong experimental baseline, not a final algorithm.

The most provisional areas are:
- lattice inference on real, messy generated imagery
- the exact loss weighting in the continuous stage
- the strength and shape of discrete cleanup heuristics
- compare-mode metrics as a proxy for actual visual quality

## Near-term priorities

## High-value regression cases

### AI badge emblem with baked checkerboard background

This locally generated AI badge has become a useful stress case because it fails in a very specific, repeatable way.

Why it is useful:
- the global target size is broadly plausible, so the failure is not just "picked the wrong lattice"
- the lower sword-tip region contains sharp single-pixel and near-single-pixel features at multiple orientations
- those features are exactly the kind of local adjacency pattern Repixelizer is supposed to preserve

Current failure modes:
- the right side of the sword tip develops a wobbling outline instead of a smooth taper
- thin oblique outline features around the tip get collapsed into mushy transitions
- local cell assignments near the tip look inconsistent even when the overall badge size feels right
- the right-hand guard wing loses short dark-light-dark adjacency motifs and collapses into a chunky gold slab
- some guard closeups look nearly identical across `snap`, `relaxed`, and `final`, which suggests the wrong local motif is often chosen before later refinement even starts

Important note:
- the checkerboard is baked into the source image, not transparency
- when projected onto a `122x122` output grid it aliases into a visible `2-3-2` cadence
- that `2-3-2` background rhythm is expected from resampling the background pattern and should not be mistaken for proof that the inferred lattice itself is varying in size
- it is still a problem, because the background is treated as real source structure and can contaminate local scoring near the blade

What future work should improve here:
- better weighting of thin local outline/motif preservation relative to high-contrast cell transitions
- some form of coordinated local relaxation so nearby cells can move together toward a globally better contour
- optional background suppression or de-weighting when a baked checkerboard is clearly not semantic content

Repository fixtures:
- `tests/fixtures/real/ai-badge-cleaned.png` is the manually cleaned transparent version of the same emblem
- `tests/fixtures/real/ai-badge-cleaned.json` records why it matters
- use the cleaned fixture when isolating lattice and contour failures from background-removal failures
- `scripts/render_focus_crop.py` generates reproducible docs/debug closeups from output-grid cell coordinates
- the current docs-facing guard crop uses `--cell-bbox 73 20 107 44`

### 1. Build a real benchmark corpus

Add curated real-world cases, not just synthetic ones:
- emblems and logos with transparency
- icons with strong outlines
- simple sprite-like figures
- examples that are known to fail naive resize-and-dither

Each case should include:
- source image
- expected target size or plausible size range
- short notes on known failure modes

### 2. Improve lattice inference

The current inference is workable but still heuristic.

Next improvements should focus on:
- better periodicity detection for noisy or partially aligned images
- stronger penalties against oversized candidate grids
- confidence calibration that reflects ambiguity more honestly

### 3. Strengthen the discrete cleanup stage

Current cleanup is local and generic.

Next version should add:
- stronger outline continuity rules
- better preservation of highlight bands
- better handling of metallic or gem-like materials
- smarter alpha-edge cleanup for transparent icons

### 4. Make compare mode more trustworthy

The current metrics are useful but still internal.

Next work should:
- expand score breakdowns
- validate whether the metrics correlate with actual human preference
- make it easier to compare runs across parameter sweeps

## Suggested next milestones

### Milestone 1: Real-image evaluation harness

- add curated real-image fixtures
- add golden metadata for expected target-size ranges
- add a repeatable batch runner
- emit one summary table per corpus run

### Milestone 2: Solver tuning pass

- tune continuous-stage loss weights against the benchmark corpus
- test more robust anti-speckle and cluster-preservation behavior
- reduce cases where the solver underperforms the naive baseline

### Milestone 3: Better cleanup heuristics

- add contour-aware cleanup
- add material-sensitive smoothing for highlights and trim
- reduce the number of visually good-but-metric-weak outputs

### Milestone 4: Human-in-the-loop v2 planning

Only after the automatic baseline is strong:
- explore optional region hints
- evaluate a minimal GUI or notebook layer
- consider multi-frame support

## Risks

- The solver can still produce outputs that score well but feel over-smoothed.
- Real generated images may contain multiple inconsistent local lattice cues.
- Continuous optimization is flexible, but easy to overfit to the reconstruction term.
- Metrics that reward coherence can accidentally reward blandness if not balanced carefully.

## Assumptions

- Single-image quality matters more than batch throughput in the current phase.
- Palette constraints are optional and should stay optional.
- The project is better served by improving automatic behavior first than by rushing into manual tooling.
- Python plus PyTorch remains the right stack for the current research/prototyping phase.
