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
