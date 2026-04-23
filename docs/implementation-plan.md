# Repixelizer Implementation Plan

## Current machine

Repixelizer now has two real reconstruction engines:

- `phase-field`: the default optimizer in `src/repixelizer/phase_field.py`
- `tile-graph`: the source-owned alternate solver in `src/repixelizer/tile_graph.py`

The old tray-based optimizer is gone. Good riddance.

The live pipeline is:

`source image -> lattice inference -> edge analysis -> phase-field or tile-graph reconstruction -> cleanup -> optional palette fit -> diagnostics`

## What is working

- lattice size and phase inference are still shared and still CUDA-capable
- `phase-field` is now the main optimizer path and produces the best-looking badge result in the repo so far
- `tile-graph` still provides a useful alternate path when literal source ownership matters more than smooth field behavior
- compare mode, benchmark mode, diagnostics writing, and tuning all still work after the optimizer cutover
- the repo now reports both:
  - `source_fidelity`: agreement with the inferred lattice portrait
  - `source_structure`: visible structural agreement at source size

That second metric exists because the first one was happily slandering the better-looking output.

## Current evidence

### Phase-field

Pinned badge case:

- fixture: `tests/fixtures/real/ai-badge-cleaned.png`
- lattice: `126x126`
- phase: `(0.0, -0.2)`
- steps: `48`

Useful artifacts:

- first pinned badge run: `artifacts/phase-field-v1-badge-126/`
- tip-focused follow-up: `artifacts/phase-field-v2-badge-126/`
- structure-metric sanity check: `artifacts/phase-field-metric-check/`

What we know:

- the output preserves the important badge structure better than the deleted optimizer did
- the tracked blemish is now narrow and specific: the sword-tip stroke widens too much in one local region
- the tracked focus fixture is `tests/fixtures/real/ai-badge-tip-focus.json`

### Tile-graph

Useful artifacts:

- full emblem win: `artifacts/full-emblem-tile-graph-atomic-v3-cuda/`
- pinned badge speed/quality checkpoint: `artifacts/tile-graph-reducebykey-v1-badge-126/`

What we know:

- source ownership is real now
- large-fixture cold-build cost is still the main pain
- the dominant remaining speed problem is connected-component labeling / region extraction, not the parity solver loop

## Maps

The repo now keeps the living maps only:

- `docs/lean-optimizer-algorithm-map.md`
- `docs/tile-graph-algorithm-map.md`

If a future pass cannot be explained cleanly against one of those maps, it should not land.

## Current priorities

### 1. Fix the phase-field sword-tip blemish

Goal:

- keep the current structural win
- stop widening the dark contour near the tapered sword tip

Current hypothesis:

- scalar weight nudges are not enough
- the field needs better anisotropic behavior near sharp tapered contours
- it likely needs to distinguish motion along a stroke from motion across a stroke

Tracked fixture:

- `tests/fixtures/real/ai-badge-tip-focus.json`

### 2. Keep phase-field honest

Rules:

- do not reintroduce portrait layers, candidate trays, or solver-stage religions
- if a new term lands, it must belong cleanly in the one-field objective
- if a result looks better and the metrics disagree, fix the metrics rather than worshipping them

### 3. Make tile-graph less slow

Goal:

- reduce cold-build latency on badge-scale inputs without reintroducing fake ownership or averaged patch lies

Most likely targets:

- connected-component labeling
- source-region extraction
- cache reuse across repeated fixed-lattice runs

## Guardrails

- prefer one clear hypothesis per pass
- show the output after every real reconstruction run
- keep the maps updated
- revert or delete machinery that does not visibly earn its keep

This repo already spent enough time as a Jenga tower of maybe-useful cleverness. The machine has to deserve every part.
