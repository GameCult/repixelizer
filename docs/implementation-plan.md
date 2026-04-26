# Repixelizer Implementation Plan

## What this file is

This file is the forward plan and active hypothesis ledger for Repixelizer.

It is not the authoritative stage-by-stage control-flow map. That lives in
`docs/lean-optimizer-algorithm-map.md`.

If these two notes disagree about what the code does right now, trust the
algorithm map and the source, then fix this plan instead of hand-waving about
intent.

## Current machine

Repixelizer now has one live reconstruction engine:

- `phase-field`: the canonical reconstruction engine in `src/repixelizer/phase_field.py`

The live pipeline is:

`source image -> lattice inference -> edge analysis -> phase-field reconstruction -> cleanup -> optional palette fit -> diagnostics`

## What is working

- lattice size and phase inference are still shared and still CUDA-capable
- `phase-field` is the only canonical reconstruction path and produces the best-looking badge result in the repo so far
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

## Maps

The repo now keeps the living map:

- `docs/lean-optimizer-algorithm-map.md`

If a future pass cannot be explained cleanly against that map, it should not land.

## Current priorities

### 1. Fix the phase-field sword-tip blemish

Goal:

- keep the current structural win
- stop widening the dark contour near the tapered sword tip

Current hypothesis:

- scalar weight nudges are not enough
- the field needs better anisotropic behavior near sharp tapered contours
- it likely needs to distinguish motion along a stroke from motion across a stroke

What we just learned:

- the later `d9fa411` phase-field tuning pass was a real regression, not paranoia
- that pass strengthened the local edge penalty, added a spacing loss plus upper-spacing clamp, and made edge gating more aggressive
- on the pinned badge case, it removed internal linework and changed `2433 / 15876` output cells relative to the original good `phase-field` run
- reverting that tuning pass restores the original good badge result exactly; the fresh fixed run under `artifacts/phase-field-regression-fix-badge-126/` is byte-identical to the original `artifacts/phase-field-v1-recheck-badge-126/`

Tracked fixture:

- `tests/fixtures/real/ai-badge-tip-focus.json`

### 2. Keep phase-field honest

Rules:

- do not reintroduce portrait layers, candidate trays, or solver-stage religions
- if a new term lands, it must belong cleanly in the one-field objective
- if a result looks better and the metrics disagree, fix the metrics rather than worshipping them

## Guardrails

- prefer one clear hypothesis per pass
- show the output after every real reconstruction run
- keep the maps updated
- revert or delete machinery that does not visibly earn its keep

This repo already spent enough time as a Jenga tower of maybe-useful cleverness. The machine has to deserve every part.
