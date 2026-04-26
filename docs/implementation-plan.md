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

Inference now splits three ways:

- fixed lattice when the caller pins size and/or phase explicitly
- searched lattice for the full local workflow
- direct autocorr lattice for the hosted demo path: one consensus size, one cheap phase probe, no size search, no phase rerank

The hosted web shell now also has an auth seam:

- `src/repixelizer/access.py` owns the app-local access boundary and local ownership checks
- `src/repixelizer/gui.py` owns hosted route policy, landing-page behavior, and queue/job stamping
- self-host and local runs stay permissive by default unless `GC_ACCESS_*` turns the seam on

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

### 2. Keep lattice inference honest

Current stance:

- keep the shipped searched path grounded in autocorrelation only
- let autocorr keep a small near-best lag plateau and use cross-axis consensus before collapsing to one lattice family
- the hosted demo path now skips multi-size search entirely and uses one autocorr-consensus size plus the cheap inferred phase, capped by the hosted output limit
- do not trust the old pixel-walk / change-interval spacing path; it was cut after it kept pruning the badge away from the `126` family
- do not keep the projected edge-energy spectral pass; it floor-hugged to `2px` / `626`-family nonsense on the dense landscape fixture and did not earn its keep
- treat phase rerank as optional scaffolding, not sacred machinery

Future options worth exploring, if autocorr stops carrying its weight:

- distance-transform / medial-radius blob sizing on softened interiors
- small scale-space blob-size detection (`LoG` / `DoG`)
- blur-aware correction of observed blob width before converting to lattice size
- tiny learned size-and-phase prior trained on synthetic fake-pixel-art fixtures

### 3. Keep phase-field honest

Rules:

- do not reintroduce portrait layers, candidate trays, or solver-stage religions
- if a new term lands, it must belong cleanly in the one-field objective
- if a result looks better and the metrics disagree, fix the metrics rather than worshipping them

### 4. Keep hosted auth surgery out of the solver

Current stance:

- Heimdall integration belongs in the hosted web layer, not in the reconstruction pipeline
- queued jobs now have local owner metadata so future auth can resolve job access from local session/account instead of provider ids
- the repo is only "auth-ready", not "auth-landed"
- `GC_ACCESS_MODE=trusted-header` exists as a thin local seam for future integration and route-policy tests; it is not the final Heimdall story

## Guardrails

- prefer one clear hypothesis per pass
- show the output after every real reconstruction run
- keep the maps updated
- revert or delete machinery that does not visibly earn its keep

This repo already spent enough time as a Jenga tower of maybe-useful cleverness. The machine has to deserve every part.
