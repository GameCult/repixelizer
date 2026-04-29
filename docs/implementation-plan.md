# Repixelizer Implementation Plan

## What this file is

This file is the forward plan for Repixelizer.

It is not the authoritative stage-by-stage control-flow map. That lives in
`docs/lean-optimizer-algorithm-map.md`.

If these two notes disagree about what the code does right now, trust the
algorithm map and the source, then fix this plan.

## Current machine

Repixelizer has one live reconstruction engine:

- `phase-field`: the canonical reconstruction engine in `src/repixelizer/phase_field.py`

The live pipeline is:

`source image -> lattice inference -> diagnostic edge preview -> fixed lattice centers -> hierarchical flatness signal -> explicit local search + lattice relaxation -> nearest source sample -> cleanup -> optional palette fit -> diagnostics`

Inference splits two ways:

- fixed lattice when the caller pins size explicitly
- autocorr lattice when size is inferred automatically; autocorr can surface multiple candidates, and low-confidence candidates can be preview-reranked

The hosted web shell has an auth seam:

- `src/repixelizer/access.py` owns the app-local access boundary and local ownership checks
- `src/repixelizer/gui.py` owns hosted route policy, landing-page behavior, and queue/job stamping
- `GC_ACCESS_MODE=heimdall` supports provider start, backend callback receipt, local JWT verification, and httpOnly session adoption
- self-host and local runs stay permissive by default unless `GC_ACCESS_*` turns the seam on

## What is working

- `phase-field` is the only canonical reconstruction path.
- The current explicit solver is fast enough for hosted interaction.
- Compare mode, benchmark mode, diagnostics writing, GUI observer previews, and tuning harnesses exist.
- The repo reports both `source_fidelity` and `source_structure` so lattice agreement and visible structure can be inspected separately.
- Hosted landing/app flow is in place and should be the immediate product path.

## Current solver shipping state

The current solver config is shippable, not final.

Useful current values:

- `phase_field_grid_alignment_weight=4.75`
- `phase_field_local_search_radius_ratio=0.215`
- `phase_field_local_search_blend=0.24`
- `phase_field_local_search_move_weight=0.09`
- `phase_field_signal_energy_power=1.4`
- `phase_field_signal_peak_power=3.25`

Tuning lesson:

- broad local-search radius caused samples to poach neighboring features
- the focused useful radius basin is around `0.20-0.22`
- at that tighter radius, stronger blend became useful
- signal knobs were not fully explored around the new basin

Detailed tuning notes live in `docs/solver-tuning-lessons-2026-04-29.md`.

## Current priorities

### 1. Ship the hosted experiment

Goal:

- get Repixelizer online
- stop blocking Heimdall
- stop Heimdall from blocking StreamPixels

Practical stance:

- do not spend more time polishing solver knobs before hosted launch unless a blocking quality regression appears
- keep hosted auth and access work isolated to `src/repixelizer/access.py` and `src/repixelizer/gui.py`
- keep solver and pipeline code out of auth work

### 2. Keep lattice inference honest

Current stance:

- keep automatic inference grounded in autocorrelation
- let autocorr keep a small near-best lag plateau and use cross-axis consensus before collapsing to one lattice family
- keep hosted demo inference on the autocorr path with candidate rerank available when confidence is low
- keep candidate rerank as optional support for low-confidence autocorr choices

### 3. Later: retune the phase-field solver deliberately

The next solver pass should begin from the current live config and its narrow
search geometry.

Recommended order:

1. Sweep signal knobs around `radius=0.215, blend=0.24`.
2. Sweep movement/coherence knobs around the best signal candidate.
3. Test combinations only after single-knob effects are understood.
4. Validate on focused artificial cases, real character glint/symbol cases, and the badge fixture.
5. Run full-corpus validation at the end, not during every exploratory twitch.

Knobs still underexplored in the new basin:

- `phase_field_signal_weight`
- `phase_field_signal_gradient_weight`
- `phase_field_signal_curvature_weight`
- `phase_field_signal_level_decay`
- `phase_field_signal_pyramid_levels`
- `phase_field_signal_energy_power`
- `phase_field_signal_peak_power`
- `phase_field_local_search_move_weight`
- `phase_field_grid_alignment_weight`
- `phase_field_local_search_grid_weight`

## Guardrails

- Prefer one clear hypothesis per pass.
- Show the output after every real reconstruction run when doing visual tuning.
- Keep maps updated after code changes.
- Revert or delete machinery that does not visibly earn its keep.
- Keep solver changes inside the one-field / one-signal / one-relaxation model unless the design map changes first.
- If a new term lands, it must belong cleanly in the one-field / one-signal / one-relaxation model.
- Treat metrics as evidence about the output, not as a replacement for visual inspection.
