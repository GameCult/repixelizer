# Fresh Workspace Handoff

This is the compact re-entry packet for `E:\Projects\repixelizer`.

Historical proof belongs in git history, diagnostics, benchmark artifacts, and
`state/evidence.jsonl`. Exact control flow belongs in
`docs/lean-optimizer-algorithm-map.md`. Detailed tuning lessons belong in
`docs/solver-tuning-lessons-2026-04-29.md`.

## Rehydrate

From the repo root:

```powershell
.\.venv\Scripts\python .\tools\repixelizer_state.py status
Get-Content '.\state\map.yaml'
Get-Content '.\notes\fresh-workspace-handoff.md'
Get-Content '.\docs\lean-optimizer-algorithm-map.md'
Get-Content '.\docs\implementation-plan.md'
git status --short --branch
git log --oneline -5
Get-Content '.\state\evidence.jsonl' -Tail 8
```

Do not trust this file for the exact live HEAD. Always check git before committing or
shipping.
Do not continue implementation automatically from a rehydrate-only request.

## Current machine

- `phase-field` is the only live reconstruction engine.
- The live pipeline is `source -> lattice inference -> diagnostic edge preview -> fixed lattice centers -> hierarchical flatness signal -> explicit local search + lattice relaxation -> nearest source sample -> cleanup / diagnostics`.
- The current solver starts from fixed lattice centers, adds tiny seeded noise, chooses local preferred positions from a 5x5 candidate window, blends toward those positions, then relaxes lattice springs for row/column coherence.
- The final image is emitted by nearest-source sampling from final source positions. Bilinear interpolation is not the output path.
- The signal is direct hierarchical flatness: multiscale luminance gradient/Laplacian energy, one normalization, inverted to flatness, then sharpened.
- `source_structure` exists alongside `source_fidelity` so visible structure and lattice agreement can be inspected separately.
- Cleanup is alpha snap plus diagnostics.
- `src/repixelizer/spritesheet.py` owns spritesheet region detection and packing.
  In automatic mode it probes crop sizes, chooses one shared
  source-pixels-per-texel density for the sheet, then calls
  `run_pipeline_rgba(...)` once per detected crop. Pinned target sizes apply per
  sprite and bypass shared-density sizing.

## Hosted/web state

- Hosted demo inference uses autocorr candidates, capped by hosted output limits.
- Low-confidence autocorr candidates can be preview-reranked; hosted mode does not forcibly disable rerank.
- Hosted access lives in `src/repixelizer/access.py` and `src/repixelizer/gui.py`.
- `GC_ACCESS_MODE=heimdall` verifies Heimdall Ed25519 access tokens locally and adopts httpOnly local sessions.
- Heimdall mode stores a separate httpOnly refresh cookie
  (`gc_access_token_refresh` by default) and calls Heimdall's
  `/v1/apps/repixelizer/sessions/refresh` endpoint with Repixelizer-owned
  Discord/Patreon entitlement policies before sending the user through provider
  OAuth again.
- Self-host/local runs stay permissive by default unless `GC_ACCESS_*` enables access policy.
- Do not smear auth logic into `src/repixelizer/pipeline.py`, `src/repixelizer/inference.py`, or the solver stack.

## Current shippable solver config

The project is shipping a usable trial config so Repixelizer stops blocking
Heimdall and StreamPixels.

Live values in `config/solver_params.json`:

- `phase_field_grid_alignment_weight=4.75`
- `phase_field_local_search_radius_ratio=0.215`
- `phase_field_local_search_blend=0.24`
- `phase_field_local_search_move_weight=0.09`
- `phase_field_signal_energy_power=1.4`
- `phase_field_signal_peak_power=3.25`

What the tuning pass proved:

- Wide local-search windows caused neighbor feature poaching.
- The useful radius basin is around `0.20-0.22`, with `0.215` best in the focused pass.
- Once radius is tight, stronger blend helps; `0.24` beat lower blend settings on the focused artificial set.

What it did not prove:

- The signal parameters are not fully retuned around the new motion basin.
- The current config is shippable, not final.

See `docs/solver-tuning-lessons-2026-04-29.md` before doing more solver tuning.


## Immediate Re-entry Instruction

Ship the hosted experiment path next. The source hygiene pass is complete:
diagnostic analysis stays as a GUI/observer gauge, the solver does not accept
`analysis`, `docs/spec.md` matches the live explicit solver, and
`config/solver_params.json` matches the documented shipping basin.

## Current priority

Ship the hosted experiment and unblock Heimdall / StreamPixels.

Solver follow-up is later work unless the hosted launch exposes a blocking
quality regression. The next solver tuning pass should start from the current
live config, sweep signal knobs first, then movement/coherence knobs, then
combinations, and only then run final full-corpus validation.

## Guardrails

- Persistent state is the agent's mind; stale state is bad thought, not harmless clutter.
- Keep the algorithm map grounded in source.
- Do not trust a metric win that makes the image look worse.
- Revert or discard tuning changes that do not clearly improve the real outcome.
- Prefer shipping the hosted path over polishing solver knobs right now.

## State surfaces

- `state/map.yaml`: canonical durable current truth.
- `state/scratch.md`: disposable active scratch.
- `state/evidence.jsonl`: distilled durable belief ledger.
- `state/branches.json`: active hypothesis tracking.
- `docs/lean-optimizer-algorithm-map.md`: live control-flow map.
- `docs/solver-tuning-lessons-2026-04-29.md`: latest solver tuning knowledge.
- `docs/implementation-plan.md`: forward plan.
- `tools/repixelizer_prepare_compaction.py`: pre-compaction hygiene check.
