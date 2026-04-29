# Scratch: hosted shipping

## Current Subgoal
Ship the hosted experiment path. Source hygiene is complete unless a fresh,
concrete stale signature appears.

## Completed in this cleanup pass
- Removed unused solver hyperparams from `src/repixelizer/params.py`.
- Removed stale tuning-campaign knobs from `tools/solver_tuning_campaign.py`.
- Removed `analysis` / `edge_map` from the live phase-field solver path.
- Simplified `cleanup_pixels(...)` to alpha snap plus zero heatmap.
- Updated tests that still passed `analysis` into the solver/candidate-selection path.
- Renamed the live observer/GUI flatness artifact from `guidance` to `signal`.
- Fixed the signal payload rename bug where the observer still copied stale `guidance`.
- Reworded docs/state from `edge scout` / solver `guidance` to diagnostic edge preview plus hierarchical flatness signal.
- Updated GUI event listeners to the live `lattice_inference_*` events.
- Reconciled `config/solver_params.json` with the documented shipping basin: `grid=4.75`, `radius=0.215`, `blend=0.24`, `move=0.09`, `energy=1.4`, `peak=3.25`.
- Updated `docs/spec.md` so it describes the live explicit local-search solver and alpha snapping.

## Verification already run
- AST parse across `src`, `scripts`, `tools`, and `tests`: passed.
- Targeted stale-string scan for solver params, cleanup hooks, guidance names, and solver-analysis call sites: clean.
- Full source audit read 60 source files under `src`, `scripts`, `tools`, and `tests`; the only remaining `analysis=analysis` hit is the intentional `RunResult` diagnostic field.
- Focused suite passed: `tests/test_pipeline.py tests/test_gui.py tests/test_inference.py tests/test_phase_field.py`.
- Full suite passed: `89 passed`.
- `tools/repixelizer_prepare_compaction.py`: 19 ok, 2 expected warnings, 0 errors.

## Remaining notes
- `PhaseFieldSourceAnalysis` intentionally remains as a diagnostic/GUI gauge for now. It does not steer the solver.
- The remaining compaction warnings are expected: evidence ledger size, dirty worktree, and this scratch file existing.
- Next real project move is shipping/commit hygiene and hosted deployment, not continuing this cleanup pass, unless a fresh scan finds a new concrete vestige.
