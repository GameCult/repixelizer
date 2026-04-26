# Fresh Workspace Handoff

This is the re-entry packet for `E:\Projects\repixelizer`.

It is intentionally short. Historical proof belongs in git history,
diagnostics, benchmark artifacts, and the distilled `state/evidence.jsonl`
ledger; exact control flow belongs in `docs/lean-optimizer-algorithm-map.md`;
forward planning belongs in `docs/implementation-plan.md`.

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

Do not trust this file for the exact live HEAD. Always check git.

## Current Orientation

- `phase-field` is the only live reconstruction engine.
- The live pipeline is `source -> lattice inference -> edge analysis -> phase-field reconstruction -> cleanup -> optional palette fit -> diagnostics`.
- The low-confidence phase rerank path is a short preview solve, not a second full optimizer.
- Cleanup is secondary and usually a no-op; the core result is supposed to come from the solver, not cleanup cosplay.
- The current tracked weakness is the widened dark contour near the badge sword tip on tapered shapes.
- `source_structure` exists because lattice-only `source_fidelity` could call visibly better outputs worse.
- The old tray optimizer is dead and should stay dead unless the entire machine map changes for a real reason.

## Critical Doctrine

- Persistent state is the agent's mind.
- Cut persistent memory as ruthlessly as code; stale context is bad thought, not harmless clutter.
- Remember Jenga: growing diffs, growing notes, and growing confidence are not proof that the system still makes sense.
- If compaction hits while source gathering or slice planning is still unpersisted, that work is gone. Re-gather it instead of pretending continuity happened.
- Keep the machine explainable in plain language and anchored to code. If a change cannot be explained against the algorithm map, it is not ready to land.

## Landed Machine

The current spine:

- lattice inference with fixed-size and searched-size paths
- low-confidence phase rerank through short preview reconstruction
- edge analysis feeding one projected displacement-field optimizer
- nearest-source final sampling from `uv0_px + disp_t`
- cleanup, optional palette fit, diagnostics, compare mode, tuning, and GUI observer events on the live path
- `source_structure` plus `source_fidelity` in run summaries and comparisons
- focused sword-tip fixture in `tests/fixtures/real/ai-badge-tip-focus.json`

The exact current control flow is documented in
`docs/lean-optimizer-algorithm-map.md`.

## Boundaries

- Do not reintroduce tray optimizers, candidate sets, portrait layers, or multi-stage solver religions casually.
- Do not trust a metric win that makes the image look worse.
- Do not let cleanup become the real solver.
- Do not let `state/evidence.jsonl` turn into an activity feed.
- Do not restart broad exploratory surgery when the current weak spot is still one bounded tapered-contour blemish.

## Verification Guardrails

- For scaffolding or note changes, run the repo-local state helper and compaction helper.
- For Python changes, run:

```powershell
.\.venv\Scripts\python -m pytest -q
```

- For solver behavior changes, reproduce the pinned badge case from
  `docs/implementation-plan.md`, inspect the sword-tip focus fixture, and
  compare both the image and `source_structure`.

## Persistent State Hygiene

Rules now in force:

- `state/map.yaml` is canonical current truth.
- `state/scratch.md` is disposable scratch.
- `state/evidence.jsonl` is a distilled durable belief ledger.
- `tools/repixelizer_prepare_compaction.py` is the pre-compaction persistence check; run it before and after imminent-compaction persistence passes.
- this handoff is a compact re-entry packet.
- `docs/implementation-plan.md` is the forward plan.
- `docs/lean-optimizer-algorithm-map.md` is the source-grounded control-flow map.

Do not let any one note become all of those things. That is how the tower grows
sideways and starts calling itself architecture.

## Next Real Move

Do not continue implementation automatically from a rehydrate-only request.

If the user asks to continue, the current next move is to take one bounded
hypothesis for tapered-contour behavior in `src/repixelizer/phase_field.py`,
check it against the pinned badge case and
`tests/fixtures/real/ai-badge-tip-focus.json`, and revert it if the visual
result or `source_structure` does not clearly improve.

## Immediate Re-entry Instruction

After compaction, first rehydrate and reorient from the listed files and git
state. Do not continue implementation merely because the state names a next
move. Wait for the user's next instruction unless they explicitly say to
continue.
