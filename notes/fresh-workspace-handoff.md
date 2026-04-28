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
- searched lattice inference now trusts autocorrelation only; it keeps a tiny near-best lag plateau and looks for cross-axis consensus before collapsing to one lattice family. the old change-interval / pixel-walk sizing path was cut after it kept steering the badge away from the `126` family, and the projected edge-energy spectral pass was also removed after floor-hugging to 2px / 626-family aliases on the dense landscape fixture
- hosted demo inference now uses one direct autocorr-consensus size, capped by the hosted output limit. it no longer silently pins unspecified jobs to `max_output_dimension`
- hosted web access now has an explicit seam in `src/repixelizer/access.py`; `src/repixelizer/gui.py` binds request subjects, guards hosted routes, and stamps queued jobs with local owner metadata
- `GC_ACCESS_MODE=heimdall` is now landed in that web layer: Repixelizer creates auth attempts, asks Heimdall to start provider OAuth, accepts the direct backend callback, verifies Heimdall Ed25519 access tokens locally, and adopts an httpOnly local session cookie
- The low-confidence candidate rerank path is a short preview solve, not a second full optimizer.
- the phase field now has a wider displacement budget, so moderate starting-grid mistakes are allowed to get corrected by the field instead of relying purely on candidate-rerank ceremony
- Cleanup is secondary and usually a no-op; the core result is supposed to come from the solver, not cleanup cosplay.
- The current tracked weakness is the widened dark contour near the badge sword tip on tapered shapes.
- `source_structure` exists because lattice-only `source_fidelity` could call visibly better outputs worse.
- The old tray optimizer is dead and should stay dead unless the entire machine map changes for a real reason.
- `E:\Projects\Heimdall\docs\architecture.md` is the future shared design for reusable GameCult-hosted access across experiments.
- `E:\Projects\Heimdall\docs\app-profiles\repixelizer.md` is the Repixelizer-specific binding onto that future shared access layer.
- current hosted-demo truth lives in `src/repixelizer/gui.py`: queue and limit machinery, hosted direct-autocorr inference defaults, provider-start / callback / adopt / logout auth routes, and a hosted-only landing page at `/` while self-host/local runs still redirect root straight to `/app/`.
- product strategy is hosted convenience first: keep the code open, let self-hosters suffer voluntarily, and do not burn time on native desktop packaging unless it clearly beats hosting on revenue, support, or strategy.

## Critical Doctrine

- Persistent state is the agent's mind.
- Cut persistent memory as ruthlessly as code; stale context is bad thought, not harmless clutter.
- Remember Jenga: growing diffs, growing notes, and growing confidence are not proof that the system still makes sense.
- If compaction hits while source gathering or slice planning is still unpersisted, that work is gone. Re-gather it instead of pretending continuity happened.
- Keep the machine explainable in plain language and anchored to code. If a change cannot be explained against the algorithm map, it is not ready to land.

## Landed Machine

The current spine:

- lattice inference with fixed-size and searched-size paths
- searched lattice inference currently trusts autocorrelation only, not the rejected pixel-walk or spectral priors
- hosted demo inference uses the direct autocorr path and keeps candidate rerank off
- hosted route auth belongs in `src/repixelizer/access.py` + `src/repixelizer/gui.py`, not in the solver or pipeline
- the first Heimdall consumer path is live there: provider buttons on the landing page, direct backend callback handoff, local JWT verification, and cookie adoption
- low-confidence candidate rerank through short preview reconstruction
- edge analysis feeding one projected displacement-field optimizer with a wider displacement leash
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
- Do not drift into desktop-app fantasies just because the web GUI is nice now. The default commercial path is still hosted access.
- Do not let future Heimdall work smear auth logic into `src/repixelizer/pipeline.py`, `src/repixelizer/inference.py`, or the solver stack. The seam is already in the hosted web layer; keep using it.

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
- `E:\Projects\Heimdall\docs\architecture.md` is future shared auth/access design, not live implementation truth.
- `E:\Projects\Heimdall\docs\app-profiles\repixelizer.md` is the app-specific future binding, also not live implementation truth.
- local `docs/gamecult-hosted-access-architecture.md` and
  `docs/repixelizer-hosted-access-profile.md` are redirect stubs.

Do not let any one note become all of those things. That is how the tower grows
sideways and starts calling itself architecture.

## Next Real Move

Do not continue implementation automatically from a rehydrate-only request.

If the user asks to continue, choose the branch deliberately:

- auth branch: keep working only in `src/repixelizer/access.py` and
  `src/repixelizer/gui.py` for bounded Heimdall polish such as provider
  expansion, session UX, or persistence hardening
- solver branch: take one bounded hypothesis for tapered-contour behavior in
  `src/repixelizer/phase_field.py`, check it against the pinned badge case and
  `tests/fixtures/real/ai-badge-tip-focus.json`, and revert it if the visual
  result or `source_structure` does not clearly improve

## Immediate Re-entry Instruction

After compaction, first rehydrate and reorient from the listed files and git
state. Do not continue implementation merely because the state names a next
move. Wait for the user's next instruction unless they explicitly say to
continue.

## Active Manual Tuning State

Manual tuning is in progress; see `state/scratch.md` for the full score ledger. Do not jump straight to editing `config/solver_params.json`.

Current evidence:
- Full validation, 38 cases x soft/crisp/ai, seed 23, steps 48: baseline score `0.7702`; `radius=0.60, blend=0.07` score `0.5790`, mean `0.2814`, worst `1.2734`, improved `97/114` rows.
- Regression guard on known bad rows says tighter radius may be safer: `radius=0.50, blend=0.07` scored `0.4075` with worst `0.6949` on the guard set, beating `radius=0.60, blend=0.07` at `0.6278` / worst `1.3589`.

Immediate re-entry for tuning:
- full-validate `radius=0.50, blend=0.07`
- full-validate `radius=0.55, blend=0.05`
- compare against the existing full-validation `radius=0.60, blend=0.07` results in `artifacts/tuning-validation-01`
- only then update `config/solver_params.json`
