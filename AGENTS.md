# Repixelizer Instructions

## Project Purpose

Repixelizer is a standalone Python CLI for forcing fake pixel art onto a real
pixel lattice without turning the solver into a landfill of compensating
heuristics.

The live machine is intentionally lean: infer a lattice, analyze structure,
optimize one displacement field, sample real source pixels, then write
diagnostics good enough to catch when we are lying to ourselves.

## Canonical State

- Treat `state/map.yaml` as the canonical current project map.
- Treat `state/scratch.md` as disposable working memory for one bounded subgoal.
- Treat `state/evidence.jsonl` as the durable distilled ledger of decisions,
  regressions, verified findings, and rejected paths.
- Treat `state/branches.json` as hypothesis tracking, not phase/status prose.
- Treat `notes/fresh-workspace-handoff.md` as the compact re-entry packet.
- Treat `docs/implementation-plan.md` as the forward plan.
- Treat `docs/lean-optimizer-algorithm-map.md` as the source-grounded live
  algorithm map.
- Treat `docs/gamecult-hosted-access-architecture.md` as the future shared
  access architecture for GameCult-hosted experiments, not as proof that auth
  machinery already exists here already.
- Treat `docs/repixelizer-hosted-access-profile.md` as the Repixelizer-specific
  binding onto that future shared access architecture.
- Update `state/map.yaml` when project understanding changes.
- Add evidence after meaningful research, implementation, verification, or
  rejected paths, but keep it distilled. Routine "I just did this" proof belongs
  in git history, diagnostics, benchmark artifacts, or commit messages unless it
  changes what the next agent should believe.

## Important Paths

- Project root: `E:\Projects\repixelizer`
- Implementation plan: `E:\Projects\repixelizer\docs\implementation-plan.md`
- Algorithm map: `E:\Projects\repixelizer\docs\lean-optimizer-algorithm-map.md`
- Shared hosted access architecture: `E:\Projects\repixelizer\docs\gamecult-hosted-access-architecture.md`
- Repixelizer hosted access profile: `E:\Projects\repixelizer\docs\repixelizer-hosted-access-profile.md`
- Handoff summary: `E:\Projects\repixelizer\notes\fresh-workspace-handoff.md`
- State CLI: `E:\Projects\repixelizer\tools\repixelizer_state.py`
- Pre-compaction helper: `E:\Projects\repixelizer\tools\repixelizer_prepare_compaction.py`
- Live pipeline: `E:\Projects\repixelizer\src\repixelizer\pipeline.py`
- Live solver: `E:\Projects\repixelizer\src\repixelizer\phase_field.py`
- Metrics: `E:\Projects\repixelizer\src\repixelizer\metrics.py`

## Useful Commands

Use the repo virtualenv when available:

```powershell
.\.venv\Scripts\python .\tools\repixelizer_state.py status
.\.venv\Scripts\python .\tools\repixelizer_state.py add-evidence --type research --status ok --note "..."
.\.venv\Scripts\python .\tools\repixelizer_prepare_compaction.py
.\.venv\Scripts\python -m pytest -q
```

Fallback if the virtualenv is unavailable:

```powershell
python .\tools\repixelizer_state.py status
python .\tools\repixelizer_prepare_compaction.py
```

## Session Bootstrap And Re-entry Protocol

On fresh session load, do this before editing:

1. read:
   - `state/map.yaml`
   - `notes/fresh-workspace-handoff.md`
   - `docs/lean-optimizer-algorithm-map.md`
   - `docs/implementation-plan.md`
2. run:
   - `.\.venv\Scripts\python .\tools\repixelizer_state.py status`
   - `git status --short --branch`
   - `git log --oneline -5`
3. restate the current next action from persisted state before touching code

After compaction, resume, or suspicious continuity loss:

1. rerun `repixelizer_state.py status`
2. reread `state/map.yaml` and `notes/fresh-workspace-handoff.md`
3. treat the persisted next action as authoritative unless fresh repo evidence
   contradicts it

When context pressure is rising:

1. stop broad exploration
2. narrow the active move to one bounded landing zone
3. persist map/handoff updates plus distilled evidence before continuity gets
   atomized

Do not wait for the blackout and then act surprised.

When the user says to prepare for imminent compaction:

1. run `tools/repixelizer_prepare_compaction.py` before editing persistence surfaces
2. use its warnings as the checklist for map, handoff, scratch, evidence, and git hygiene
3. update only the state that actually changed
4. run `tools/repixelizer_prepare_compaction.py` again after edits
5. fix errors, address warnings, and commit the completed persistence pass unless the work is deliberately mid-surgery

## Operating Discipline

- Before substantial edits, restate the current mechanism and intended change.
- Prefer one clear hypothesis per pass.
- Validate against the real objective: pinned badge runs, focused crops, visual
  diffs, and `source_structure`, not just a convenient proxy metric.
- Revert failed regression or benchmark fixes immediately before trying the next
  idea.
- If the diff grows while understanding shrinks, stop implementation and switch
  to diagnosis, mapping, or simplification.
- Keep prose and maps grounded in concrete source paths.
- Do not reintroduce trays, portrait layers, candidate sets, or multi-stage
  solver religions unless the design has actually changed and the map says so.
- Before handoff or compaction, sync `state/map.yaml`, refresh
  `notes/fresh-workspace-handoff.md`, add distilled evidence when future belief
  changed, and make the next action explicit.
