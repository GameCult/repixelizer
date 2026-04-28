# Scratch: manual solver tuning in progress

## Current Subgoal
Manual solver tuning is in progress; full-validate tighter/slower local search candidates before shipping params.

## Current bounded subgoal
Find a shippable `config/solver_params.json` update for the simplified explicit phase-field solver. Do not resume random tuner blindly. Continue deliberate parameter nudging with benchmark evidence.

## Baseline
Current committed config before tuning:
- `phase_field_local_search_radius_ratio`: `0.80`
- `phase_field_local_search_blend`: `0.18`
- `phase_field_signal_weight`: `0.25`
- `phase_field_local_search_grid_weight`: `0.0`

## Timing / benchmark setup
Manual batches use `run_roundtrip_benchmark(...)`, CPU, `steps=48`, `variants=1`, usually pinned ground-truth size (`infer_size=False`).
Progress/log artifacts are under `artifacts/manual-tuning-*`, `artifacts/tuning-validation-01`, and `artifacts/tuning-regression-guard-01`.

## Key findings so far
1. Blindly increasing signal made results worse.
   - `more-signal`: score `0.7339` vs baseline `0.6014` on 8-case soft+ai slice.
   - `signal-and-spring`: score `0.7090`.
   - Stronger attraction causes feature crowding / artifacts; do not chase bigger signal.

2. Increasing spring weight above current `4.0` did nothing on the slice.
   - `more-spring`: exactly same score as baseline.
   - Reason found in source: `_relax_lattice_springs_np` clamps spring step at `0.22`, and current `4.0` is already above the clamp threshold.

3. Smaller local search radius is the first real improvement.
   - 8-case soft+ai slice baseline: score `0.6014`, mean `0.3570`, worst `1.1717`.
   - `radius=0.60`: score `0.3746`, mean `0.2104`, worst `0.7575`.
   - Biggest wins were soft enemy cases that baseline lost to naive.

4. Slower local search blend improves further.
   - `radius=0.60, blend=0.10`: score `0.2804`, mean `0.1643`, worst `0.5513` on same slice.
   - Blend bracket showed `0.06`, `0.07`, `0.08` are basically a plateau.
   - Best small-slice objective: `radius=0.60, blend=0.07`, score `0.2687356`, mean `0.1616738`, worst `0.5185465`.
   - `radius=0.60, blend=0.08` is almost tied: score `0.2687747`.

5. Full-corpus validation across all 38 cases x soft/crisp/ai chose `radius=0.60, blend=0.07` over baseline and `blend=0.08`.
   - Baseline: score `0.7701765`, mean `0.4552439`, worst `1.5050192`.
   - `r060-b007`: score `0.5790363`, mean `0.2814493`, worst `1.2734060`.
   - `r060-b008`: score `0.5938681`, mean `0.2864235`, worst `1.3112390`.
   - `r060-b007` improved 97 / 114 rows, regressed 14 / 114, tied 3 / 114.
   - Mean deltas by profile: ai `-0.1981`, crisp `-0.1836`, soft `-0.1396`.

6. Known regression guard says `radius=0.60` still hurts specific crisp cases, especially `disciple` crisp.
   Full validation worst regression for `r060-b007`:
   - `disciple` crisp: row score `0.6688 -> 1.2734`, delta `+0.6046`.
   Other notable regressions:
   - `dagrons5-08` crisp: `1.1013 -> 1.2413`, delta `+0.1400`.
   - `dagrons5-10` ai: `0.8494 -> 0.9664`, delta `+0.1170`.

7. Regression-guard batch on the known bad cases prefers tighter radius.
   Cases: `disciple`, `dagrons5-08`, `dagrons5-10`, `dagrons5-11`, `9rpgenemies-02`, `9rpgenemies-06`, `minion`, `more-rpg-enemies-06`; profiles soft/crisp/ai; seed 23.
   - Guard baseline: score `0.8112430`, mean `0.5648541`, worst `1.3861505`.
   - `r060-b007`: score `0.6277928`, mean `0.3144768`, worst `1.3588636`.
   - `r055-b007`: score `0.5313045`, mean `0.3039784`, worst `1.0617322`.
   - `r050-b007`: score `0.4075140`, mean `0.2843421`, worst `0.6949151`.
   - `r055-b005`: score `0.4298573`, mean `0.2982637`, worst `0.7369090`.
   - `r060-b007-grid004`: score `0.6339122`, mean `0.3133518`, worst `1.3818866`.
   This means `r050-b007` or `r055-b005` need full-corpus validation before shipping. `r060-b007` is no longer automatically safe despite winning full validation, because the guard set exposes known bruises.

## Recommended next action after compaction
Run full-corpus validation for:
- baseline
- `radius=0.60, blend=0.07` if not already using existing results for comparison
- `radius=0.50, blend=0.07`
- `radius=0.55, blend=0.05`
- optionally `radius=0.55, blend=0.07`

Use all 38 cases, profiles `soft/crisp/ai`, seed `23`, steps `48`, CPU, pinned size. If `r050-b007` generalizes, it is the leading ship candidate. If it overfits the guard set, compare `r055-b005` and `r060-b007` on mean/worst/regression count.

Do not edit `config/solver_params.json` yet. The current best full-corpus validated params are `r060-b007`, but the regression guard has created a better hypothesis that still needs full validation.
