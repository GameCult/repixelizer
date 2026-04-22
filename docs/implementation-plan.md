# Repixelizer Implementation Plan

## Current state

The project is bootstrapped and runnable as a standalone Python repo.

Implemented:
- package metadata and editable install
- CLI entrypoint with `run` and `compare` modes
- lattice/phase inference with coherence scoring plus source-period prior
- source-image structural analysis
- PyTorch-based continuous UV-field optimization
- shared source-lattice reference built from actual inferred cell assignments
- source-first snap/refine scoring with explicit source-vs-representative weights
- low-confidence phase reranking with soft size penalties instead of hard size-jump rejection
- discrete cleanup on the output grid
- optional palette loading and quantization modes
- compare mode with naive and diffusion baselines
- stage-aware diagnostics writing, including per-stage source-fidelity and rerank traces
- synthetic test fixtures and automated tests

Verified:
- local editable install in a dedicated venv
- full test suite currently passing
- compare-mode smoke run against a real emblem image
- focused real-fixture probe on `tests/fixtures/real/ai-badge-cleaned.png` now beats naive resize on source-lattice consistency under the selected lattice (`0.0832` final vs `0.1304` naive with the current `126x126` / `(0.0, -0.2)` pick)

## Status after adjacency-first pass

What changed:
- `src/repixelizer/source_reference.py` now owns the lattice-indexed source reference used by metrics, the solver, and phase reranking
- snap and refine now prefer sharp per-cell source evidence and keep the relaxed-mode neighborhood in the final discrete choice
- `run.json` now includes `source_fidelity` for `snap_initial`, `solver_target`, and `final_output`, plus `phase_rerank_candidates`
- `scripts/render_focus_crop.py` now prints stage-level source-fidelity when generating the guard/tip crop sheet

What that fixed:
- the cleaned badge regression no longer loses badly to naive resize on adjacency and motif preservation once the solver starts from the selected lattice
- the solver no longer regresses relative to its own snap stage on the added thin-feature regression case

What is still provisional:
- the cleaned badge still lands on the conservative `126x126` lattice candidate, just with a much lower confidence than before; that means candidate selection and weight tuning are improved but not “done”
- tuning has not been rerun yet against the new source-first hyperparameters, so the benchmark/tuning baseline should be treated as stale until the next sweep
- the baked-checkerboard version of the badge remains a separate background-suppression problem

## What is still provisional

The current implementation should be treated as a strong experimental baseline, not a final algorithm.

The most provisional areas are:
- lattice inference on real, messy generated imagery
- the exact loss weighting in the continuous stage
- the strength and shape of discrete cleanup heuristics
- compare-mode metrics as a proxy for actual visual quality

## Near-term priorities

## High-value regression cases

### AI badge emblem with baked checkerboard background

This locally generated AI badge has become a useful stress case because it fails in a very specific, repeatable way.

Why it is useful:
- the global target size is broadly plausible, so the failure is not just "picked the wrong lattice"
- the lower sword-tip region contains sharp single-pixel and near-single-pixel features at multiple orientations
- those features are exactly the kind of local adjacency pattern Repixelizer is supposed to preserve

Current failure modes:
- the right side of the sword tip develops a wobbling outline instead of a smooth taper
- thin oblique outline features around the tip get collapsed into mushy transitions
- local cell assignments near the tip look inconsistent even when the overall badge size feels right
- the right-hand guard wing loses short dark-light-dark adjacency motifs and collapses into a chunky gold slab
- some guard closeups look nearly identical across `snap`, `relaxed`, and `final`, which suggests the wrong local motif is often chosen before later refinement even starts

Current status after this pass:
- on the cleaned fixture, `snap`, `relaxed`, and `final` now all preserve source-lattice structure substantially better than the old baseline, with `final` currently landing at `0.0832` source-fidelity score in the reproducible probe under `artifacts/badge-final-probe/`
- that is a meaningful improvement over both the previous documented `0.1494` final score and the naive resize score under the same selected lattice
- the remaining weakness is candidate choice confidence, not a collapse inside the snap/refine handoff

Important note:
- the checkerboard is baked into the source image, not transparency
- when projected onto a `122x122` output grid it aliases into a visible `2-3-2` cadence
- that `2-3-2` background rhythm is expected from resampling the background pattern and should not be mistaken for proof that the inferred lattice itself is varying in size
- it is still a problem, because the background is treated as real source structure and can contaminate local scoring near the blade

What future work should improve here:
- better weighting of thin local outline/motif preservation relative to high-contrast cell transitions
- some form of coordinated local relaxation so nearby cells can move together toward a globally better contour
- optional background suppression or de-weighting when a baked checkerboard is clearly not semantic content

Current optimizer diagnosis:
- a lot of the so-called `source_*` terms in the solver are still measured against the smoothed representative lattice, not raw source-side candidate evidence
- that means the optimizer often "preserves" a softened, already-collapsed interpretation of the source instead of the sharper local adjacency patterns we actually care about
- snap is still especially guilty here: its base match and neighbor deltas are dominated by the representative lattice, so the wrong local motif can be locked in before relaxation or hard refinement even starts
- relaxation can often descend to a much lower expected energy basin, but the hardening/refine handoff then projects that soft solution back into a worse discrete assignment
- diagonal structure is also underrepresented in the final structure score, which is a bad fit for curved one-pixel outlines and hooks

Most likely next fixes:
- rerun tuning so the new source-first weights are benchmarked against the corpus instead of being judged only by defaults
- keep pressure on low-confidence badge-like cases where larger lattice candidates have better support but still lose on the combined rerank score
- evaluate whether the compare-mode summary should surface per-stage source-fidelity directly, not only the final output metrics

Repository fixtures:
- `tests/fixtures/real/ai-badge-cleaned.png` is the manually cleaned transparent version of the same emblem
- `tests/fixtures/real/ai-badge-cleaned.json` records why it matters
- use the cleaned fixture when isolating lattice and contour failures from background-removal failures
- `scripts/render_focus_crop.py` generates reproducible docs/debug closeups from output-grid cell coordinates
- the current docs-facing guard crop uses `--cell-bbox 73 20 107 44`

### 1. Build a real benchmark corpus

Add curated real-world cases, not just synthetic ones:
- emblems and logos with transparency
- icons with strong outlines
- simple sprite-like figures
- examples that are known to fail naive resize-and-dither

Each case should include:
- source image
- expected target size or plausible size range
- short notes on known failure modes

### 2. Improve lattice inference

The current inference is workable but still heuristic.

Next improvements should focus on:
- better periodicity detection for noisy or partially aligned images
- stronger penalties against oversized candidate grids
- confidence calibration that reflects ambiguity more honestly

### 3. Strengthen the discrete cleanup stage

Current cleanup is local and generic.

Next version should add:
- stronger outline continuity rules
- better preservation of highlight bands
- better handling of metallic or gem-like materials
- smarter alpha-edge cleanup for transparent icons

### 4. Make compare mode more trustworthy

The current metrics are useful but still internal.

Next work should:
- expand score breakdowns
- validate whether the metrics correlate with actual human preference
- make it easier to compare runs across parameter sweeps

## Suggested next milestones

### Milestone 1: Real-image evaluation harness

- add curated real-image fixtures
- add golden metadata for expected target-size ranges
- add a repeatable batch runner
- emit one summary table per corpus run

### Milestone 2: Solver tuning pass

- tune continuous-stage loss weights against the benchmark corpus
- test more robust anti-speckle and cluster-preservation behavior
- reduce cases where the solver underperforms the naive baseline

### Milestone 3: Better cleanup heuristics

- add contour-aware cleanup
- add material-sensitive smoothing for highlights and trim
- reduce the number of visually good-but-metric-weak outputs

### Milestone 4: Human-in-the-loop v2 planning

Only after the automatic baseline is strong:
- explore optional region hints
- evaluate a minimal GUI or notebook layer
- consider multi-frame support

## Risks

- The solver can still produce outputs that score well but feel over-smoothed.
- Real generated images may contain multiple inconsistent local lattice cues.
- Continuous optimization is flexible, but easy to overfit to the reconstruction term.
- Metrics that reward coherence can accidentally reward blandness if not balanced carefully.

## Assumptions

- Single-image quality matters more than batch throughput in the current phase.
- Palette constraints are optional and should stay optional.
- The project is better served by improving automatic behavior first than by rushing into manual tooling.
- Python plus PyTorch remains the right stack for the current research/prototyping phase.
