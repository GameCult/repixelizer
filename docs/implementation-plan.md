# Repixelizer Implementation Plan

## Current state

The project is bootstrapped and runnable as a standalone Python repo.

Implemented:
- package metadata and editable install
- CLI entrypoint with `run` and `compare` modes
- lattice/phase inference with coherence scoring plus source-period prior
- source-image structural analysis, now with an optional Torch/CUDA path
- PyTorch-based continuous UV-field optimization
- shared source-lattice reference built from actual inferred cell assignments, now with an optional Torch/CUDA path
- source-first snap/refine scoring with explicit source-vs-representative weights
- low-confidence phase reranking with soft size penalties instead of hard size-jump rejection
- GPU-friendly tile-graph proposal building from lattice-assigned raw source pixels plus per-cell edge/cluster scoring
- discrete cleanup on the output grid
- optional palette loading and quantization modes
- compare mode with naive and diffusion baselines
- stage-aware diagnostics writing, including per-stage source-fidelity and rerank traces
- synthetic test fixtures and automated tests

Verified:
- local editable install in a dedicated venv
- full test suite currently passing on the last completed pass (`54 passed`)
- compare-mode smoke run against a real emblem image
- focused real-fixture probe on `tests/fixtures/real/ai-badge-cleaned.png` now beats naive resize on source-lattice consistency under the selected lattice (`0.0832` final vs `0.1304` naive with the current `126x126` / `(0.0, -0.2)` pick)
- end-to-end tile-graph CUDA smoke runs on both a `24x24` emblem case and the cleaned badge fixture; the small emblem case currently drops from about `2.57s` on CPU to `0.61s` on CUDA on this machine

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
- the cleaned badge still lands on the conservative `126x126` lattice candidate, just with a much lower confidence than before; that means candidate selection and weight tuning are improved but not "done"
- tuning has not been rerun yet against the new source-first hyperparameters, so the benchmark/tuning baseline should be treated as stale until the next sweep
- the baked-checkerboard version of the badge remains a separate background-suppression problem

## What is still provisional

The current implementation should be treated as a strong experimental baseline, not a final algorithm.

The most provisional areas are:
- lattice inference on real, messy generated imagery
- edge-cell reliability and candidate coverage in the continuous stage
- the new experimental tile-graph reconstruction path, especially on real noisy inputs
- low-confidence reranking once multiple plausible lattice sizes survive the new soft penalty
- compare-mode metrics as a proxy for actual visual quality

## Near-term priorities

### Active pass

This pass refactors the experimental tile-graph mode into a GPU-friendly codepath.

What landed:
- `analysis.py` and `source_reference.py` now have Torch device paths, so the expensive per-pixel reductions can run on CPU or CUDA without changing the public CLI
- `tile_graph.py` no longer walks connected components and destructively consumes windows during model building
- instead, tile-graph now builds a small per-cell proposal pool from lattice-assigned raw source pixels, sharp exemplars, edge peaks, and cluster-aware proposal scoring
- the full tile-graph path now honors `device`, not just the final solver loop, and diagnostics now record both the model-build device and the solver device

Current implementation note:
- the destructive cluster walk was the main thing fighting CUDA; replacing it with lattice-assigned per-cell proposals keeps the “one cell, one local candidate set” design while turning extraction into top-k scoring and grouped reductions
- the new proposal builder is intentionally local-first: it keeps candidates tied to the inferred output coord and uses edge/cluster signals only to rank raw source pixels already assigned to that lattice cell

Experimental tile-graph note:
- there is now a separate `tile-graph` reconstruction mode that builds literal source-pixel candidates from lattice assignments instead of cell-averaged patches or globally reusable labels
- the raw-pixel rewrite fixed an implementation mismatch where candidates had drifted into cell-averaged patch colors instead of actual source pixels
- the later per-cell-local rewrite fixed the repeated-label and opaque-black-background failures by making candidates output-coord-scoped
- the newest GPU-friendly rewrite keeps that local-candidate behavior but replaces the old component walk with per-cell top-k proposal scoring, which is fast enough to run end-to-end on CUDA
- the current badge CUDA probe under `artifacts/tile-graph-cuda-pass/badge-cuda/` completes end-to-end at the selected `126x126` lattice with `24142` candidates and `0.0283` final source-fidelity
- the latest hard-edge candidate pass widens edge-cell proposal sets with same-cell edge neighbors and strongest same-cell edge pixels, and it does reduce some medoid-like contour picks locally
- that hard-edge pass is not a clean win yet: on the full badge fixture it increases candidate count to about `30k` and average choices to about `1.92`, but regresses full-image source-fidelity to about `0.0377`, so it remains exploratory

Next after that:
- rerank low-confidence top lattice candidates with short real solver probes instead of relying only on the cheapest preview
- improve per-cell tile-graph proposal diversity so thin contours get more than “sharp vs edge-peak” in hard cells
- separate “edge-side selection” from “more edge candidates” so we can test contour-targeted scoring without paying the current whole-image fidelity penalty
- evaluate coordinated local moves when single-cell refinement still breaks thin contours
- rerun tuning now that the tile-graph proposal and device path have changed materially
- decide whether the tile-graph path should stay a diagnostics-only experiment, become a refine-stage candidate generator, or mature into a full alternate solver

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

Current status after the adjacency-first pass:
- on the cleaned fixture, `snap`, `relaxed`, and `final` now all preserve source-lattice structure substantially better than the old baseline, with `final` currently landing at `0.0832` source-fidelity score in the reproducible probe under `artifacts/badge-final-probe/`
- that is a meaningful improvement over both the previous documented `0.1494` final score and the naive resize score under the same selected lattice
- the remaining weakness is edge-cell trust and candidate coverage, not the old snap/refine handoff collapse

Important note:
- the checkerboard is baked into the source image, not transparency
- when projected onto a `122x122` output grid it aliases into a visible `2-3-2` cadence
- that `2-3-2` background rhythm is expected from resampling the background pattern and should not be mistaken for proof that the inferred lattice itself is varying in size
- it is still a problem, because the background is treated as real source structure and can contaminate local scoring near the blade

What future work should improve here:
- edge-aware source reliability so high-dispersion but high-contrast cells keep trusting source evidence
- richer candidate generation so snap/refine can reach sharp exemplars, cell edge peaks, and gradient-guided offsets instead of only a fixed local grid
- short top-k refine probes for low-confidence lattice candidates before the final rerank decision
- a retuning sweep after the edge-aware/source-guided changes land
- some form of coordinated local relaxation so nearby cells can move together toward a globally better contour
- optional background suppression or de-weighting when a baked checkerboard is clearly not semantic content
- for the experimental tile-graph path specifically, better local proposal diversity and stronger adjacency evidence are still needed even after switching candidates back to literal source pixels
- for the experimental tile-graph path specifically, edge-cell diversity alone is not enough; the current hard-edge candidate pass shows that simply widening edge-cell choices can sharpen some contour pixels while still making the global arrangement worse
- for the experimental tile-graph path specifically, the next useful step is pruning and richer proposal families rather than more component-walk optimization, because the old CPU walk is now gone

Current optimizer diagnosis:
- the shared source-lattice reference, source-first snap/refine scoring, and relaxed-mode handoff are now in place, so the biggest remaining losses are no longer caused by the old representative-lattice collapse
- the current weak spot is that difficult thin features often live in high-dispersion cells, which still makes the solver partially distrust the source exactly where the contrast is most informative
- candidate search is also still too centered on the regular UV neighborhood; if the right edge pixel never enters the candidate set, the solver can preserve the wrong motif very consistently
- low-confidence reranking is softer than before, but it still relies on cheap probes, so larger lattice candidates can remain underexplored even when they support thin contours better

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

### 3. Improve edge-cell reconstruction

The next quality gap is no longer generic cleanup; it is preserving thin, high-contrast local structure.

Next version should add:
- edge-aware source reliability overrides for thin outlines and hard transitions
- source-guided candidate pools that are not limited to a uniform local offset lattice
- richer diagnostics for which candidate families win on real fixtures
- later, coordinated block or contour moves when single-cell greedy updates still break lines

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

- tune continuous-stage loss weights against the benchmark corpus after the edge-aware/candidate-guided pass lands
- test more robust thin-feature and contour preservation behavior
- reduce cases where the solver still underperforms the naive baseline or settles on the wrong low-confidence lattice

### Milestone 3: Coordinated contour moves

- add contour-aware multi-cell refinement moves
- protect short dark-light-dark edge motifs during block updates
- reduce cases where locally correct cells still fail to form a globally smooth outline

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
