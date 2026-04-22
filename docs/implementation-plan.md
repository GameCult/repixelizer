# Repixelizer Implementation Plan

## Current state

The project is bootstrapped and runnable as a standalone Python repo.

Implemented:
- package metadata and editable install
- CLI entrypoint with `run` and `compare` modes
- lattice/phase inference with coherence scoring plus source-period prior
- source-image structural analysis, with optional Torch/CUDA paths
- PyTorch-based continuous UV-field optimization
- a shared source-lattice reference built from actual inferred cell assignments
- source-first snap/refine scoring with explicit source-vs-representative weights
- low-confidence phase reranking with soft size penalties instead of hard size-jump rejection
- an experimental `tile-graph` reconstruction path with coord-local literal source-pixel candidates
- stage-aware diagnostics writing, including per-stage source-fidelity and rerank traces
- synthetic test fixtures and automated tests

Verified:
- local editable install in a dedicated venv
- full test suite currently passing (`56 passed`)
- compare-mode smoke run against a real emblem image
- the cleaned real badge fixture in `tests/fixtures/real/ai-badge-cleaned.png` still beats naive resize on source-lattice consistency on the continuous path
- the latest full-emblem tile-graph atomic probe under `artifacts/full-emblem-tile-graph-atomic-v3-cuda/` lands at `0.0224` source-fidelity, beating the older full-emblem tile-graph CUDA baseline at `0.0283`

## Status after adjacency-first pass

What changed:
- `src/repixelizer/source_reference.py` now owns the lattice-indexed source reference used by metrics, the solver, and phase reranking
- snap and refine prefer sharp per-cell source evidence and can keep the relaxed-mode neighborhood in the final discrete choice
- `run.json` now includes `source_fidelity` for `snap_initial`, `solver_target`, and `final_output`, plus `phase_rerank_candidates`
- `scripts/render_focus_crop.py` prints stage-level source-fidelity when generating the guard/tip crop sheet

What that fixed:
- the cleaned badge regression no longer loses badly to naive resize once the continuous solver starts from the selected lattice
- the continuous solver no longer regresses relative to its own snap stage on the added thin-feature regression case

What is still provisional:
- the cleaned badge still lands on the conservative `126x126` lattice candidate with low confidence, so lattice selection is improved but not "done"
- tuning has not been rerun against the newer source-first weights
- the baked-checkerboard badge background remains a separate background-suppression problem

## What is still provisional

The current implementation should still be treated as a strong experimental baseline, not a final algorithm.

The most provisional areas are:
- lattice inference on real messy generated imagery
- edge-cell reliability and candidate coverage in the continuous stage
- the experimental tile-graph reconstruction path, especially on real noisy inputs
- low-confidence reranking once multiple plausible lattice sizes survive the soft penalty
- compare-mode metrics as a proxy for actual visual quality

## Near-term priorities

### Active pass

This pass replaced the custom Python source-region flood fill with a device-side connected-components pass, which is the right seam for the GPU labeling work the tile-graph path needed.

What landed:
- `tile_graph.py` now labels source-side atomic regions through a Torch CCL pass instead of a Python BFS when the source-region extractor runs on a device-backed path
- the region builder still projects those source regions onto output coords and layers them together with the older sharp/edge anchors
- tile-graph still keeps its initial assignment whenever the propagation loop would worsen source-lattice fidelity
- the focused CUDA-vs-CPU regression still matches after the CCL swap, and the full suite is green at `56 passed`

Current implementation note:
- source-region labeling is now device-backed, but the one-cell window cutting pass still runs as a CPU queue walk over the labeled members, so candidate extraction remains the dominant runtime cost on large real fixtures
- on the cleaned badge, the current tile-graph model build now takes about `9.48s` on CPU versus `6.37s` on CUDA while producing the same `2499` source regions and `23223` candidates
- the latest full-emblem atomic probe under `artifacts/full-emblem-tile-graph-atomic-v3-cuda/` lands at `0.0224` source-fidelity with `34278` candidates and about `2.16` average choices per cell
- that beats the earlier full-CUDA tile-graph baseline (`0.0283`) and materially improves on the first atomic-only attempt (`0.1571`), which chose atomic regions too eagerly and then let the solver blur them out again
- the latest cleaned-badge tile-graph CUDA probe under `artifacts/badge-tile-graph-ccl-cuda/` completes end-to-end and keeps its initial assignment at `0.1800` source-fidelity, which is not a quality win but confirms the new CCL seam is stable on the real stress fixture

Next after that:
- move the one-cell window cutting pass closer to the user's original "consume one-cell regions and step through larger regions" design now that the region ownership itself is source-side
- port the per-component window cutting pass back toward a fully device-friendly formulation instead of only accelerating the labeling step
- rerank low-confidence top lattice candidates with short real solver probes instead of relying only on the cheapest preview
- evaluate whether the tile-graph propagation loop should become more contour-aware or simply stay more conservative now that the initial assignment is stronger
- rerun tuning after the tile-graph objective and candidate sets stabilize
- decide whether the tile-graph path should stay an alternate solver or become a candidate generator for the continuous refine stage

## High-value regression case

### AI badge emblem with baked checkerboard background

This locally generated AI badge remains a useful stress case because it fails in a specific, repeatable way.

Why it is useful:
- the global target size is broadly plausible, so the failure is not just "picked the wrong lattice"
- the sword tip and guard contain sharp single-pixel and near-single-pixel features at multiple orientations
- those features are exactly the kind of local adjacency pattern Repixelizer is supposed to preserve

Current failure modes:
- the right side of the sword tip can still wobble instead of tapering cleanly
- thin oblique outline features can still collapse into mushy transitions
- the right-hand guard wing can still lose short dark-light-dark adjacency motifs
- on tile-graph specifically, the remaining failure is less about wrong colors entering the candidate pool and more about the propagation step preferring a smoother local arrangement than the sharper initial assignment

Current status:
- on the cleaned fixture, the continuous path now lands at `0.0832` source-fidelity in the reproducible probe under `artifacts/badge-final-probe/`, which is materially better than the older documented `0.1494` result and better than naive resize under the same lattice
- on the full emblem, the tile-graph atomic path under `artifacts/full-emblem-tile-graph-atomic-v3-cuda/` now reaches `0.0224`, which is the best tile-graph full-emblem result so far in this repo
- the remaining weakness on tile-graph is not candidate color purity anymore; it is candidate extraction cost plus deciding how aggressive the propagation step should be once the initial assignment is already strong
- on the cleaned badge, the new CCL-backed tile-graph run under `artifacts/badge-tile-graph-ccl-cuda/` still lands at a weak `0.1800`, so the current CCL work should be treated as an infrastructure/performance step rather than a visual-quality fix

Important note:
- the checkerboard is baked into the source image, not transparency
- when projected onto a `122x122` output grid it aliases into a visible `2-3-2` cadence
- that background rhythm is expected from resampling the background pattern and should not be mistaken for proof that the inferred lattice itself is varying in size

What future work should improve here:
- edge-aware source reliability so high-dispersion but high-contrast cells keep trusting source evidence
- richer candidate generation so continuous snap/refine can reach sharp exemplars, edge peaks, and guided offsets instead of only a fixed local grid
- short top-k refine probes for low-confidence lattice candidates before the final rerank decision
- an atomic candidate builder that more closely matches real source-region ownership and consumption
- a GPU-friendly reformulation of the atomic candidate extraction stage
- some form of coordinated local relaxation so nearby cells can move together toward a globally better contour

## Suggested next milestones

### Milestone 1: Real-image evaluation harness

- add curated real-image fixtures
- add golden metadata for expected target-size ranges
- add a repeatable batch runner
- emit one summary table per corpus run

### Milestone 2: Tile-graph extraction pass

- extract atomic source regions in a way that better matches actual source texel ownership
- preserve the current sharp/edge anchors as a safety net
- port the heavy extraction work back toward a GPU-friendly formulation

### Milestone 3: Solver tuning pass

- tune continuous-stage loss weights against the benchmark corpus after the next edge-aware and candidate-guided pass lands
- tune tile-graph propagation aggressiveness now that the initial-assignment fallback is in place
- reduce cases where either solver underperforms the naive baseline or settles on the wrong low-confidence lattice

### Milestone 4: Coordinated contour moves

- add contour-aware multi-cell refinement moves
- protect short dark-light-dark edge motifs during block updates
- reduce cases where locally correct cells still fail to form a globally smooth outline

## Risks

- the solver can still produce outputs that score well but feel over-smoothed
- real generated images may contain multiple inconsistent local lattice cues
- metrics that reward coherence can accidentally reward blandness if not balanced carefully
- the current atomic candidate extraction can become too expensive on large fixtures unless it is reworked again

## Assumptions

- single-image quality matters more than batch throughput in the current phase
- palette constraints are optional and should stay optional
- the project is better served by improving automatic behavior first than by rushing into manual tooling
- Python plus PyTorch remains the right stack for the current research/prototyping phase
