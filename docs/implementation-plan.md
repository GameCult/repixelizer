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
- full test suite currently passing before the current speed/refactor pass (`69 passed` once the new grouped-extraction path landed)
- compare-mode smoke run against a real emblem image
- the cleaned real badge fixture in `tests/fixtures/real/ai-badge-cleaned.png` still beats naive resize on source-lattice consistency on the continuous path
- the latest full-emblem tile-graph atomic probe under `artifacts/full-emblem-tile-graph-atomic-v3-cuda/` lands at `0.0224` source-fidelity, beating the older full-emblem tile-graph CUDA baseline at `0.0283`
- the new algorithm map in `docs/tile-graph-algorithm-map.md` confirms that the pinned `126x126` tile-graph badge collapse is already present in the tile-graph initial assignment; the fixed-lattice pipeline path itself is reproducing that bad state faithfully rather than introducing it, and the newer extraction coverage fix now guarantees that occupied output cells are not silently losing their source-region bucket under the corrected full-size lattice mapping
- the new architectural split kept the fixed synthetic emblem baselines byte-identical on both engines at the moment it landed: the pinned tile-graph output hashed to `31e3bc2c...6ae9` and the pinned continuous output hashed to `404748af...7903`

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

This pass is the first serious speed pass after the algorithm cleanup: replace the expensive per-component window-walk extractor with a grouped `(component_id, output_cell)` overlap reducer and then measure the tradeoff honestly.

What landed:
- `tile-graph` no longer participates in pipeline phase rerank probes; rerank is now a continuous-only wrapper
- the experimental `hybrid` path and its geometry-prior wiring have been removed
- `tile_graph.py` no longer carries geometry-prior fields through `TileGraphModel` or tile-graph unary scoring
- `tile-graph` now skips source clustering entirely and uses only edge analysis plus direct per-cell source summaries during unary scoring
- `analysis.py` now has separate prep paths, but both are edge-only after the first optimizer cut; the continuous k-means cluster scout was removed because its only production job was turning coarse color partitions into fake edge guidance
- `source_reference.py` was split so tile-graph could have its own lean prep contract, but the next cut removed tile-graph's need for any separate source-reference object at all
- `TileGraphModel` is now solver-only state; cache metadata and build diagnostics live in a separate `TileGraphBuildStats` sidecar
- `cli.py` now exposes only `continuous` and `tile-graph` as reconstruction engines
- `tests/test_pipeline.py` now checks that low-confidence tile-graph runs skip rerank probes instead of rebuilding probe candidates
- `tests/test_analysis.py` and `tests/test_tile_graph.py` now cover the new edge-only analysis path
- `tests/test_pipeline.py` now locks in fixed continuous and tile-graph output hashes from a pre-refactor baseline so this architectural cut cannot silently drift behavior
- `docs/tile-graph-algorithm-map.md` now describes the smaller machine directly rather than documenting the removed hybrid sidecar path
- the old per-component queue/seed/window extractor is gone; `_extract_source_region_tiles(...)` now directly reduces `(component_id, output_cell)` overlaps into candidate shards
- the old stroke-aware slicing machinery and its tuning knobs are gone along with that extractor
- same-component continuation now defines tile-graph adjacency directly, replacing the previous search-heavy neighbor heuristic inside extraction

Current implementation note:
- the earlier profiling result still stands conceptually: on the searched badge path, the solver loop is not the problem
- the direct-control path is specifically for dodging those costs during iteration when we already know which lattice we want to inspect
- tile-graph now skips pipeline rerank entirely, so iteration no longer rebuilds multiple probe candidates before the final tile-graph solve
- on the cleaned badge at pinned `126x126` / phase `(0.0, -0.2)`, the new grouped-extraction CUDA path now takes about `16.1s` on the first same-process run and about `6.0s` on the cached rerun, with `tile_graph_model_cache_hit=True` on the second pass
- the current deep-dive diagnosis is that the fixed-lattice garbling is not a wrapper bug and not mainly a parity-solver bug; it is born in tile cutting, candidate truncation, local source support, and the first unary argmin assignment
- the extraction coverage bug turned out to be real but secondary: the new `_extract_source_region_tiles(...)` overlap fill pass now guarantees that any output cell containing opaque sampled source pixels gets at least one extracted region bucket, but the pinned `126x126` badge output remains numerically unchanged (`0.4998626`), which narrows the remaining failure to candidate ranking and lattice-conditioned reference usage rather than empty region buckets
- the latest full-emblem atomic probe under `artifacts/full-emblem-tile-graph-atomic-v3-cuda/` lands at `0.0224` source-fidelity with `34278` candidates and about `2.16` average choices per cell
- that beats the earlier full-CUDA tile-graph baseline (`0.0283`) and materially improves on the first atomic-only attempt (`0.1571`), which chose atomic regions too eagerly and then let the solver blur them out again
- the latest core algorithm cut removes `source_region_stride`, removes portrait-based candidate ranking, removes sharp/edge foreground fallback injection, and replaces sampled-delta pairwise scoring with adjacency learned from extracted tiles
- that cut also removes the dedicated `TileGraphSourceReference` entirely: tile-graph now grounds itself in full-resolution extracted tiles plus direct per-cell source summaries (`cell_mean_rgba`, `cell_alpha_mean`, `cell_edge_strength`)
- on the cleaned badge at pinned `126x126` / phase `(0.0, -0.2)`, the portrait-free cut first recovered the fixed-lattice run from roughly `0.5055` source-fidelity down to `0.1814`; the new grouped-extraction speed pass regresses a bit to `0.2036` while making the same case roughly `6x` faster cold on CUDA
- a fresh CPU profile of the grouped extractor shows the bottleneck has moved: `build_tile_graph_model(...)` now takes about `61.0s` instead of `145.8s`, and `_extract_source_region_tiles(...)` no longer dominates; the new hot spot is `_segment_atomic_source_regions(...)` at about `59.5s`

Next after that:
- split tile-graph iteration into a cached model-build phase and a near-free solve phase so weight tuning does not keep paying the `~131s` source-region cutting bill
- extend the current fixed-lattice path so repeated CLI runs can reuse cached model artifacts across processes instead of only within one Python process
- replace or accelerate `_segment_atomic_source_regions(...)`, because that is now the cold-build bottleneck rather than extraction
- decide whether the continuous path should stay the only global search/rerank engine or whether tile-graph eventually needs its own lighter lattice chooser
- evaluate whether the tile-graph propagation loop should become more contour-aware or simply stay more conservative now that the initial assignment is stronger
- rerun tuning after the tile-graph objective and candidate sets stabilize
- decide whether the tile-graph path should stay an alternate solver or become a candidate generator for the continuous refine stage

### Optimizer map

That documentation pass is now in `docs/optimizer-algorithm-map.md`.

The new map does for `continuous.py` what `docs/tile-graph-algorithm-map.md` does for tile-graph:

- walks the optimizer from pipeline entry to final output
- names the real state variables the machine carries (`uv0_t`, `initial_representative_t`, `source_lattice_reference`, `source_reliability_t`, `snap_t`)
- explains the regular UV lattice, representative portrait, source lattice portrait, source-first snap, relaxed handoff, and discrete refine stages in plain language
- identifies the main contradictions still living in the optimizer, especially:
  - the machine maintaining two overlapping portraits of the same lattice and spending much of its complexity mediating between them
  - adjacency / motif / line structure being expressed repeatedly in several slightly different dialects

The first optimizer cut landed immediately after that map:

- `optimize_uv_field(...)` was renamed to `optimize_lattice_pixels(...)`
- the unused `_exemplar_colors(...)` helper and its test were removed
- continuous analysis stopped building a k-means `cluster_map`
- `continuous.py` now uses only real source edges for guidance instead of blending in coarse cluster boundaries
- the pinned cleaned-badge continuous smoke run under `artifacts/optimizer-cut-v1-badge-126/` still behaves coherently after the cut: snap scores `0.08100`, final scores `0.07535`, and the generated preview is `diagnostics/output-preview.png`

The second optimizer cut is structural:

- `_prepare_optimizer(...)` now owns source tensor creation, UV construction, edge gradients, source lattice reference building, detail reference building, source reliability, source deltas, guidance, and cell geometry
- `_source_detail_delta_tensors(...)` computes premultiplied source-detail deltas once instead of repeating the same `premultiply(...)` calls for each axis
- `optimize_lattice_pixels(...)` is now an orchestration shell around prep, snap, refine, the source-fidelity guardrail, and artifact packaging
- the pinned cleaned-badge continuous smoke run under `artifacts/optimizer-cut-v2-prep-badge-126/` is byte-size identical to the previous preview and preserves the same source-fidelity numbers: snap `0.08100`, final `0.07535`

Next after that:

- add a focused ablation runner for pinned badge/emblem cases before cutting more behavior
- collapse duplicated adjacency / motif / line voices only after ablation shows which stage owns each idea
- put relax on trial now that refine itself is source-first

The third optimizer cut is behavioral:

- refine no longer scores candidates against the representative portrait
- `_structure_score(...)` no longer includes representative-match as a final tiebreaker
- the dead refine/structure representative knobs were removed from `SolverHyperParams` and the tuning harness
- the pinned cleaned-badge continuous smoke run under `artifacts/optimizer-cut-v3-source-refine-badge-126/` improved slightly: snap stayed `0.08100`, final moved from `0.07535` to `0.07485`

The fourth optimizer cut trims relax rather than removing it:

- the old `refine_relaxed_mode_weight` bonus is gone, so greedy refine is no longer biased toward the relaxed-mode selection on every candidate score
- the core relax stage is still kept, because the earlier ablation showed that turning relax off entirely made the pinned badge worse
- the pinned cleaned-badge continuous smoke run under `artifacts/optimizer-cut-v4-relax-bonus-badge-126/` improved again: snap stayed `0.08100`, final moved from `0.07485` to `0.07451`

Next after that:

- keep putting relax on trial, but now term-by-term instead of as a monolith
- add a focused ablation runner for pinned badge/emblem cases before cutting more behavior
- collapse duplicated adjacency / motif / line voices only after ablation shows which stage owns each idea

The next pass turned that "put relax on trial" note into an actual measurement seam:

- `SolverArtifacts` now carries stage diagnostics for continuous runs, including the exact selected-source displacement field for `snap`, `relax_handoff`, `relax_mode`, and `final_output`
- `run.json` now includes `optimizer_displacement` metrics, and diagnostics directories now write color-coded displacement previews for those stages
- the pinned cleaned-badge comparison under `artifacts/optimizer-relax-purpose-check/` runs the same fixed lattice twice: once with normal relax and once with `relax_iterations=0`

What that test says:

- `relax` still helps a little on the real badge objective: final source-fidelity improves from `0.07549` without relax to `0.07451` with relax
- but the displacement field does **not** become clearly smoother in the way the original design story promised
- on that same pinned badge run, final orthogonal displacement jitter is `1.7154` with relax versus `1.6783` without it, and final local residual is `1.1548` with relax versus `1.1312` without it
- the honest read is that the current relax stage behaves more like a soft consensus / handoff stabilizer over fixed local candidate trays than a true large-swath phase adjustment mechanism

That means the next optimizer cuts should treat `relax` as a measurable subsolver with a narrower actual job, not as a sacred phase-field story we keep around out of nostalgia.

The next optimizer simplification pass should use that map as its cutting checklist instead of guessing from scattered helper names.

### Lean optimizer restart target

The next real optimizer move is not another incremental trim on the old tray-based solver. It is a restart toward the smaller machine now mapped in `docs/lean-optimizer-algorithm-map.md`.

What that new map says plainly:

- keep lattice inference and edge scouting
- keep one fixed grid of cell centers
- attach one displacement vector `(dx, dy)` to each output cell, initialized to zero
- optimize that displacement field directly
- use a tiny objective with only:
  - local solidity / coherence
  - smoothness
  - edge-aware smoothness
  - anti-collapse / anti-fold constraints
  - displacement magnitude regularization
- sample the final output once from the optimized field

What that restart explicitly rejects:

- `representative_t`
- `source_lattice_reference`
- `source_reliability_t`
- tray-based snap / relax / refine as the core optimizer state
- repeated adjacency / motif / line dialects trying to compensate for each other

Why this is the right next cut:

- the current optimizer map now makes the contradiction impossible to ignore: the code is not really optimizing phase directly anymore
- the relax diagnostics proved that one of the old justifications was overstated; the machine gets a tiny score win from relax, but not the promised broad-swatch phase drift
- the user's target description is cleaner than the current code: find cell size, initialize a phase field at zero, wiggle it into solid areas coherently, sample, done

Next implementation steps:

1. compare the new `phase-field` mode against pinned badge and emblem runs before porting any old heuristics
2. improve the local evidence term so the field is pulled harder into solid fake-pixel interiors
3. tune the projection / spacing story so the field can drift farther without collapsing or freezing
4. only then consider importing a small number of old heuristics if they clearly help

What has already landed:

- a brand-new `src/repixelizer/phase_field.py` module
- an experimental `--reconstruction-mode phase-field` pipeline path
- a direct displacement-field optimizer using only:
  - local coherence
  - local edge avoidance
  - edge-aware smoothness
  - anti-collapse spacing
  - displacement magnitude regularization
- pinned-badge smoke artifacts under `artifacts/phase-field-v1-badge-126/`

First honest result:

- on the cleaned badge at fixed `126x126`, phase `(0.0, -0.2)`, `steps=48`, CUDA
- the new `phase-field` mode lands at `0.1461` final source-fidelity
- that is much worse than the current continuous path (`0.0745` on the same pinned case), but dramatically better than the earlier broken tile-graph collapse and, more importantly, it is being produced by a machine that actually matches its own description

That is a respectable first brick: not good enough, but finally the right shape.

Important metric correction:

- that same first phase-field badge run is also the clearest proof that `source_fidelity` is not enough as the repo's north star
- visually, the phase-field output preserves the important internal badge structure better than the pinned continuous output
- numerically, the old `source_fidelity` score still calls it worse because it rewards agreement with the lattice reference's mean / sharp portraits
- the repo now also reports `source_structure`, a source-size structural metric that combines:
  - foreground reconstruction error
  - edge-position error
  - stroke wobble
  - edge support precision / recall / F1
  - exact-match ratio
- on the pinned badge comparison under `artifacts/phase-field-metric-check/`, that new metric finally agrees with human judgment: `phase-field` scores `0.4244` versus continuous `0.4291` where lower is better

First focused blemish pass:

- the tracked sword-tip focus fixture now lives in `tests/fixtures/real/ai-badge-tip-focus.json`
- the next phase-field pass tried exactly the three obvious fixes:
  - stronger local evidence
  - less timid edge-aware smoothness gating
  - a better spacing story with both preferred spacing and an upper spacing clamp
- the resulting pinned badge run is under `artifacts/phase-field-v2-badge-126/`
- this did move the specific blemish a little in the right direction, but only a little:
  - on the tip focus crop, `source_structure` improved from `0.4187` to `0.4179`
  - on the whole image, `source_structure` regressed slightly from `0.4244` to `0.4248`
- honest read: the pass shaved the targeted contour swelling a bit, but not enough to call the issue solved, and it did so by spending a little of the whole-image win

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
- the new phase-field path has one obvious blemish on the pinned badge: the dark contour along the sword tip widens too much in a localized patch now recorded in `tests/fixtures/real/ai-badge-tip-focus.json`

Current status:
- on the cleaned fixture, the continuous path now lands at `0.0832` source-fidelity in the reproducible probe under `artifacts/badge-final-probe/`, which is materially better than the older documented `0.1494` result and better than naive resize under the same lattice
- on the full emblem, the tile-graph atomic path under `artifacts/full-emblem-tile-graph-atomic-v3-cuda/` now reaches `0.0224`, which is the best tile-graph full-emblem result so far in this repo
- the remaining weakness on tile-graph is not candidate color purity anymore; it is candidate extraction cost plus deciding how aggressive the propagation step should be once the initial assignment is already strong
- on the cleaned badge, the new CCL-backed tile-graph run under `artifacts/badge-tile-graph-ccl-cuda/` still lands at a weak `0.1800`, so the current CCL work should be treated as an infrastructure/performance step rather than a visual-quality fix
- the new grouped-extraction badge run under `artifacts/tile-graph-reducebykey-v1-badge-126/` is the first speed-focused cut that actually changes the preprocessing profile in the right direction: cold CUDA time drops to about `16.1s`, warm cached time is about `6.0s`, and the remaining quality problem is contour precision rather than total collapse
- the latest CPU profile shows that preprocessing is still the story, but the story changed: extraction is no longer the monster, and connected-component labeling / label compression is now the thing to attack next

Important note:
- the checkerboard is baked into the source image, not transparency
- when projected onto a `122x122` output grid it aliases into a visible `2-3-2` cadence
- that background rhythm is expected from resampling the background pattern and should not be mistaken for proof that the inferred lattice itself is varying in size

What future work should improve here:
- edge-aware source reliability so high-dispersion but high-contrast cells keep trusting source evidence
- richer candidate generation so continuous snap/refine can reach sharp exemplars, edge peaks, and guided offsets instead of only a fixed local grid
- short top-k refine probes for low-confidence lattice candidates before the final rerank decision
- an atomic candidate builder that more closely matches real source-region ownership and consumption
- a faster labeler or grouped label-reduction path so tile-graph can keep the new simpler overlap model without paying for label propagation so heavily
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
