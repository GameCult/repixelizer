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
- full test suite currently passing before the current algorithm-cut pass (`70 passed`)
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

This pass is the first pruning pass after the new algorithm map: strip away the optimizer-era sidecars that are not part of the core tile-graph machine.

What landed:
- `tile-graph` no longer participates in pipeline phase rerank probes; rerank is now a continuous-only wrapper
- the experimental `hybrid` path and its geometry-prior wiring have been removed
- `tile_graph.py` no longer carries geometry-prior fields through `TileGraphModel` or tile-graph unary scoring
- `tile-graph` now skips source clustering entirely and uses only edge analysis plus direct per-cell source summaries during unary scoring
- `analysis.py` now has separate prep paths: continuous gets edge plus cluster analysis, while tile-graph gets an edge-only scout report
- `source_reference.py` was split so tile-graph could have its own lean prep contract, but the next cut removed tile-graph's need for any separate source-reference object at all
- `TileGraphModel` is now solver-only state; cache metadata and build diagnostics live in a separate `TileGraphBuildStats` sidecar
- `cli.py` now exposes only `continuous` and `tile-graph` as reconstruction engines
- `tests/test_pipeline.py` now checks that low-confidence tile-graph runs skip rerank probes instead of rebuilding probe candidates
- `tests/test_analysis.py` and `tests/test_tile_graph.py` now cover the new edge-only analysis path
- `tests/test_pipeline.py` now locks in fixed continuous and tile-graph output hashes from a pre-refactor baseline so this architectural cut cannot silently drift behavior
- `docs/tile-graph-algorithm-map.md` now describes the smaller machine directly rather than documenting the removed hybrid sidecar path

Current implementation note:
- the earlier profiling result still stands: on the searched badge path, the solver loop is not the problem; the heavy costs are `infer_lattice(...)`, continuous-mode phase rerank probes, and `_extract_source_region_tiles(...)`
- the direct-control path is specifically for dodging those costs during iteration when we already know which lattice we want to inspect
- tile-graph now skips pipeline rerank entirely, so iteration no longer rebuilds multiple probe candidates before the final tile-graph solve
- on the cleaned badge at pinned `126x126` / phase `(0.0, -0.2)`, a fixed-lattice CUDA `tile-graph` run now took about `10.2s` on the first same-process run and about `2.1s` on the cached rerun, with the same output and a reported `tile_graph_model_cache_hit` on the second pass
- the current deep-dive diagnosis is that the fixed-lattice garbling is not a wrapper bug and not mainly a parity-solver bug; it is born in tile cutting, candidate truncation, local source support, and the first unary argmin assignment
- the extraction coverage bug turned out to be real but secondary: the new `_extract_source_region_tiles(...)` overlap fill pass now guarantees that any output cell containing opaque sampled source pixels gets at least one extracted region bucket, but the pinned `126x126` badge output remains numerically unchanged (`0.4998626`), which narrows the remaining failure to candidate ranking and lattice-conditioned reference usage rather than empty region buckets
- the latest full-emblem atomic probe under `artifacts/full-emblem-tile-graph-atomic-v3-cuda/` lands at `0.0224` source-fidelity with `34278` candidates and about `2.16` average choices per cell
- that beats the earlier full-CUDA tile-graph baseline (`0.0283`) and materially improves on the first atomic-only attempt (`0.1571`), which chose atomic regions too eagerly and then let the solver blur them out again
- the latest core algorithm cut removes `source_region_stride`, removes portrait-based candidate ranking, removes sharp/edge foreground fallback injection, and replaces sampled-delta pairwise scoring with adjacency learned from extracted tiles
- that cut also removes the dedicated `TileGraphSourceReference` entirely: tile-graph now grounds itself in full-resolution extracted tiles plus direct per-cell source summaries (`cell_mean_rgba`, `cell_alpha_mean`, `cell_edge_strength`)
- on the cleaned badge at pinned `126x126` / phase `(0.0, -0.2)`, the new cut recovers the fixed-lattice run from roughly `0.5055` source-fidelity down to `0.1814`; that is still weak, but it is no longer total algorithmic collapse

Next after that:
- split tile-graph iteration into a cached model-build phase and a near-free solve phase so weight tuning does not keep paying the `~131s` source-region cutting bill
- extend the current fixed-lattice path so repeated CLI runs can reuse cached model artifacts across processes instead of only within one Python process
- port or reformulate `_extract_source_region_tiles(...)` so it no longer spends most of the badge runtime in Python/NumPy window cutting
- decide whether the continuous path should stay the only global search/rerank engine or whether tile-graph eventually needs its own lighter lattice chooser
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
- the new stroke-aware badge run under `artifacts/badge-tile-graph-stroke-v2-cuda/` still does not solve the guard-contour problem and slightly regresses to `0.1832`, so the next stroke pass should be judged on whether it changes the source-side path model more fundamentally
- older searched-path profiling showed that iteration was dominated by preprocessing work rather than the tile-graph parity solver: about `44.1s` in inference, `223.9s` in low-confidence rerank probes, and `142.3s` in the final selected-candidate build; the pruned tile-graph path no longer pays those rerank probes, but `_extract_source_region_tiles(...)` is still the heavy remaining step

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
