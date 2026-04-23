# Optimizer Algorithm Map

## Why this map exists

The optimizer path is trying to do one slippery thing:

Lay a sheet of graph paper over the source image, imagine what each output cell ought to feel like, then force every final cell to choose a real source pixel nearby without letting the whole image turn into soup.

When it works, the result feels like the image finally committed to one lattice.

When it fails, this map is here to answer:

Where did the machine start negotiating with itself too much?

This is a map of the **current** optimizer in `src/repixelizer/continuous.py`.

If you want the replacement target instead of the autopsy, read `docs/lean-optimizer-algorithm-map.md` too. That companion map describes the smaller displacement-field machine this repo should grow next.

## One-sentence machine

The current optimizer machine is:

`source image -> edge scout -> regular UV grid -> soft representative portrait + hard source lattice portrait -> source-first snap -> source-first refine -> keep snap if refine made things worse`

Or, in pictures:

- the source image is the mural
- the inferred lattice is the sheet of graph paper
- the UV field is where the graph-paper cell centers land on the mural
- the representative portrait is a soft watercolor guess of each cell
- the source lattice portrait is a ledger of which real source pixels belong to each cell
- snap is the mason placing one actual stone into each slot
- refine is the foreman walking the wall, nudging stones to improve the joints

## How to read this machine

The optimizer carries five important pieces of cargo all the way through:

- `uv_t`: the regular lattice centers, in normalized sampling coordinates
- `representative_t`: the soft portrait made by sampling little patches around each UV center
- `source_lattice_reference`: the hard portrait built by assigning every source pixel to one inferred output cell
- `source_reliability_t`: the per-cell trust dial that decides how much the machine should listen to the hard portrait instead of the soft one
- `snap_t`: the first fully discrete output, used as both a candidate answer and the anchor for later refinement

If we lose track of one of those, the optimizer becomes impossible to reason about.

The first structural cut now gathers that cargo into `_OptimizerPrep` before snap and refine begin. That does not make the algorithm better by itself, but it makes the machine easier to see: preparation builds the map, snap and refine walk it.

There is one extra sidecar cargo too:

- `guidance`: a downsampled edge map saved into `SolverArtifacts` for the later cleanup stage

It matters to the pipeline contract, but it is not part of the optimizer's decision-making spine.

For every stage below, read it in two layers:

- `What the source actually does`: the code-shaped account, using the real function names and variables.
- `Metaphor`: the image to keep in your head so the data flow stays legible instead of dissolving into a swamp of tensors.

## Stage 0: The pipeline picks the ruler

Functions:

- `run_pipeline(...)` in `src/repixelizer/pipeline.py`
- `_prepare_analysis(...)` in `src/repixelizer/pipeline.py`
- `_select_phase_candidate_with_reconstruction(...)` in `src/repixelizer/pipeline.py`
- `_run_reconstruction(...)` in `src/repixelizer/pipeline.py`

Inputs:

- `source_rgba`
- `InferenceResult`
  - `target_width`
  - `target_height`
  - `phase_x`
  - `phase_y`

Meaning:

Before the optimizer does anything clever, the pipeline chooses the ruler, and for low-confidence continuous runs it may give that ruler one last suspicious look before letting the solver touch it.

This fixes:

- how many cells the output wall will have
- where that cell grid is phased against the source image

Every later judgment is conditioned on that ruler. If it is wrong, the optimizer is making careful decisions against the wrong graph paper.

### What the source actually does

In `run_pipeline(...)`, the pipeline first resolves whether the user pinned a target size and phase through `_resolve_requested_target_dims(...)`. If the user did not, it calls `infer_lattice(...)`; if the user did, it calls `infer_fixed_lattice(...)`. Then `_prepare_analysis(...)` builds the edge scout report. After that, `_select_phase_candidate_with_reconstruction(...)` may rerank the inferred candidates, but only when `reconstruction_mode == "continuous"`, reranking is enabled, confidence is low enough, and there is more than one top candidate. That rerank path actually runs `_run_reconstruction(...)` with `steps=0` on up to eight candidates and scores them with source-lattice support, edge-position error, stroke wobble, edge concentration, size delta, and inference penalty. Only after that does `_run_reconstruction(...)` hand the final `InferenceResult` to `optimize_lattice_pixels(...)`. Once the optimizer itself starts, the ruler is fixed.

### Metaphor

This is the surveyor hammering stakes into the ground before anyone starts building, then having one brief, paranoid committee meeting about whether the stakes were hammered in the wrong place.  
After that, the argument is over. The masons do not get to kick the stakes sideways halfway through the wall just because they are having feelings.

## Stage 1: The edge scout walks the mural

Functions:

- `analyze_continuous_source(...)` in `src/repixelizer/analysis.py`
- `_compute_edge_map(...)`

Output:

- `ContinuousSourceAnalysis.edge_map`

Meaning:

The optimizer sends out one scout:

- it chalks the cliff faces where luminance or alpha changes sharply

Important variables:

- `edge_map[y, x]`

Natural-language picture:

The edge scout is looking for cracks and ridges.  

The old k-means paint-by-number scout has been cut. It was letting coarse color partitions masquerade as geometry, and that did not fit the optimizer's core story.

### What the source actually does

`analyze_continuous_source(...)` in `src/repixelizer/analysis.py` now does one thing: it builds `edge_map`. `_compute_edge_map(...)` computes luminance, reads alpha, forms `dx` and `dy` from neighboring differences, and normalizes the result into one float map. The GPU path mirrors the same logic in `_compute_edge_map_torch(...)`. The output is not a segmentation, not a palette, not a region graph. It is a single heat map saying where the image folds sharply.

### Metaphor

This scout is not painting the mural into numbered regions anymore. That old idiot got fired.  
The surviving scout just runs charcoal over the wall and leaves dark marks where the plaster catches: ridges, cracks, cliff edges, places where the picture changes its mind abruptly. That is all.

## Stage 2: The optimizer lays down a regular UV grid

Functions:

- `_make_regular_uv(...)`
- `_make_patch_offsets(...)`

Key outputs:

- `uv`
- `uv_t`
- `offsets_t`

Meaning:

This stage drops the graph paper onto the mural.

`uv` is not an optimized warp. It is a perfectly regular lattice built directly from:

- source width and height
- inferred output width and height
- inferred phase

Each output cell gets one normalized coordinate pair saying, "my center currently lands here."

Then `_make_patch_offsets(...)` builds a little stencil of nearby offsets around each center. Those offsets are used to ask, "what does the neighborhood around this cell center look like?"

Important truth:

Despite the old name `optimize_uv_field(...)`, the current path never actually moves the UV field. It lays down a regular grid and keeps it.

### What the source actually does

`_make_regular_uv(...)` computes `cell_x` and `cell_y` from the source size and the chosen output size, builds the cell-center coordinates with `(index + 0.5 + phase) * cell_size - 0.5`, clamps them to the image bounds, then normalizes them to `[-1, 1]` sampling coordinates. `_make_patch_offsets(...)` then creates a small fixed stencil of local offsets, scaled by cell size, around each UV center. Nothing in this stage optimizes those centers. It manufactures a ruler and a sampling stencil, then freezes both.

### Metaphor

This is the crew lowering a sheet of graph paper over a fresco and taping it flat.  
The UV coordinates are the pins through the paper. The patch offsets are a tiny clear plastic stencil you can place around each pin to peek at nearby paint without moving the pin itself.

## Stage 3: The machine paints a soft portrait of the lattice

Functions:

- `_sample_cell_patches(...)`
- `_representative_colors(...)`

Key variables:

- `initial_patches`
- `representative_t`

Meaning:

For each output cell center, the optimizer samples a small cloud of nearby colors from the source image.

Then it asks:

If I blur these samples together just enough to ignore local jitter, what is the most representative color for this cell?

That answer becomes the representative portrait.

Natural-language picture:

This is the machine squinting at the mural from a few steps back, trying to see what each graph-paper square "mostly wants to be."

Why it exists:

- the representative portrait is smooth and stable
- it gives the later discrete stages something coherent to aim at

Why it is dangerous:

- this portrait is also exactly where softness can sneak in
- if the machine listens to it too much, it starts preserving vibes instead of preserving source-owned structure

### What the source actually does

`_sample_cell_patches(...)` uses `grid_sample` to pull a little cloud of nearby RGBA samples around each UV center, flattening and reshaping the offset stencil so the source image can be sampled in one batched pass. `_representative_colors(...)` then computes `patch_mean`, measures each sampled patch color's distance from that mean, applies a softmax over the negative distances, and blends the patch colors with those weights. The result is `representative_t`: a premultiplied, smoothed per-cell color portrait. The code literally computes "which nearby samples are most typical for this cell?" and then averages toward that typicality.

### Metaphor

This is the machine stepping back, squinting, and painting a watercolor thumbnail over the graph paper.  
It does not ask, "what exact shard of pigment lives here?" It asks, "if I blur out the noise and half-close my eyes, what color does this square mostly hum with?" Useful. Also suspicious. Watercolor is where the mush sneaks in wearing respectable shoes.

## Stage 4: The machine builds a hard ledger from the actual source assignment

Functions:

- `build_source_lattice_reference(...)` in `src/repixelizer/source_reference.py`
- `lattice_indices(...)`
- `_build_source_detail_reference(...)`
- `_build_source_reliability(...)`
- `_edge_gradient_maps(...)`

Key outputs:

- `source_lattice_reference`
- `source_detail_reference`
- `source_reliability_t`
- `source_delta_x_t`
- `source_delta_y_t`
- `source_delta_diag_t`
- `source_delta_anti_t`

Meaning:

Now the optimizer stops squinting and starts bookkeeping.

Every source pixel is assigned to exactly one inferred output cell. Once that assignment exists, the machine can compute:

- `mean_rgba`: the average contents of the cell
- `sharp_rgba`: the most typical real pixel in that cell
- `cell_dispersion`: how mixed or unstable the cell is
- `cell_support`: how much source support the cell has
- `cell_alpha_max`: whether the cell ever gets truly opaque
- `edge_peak_x`, `edge_peak_y`: where the strongest edge inside that cell lives
- `edge_strength`: how strong that edge is
- `edge_grad_x`, `edge_grad_y`: which way the edge points locally

Then `_build_source_detail_reference(...)` mixes `sharp_rgba` with the color at the cell's edge peak, producing a more edge-aware source portrait.

Then `_build_source_reliability(...)` decides how much to trust that hard portrait.

Natural-language picture:

If the representative portrait is a watercolor painting, the source lattice reference is an accountant's ledger.

It says:

- these are the pixels that actually live under this cell
- this is how messy they are
- this is where the sharpest internal edge sits
- this is how much confidence we should have that the cell really knows what it is

Important seam:

The optimizer now has two different stories about the same lattice:

- a soft patch-sampled story
- a hard source-assignment story

Much of the optimizer's complexity is the cost of arbitrating between those two stories.

Current code shape:

- `_prepare_optimizer(...)` owns this whole setup stage now
- `_source_detail_delta_tensors(...)` computes the source-detail deltas once instead of rebuilding premultiplied references several times
- `optimize_lattice_pixels(...)` now receives a prepared bundle and orchestrates snap, refine, and the final guardrail

### What the source actually does

`build_source_lattice_reference(...)` in `src/repixelizer/source_reference.py` calls `_build_reference_payload(...)`, which first uses `lattice_indices(...)` to assign every source pixel to one output cell under the chosen size and phase. It premultiplies the source, bins pixel contributions by cell index, and computes `mean_rgba`, `cell_dispersion`, `cell_support`, and `cell_alpha_max`. Then it chooses one exemplar-like pixel per cell with `_argbest_linear_indices(...)` over `exemplar_cost_t`, where that cost is basically "distance from the cell's premultiplied mean, plus a tiny penalty if alpha falls below that mean." That produces `sharp_rgba`, `sharp_x`, and `sharp_y`. In parallel it picks edge-peak locations using the strongest `edge_hint`, giving `edge_peak_x`, `edge_peak_y`, `edge_strength`, `edge_grad_x`, and `edge_grad_y`. Back in `continuous.py`, `_build_source_detail_reference(...)` mixes the sharp pixel with the edge-peak pixel where `edge_strength` clears the configured threshold. `_build_source_reliability(...)` then turns dispersion, support, alpha, and edge strength into a trust map, while `_source_detail_delta_tensors(...)` computes the horizontal, vertical, and diagonal color-jump tensors from that hard portrait.

### Metaphor

This stage stops squinting and opens the books.  
Every real source pixel gets marched into exactly one cell's ledger like a tax record. The machine counts how many citizens live there, how similar they are, who the most typical resident is, where the loudest knife-edge cuts through town, and whether this district is solid enough to trust. Then it writes down not just what each cell is, but how violently it expects the color to jump when you step east, south, or diagonally into the next cell.

## Stage 5: Snap every output cell to one real nearby source pixel

Function:

- `_snap_output_to_source_pixels(...)`

Key inputs:

- `uv_t`
- `representative_t`
- `source_reference_t`
- `source_reliability_t`

Key working variables:

- `candidate_x`, `candidate_y`
- `candidate_colors`
- `representative_match`
- `source_match`
- `base_energy`
- `desired_delta_x`
- `desired_delta_y`
- `desired_delta_diag`
- `desired_delta_anti`
- `selected`

Meaning:

This is the first moment the optimizer has to stop talking and actually place stones in the wall.

For each output cell:

1. it looks at a fixed `5x5` neighborhood of nearby source pixels around the UV center
2. it scores those real pixels against both portraits
3. it blends those portrait scores using `source_reliability_t`
4. it repeatedly asks whether the chosen pixel agrees with the colors its neighbors are choosing
5. it picks one real source pixel

The delta maps matter here.

They encode the desired color jump from one cell to the next:

- left to right
- top to bottom
- main diagonal
- anti-diagonal

Those desired jumps are themselves blended between:

- the soft representative portrait
- the hard source-detail portrait

Natural-language picture:

Snap is a mason walking across the empty wall, picking one real stone for each slot while glancing left, right, up, and diagonally to keep the joints plausible.

Then `_harden_binary_alpha_selection(...)` makes an extra binary decision for uncertain alpha cells:

- if this cell should really be foreground, force the best opaque candidate
- if this cell should really be background, force the best transparent candidate

That stops fuzzy half-alpha picks from surviving into the final raster.

### What the source actually does

`_snap_output_to_source_pixels(...)` turns each UV center into a fixed `5x5` local neighborhood by adding the `fractions = [-0.42, -0.21, 0.0, 0.21, 0.42]` offsets scaled by `cell_x` and `cell_y`, then rounds those positions to real source pixels to get `candidate_x`, `candidate_y`, and `candidate_colors`. It measures `representative_match` against `representative_t` and `source_match` against `source_reference_t`, then fuses them with `_reference_match_energy(...)` using `source_reliability_t`. It also builds `desired_delta_x`, `desired_delta_y`, `desired_delta_diag`, and `desired_delta_anti` by blending the soft portrait's deltas with the hard portrait's deltas through `_blend_reference_delta_map(...)`. Then it does not just choose once. It starts from `selected = argmin(base_energy)`, then runs four synchronous neighbor-consistency sweeps, rebuilding `energy` from the currently selected colors and the desired deltas before taking a new per-cell argmin each time. Finally, `_harden_binary_alpha_selection(...)` snaps indecisive alpha picks to an explicitly opaque or transparent candidate when the ranking supports it.

### Metaphor

This is the first time the machine has to stop daydreaming and pick actual stones.  
For each slot in the wall, it kneels over a tray of twenty-five nearby stones, weighs each one against the watercolor sketch and the accountant's ledger, then does four quick passes across the wall tightening the joints before the mortar sets. At the end it slaps the hand away from any damp half-ghost stone and says, no, pick rock or air. Commit.

## Stage 6: Expand the search space for refine

Functions:

- `_build_candidate_positions(...)`
- `_reference_match_energy(...)`
- `_blend_reference_delta_map(...)`

Key outputs:

- `candidate_x`, `candidate_y`
- `candidate_colors`
- `base_energy`
- `relax_base_energy`

Meaning:

After snap, the optimizer opens a larger local search window for each cell.

These candidates are not just a fixed square anymore. The builder also adds:

- the strongest edge peak for the cell
- gradient-guided offsets along the local edge direction

So the refine stage can reach sharper alternatives than the original snap neighborhood would have seen.

Each candidate now accumulates several costs:

- anchor match: how far it strays from the snapped output
- source match: how well it fits the hard source portrait
- alpha match
- distance from the UV center

Natural-language picture:

Snap chose one stone per slot.  
Now refine lays several nearby stones on the scaffold around each slot so it can ask whether a slightly different local arrangement would make a cleaner wall.

### What the source actually does

`_build_candidate_positions(...)` starts from the fixed UV centers again, but now it builds a denser candidate set. It lays down a grid of fraction offsets from `refine_candidate_levels` and `refine_candidate_extent`, then appends edge-aware positions: the per-cell `edge_peak_x`/`edge_peak_y`, plus guided offsets along `edge_grad_x` and `edge_grad_y` when `edge_strength` and `cell_dispersion` say the cell is both edgy and unstable. `candidate_colors` are sampled directly from the source at those positions. The stage then computes `anchor_energy`, `source_reference_energy`, `alpha_energy`, and `distance_energy`, producing both `base_energy` for refine and the slightly softened `relax_base_energy` for the relax stage.

### Metaphor

Snap picked one stone and set it in mortar. Refine shows up with a wider tray.  
Now the foreman has the snapped stone, a few neighboring stones, the stone sitting right on the local crack line, and a couple more dragged along the crack's direction. The question stops being "what is nearby?" and becomes "what nearby alternative might preserve the cut in the image without wandering off the ruler?"

## Stage 7: Relax into a soft field before going fully discrete again

Function:

- `_relax_candidate_selection(...)`

Key variables:

- `probs`
- `context_colors`
- `final_energy`
- `relaxed_context`
- `handoff_energy`
- `mode_selected`

Meaning:

This is the optimizer taking a deep breath before it starts making hard yes/no choices again.

Instead of choosing one candidate per cell immediately, it keeps a soft probability distribution over the candidates.

At each relaxation step:

1. it computes the expected color for each cell from those probabilities
2. it scores every candidate against that soft neighborhood
3. it adds structure terms for adjacency, motif, and line continuation
4. it cools the temperature and sharpens the distribution

There are two families of structure terms here:

- anchor-facing terms: stay consistent with the snapped lattice
- source-facing terms: stay consistent with the hard source portrait

Natural-language picture:

This is wet plaster. Nothing is set yet, but the stones are starting to lean into a coherent arrangement.

At the end of relax, the optimizer keeps two things:

- `selected`: the best discrete choice after the handoff energy
- `mode_selected`: the pure modal choice from the soft distribution

That is the machine hedging its bets before the next stage hardens everything again.

Recent cut:

- the old extra `relaxed_mode` bonus in refine is gone
- the machine still compares the best refined state against the relaxed mode state at the end, but it no longer biases every greedy candidate score toward the relaxed mode

### What the source actually does

`_relax_candidate_selection(...)` initializes `probs` as a softmax over `-base_energy / start_temp`. On each iteration it converts those probabilities into `context_colors`, the expected color field implied by the current soft choices. It then rebuilds `final_energy` by adding pairwise delta agreement terms, optional source-facing adjacency terms, optional motif terms against both `anchor` and `source_reference`, and optional line-continuation terms. After each step, it damps and cools the distribution so the probabilities get sharper over time. When the loop ends, it builds `relaxed_context`, then forms `handoff_energy` by combining `final_energy`, a penalty for straying from `relaxed_context`, pairwise delta agreement against `relaxed_context`, and motif/line terms that now treat `relaxed_context` itself as the local template for the anchor-facing side of the handoff. It keeps both the best handoff selection `selected` and the pure modal pick `mode_selected`. The relax-mode bonus that used to bias refine directly is gone; only the end-of-stage hedge remains.

### Metaphor

This is wet plaster and half-seated stones.  
Nothing is fully committed yet. Each slot is allowed to say, "I'm 60% this dark edge stone, 30% the slightly duller one, 10% the safer interior stone." The wall is still trembling, but the trembling starts to synchronize. Temperature drops. The slurry thickens. Shapes that looked independent begin leaning into each other like bones finding their sockets.

### Measured reality

The original design goal for `relax` was larger than "soften local choices." It was supposed to let the machine adjust phase smoothly over broad swaths of the canvas instead of only making tiny local corrections.

The current instrumentation in `continuous.py` says the truth is weaker than that story.

On the pinned cleaned-badge run at `126x126`, phase `(0.0, -0.2)`, `steps=48`:

- with `relax_iterations=24`, final source-fidelity lands at `0.07451`
- with `relax_iterations=0`, final source-fidelity lands at `0.07549`
- so `relax` does help a little

But the displacement-field diagnostics do **not** show a dramatic large-scale phase-smoothing effect:

- final orthogonal displacement jitter is `1.7154` with relax versus `1.6783` without it
- final local residual is `1.1548` with relax versus `1.1312` without it
- final dominant-offset ratio is `0.6296` with relax versus `0.6380` without it

So the current `relax` earns a small fidelity gain, but it does not look like a true broad-swatch phase transport stage. It behaves more like a soft consensus pass over a fixed tray of local candidates, then hands that consensus to greedy refine.

## Stage 8: Greedy discrete refine

Function:

- `_discrete_refine_output(...)`

Key variables:

- `selected`
- `best_selected`
- `best_score`
- `candidate_energy(...)`
- `loss_history`

Meaning:

Now the optimizer goes cell-by-cell again, but with a richer energy than snap had.

It starts from:

- the relaxed handoff selection
- the relaxed mode selection

Then it repeatedly:

1. computes the current selected colors
2. scores every candidate against the current neighborhood
3. takes the per-cell argmin
4. measures the full structure score of the resulting lattice
5. remembers the best discrete state it has seen

The structure score in `_structure_score(...)` mixes:

- source-boundary agreement from direct source sampling
- adjacency agreement with the snapped anchor
- motif agreement with the snapped anchor
- line agreement with the snapped anchor
- adjacency agreement with the hard source portrait
- motif agreement with the hard source portrait
- line agreement with the hard source portrait

Natural-language picture:

This is the foreman walking the wall in passes, replacing one stone at a time wherever a better local fit appears, but still judging the whole wall by whether the mortar lines and repeated motifs feel right.

When refine finishes, it compares:

- the best greedy-refined state
- the relaxed mode state

and keeps whichever one has the better full structure score.

Then it hardens alpha again before producing the final `target_rgba`.

### What the source actually does

`_discrete_refine_output(...)` starts from the relaxed handoff and relaxed mode states, scores both with `_structure_score(...)`, and keeps the better starting point. Then for up to `iterations` passes it computes `candidate_energy(context_colors)` from `refine_base_energy`, pairwise delta terms, optional motif and line terms against the snapped anchor, and optional source-facing motif, line, and adjacency terms against the hard portrait. `_structure_score(...)` separately judges whole-lattice boundary agreement by directly sampling the source around the UV field with `_boundary_pattern_loss(...)`, then adds anchor-facing adjacency/motif/line losses and source-facing adjacency/motif/line losses. Each pass takes the per-cell argmin, scores the full discrete lattice with `_structure_score(...)`, and remembers the best whole-state selection it has seen in `best_selected`. The loop deliberately runs `passes + 1` scoring rounds so the current state is always scored before the next update is applied. At the end it compares that best greedy state against `relaxed_mode_selected`, picks the lower full-structure score, and hardens alpha one more time. After the recent cuts, this stage no longer consults the representative portrait directly; refine answers to the snapped anchor and the hard source portrait.

### Metaphor

Now the plaster has set enough that the foreman starts walking the wall with a hammer and a bad attitude.  
He taps one stone loose, tries another, steps back, judges the seam, repeats. He does not repaint the watercolor sketch anymore. He judges the wall by the actual stone joints: edge runs, repeated motifs, line continuation, whether the neighboring cuts still read like they belong to the same carving instead of a committee.

## Stage 9: Final sanity check against snap

Functions:

- `source_lattice_consistency_breakdown(...)`
- final guardrail in `optimize_lattice_pixels(...)`

Key variables:

- `snap_score`
- `target_score`

Meaning:

At the very end, the optimizer asks a brutally practical question:

Did all this extra cleverness actually make the result more faithful to the source lattice?

If not, it throws the refined answer away and keeps the snapped one.

Natural-language picture:

This is the supervisor looking at the finished wall and saying:

"Nice theory. But if the first honest stone placement was better, we are keeping that."

This guardrail matters because the refine stage is absolutely capable of making the wall feel smoother while also making it less true.

### What the source actually does

Back in `optimize_lattice_pixels(...)`, the code computes `snap_score` and `target_score` with `source_lattice_consistency_breakdown(...)`. That metric asks how well the output agrees with the inferred source lattice under the chosen size and phase. If `target_score > snap_score + 1e-6`, the refined result is discarded and `target_rgba = snap_rgba`. The machine literally keeps a receipt from before it got clever, then checks whether cleverness paid rent.

### Metaphor

This is the last adult in the room.  
The crew may have spent hours lovingly fussing over the wall, but the supervisor still walks over with a clipboard and says: if your improved version is prettier but less faithful to the stone yard we pulled from, it goes in the bin. We keep the first honest wall.

## What fits the machine

These parts fit the optimizer's core story:

- using a regular inferred lattice as the starting ruler
- building a soft representative portrait from local patch samples
- building a hard source-assignment portrait from the actual lattice ownership
- using per-cell reliability to blend those two portraits instead of always trusting one
- forcing snap and refine to choose real source pixels, not synthesized colors
- letting the soft representative portrait stabilize snap, then making refine answer only to the snapped anchor and the hard source portrait
- keeping snap if refine makes source-lattice fidelity worse

Those are coherent. They all serve the same broad goal: make one global lattice without drifting too far from real local source evidence.

## What does not fit the machine cleanly

These are the places where the optimizer still speaks in an overly complicated or self-contradictory voice:

- the machine still maintains two overlapping portraits of the same lattice, even though refine is now source-first and only snap still arbitrates between them
- snap, relax, and refine all carry slightly different versions of adjacency, motif, line, and delta agreement; the same idea is being said three times with different accents
- the candidate generator is local and the solver is discrete, but the vocabulary around it still sounds like a continuous deformation engine
- the main entry point is now honestly named `optimize_lattice_pixels(...)`, the k-means boundary scout is gone, prep is split from decision-making, refine no longer consults the representative portrait, and the relax-mode bonus is gone; those cuts removed five contradictions from the previous map

This is probably the real cutting checklist for the next optimizer pass.

## The current honest shape

If we strip the optimizer down to what it is really doing today, the machine is:

`source image -> edge scout -> fixed lattice centers -> soft portrait + hard portrait -> source-first snap -> soft neighborhood relax -> source-first greedy refine -> keep snap if refine lied`

That is much simpler than the surrounding names make it sound.

## Likely next cuts

If we keep pruning, the highest-value cuts look like this:

1. Put relax on trial.
   The soft relaxation stage still seems useful, but the current measurements say it is not really acting like a large-scale phase smoother. The next question is which remaining relax terms are actually earning their keep, and whether the surviving benefit comes from handoff stabilization rather than regional phase drift.
2. Collapse duplicate structure voices.
   Adjacency, motif, and line are important, but they should not need three near-parallel dialects unless they are truly doing different work.
3. Decide whether snap still needs both portraits everywhere.
   Refine is already source-first now, so the next big question is whether snap still needs all of its portrait arbitration machinery or only the parts that genuinely stabilize hard cases.

That is the next place to swing the axe.
