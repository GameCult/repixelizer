# Optimizer Algorithm Map

## Why this map exists

The optimizer path is trying to do one slippery thing:

Lay a sheet of graph paper over the source image, imagine what each output cell ought to feel like, then force every final cell to choose a real source pixel nearby without letting the whole image turn into soup.

When it works, the result feels like the image finally committed to one lattice.

When it fails, this map is here to answer:

Where did the machine start negotiating with itself too much?

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

- `uv0_t`: the regular lattice centers, in normalized sampling coordinates
- `initial_representative_t`: the soft portrait made by sampling little patches around each UV center
- `source_lattice_reference`: the hard portrait built by assigning every source pixel to one inferred output cell
- `source_reliability_t`: the per-cell trust dial that decides how much the machine should listen to the hard portrait instead of the soft one
- `snap_t`: the first fully discrete output, used as both a candidate answer and the anchor for later refinement

If we lose track of one of those, the optimizer becomes impossible to reason about.

The first structural cut now gathers that cargo into `_OptimizerPrep` before snap and refine begin. That does not make the algorithm better by itself, but it makes the machine easier to see: preparation builds the map, snap and refine walk it.

## Stage 0: The pipeline picks the ruler

Functions:

- `run_pipeline(...)` in `src/repixelizer/pipeline.py`
- `_run_reconstruction(...)` in `src/repixelizer/pipeline.py`

Inputs:

- `source_rgba`
- `InferenceResult`
  - `target_width`
  - `target_height`
  - `phase_x`
  - `phase_y`

Meaning:

Before the optimizer does anything clever, the pipeline chooses the ruler.

This fixes:

- how many cells the output wall will have
- where that cell grid is phased against the source image

Every later judgment is conditioned on that ruler. If it is wrong, the optimizer is making careful decisions against the wrong graph paper.

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

## Stage 2: The optimizer lays down a regular UV grid

Functions:

- `_make_regular_uv(...)`
- `_make_patch_offsets(...)`

Key outputs:

- `uv0`
- `uv0_t`
- `offsets_t`

Meaning:

This stage drops the graph paper onto the mural.

`uv0` is not an optimized warp. It is a perfectly regular lattice built directly from:

- source width and height
- inferred output width and height
- inferred phase

Each output cell gets one normalized coordinate pair saying, "my center currently lands here."

Then `_make_patch_offsets(...)` builds a little stencil of nearby offsets around each center. Those offsets are used to ask, "what does the neighborhood around this cell center look like?"

Important truth:

Despite the old name `optimize_uv_field(...)`, the current path never actually moves the UV field. It lays down a regular grid and keeps it.

## Stage 3: The machine paints a soft portrait of the lattice

Functions:

- `_sample_cell_patches(...)`
- `_representative_colors(...)`

Key variables:

- `initial_patches`
- `initial_representative_t`

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
   The soft relaxation stage still seems useful, but the relax-mode bonus was dead weight. The next question is which remaining relax terms are actually earning their keep.
2. Collapse duplicate structure voices.
   Adjacency, motif, and line are important, but they should not need three near-parallel dialects unless they are truly doing different work.
3. Decide whether snap still needs both portraits everywhere.
   Refine is already source-first now, so the next big question is whether snap still needs all of its portrait arbitration machinery or only the parts that genuinely stabilize hard cases.

That is the next place to swing the axe.
