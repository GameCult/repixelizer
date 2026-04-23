# Tile-Graph Algorithm Map

## Why this map exists

The tile-graph path is trying to do one stubbornly honest thing:

Take a fake-pixel mural, split it into real opaque paint islands, slice those islands by the chosen output lattice, learn which slices continue into neighboring cells, and then reassemble the image using only those real slices.

When the output looks wrong, this map is here to answer:

Where did the machine start lying?

## One-sentence machine

The current tile-graph machine is:

`source image -> edge scout -> connected components -> component-cell overlap reduction -> per-cell candidate buckets -> learned same-component adjacency -> local discrete assignment`

Or, in pictures:

- the source image is the mural
- connected components are paint islands
- the chosen lattice is a sheet of graph paper laid over the mural
- the overlap reducer is a cookie cutter that stamps each island against each graph-paper square it actually touches
- the solver is the mason choosing which stamped shard goes in each final slot

## What was cut away

These ideas are gone:

- subsampling the mural before extraction with `source_region_stride`
- per-component seed walking and window stepping
- stroke-specific PCA slicing
- ranking candidates by resemblance to a lattice portrait (`sharp_rgba`, `edge_rgba`)
- injecting `sharp_pixel` / `edge_pixel` when extraction failed
- sampled one-cell-away RGBA deltas pretending to be a graph
- a dedicated tile-graph source-reference object

That matters because all of those were ways for the machine to stop listening to the actual source-owned tiles.

## Stage 0: The pipeline picks a ruler

Function:

- `run_pipeline(...)` in `src/repixelizer/pipeline.py`

Inputs:

- `source_rgba`
- chosen `InferenceResult`
  - `target_width`
  - `target_height`
  - `phase_x`
  - `phase_y`

Meaning:

Before anything else, the system chooses the graph paper.

If the graph paper is wrong, every later step is cutting the mural against the wrong ruler.

Once the lattice is pinned, the tile-graph path now runs directly on that lattice. It no longer burns time on phase-rerank probes or hybrid sidecars.

## Stage 1: Edge scout

Function:

- `analyze_tile_graph_source(...)` in `src/repixelizer/analysis.py`

Output:

- `TileGraphSourceAnalysis.edge_map`

Meaning:

This is a scout walking over the mural with a lantern and chalk, marking where the paint changes sharply.

It is not deciding colors. It is only measuring local cliff faces.

Important variable:

- `edge_map[y, x]`

Why it exists:

- connected components need a detail hint
- candidate budgeting still uses edge-heavy cells as "hard mode"

## Stage 2: Every source pixel is assigned to an output cell

Function:

- `build_tile_graph_model(...)`

Key variables:

- `cell_w`
- `cell_h`
- `projected_flat_index`
- `cell_mean_rgba_flat`
- `cell_alpha_mean_flat`
- `cell_alpha_support_flat`
- `cell_edge_strength_flat`

Meaning:

Imagine dropping transparent graph paper over the mural and writing the output cell index on every source pixel.

That gives us direct per-cell source summaries:

- average color already under the slot
- average alpha already under the slot
- strongest alpha under the slot
- strongest edge under the slot

These are not a separate portrait of what the slot "should" be. They are literally summaries of the pixels already living under that slot.

## Stage 3: Connected components

Functions:

- `_segment_atomic_source_regions(...)`
- `_segment_atomic_source_regions_cpu(...)`

Output:

- `AtomicRegionLabeling`
  - `pixel_linear`
  - `component_ids`
  - `component_sizes`
  - `component_count`

Meaning:

Now the mural is split into paint islands.

Pixels join the same island only if:

- they are opaque enough
- their premultiplied colors are similar enough
- their alpha is similar enough

This is literal connected-component labeling over the full-resolution source.

Important consequence:

- every opaque source pixel belongs to exactly one component

## Stage 4: Reduce component-cell overlaps

Function:

- `_extract_source_region_tiles(...)`

This is the new heart of the machine.

Instead of walking each component with seed queues and cell-sized windows, the reducer does something much simpler:

1. take every opaque labeled source pixel
2. project it onto the chosen output lattice
3. form a compound key: `(component_id, output_cell)`
4. sort by that key
5. reduce each run into one candidate shard

This is the same basic mental pattern as `reduce_by_key` / `segment_reduce`: sort once, then aggregate consecutive runs.

What each reduced shard stores:

- `rep_linear`
- `rep_rgba`
- `area_ratio`
- `coverage`
- `edge_peak`
- `source_center_x`
- `source_center_y`
- `coord_x`, `coord_y`, `flat_index`
- `component_id`

Natural-language picture:

Instead of hand-carving the island with a pocketknife, we stamp every island against the graph paper and keep one shard per `(island, cell)` overlap.

## Stage 5: Candidate buckets

Still inside:

- `_extract_source_region_tiles(...)`
- `_select_source_region_candidates(...)`

Meaning:

Each output cell now gets a bucket of legal shards.

The reducer first creates every real `(component, cell)` overlap shard.

Then bucket pruning keeps the best local candidates using only tile-owned evidence:

- area fit
- coverage
- edge peak

Important rule:

- if a cell has opaque source support, it must have at least one real extracted shard
- if pruning would empty a supported cell, the best overlap shard for that cell is kept anyway
- background or near-background cells can also get a transparent candidate

What this prevents:

- silent empty foreground buckets
- fake sharp/edge fallback pixels

## Stage 6: Learn adjacency from the reduced shards

Still inside:

- `_extract_source_region_tiles(...)`

Meaning:

Each shard now asks a simpler question than before:

"Does my same component continue into the cell to the right, down, left, or up?"

If a same-component overlap shard exists in the neighboring cell, that shard's representative color becomes the expected neighbor color for that direction.

Stored per shard:

- `neighbor_rgba[4, 4]`
- `neighbor_mask[4]`

This is an intentionally narrower graph than the earlier heuristic:

- it only learns continuity that the same connected paint island actually exhibits
- it does not try to smooth across boundaries between different components

That is a carefully chosen bias:

- continuity within a real island is trustworthy
- continuity across a contour is where fake smoothing usually starts

## Stage 7: Solver model

Dataclass:

- `TileGraphModel`

Current essential fields:

- `candidate_rgba`
- `candidate_area_ratio`
- `candidate_coverage`
- `candidate_edge_peak`
- `candidate_neighbor_rgba`
- `candidate_neighbor_mask`
- `cell_candidate_offsets`
- `cell_candidate_indices`
- `cell_mean_rgba`
- `cell_alpha_mean`
- `cell_edge_strength`

Meaning:

This is the tray of legal shards plus the local evidence under each output slot.

## Stage 8: Unary cost

Function:

- `_tile_graph_unary_cost_torch(...)`

Terms:

- `color_error`
  - candidate versus actual mean color already under the slot
- `area_error`
  - how close the shard is to one full cell
- `alpha_error`
  - candidate alpha versus actual mean alpha under the slot
- `coverage_error`
  - how incomplete the overlap is
- `edge_error`
  - candidate edge peak versus strongest edge under the slot

Meaning:

The unary cost is the bouncer at each slot asking:

"Does this shard honestly fit the patch of mural under this square?"

Important difference from the old machine:

- this is local source support
- not a pre-baked portrait from another reference object

## Stage 9: Pairwise cost

Functions:

- `_pair_penalty_selected_torch(...)`
- `_pair_penalty_option_right_torch(...)`
- `_pair_penalty_option_left_torch(...)`
- `_pair_penalty_option_down_torch(...)`
- `_pair_penalty_option_up_torch(...)`

Meaning:

Now the mason cares about seams.

If a shard says "my component really continues to the right as color X," then placing a very different right neighbor costs something.

This is narrower than the old fake graph, but more honest:

- it rewards real within-component continuation
- it stops encouraging smoothing across unrelated components

## Stage 10: Parity refinement

Function:

- `optimize_tile_graph(...)`

What it does:

1. build candidate buckets
2. compute unary cost
3. choose an initial argmin candidate per cell
4. alternate parity updates across the grid
5. if refinement worsens source-fidelity, keep the initial assignment

Meaning:

The mason lays an initial mosaic, then alternates black and white squares of the chessboard trying to improve local seams without moving everything at once.

Important truth:

- this is still a local discrete optimizer
- it is not globally optimal
- if the buckets are wrong, refinement will not save the image

## Data dictionary

### `AtomicRegionLabeling.pixel_linear`

- flattened source-pixel indices for every opaque labeled pixel

### `AtomicRegionLabeling.component_ids`

- component id for each `pixel_linear`

### `projected_flat_index`

- output-cell index for each source pixel under the chosen lattice

### `compound_key`

- `(component_id * output_area) + output_cell`

This is the reducer's key. Every identical key means "these pixels belong to the same component-cell overlap shard."

### `region_buckets[flat_index]`

- all legal source-owned shards currently available for one output cell

### `candidate_neighbor_rgba`

- expected neighboring shard colors learned from same-component continuation

## What this machine is optimizing for

In plain language:

- choose only real source-owned shards
- prefer shards that genuinely fit the local patch of mural
- preserve continuation within real paint islands
- refuse to invent a foreground shard when extraction did not produce one

## What can still go wrong

The likely remaining lies are now:

- the lattice is wrong
- component labeling merged or split the wrong source regions
- the direct overlap shard for a difficult contour cell is too coarse
- candidate truncation threw away the only good shard in a crowded cell
- same-component adjacency is too narrow to express a more complex local motif
- the parity solver settles into a mediocre local arrangement

## Current reality check

Pinned cleaned badge run after the reduce-by-key extraction rewrite:

- target: `126x126`
- phase: `(0.0, -0.2)`
- mode: `tile-graph`
- phase rerank: off

Measured result:

- initial source-fidelity: `0.2036`
- final source-fidelity: `0.2036`
- the solver kept the initial assignment

That is a small quality regression from the previous `0.1814` cut, but it is still far better than the earlier fixed-lattice collapse around `0.5055`.

## Big-picture judgment

Does the current machine make more sense than the old one?

Yes.

The dataflow is now much cleaner:

1. observe the mural
2. label the paint islands
3. stamp those islands against the chosen lattice
4. keep the resulting overlap shards
5. learn only the continuation those shards actually exhibit
6. assemble the output from those shards

That is the right shape.

The remaining problem is no longer "why is the machine inventing stories?"

It is:

"Are these overlap shards expressive enough for the finest contours, and can we compute the labels fast enough?"
