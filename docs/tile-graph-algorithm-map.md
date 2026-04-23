# Tile-Graph Algorithm Map

## Why this map exists

The tile-graph path is trying to do one simple, stubborn thing:

Take a fake-pixel mural, cut it into honest one-cell tiles that really belong to the source image, learn which tiles naturally sit beside which, and then lay those tiles back down on a clean output lattice.

That is the dream.

When the output looks wrong, this map is here to answer one question:

Where, exactly, did the machine start lying?

## One-sentence machine

The current tile-graph machine is:

`source image -> edge scout -> atomic opaque regions -> one-cell tile cuts -> per-cell candidate buckets -> learned tile adjacency -> local discrete assignment`

Or in more visual language:

- the source image is the mural
- connected components are the paint islands
- extracted tiles are the stones we chip out of that mural
- candidate buckets are the bins beside each slot in the final mosaic
- the solver is the mason choosing which chipped stone goes in each slot while trying to keep neighboring stones compatible

## What was cut away

The machine used to carry several ideas that did not belong:

- subsampling the mural before extraction with `source_region_stride`
- ranking region candidates by resemblance to a lattice portrait (`sharp_rgba`, `edge_rgba`)
- papering over missing foreground tiles by injecting `sharp_pixel` or `edge_pixel`
- pretending a learned tile graph existed while actually using sampled color deltas from one cell away in source space
- building and hauling a dedicated tile-graph source-reference object through the core loop

Those are gone.

That matters because they were smuggling continuous-solver assumptions back into a path that is supposed to be source-owned and discrete.

## The living machine

### Stage 0: Pipeline chooses a lattice

Function:

- `run_pipeline(...)` in `src/repixelizer/pipeline.py`

Inputs:

- `source_rgba`
- chosen `InferenceResult`
  - `target_width`
  - `target_height`
  - `phase_x`
  - `phase_y`

What it means:

The pipeline chooses the grid we are going to commit to. This is not the tile-graph solver yet. This is just choosing the size of the chessboard before we decide which pieces go on it.

Important truth:

- if the lattice is wrong, every later step is forced to cut the mural against the wrong ruler
- but once the lattice is pinned, the tile-graph path now runs directly on that lattice without phase-rerank probes or hybrid sidecars

### Stage 1: Edge scout

Function:

- `analyze_tile_graph_source(...)` in `src/repixelizer/analysis.py`

Output:

- `TileGraphSourceAnalysis.edge_map`

What it means:

This stage is the scout walking over the mural with a lantern, marking where the sharp cliffs are. It is not deciding colors. It is only pointing out where the paint changes abruptly.

Important variable:

- `edge_map[y, x]`
  - a per-source-pixel edge strength in `[0, 1]`

Why it exists:

- region extraction needs to know where a component has its sharpest internal detail
- candidate budgeting uses edge strength to give difficult cells more room

### Stage 2: Source pixels are projected onto the output lattice

Function:

- `build_tile_graph_model(...)` in `src/repixelizer/tile_graph.py`

Key variables:

- `cell_w`
- `cell_h`
- `projected_coord_x`
- `projected_coord_y`
- `projected_flat_index`

What it means:

Every source pixel is asked:

"If this final grid is the one we believe in, which output cell do you live under?"

This is like dropping a transparent graph-paper sheet over the mural and writing the output cell index on every source pixel.

Derived summaries:

- `cell_counts_flat`
  - how many source pixels fall into each output cell
- `cell_mean_rgba_flat`
  - the average source color already living under that cell
- `cell_alpha_support_flat`
  - the strongest alpha seen in that cell
- `cell_alpha_mean_flat`
  - the average alpha in that cell
- `cell_edge_strength_flat`
  - the strongest edge signal seen in that cell

These summaries are not a tile portrait from a separate source-reference system. They are direct summaries of the actual source pixels already assigned to that output cell under the chosen lattice.

### Stage 3: Atomic opaque regions

Function:

- `_segment_atomic_source_regions(...)`
- CPU path: `_segment_atomic_source_regions_cpu(...)`

Inputs:

- full-resolution `source_rgba`
- full-resolution `edge_map`
- alpha/color join thresholds

What it means:

Now the mural is split into paint islands.

Pixels join the same island only if:

- they are opaque enough
- their premultiplied colors are similar enough
- their alpha is similar enough

This is not a "cluster by vibes" step. It is literal connected-component labeling over the full-resolution mural.

Important consequence:

- every opaque source pixel belongs to some component
- if an occupied output cell later ends up with no real extracted tile, that is not normal; it is a bug or a failure in cutting/projection logic

### Stage 4: Cut one-cell tiles out of each component

Function:

- `_extract_source_region_tiles(...)`

Inputs:

- `components`
- `flat_rgba`
- `flat_edge`
- `flat_x`
- `flat_y`
- lattice geometry: `cell_w`, `cell_h`, `phase_x`, `phase_y`

What it means:

This is the quarry.

For each paint island, we try to chip out cell-sized stones.

There are two modes:

- ordinary components:
  - start from the centroid and edge peak
  - accept a one-cell window if it contains enough of the component
  - march outward one cell at a time
- elongated stroke-like components:
  - estimate the component’s principal axis
  - seed along that axis
  - cut tiles in bands so a long stroke is not immediately turned into a rectangular blob

Important variables inside each extracted tile:

- `rep_linear`
  - the literal source pixel index chosen to represent the tile
- `rep_rgba`
  - the tile color
- `area_ratio`
  - how much component area the tile owns relative to one output cell
- `coverage`
  - `area_ratio` clipped to `[0, 1]`
- `edge_peak`
  - strongest edge value inside that tile footprint
- `source_center_x`, `source_center_y`
  - where this tile lives in source space
- `coord_x`, `coord_y`, `flat_index`
  - which output cell this tile projects into
- `component_id`
  - which paint island it came from

Natural-language picture:

The component is a strip of stained glass. We are cutting out little panes, each about one output cell wide, and tagging where each pane came from.

### Stage 5: Fill empty occupied cells honestly

Still inside:

- `_extract_source_region_tiles(...)`

What it means:

After the first pass of tile cutting, some output cells may still have no tile in their bucket even though component pixels really do overlap them.

To fix that, the builder does an overlap-based empty-cell fill:

- for each component, look at every output cell it physically overlaps
- if that bucket is still empty, add one overlap tile there

This matters because:

- a foreground cell is allowed to have multiple candidates
- a foreground cell is not allowed to have zero real extracted candidates

Background-only cells are different:

- they have no opaque source ownership
- they get a transparent candidate later in model building

### Stage 6: Learn adjacency from extracted tiles

Still inside:

- `_extract_source_region_tiles(...)`

What it means:

Each extracted tile now looks one cell away in all four directions and asks:

"Among the tiles that actually landed in that neighboring output slot, which one looks like my real neighbor back in source space?"

For each direction, we search the neighboring bucket and choose the best observed neighbor by:

- source-center displacement closeness
- same-component tie preference
- area fit
- coverage
- edge strength

Stored per tile:

- `neighbor_rgba[4, 4]`
- `neighbor_mask[4]`

This is the first honest adjacency graph the tile-graph path has had.

It is not:

- sampled one-cell-away colors from the source image
- a lattice portrait
- a guessed delta field

It is:

- "this extracted stone most naturally sat beside that extracted stone"

### Stage 7: Build per-cell candidate buckets

Function:

- `build_tile_graph_model(...)`
- helper: `_select_source_region_candidates(...)`

What it means:

Each output cell gets a bucket of legal stones.

Candidate selection now keeps only tile-owned signals:

- area fit
- coverage
- edge peak

It does not use:

- `sharp_rgba` similarity
- `edge_rgba` similarity
- injected sharp/edge fallback pixels

Important rule:

- if a cell has real source-owned region tiles, those are the truth
- if a cell has no opaque source support, it gets a transparent candidate
- if a cell has opaque source support but no extracted region candidate, the build fails loudly

That last rule is deliberate. The old machine would quietly make something up. The new one would rather stop the line than lie.

Additional background option:

- if `cell_alpha_mean_flat < 0.98`, a transparent candidate is also added

This lets the solver choose "leave this slot empty" in partially occupied or fringe cells instead of being forced to place an opaque stone just because a tiny sliver of source detail touched the slot.

### Stage 8: The solver model

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

What it means:

This is the parts tray beside the machine.

- the first block describes the legal stones
- the second block says which stones belong to which output slot
- the last block is the local source evidence already living under each slot

### Stage 9: Unary cost

Function:

- `_tile_graph_unary_cost_torch(...)`

Current unary terms:

- `color_error`
  - how far the candidate color is from the actual mean source color already under that output cell
- `area_error`
  - how far the candidate footprint is from one cell
- `alpha_error`
  - how far the candidate alpha is from the actual mean alpha under that cell
- `coverage_error`
  - how incomplete the candidate’s source ownership is
- `edge_error`
  - how far the candidate’s edge peak is from the strongest edge already under that cell

Natural-language picture:

This is the bouncer at each slot saying:

"Does this stone even look like it belongs in the patch of mural that sits under this square?"

Important difference from the old machine:

- this is local source support
- not a separate lattice portrait pretending to know the answer in advance

### Stage 10: Pairwise adjacency cost

Functions:

- `_pair_penalty_selected_torch(...)`
- `_pair_penalty_option_right_torch(...)`
- `_pair_penalty_option_left_torch(...)`
- `_pair_penalty_option_down_torch(...)`
- `_pair_penalty_option_up_torch(...)`

What it means:

Now the mason cares about seams.

If tile A says it naturally expects tile B to the right, then choosing a very different right neighbor should cost something.

The current pairwise cost compares:

- the actually chosen neighbor color
- against the neighbor color learned from extracted tiles

This is now a real observed compatibility signal.

It is no longer:

- "the source image one cell away from this center happened to look like X, so maybe that is the right delta"

### Stage 11: Checkerboard parity refinement

Function:

- `optimize_tile_graph(...)`

What it does:

1. build the model
2. compute unary cost
3. choose an initial argmin candidate per cell
4. alternate parity updates over the grid
5. keep the initial assignment if the refinement step makes source-lattice fidelity worse

Natural-language picture:

The mason lays down a first draft, then alternates black squares and white squares of the chessboard, trying to improve local seam compatibility without moving everything at once.

Important truth:

- this is still a local greedy/discrete smoother
- it is not a global optimal graph solver
- if the initial candidate buckets are wrong, parity updates will not save the image

## Data dictionary

### `flat_rgba_np`

- full-resolution source pixels flattened into `[pixel, rgba]`

### `projected_flat_index`

- for each source pixel, which output cell it belongs to under the chosen lattice

### `components`

- connected opaque paint islands in the source

### `region_buckets[flat_index]`

- all extracted real tiles currently legal for one output cell

### `candidate_rgba`

- literal colors the solver is allowed to place

### `candidate_neighbor_rgba`

- per-candidate remembered neighbor colors from real extracted-tile adjacency

### `cell_mean_rgba`

- actual average source color already living under each output cell

### `cell_alpha_mean`

- actual average source alpha already living under each output cell

### `cell_edge_strength`

- strongest edge signal already living under each output cell

## What this machine is trying to maximize

In plain language:

- choose only real, source-owned stones
- prefer stones that genuinely fit the patch of mural beneath the slot
- prefer neighboring stones that were actually observed beside each other
- never silently hallucinate a foreground tile just because the lattice wanted one

## What would make the machine lie now

If the output is still wrong after this pass, the likely failure points are no longer the old ones.

The remaining likely lies are:

- region cutting still slices a component into the wrong one-cell panes
- candidate truncation still throws away the only good tile for a difficult slot
- learned adjacency still chooses the wrong observed neighbor when multiple neighbors are plausible
- the parity solver is still too local and can settle into a bad arrangement even with legal tiles
- the lattice itself is wrong, which means the whole quarry is being cut against the wrong ruler

## Current reality check

Pinned cleaned badge run:

- target: `126x126`
- phase: `(0.0, -0.2)`
- mode: `tile-graph`
- phase rerank: off

Current result after the latest cut:

- initial source-fidelity: `0.1814`
- final source-fidelity: `0.1814`
- the solver kept the initial assignment because refinement did not improve it

That is still not good enough, but it is dramatically more honest than the earlier architectural-collapse run around `0.5055`.

In other words:

- the machine is no longer babbling nonsense
- it is still not carving the right stones often enough

## Big-picture judgment

Does the current machine make sense?

Yes, much more than before.

The data now flows in one believable direction:

1. observe the mural
2. find real paint islands
3. cut one-cell stones out of those islands
4. learn which stones really sat beside which
5. choose among those stones for each output slot

That is a coherent mental model.

What is still missing is not conceptual honesty.

What is still missing is quality in the stone cutting and compatibility choice.
