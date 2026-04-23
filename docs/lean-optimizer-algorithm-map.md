# Lean Optimizer Replacement Map

## Why this map exists

The current optimizer is useful, but it is also a fussy little parliament.

Too many portraits. Too many local trays. Too many side negotiations about adjacency, motif, line, handoff, guardrails, and whether the machine still trusts its own decisions.

This document is not a map of the code we have.

It is a map of the optimizer we should replace it with:

`infer one lattice -> attach one displacement vector to each output cell -> nudge those vectors toward solid source regions while keeping the field coherent -> sample once`

That is the whole dream. No watercolor portrait. No accountant's ledger. No snap religion and refine religion living in the same cathedral. Just one field, one objective, one final sample.

The first implementation of this restart now exists in `src/repixelizer/phase_field.py` behind `--reconstruction-mode phase-field`.

That implementation is intentionally small and slightly stupid:

- one displacement vector per output cell
- one local solidity term
- one edge-aware smoothness term
- one anti-collapse spacing term
- one displacement magnitude prior
- one final nearest source sample

It is not yet beating the old continuous optimizer on the pinned badge stress case, but it is finally the right kind of machine to improve.

## One-sentence machine

The lean optimizer should be:

`source image -> edge scout -> fixed lattice centers -> zero displacement field -> direct phase-field optimization -> nearest source sampling -> done`

Or, in pictures:

- the source image is the mural
- the inferred lattice is the graph paper
- the displacement field is each graph-paper cell deciding how far to slide before it bites into real paint
- the objective is trying to make those slides land inside solid fake-pixel regions without the whole field folding into a heap

## How to read this machine

The replacement optimizer should carry only four important pieces of cargo:

- `uv0`: the fixed lattice centers chosen by inference
- `d`: one displacement vector `(dx, dy)` per output cell, initialized to zero
- `edge_map`: one edge scout report for the source image
- `sampled_rgba`: the current source color under `uv0 + d`

Everything else is a term in the objective, not a new ontology.

If a future version introduces more than those four core state variables, it should have to explain itself like a defendant.

## What survives from the current code

These existing pieces are still worth keeping:

- lattice inference in `src/repixelizer/inference.py`
- edge analysis in `src/repixelizer/analysis.py`
- source-lattice consistency metrics in `src/repixelizer/metrics.py`
- diagnostics and compare-mode output in `src/repixelizer/pipeline.py` and `src/repixelizer/diagnostics.py`
- the current optimizer in `src/repixelizer/continuous.py` as a benchmark and fallback during the replacement

These pieces should **not** be part of the new core:

- `representative_t`
- `source_lattice_reference`
- `source_reliability_t`
- snap candidate trays
- relax candidate probabilities
- greedy refine candidate trays
- the current layered boundary / motif / line dialect stack

The new machine should optimize one thing directly: the displacement field.

## Stage 0: The pipeline picks the ruler

Source that survives:

- `run_pipeline(...)` in `src/repixelizer/pipeline.py`
- `infer_lattice(...)` and `infer_fixed_lattice(...)` in `src/repixelizer/inference.py`

Meaning:

The new optimizer does **not** get to argue about target size or global phase after the pipeline hands it the ruler.

The ruler still decides:

- output width and height
- the initial phase offset
- the initial cell spacing in source pixels

That part of the machine is already a separate concern and should stay that way.

### Metaphor

The survey crew hammers the stakes in once.  
After that, the new optimizer is not a politician. It is a field crew. It works inside the stakes or it goes home.

## Stage 1: Lay down the fixed lattice centers

Current source to reuse:

- `_make_regular_uv(...)` in `src/repixelizer/continuous.py`

New state:

- `uv0[y, x]`: the base source-space center for each output cell

Meaning:

This is the unmoving graph paper.

Unlike the current optimizer, the new machine should be honest about what is fixed and what is variable:

- `uv0` is fixed
- `d` is the thing being optimized

There should be no fake "UV optimizer" language once that distinction exists.

### Metaphor

This is the empty skeleton of the machine.  
The graph paper is nailed to the mural. Nothing clever has happened yet. Each cell is just a little pinned square waiting to decide how much to slide before it commits.

## Stage 2: Initialize one displacement vector per cell

New state:

- `d[y, x] = (0, 0)` initially

Meaning:

Every output cell starts by targeting the exact center of its inferred source cell.

That is the null hypothesis:

- maybe the inferred phase is already fine
- maybe some regions need to drift

The machine starts from stillness and earns every movement.

### Metaphor

Every cell begins with its feet planted under its shoulders.  
No lunging. No mystical premonitions about where it "really wants" to go. Just a row of workers standing still on the scaffold until the foreman starts shouting.

## Stage 3: Sample the source at the displaced centers

Derived state:

- `uv = uv0 + d`
- `sampled_rgba[y, x]`

Meaning:

At any moment, each output cell points at one source location:

- start from the fixed center
- add the current displacement
- sample the source there

This is the only color the cell currently "believes in."

There should be no alternate portrait layer describing what the cell vaguely feels like. The sample is the truth of the current hypothesis.

### Metaphor

Each worker shines a laser pointer from the scaffold down onto the mural.  
Where the dot lands, that is the color they are claiming. No watercolor sketch. No emotional support ledger. Just a dot on paint.

## Stage 4: Compute the local evidence term

New objective family:

- prefer displacement targets that land inside solid or internally coherent local regions

Candidate implementation options worth testing:

- low local variance in a tiny neighborhood around `uv`
- low edge energy in a tiny neighborhood around `uv`
- high agreement between the center sample and nearby samples inside a fractional cell window

Meaning:

This is the term that tries to make each cell land inside a fake-pixel interior instead of on a messy boundary.

The purpose is simple:

- solid colored areas are safe
- high-conflict mixed areas are dangerous

This term is the main answer to "why move at all?"

### Metaphor

The worker is probing the mural with a boot before putting full weight down.  
Some spots are firm stone. Some are crumbling ledges. This term tells the field to step onto the stable bits instead of balancing on the edge of a crack like an idiot.

## Stage 5: Compute coherence and topology terms

New objective family:

- neighboring cells should have similar displacements unless a real edge justifies a break
- nearby output cells should not collapse onto the same source pixel
- the field should not fold, cross, or invert local order
- displacement magnitude should stay modest unless the data term really wants motion

Meaning:

This is where the optimizer stops being a mob of independent little boot-probes.

These terms are what make the displacement field a field:

- smoothness keeps neighboring cells moving together
- edge-aware gating prevents smoothness from smearing through sharp source boundaries
- anti-collapse / anti-fold constraints stop the grid from crumpling into duplicated samples
- magnitude regularization keeps the field from wandering off because one local patch looked briefly tempting

This is the actual home for the design goal that `relax` only half fulfilled.

### Metaphor

Now imagine the workers are connected by springs, but the springs are smart enough to loosen at real cracks in the wall.  
They can drift together like a sheet, but they are not allowed to pile into the same footprint or climb over each other like rats in a drain.

## Stage 6: Update the displacement field directly

New core operation:

- optimize `d` itself, not a tray of discrete local candidate picks

Candidate solver shapes worth trying:

- gradient descent or Adam on a differentiable approximation
- projected gradient steps with explicit clamp / anti-fold projection
- alternating local smoothing plus data-term descent

Hard invariants:

- `d` remains bounded
- local topology constraints are enforced every iteration
- edge-aware coherence remains part of the update, not a decorative afterthought

Meaning:

This is the heart of the replacement.

The field should move directly under one unified objective, not by:

- inventing a portrait
- turning it into candidate trays
- softening the trays
- hardening the trays
- comparing against a previous tray-based religion

If the optimizer exists, it should optimize.

### Metaphor

This is finally the graph paper itself breathing and sliding over the mural.  
Not stones being swapped in and out of trays. Not committees debating motifs. The sheet moves, the springs pull, the cracks resist, the dots settle.

## Stage 7: Quantize to final source pixels

Final sampling step:

- convert `uv0 + d` to source sampling positions
- round or nearest-sample to the final source pixel for each output cell
- emit `target_rgba`

Meaning:

Once the field has settled, the image comes from one final honest sampling step.

This is where the current machine keeps cheating by spending most of its life as a discrete chooser. The replacement should do the opposite:

- optimize a field first
- discretize once at the end

That keeps the algorithm aligned with its own story.

### Metaphor

After all the sliding and settling, each worker plants the flag once.  
No more hedging. No more "best of snap and refine." Just one final stamp where the field says the cell belongs.

## Stage 8: Evaluate against reality

Source to keep:

- `source_lattice_consistency_breakdown(...)` in `src/repixelizer/metrics.py`
- compare-mode diagnostics in `src/repixelizer/pipeline.py`

Meaning:

The new machine still needs adult supervision.

But the supervision should judge one coherent pipeline, not referee a knife fight between three subsolvers. The important diagnostics for the replacement should be:

- source-lattice fidelity
- displacement magnitude statistics
- displacement smoothness / jitter
- topology violations prevented or projected away
- maybe edge-crossing counts if we make that measurable

If the replacement loses badly to the old optimizer, we keep the old optimizer. No romance.

### Metaphor

This is the inspector with the clipboard, but now the questions are cleaner.  
Did the field settle into real cell interiors? Did the sheet drift coherently? Did it fold? Did it cheat? Did the result actually look less stupid?

## What fits this machine

These things belong:

- one fixed inferred ruler
- one displacement field
- one direct objective
- edge-aware smoothness
- anti-collapse / anti-fold constraints
- one final discrete sample

Those all speak the same language.

## What does not fit this machine

These things should be treated as suspicious imports from the old religion:

- multiple competing portraits of the same cell
- separate snap / relax / refine solver identities
- repeated adjacency / motif / line stacks in different stages
- candidate trays as the main state of the optimizer
- any scoring term that exists mainly to compensate for another scoring term

If one of those comes back, it should have to win a public trial.

## The current replacement plan

The lean optimizer should be built beside the current one, not by mutating the old whale in place.

Recommended build order:

1. Reuse lattice inference and edge scouting.
2. Create a new displacement-field optimizer module.
3. Implement the smallest viable objective:
   - local solidity
   - smoothness
   - edge-aware smoothness
   - anti-collapse
   - displacement magnitude prior
4. Add diagnostics for displacement magnitude, smoothness, and topology health.
5. Compare against the current optimizer on pinned badge and emblem cases.
6. Only port any old "fancy" idea if it clearly improves the real outputs.

## The diamond test

The replacement is worth keeping only if it passes this crude, necessary test:

Can a new reader describe the whole machine without sounding like they are defending a tax code?

If not, there is still more to cut.
