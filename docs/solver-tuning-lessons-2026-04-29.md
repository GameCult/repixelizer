# Solver tuning lessons - 2026-04-29

This note records what the tuning pass actually taught us, what it did not
teach us, and how the next pass should be run without constructing another
beautiful little swamp.

## Shipping state

The current shipped trial config is intentionally usable, not final:

```json
{
  "phase_field_grid_alignment_weight": 4.75,
  "phase_field_local_search_radius_ratio": 0.215,
  "phase_field_local_search_blend": 0.24,
  "phase_field_local_search_move_weight": 0.09,
  "phase_field_signal_energy_power": 1.4,
  "phase_field_signal_peak_power": 3.25
}
```

Everything else in `config/solver_params.json` remains as previously configured.

The important caveat: this is a motion-geometry win, not a full signal-shape
optimum. It is acceptable for shipping the hosted experiment, but it should not
be treated as the final form of the phase-field solver.

## What the pass proved

The most useful discovery was that the local-search radius was much too large.
The solver was not merely underpowered; it was letting samples inspect too much
neighboring territory. Adjacent samples could chase the same feature, which
looked like feature attraction but behaved like poaching.

Recent focused artificial results, all with:

```text
grid=4.75
blend=0.18 unless noted
move=0.09
energy=1.4
peak=3.25
```

showed this radius curve:

| Run | Radius | Score | Mean | P90 | Worst | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 068 | 0.40 | 0.187047 | 0.073598 | 0.278124 | 0.445583 | Previous best before lower bracket |
| 071 | 0.32 | 0.144824 | 0.056821 | 0.169920 | 0.363593 | Big improvement; radius was still too large |
| 075 | 0.24 | 0.128386 | 0.046622 | 0.126980 | 0.341534 | Better again |
| 080 | 0.20 | 0.119077 | 0.046775 | 0.121492 | 0.306095 | Strong lower shoulder |
| 081 | 0.22 | 0.116976 | 0.044065 | 0.116555 | 0.306714 | Strong local center |
| 084 | 0.215 | 0.116584 | 0.044071 | 0.117496 | 0.304752 | Best radius point tested |
| 085 | 0.225 | 0.121924 | 0.045445 | 0.131808 | 0.316814 | Upper side already regresses |

The model this supports:

- `radius` is the gatekeeper for whether samples solve locally or poach.
- Very small radius eventually starves movement, as seen at `0.16`.
- The useful basin is narrow, roughly `0.20-0.22`, with `0.215` best in the focused pass.
- Once the search window is tight, stronger blend becomes useful because samples can move without wandering into the neighbor's lunch.

Blend sweep at `radius=0.215`:

| Run | Blend | Score | Mean | P90 | Worst | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 087 | 0.12 | 0.161612 | 0.062461 | 0.197631 | 0.404997 | Too timid |
| 088 | 0.15 | 0.136534 | 0.050632 | 0.145291 | 0.356375 | Still timid |
| 084 | 0.18 | 0.116584 | 0.044071 | 0.117496 | 0.304752 | Radius-only best |
| 089 | 0.20 | 0.111183 | 0.041232 | 0.112633 | 0.292477 | Better |
| 090 | 0.24 | 0.103754 | 0.038329 | 0.100917 | 0.274995 | Current focused winner |

The model this supports:

- After radius is tightened, blend must be retuned.
- Low blend was compensating for an over-wide search area.
- At the new radius, `0.24` is better than `0.18` on focused metrics.
- We did not yet find the upper blend cliff. `0.24` won the tested set, but values above it still need bracketing later.

## What the pass did not prove

The pass did not exhaustively model the entire system.

Specifically, after finding the new `radius/blend` basin, the later tuning did
not sweep:

- `phase_field_signal_weight`
- `phase_field_signal_gradient_weight`
- `phase_field_signal_curvature_weight`
- `phase_field_signal_level_decay`
- `phase_field_signal_pyramid_levels`
- `phase_field_signal_energy_power` around the new basin
- `phase_field_signal_peak_power` around the new basin
- `phase_field_local_search_move_weight` around the new basin
- `phase_field_grid_alignment_weight` around the new basin
- `phase_field_local_search_grid_weight` around the new basin

Earlier experiments touched some of these in older regimes, but those results
should not be treated as final because they were gathered before the current
motion geometry was discovered. A signal knob can look useless when the solver
cannot move into the signal correctly. That is exactly how we wasted time
before.

The correct summary is:

- Current config is shippable.
- Motion geometry is much better understood.
- Signal shape remains underexplored in the new basin.
- The next tuning pass should start from `radius=0.215, blend=0.24`, not from the old wide-radius config.

## Current mental model

The phase-field solver now looks like this:

1. Build a hierarchical flatness signal from the source image.
2. Give each lattice sample a small local candidate window.
3. Let it choose the locally most comfortable source position.
4. Blend toward that local preference.
5. Use lattice coherence pressure to keep neighboring samples from tearing the grid apart.

The important part is the scale separation:

- The signal says where coherent source positions are.
- The local search radius says which nearby positions a sample is allowed to consider.
- The blend says how fast the sample moves toward the chosen position.
- The lattice coherence force says how much the grid resists local deformation.

The mistake was treating signal strength as the main issue while the search
window was still too broad. A broad search window turns a good signal into a
poaching contest.

## Config intents we can currently generate

These are educated starting points, not final recommendations:

### Hot focused winner

Use when the goal is maximum focused artificial metric improvement and the live
visual does not look too twitchy.

```json
{
  "phase_field_grid_alignment_weight": 4.75,
  "phase_field_local_search_radius_ratio": 0.215,
  "phase_field_local_search_blend": 0.24,
  "phase_field_local_search_move_weight": 0.09,
  "phase_field_signal_energy_power": 1.4,
  "phase_field_signal_peak_power": 3.25
}
```

### Balanced live trial

Use if the hot winner looks over-eager on real AI cases.

```json
{
  "phase_field_grid_alignment_weight": 4.75,
  "phase_field_local_search_radius_ratio": 0.215,
  "phase_field_local_search_blend": 0.20,
  "phase_field_local_search_move_weight": 0.09,
  "phase_field_signal_energy_power": 1.4,
  "phase_field_signal_peak_power": 3.25
}
```

### Conservative control

Use if the solver starts visibly over-moving or damaging tiny details.

```json
{
  "phase_field_grid_alignment_weight": 4.75,
  "phase_field_local_search_radius_ratio": 0.20,
  "phase_field_local_search_blend": 0.18,
  "phase_field_local_search_move_weight": 0.09,
  "phase_field_signal_energy_power": 1.4,
  "phase_field_signal_peak_power": 3.25
}
```

## Next tuning protocol

The next pass should not repeat the failed "long campaign then hope" shape.

Run it in phases:

1. Lock the current motion baseline:

```text
radius=0.215
blend=0.24
move=0.09
grid=4.75
energy=1.4
peak=3.25
```

2. Sweep every signal knob once against that baseline:

```text
signal_weight
gradient_weight
curvature_weight
level_decay
pyramid_levels
energy_power
peak_power
```

3. Sweep movement/coherence knobs once against the best signal candidate:

```text
move_weight
grid_alignment_weight
local_search_grid_weight
local_search_interval
max_displacement_ratio
```

4. Only then test combinations.

5. Persist every experiment immediately with:

```text
hypothesis
patch
score
mean
p90
worst
case-specific regressions
interpretation
next hypothesis
```

6. Run full-corpus validation only after focused cases stop moving.

7. Do visual checks on real cases before shipping:

```text
character eye glint
biohazard symbol
badge sword-tip focus
simple crisp artificial cases
soft artificial cases
AI-distorted artificial cases
```

## Failure mode from this pass

The user asked for every knob and combination to be touched first. That did not
happen. The pass found a real, valuable motion-geometry basin, but it then
declared too much confidence from too little coverage.

Do not repeat that.

The next agent should treat this note as a partial model:

- strong evidence for radius/blend behavior
- weak evidence for signal behavior in the new basin
- no final evidence for all-knob interactions

Ship the current config because the project needs to unblock Heimdall and
StreamPixels. Come back later with a disciplined signal-first pass.
