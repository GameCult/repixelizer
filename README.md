# Repixelizer

Repixelizer is a Python tool for images that are doing a pixel art impression instead of actually respecting the grid.

It is aimed at the annoying middle ground: sprites, emblems, and logos that look locally pixelated but fall apart the second you ask them to commit to one lattice like grown-ups.

Instead of pretending this is a resize problem, Repixelizer treats it as lattice inference plus local structure preservation: infer the implied grid, choose real output cells, and snap them back onto a coherent pixel lattice while preserving the source's local adjacency patterns.

## GUI First

The web GUI is the main way to use this thing unless you enjoy raw flags for sport.

It gives you:

- drag-and-drop input
- live pipeline feedback while lattice search, rerank, and solving run
- synchronized input/output inspection zoom for close comparison
- a built-in output editor for single-pixel cleanup with eyedropper and pencil tools

Run it from the repo checkout:

```powershell
.\scripts\run_gui.ps1
```

If you want to bypass the PowerShell wrapper and call Python directly:

```powershell
.\.venv\Scripts\python scripts\run_gui.py
```

If you want to support the official hosted deployment and help us scale the poor little server when people start leaning on it, join the [GameCult Patreon](https://www.patreon.com/GameCult). At `$10` per subscriber, the math gets a lot less tragic.

## Example

This closeup from the badge fixture is the whole pitch in miniature: same ugly source patch, same target grid, one output that finally quits bluffing.

![Repixelizer badge closeup](docs/readme-assets/guard-right-crop-source-final.png)

## Current Status

This repo is past the "pile of hopeful heuristics" stage and into "real machine, still experimental."

What exists now:

- a web GUI with drag-and-drop, live diagnostics, comparison tools, and pixel cleanup
- lattice inference with CUDA support
- a lean displacement-field optimizer in `src/repixelizer/phase_field.py`
- automatic diagnostics, comparisons, and benchmark runs
- a tuning harness for offline parameter sweeps
- metrics that finally care about visible structure instead of only pleasing the lattice accountant

What changed recently:

- the old tray-based optimizer is gone
- `phase-field` is now the default optimizer path
- `source_structure` is reported alongside `source_fidelity`, because the old metric was happily calling better-looking images worse
- the tracked sword-tip blemish on the AI badge has its own focused fixture in `tests/fixtures/real/ai-badge-tip-focus.json`

Current read on the engine:

- `phase-field` is the release path. It currently produces the best-looking badge result in the repo, especially on internal linework, even though it still widens the tracked sword-tip stroke a bit too much.

Current weak spots:

- `phase-field` still needs better along-stroke versus across-stroke behavior near tapered contours
- lattice selection is still low-confidence on some ugly generated inputs, so pinned-size iteration remains an important workflow

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

Launch the GUI:

```powershell
.\scripts\run_gui.ps1
```

Run the optimizer from the CLI:

```powershell
repixelize input.png --out output.png
repixelize input.png --out output.png --diagnostics-dir diagnostics --device auto
repixelize input.png --out output.png --target-width 126 --target-height 126 --phase-x 0.0 --phase-y -0.2 --device cuda
```

Run the optimizer plus baselines:

```powershell
repixelize compare input.png --out output.png --diagnostics-dir diagnostics
```

## Corpus And Benchmarks

Prepare a local Creative Commons corpus:

```powershell
repixelize prepare-corpus --corpus-dir examples/corpus
```

Run the round-trip benchmark:

```powershell
repixelize benchmark --corpus-dir examples/corpus --out-dir artifacts/benchmark
```

Useful benchmark flags:

- `--profile soft` or `--profile crisp`
- `--case <name>` repeated for a focused slice
- `--limit <n>` for the first `n` matching cases
- `--infer-size` to test automatic size inference
- `--keep-existing` if you explicitly want to preserve an old output directory

By default the benchmark clears its output directory before each run so the artifacts folder stays readable.

## Tuning

Repixelizer includes a black-box tuning loop for longer offline searches over solver weights:

```powershell
repixelize tune --corpus-dir examples/corpus --out-dir artifacts/tuning --profile soft --limit 8
```

This is intentionally a search-based tuner, not gradient descent. The objective depends on discrete argmins, thresholding, and benchmark comparisons, so it is better treated as a repeatable black-box optimization problem.

## Repository Hygiene

This repo is set up to be safe for local experimentation:

- generated outputs under `artifacts/` are ignored
- local corpus payloads under `examples/corpus/originals/` are ignored
- archived raw sprite sheets under `examples/corpus/source-sheets/` are ignored
- generated corpus metadata files are ignored

That keeps benchmark assets, attribution exports, and tuning runs from polluting upstream history while still keeping the code, docs, and corpus layout tracked.

## Repo Layout

- `src/repixelizer`: core package
- `tests`: focused regression tests
- `docs/spec.md`: product and technical spec
- `docs/implementation-plan.md`: working roadmap
- `docs/lean-optimizer-algorithm-map.md`: map of the live displacement-field optimizer
- `examples/corpus/README.md`: local corpus layout and attribution workflow

Core modules:

- `inference.py`: target size and phase inference
- `phase_field.py`: default displacement-field optimizer
- `metrics.py`: fidelity, adjacency, and motif metrics
- `benchmark.py`: corpus benchmark runner
- `tuning.py`: black-box hyperparameter search harness
- `synthetic.py`: facsimile generation for tests and benchmarks

## Validation

Run the focused test suite with:

```powershell
.venv\Scripts\python -m pytest -q
```

## Diagnostic Closeups

For docs or regression notes, you can generate reproducible source/output closeups from output-grid cell coordinates with:

```powershell
.venv\Scripts\python scripts\render_focus_crop.py --input tests\fixtures\real\ai-badge-cleaned.png --out docs\readme-assets\guard-right-crop-source-final.png --cell-bbox 73 20 107 44 --panels source,final --scale 16 --steps 48 --device cpu
```

If you want the full internal state sheet instead of the docs-facing comparison:

```powershell
.venv\Scripts\python scripts\render_focus_crop.py --input tests\fixtures\real\ai-badge-cleaned.png --out docs\readme-assets\guard-right-crop-states.png --cell-bbox 73 20 107 44 --panels source,initial,final --scale 16 --steps 48 --device cpu
```

Notes:
- `--cell-bbox` is in output-grid coordinates, not source-image pixels.
- The script maps that region back to the matching source crop automatically.
- `--device cpu` is the polite choice for docs generation if you do not want to stress the GPU for a tiny crop.

## Notes

- The local corpus is optional; the codebase is still usable without it.
- The benchmark and tuning commands are designed for iterative local work, not for shipping generated artifacts in git.
- Palette enforcement is optional and can be left off for unconstrained recovery runs.
