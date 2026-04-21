# Corpus Layout

Put Creative Commons source pixel art in [originals](</E:/Projects/repixelizer/examples/corpus/originals>).
If you import sprite sheets or collage sheets first, run `repixelize prepare-corpus --corpus-dir examples/corpus` to archive the raw sheets into `source-sheets/` and rewrite `originals/` into one-sprite-per-file inputs for benchmarking.
That step also refreshes [ATTRIBUTION.md](</E:/Projects/repixelizer/examples/corpus/ATTRIBUTION.md>) from the processed sidecars.

Recommended layout:

- `examples/corpus/originals/<name>.png`
- `examples/corpus/originals/<name>.json`

The JSON sidecar is optional, but useful for attribution and notes. Suggested fields:

```json
{
  "title": "Sprite Name",
  "author": "Artist Name",
  "license": "CC-BY 4.0",
  "source_url": "https://example.com/original",
  "notes": "Strong outline, metallic highlights."
}
```

Run the round-trip benchmark with:

```powershell
repixelize prepare-corpus --corpus-dir examples/corpus
repixelize benchmark --corpus-dir examples/corpus --out-dir artifacts/benchmark
```

By default the benchmark locks the target size to the original image dimensions so we can measure recovery quality cleanly. Add `--infer-size` if you want to evaluate automatic size inference too.
