# Repixelizer Eve Surface Lowering

Repixelizer now emits a read-only `gamecult.eve.surface.v1` projection for its
front page, single-page app workflow, runtime config, auth projection, queue
summary, and command intents.

## Authority Map

- Owner: `repixelizer.gui` still owns hosted config, route policy, queue state,
  job ownership, uploads, solver execution, and browser assets.
- Inputs: `HostedDemoConfig`, `AccessController.public_payload()`, and
  `GuiJobManager.get_queue_summary()`.
- Output: `repixelizer.eve_surface.build_repixelizer_eve_surface()` emits the
  retained Eve surface tree, style tokens, command descriptors, and renderer
  comparison notes.
- Derived state: `/eve/surface`, `repixelize eve-surface`, and
  `docs/fixtures/repixelizer.eve-surface.json` are projections. They do not
  decide queue truth.
- Forbidden writers: Eve renderers, Odin, Periwinkle, and fixtures must route
  commands back through `/api/jobs`, `/api/jobs/{job_id}`, or `/app/`; they do
  not mutate Repixelizer jobs directly.
- Shared paths: browser GUI, CLI fixture export, hosted `/eve/surface`, and
  provider advertisement all name the same surface and route authority.
- Cut line: no parallel status-card service, no renderer-owned queue summary,
  and no fake native canvas authority until Eve can lower upload and image-edit
  controls coherently.

## Published Surface

Generate the deterministic fixture:

```powershell
repixelize eve-surface --out docs/fixtures/repixelizer.eve-surface.json --updated-at 2026-06-04T00:00:00Z --public-base-url https://repixelizer.gamecult.org
```

The hosted GUI serves the live projection at:

```text
/eve/surface
```

The provider advertisement now marks `repixelizer.operator.surface` as
`available` with an HTTP JSON lowering at `/eve/surface`.

## Retro Style Contract

The Eve style profile is `repixelizer.retro.pixel`. The tokens intentionally
mirror the browser CSS instead of flattening the app into generic dashboard
chrome:

- title font: `Press Start 2P`
- body/mono font: `VT323`
- dark shell gradient and blue panel surfaces
- warm yellow/orange accent tokens
- chunky pixel borders, lower border weight, corner accents, scanline overlay,
  dashed inner frame, and pixelated image sampling

The browser lowering remains the visual oracle for this pass because it already
implements the full upload, solver inspection, comparison canvas, editor, and
retro CSS.

## Periwinkle Comparison

Periwinkle's Android Kotlin lowerer currently has useful structural support:
it connects to the Eve dashboard broker, traverses `surface.root`, renders
pane/card/text hierarchies, supports basic metric progress bars, and attaches
some command handlers.

The current gap is style and control parity. `MainActivity.kt` still hard-codes
`toneColor`, `textColorFor`, and `textSizeFor`; it does not consume
`surface.styles.tokens`, does not load Repixelizer's pixel fonts, does not
render image assets from the surface tree, and does not lower the upload picker,
comparison canvas, pan/zoom inspection, eyedropper, or paint controls.

Minimum Periwinkle parity work:

- map provider style tokens before applying fallback tone/text/font rules
- package or cache `Press Start 2P` and `VT323`
- render `assetUri` / `assetRef` as pixelated `ImageView` content
- lower Repixelizer command controls to Android document picker and job route
  calls
- add native canvas controls only after ownership remains routed through the
  existing Repixelizer job and editor command paths
