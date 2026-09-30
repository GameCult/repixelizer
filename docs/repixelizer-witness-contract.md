# Repixelizer Witness Contract

Repixelizer's hosted queue is still owned by `src/repixelizer/gui.py`.
This document defines the read-only first cut for making that state visible to
Odin, Huginn, CultMesh, and Eve without moving queue authority.

## Objective

Publish a `gamecult.eve.provider_advertisement.v1` fixture/export that names
Repixelizer's typed witness shapes and reserved CultCache `.cc` witness path.
The export is an authority map. It is not the queue store, not an Eve renderer,
and not a replacement for the current hosted routes.

Generate it with:

```powershell
repixelize witness-advertisement --out docs/fixtures/repixelizer.provider-advertisement.json
```

The checked-in fixture is:

`docs/fixtures/repixelizer.provider-advertisement.json`

## Authority Map

- Owner: `src/repixelizer/gui.py` owns queue admission, job lifecycle, progress
  events, cancellation, heartbeat expiry, and spool cleanup.
- Inputs: uploaded image bytes, normalized hosted options, queue capacity,
  worker heartbeat, cancellation requests, app-local subject binding, and
  Heimdall-derived access projection.
- Outputs: in-process job records, queue summaries, SSE progress events, output
  image artifacts, diagnostics refs, and route-level JSON projections.
- Derived state: provider advertisement JSON, future `.cc` witness export,
  CultMesh publication, Odin discovery state, and Eve operator surfaces.
- Forbidden writers: Odin, Eve renderers, provider-advertisement fixtures,
  witness exporters, and `.cc` inspection tools do not decide queue truth.
- Shared paths: browser GUI, hosted API routes, future `.cc` witness export,
  future CultMesh publication, and Eve surfaces must describe the same
  Repixelizer-owned job state.
- Cut line: this pass reserves witness names and paths only. A later pass may
  add a real `.cc` writer or move queue ownership, but that must be an explicit
  rebuild of ownership, not an exporter pretending to be persistence.

## Witness Schemas

The first advertisement names these schemas:

- `repixelizer.job.v0`: job id, account/session binding, status, timestamps,
  input artifact ref, output artifact ref, error summary, and retention state.
- `repixelizer.queue_snapshot.v0`: queue capacity, waiting/running counts,
  active job ids, oldest waiting age, and hosted-demo mode.
- `repixelizer.job_event.v0`: redacted progress event, stage, index, timestamp,
  and job id.
- `repixelizer.runtime_config.v0`: hosted flags, queue protection, visible UI
  flags, solver config hash placeholder, deployment id, and spool path.
- `repixelizer.auth_projection.v0`: auth mode, provider availability, subject
  capability summary, and Heimdall claim freshness without raw tokens.

The reserved witness path is `state/repixelizer.witness.cc`.

## Provider Advertisement

`repixelize witness-advertisement` emits:

- `schema: gamecult.eve.provider_advertisement.v1`
- `providerId: repixelizer`
- schema catalog entries for the witness shapes above
- a reserved `.cc` witness entry with bulk image bytes, raw tokens, and private
  claims marked as redacted
- an available canonical Eve operator surface projection at `/eve/surface`,
  retaining the canonical CultMesh surface key:
  `cultmesh://repixelizer/surfaces/operator`
- the current browser lowering at `/app/`
- route-owned command boundaries for web open, job submit, and own-job cancellation
- nested Verse declarations for session and hosted-operator spaces
- runtime projection from `HostedDemoConfig.from_env()` and
  `AccessController.from_env()`

This is deliberately read-only. The current queue is still in-process memory.
The advertisement only tells discovery organs where truth lives, where the
read-only Eve projection is served, and what shape future durable witnesses must
take.
