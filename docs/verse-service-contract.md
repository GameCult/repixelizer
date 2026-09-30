# Repixelizer Verse Service Contract

Repixelizer is both a local image tool and a hosted GameCult service. Heimdall
owns shared auth. Repixelizer owns repixelizing work: uploads, queue admission,
job lifecycle, progress events, generated outputs, local session/job binding,
and product presentation state. The current website is a browser lowering; Eve
is the canonical surface contract.

The current hosted service is intentionally small and single-process. The next
GameCult Verse cut is not to make it grand. The cut is to make its job and
operator state typed, inspectable, and visible through CultMesh/Eve instead of
living only inside the GUI process.

## Owner Map

- Owner: Repixelizer owns input validation, queue admission, job lifecycle,
  solver configuration, progress events, generated output paths, job/session
  ownership, hosted route policy, and local cleanup/retention decisions.
- Inputs: uploaded source image, runtime solver config, hosted-demo env flags,
  Heimdall access claims, local session cookies, queue capacity, worker
  heartbeat, cancellation requests, and filesystem spool/output paths.
- Outputs: job records, queue summaries, progress/event streams, output image
  artifacts, diagnostics, auth/session public projections, and hosted UI
  runtime config.
- Derived state: browser status panels, frontend queue polling, SSE streams,
  generated preview HTML, diagnostics JSON, and static legal pages are
  projections. They do not own job truth.
- Forbidden writers: frontend widgets, browser callback pages, Eve/TUI/native
  renderers, Heimdall, and Odin must not directly mutate job truth or own the
  product flow. They submit command intent through Repixelizer routes.
- Shared paths: local GUI, hosted GUI, API job routes, future CultCache `.cc`
  job witness, CultMesh publication, Odin discovery, and Eve/TUI operator
  surface must describe the same queue/job state.
- Deletion line: the in-process queue can remain the first owner only while the
  service is explicitly single-process. When hosted scale grows, job ownership
  must move to a typed store before extra workers are added.

## CultCache Requirement

Repixelizer does not need to store bulk image bytes in CultCache. Images,
generated outputs, and large diagnostics belong in artifact storage or the
spool/output filesystem.

Repixelizer does need typed CultCache witness documents for the state that
steers service behavior:

- `repixelizer.job.v0`: job id, account/session binding, status, timestamps,
  input artifact ref, output artifact ref, error summary, retention state.
- `repixelizer.queue_snapshot.v0`: capacity, waiting/running counts, active job
  ids, oldest waiting age, hosted-demo mode.
- `repixelizer.job_event.v0`: redacted progress event, stage, index, timestamp,
  and job id.
- `repixelizer.runtime_config.v0`: hosted flags, queue protection, visible UI
  flags, solver config hash, deployment id.
- `repixelizer.auth_projection.v0`: auth mode, provider availability, subject
  capability summary, and Heimdall claim freshness without raw tokens.

The first pass should export these as a `.cc` witness from the current
single-process manager. A later pass can decide whether the typed store becomes
the primary queue owner.

The next live cut is not speculative anymore. The Python CultLib lane already
exists in `E:\Projects\CultLib\packages\cultcache-py`, including `cultcache_py`
for `.cc` persistence and `cultnet_py` for CultNet/RUDP transport. Repixelizer
needs to wire that runtime into the hosted GUI process so the live owner can:

- write a daemon-owned `.cc` witness from `GuiJobManager`
- publish daemon health to Idunn over CultNet/RUDP
- advertise command-boundary and transport-profile state from the runtime
- keep `/api/health`, `/api/config`, nginx, and systemd as compatibility
  witnesses instead of the keepalive truth

## Current Compatibility Boundary

Today Yggdrasil still verifies Repixelizer through:

- `repixelizer-gui.service`
- `GET /api/health`
- `GET /api/config`
- nginx host-routed `/app/` and `/api/health`

Those checks prove deployment and public routing. They do not satisfy the daemon
truth contract Idunn expects.

## Eve Surface Target

Repixelizer should publish an Eve GUI/TUI DSL product surface with these panels
and flows:

1. `Runtime`: deployment mode, solver config hash, hosted-demo flags, CUDA
   availability, queue capacity, current worker state.
2. `Queue`: waiting/running/completed/failed/canceled counts, active job ids,
   queue age, stale heartbeat warnings, and rejected admission counts.
3. `Jobs`: selected job status, owner binding, stage, progress events, output
   artifact ref, diagnostics ref, and cancellation/retention command boundary.
4. `Auth`: Heimdall mode, allowed providers, current subject capability summary,
   queue protection, claim freshness, and denied access reasons.
5. `Artifacts`: spool/output roots, retention policy, recent cleanup actions,
   and artifact refs without embedding large media in service state.
6. `Image Flow`: upload, lattice inference, comparison view, zoom/pan,
   before/after inspection, cleanup tools, palette fitting, export, and
   artifact handoff.

Eve can render previews by artifact reference, but the artifacts themselves are
not the state owner. Repixelizer owns the job record that says which artifact is
current.

The existing web style should translate into Eve style tokens, canvas behavior,
tool palettes, and component variants. A Kotlin Android Eve runtime should be
able to run the cleanup flow with native controls while preserving the same
command boundary and product identity.

## Read-Only Witness And Eve Surface Export

The first read-only provider advertisement is generated by:

```powershell
repixelize witness-advertisement --out docs/fixtures/repixelizer.provider-advertisement.json
```

It names the witness schemas above, reserves `state/repixelizer.witness.cc` for
the future CultCache export, and publishes the available Eve operator surface
HTTP projection without changing queue ownership. The detailed authority maps
live in `docs/repixelizer-witness-contract.md` and
`docs/eve-surface-lowering.md`.

The read-only Eve surface fixture is generated by:

```powershell
repixelize eve-surface --out docs/fixtures/repixelizer.eve-surface.json
```

The hosted GUI serves the live projection at `/eve/surface`. It reads
`HostedDemoConfig`, `AccessController.public_payload()`, and
`GuiJobManager.get_queue_summary()`; it does not own queue truth.

## Migration Order

1. Define Repixelizer CultCache document shapes for job, queue, runtime, event,
   and auth projection witness state.
2. Publish a read-only `gamecult.eve.provider_advertisement.v1` fixture/export
   that names those shapes and the reserved `.cc` witness path.
3. Add a read-only export from the current `gui.py` queue/job manager into a
   daemon-owned `.cc` witness using `cultcache-py`.
4. Publish Repixelizer daemon health to Idunn over CultNet/RUDP from the live
   GUI process using `cultnet_py`, with a dedicated Repixelizer health contract.
5. Publish runtime-owned command-boundary and transport-profile state beside
   the witness so deploy/restart authority is typed instead of implicit ops lore.
6. Update the Yggdrasil deploy lane to ship the required CultLib Python package
   snapshot with the app artifact and install it before the Repixelizer package.
7. Publish the witness through CultMesh.
8. Add an Eve DSL provider over the witness and existing health/config routes.
   First read-only HTTP projection is live at `/eve/surface`.
9. Translate the existing browser GUI style and tool behavior into Eve style
   tokens, canvas primitives, and command descriptors.
10. Lower the current website from the Eve surface instead of treating it as the
   product UI owner.
11. Register the surface with Odin.
12. Only after the witness is stable, decide whether queue ownership should move
   from in-process memory into the typed store.

The invariant: Heimdall owns shared auth; Repixelizer owns jobs and artifacts.
CultCache/CultMesh/Eve make that state inspectable without moving product truth
into the auth service or the renderer.
