# Repixelizer Verse Service Contract

Repixelizer is both a local image tool and a hosted GameCult service. Heimdall
owns shared auth. Repixelizer owns repixelizing work: uploads, queue admission,
job lifecycle, progress events, generated outputs, local session/job binding,
and hosted UI behavior.

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
- Forbidden writers: frontend widgets, browser callback pages, Eve/TUI
  renderers, Heimdall, and Odin must not directly mutate job truth. They submit
  command intent through Repixelizer routes.
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

## Eve Surface Target

Repixelizer should publish an Eve GUI/TUI DSL operator surface with these
panels:

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

Eve can render previews by artifact reference, but the artifacts themselves are
not the state owner. Repixelizer owns the job record that says which artifact is
current.

## Migration Order

1. Define Repixelizer CultCache document shapes for job, queue, runtime, event,
   and auth projection witness state.
2. Add a read-only export from the current `gui.py` queue/job manager into a
   `.cc` witness.
3. Publish the witness through CultMesh.
4. Add an Eve DSL provider over the witness and existing health/config routes.
5. Register the surface with Odin.
6. Only after the witness is stable, decide whether queue ownership should move
   from in-process memory into the typed store.

The invariant: Heimdall owns shared auth; Repixelizer owns jobs and artifacts.
CultCache/CultMesh/Eve make that state inspectable without moving product truth
into the auth service or the renderer.
