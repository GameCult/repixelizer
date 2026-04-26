# Repixelizer Hosted Access Profile

## What this file is

This file binds Repixelizer onto the shared future design in
`docs/gamecult-hosted-access-architecture.md`.

It is still future design, not a claim that the hosted demo already has auth.

## Current repo boundary

Current live hosted-demo behavior is still just:

- hosted-demo runtime limits and UI flags in `src/repixelizer/gui.py`
- the local single-process queue in `src/repixelizer/gui.py`
- frontend runtime-config consumption in `frontend/src/app.ts`

No Discord login, Patreon login, linked identities, local sessions, or access
gates are landed yet.

## App identity

- `app_slug`: `repixelizer`
- primary hostname: `repixelizer.gamecult.org`
- runtime shape: one hosted GUI plus one queue-backed worker loop inside the app

## Capabilities

Repixelizer does not need a baroque permission lattice.

The likely first-cut capabilities are:

- `app_access`
  - may load the protected hosted GUI
- `queue_submit`
  - may create a repixelizer job
- `job_read_own`
  - may read own job state, event stream, and final output
- `job_cancel_own`
  - may cancel own queued or running job
- `admin_access`
  - may inspect operational or grant/admin surfaces later

For the first cut:

```text
queue_submit = app_access
job_read_own = app_access + resource ownership check
job_cancel_own = app_access + resource ownership check
```

## Access policy

Repixelizer should inherit the generic GameCult access pattern:

```text
app_access =
  discord.allowed_role
  || patreon.allowed_tier
  || grant.global_member
  || grant.app_access

queue_submit = app_access

admin_access =
  grant.operator
  || grant.admin_access
```

Likely Discord-allowed examples:

- Patreon supporter role
- founder role
- moderator role
- explicit alpha-tester role

Likely Patreon-allowed examples:

- any paid GameCult membership
- or one configured subset of tiers if the experiment needs a stricter gate

## Repixelizer runtime binding

When this lands, the shared access layer should integrate with the current queue
shape rather than replace it.

Suggested queue/job additions:

- `jobs.account_id`
- `jobs.session_id`
- `jobs.access_revision`
- `jobs.entitlement_checked_at`

Route policy:

- public:
  - `/api/health`
  - `/api/config`
- public or capability-gated by policy choice:
  - `/api/queue`
- requires `queue_submit`:
  - `POST /api/jobs`
- requires ownership:
  - `GET /api/jobs/{job_id}`
  - `GET /api/jobs/{job_id}/events`
  - `POST /api/jobs/{job_id}/heartbeat`
  - `DELETE /api/jobs/{job_id}`

Important invariant:

- job ownership must resolve from the local session/account, not from raw
  Discord or Patreon identifiers

## UI binding

Once access exists, the hosted landing flow should become:

- anonymous visitor sees:
  - `Log in with Discord`
  - `Log in with Patreon`
- signed-in but denied visitor sees:
  - plain-English denial reason
  - what qualifies
  - link-the-other-provider affordance
- allowed visitor sees:
  - the normal hosted GUI and queue flow

The queue/progress UI itself should stay as close as possible to the current
runtime. The auth layer should gate entry, not rewrite the machine for fun.

## Config mapping

Repixelizer-specific auth config should not grow a separate provider jungle.

Keep:

- existing runtime/queue/limit env vars under `REPIXELIZER_*`

Add shared access config under `GC_ACCESS_*`.

Repixelizer-specific app-profile config can live under either:

- code constants, or
- a small profile/config surface such as:
  - `GC_ACCESS_REPIXELIZER_ALLOWED_DISCORD_ROLE_IDS=...`
  - `GC_ACCESS_REPIXELIZER_ALLOWED_PATREON_TIER_IDS=...`
  - `GC_ACCESS_REPIXELIZER_PUBLIC_QUEUE_SUMMARY=1`

The exact encoding can be decided later. The important thing is not cloning the
provider credentials into a second app-specific namespace.

## Recommended implementation order

1. Extract or embed the shared `gamecult_access` layer described in
   `docs/gamecult-hosted-access-architecture.md`.
2. Gate `POST /api/jobs` plus per-job read/cancel/event routes through local
   sessions and ownership checks.
3. Add login/link/denied-state UI around the existing hosted GUI shell.
4. Decide whether `/api/queue` stays public summary or moves behind `app_access`.
5. Only after a second hosted experiment exists, decide whether the shared
   access layer should leave embedded-package mode and become a shared service.

## Invariants

- Repixelizer should consume the shared access layer; it should not become the
  template that future experiments copy by hand.
- The queue remains owned by local session/account ids.
- Provider tokens stay server-side only.
- Capability checks belong at route boundaries and resource ownership seams, not
  scattered through queue internals.
- Auth should gate the hosted demo, not mutate the solver model or pretend to be
  the product.
