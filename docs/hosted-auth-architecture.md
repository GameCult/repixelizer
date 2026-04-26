# Repixelizer Hosted Auth Architecture

## Summary

The hosted Repixelizer demo should support both:

- Patreon-backed access for supporters who do not use Discord
- Discord role-backed access for supporters or community members who do

Access should be granted if **either** entitlement source says yes.

That means we do **not** build two separate protected apps, two separate queue paths, or two separate user models. We build one local account system with multiple linked identities and one access policy.

## Goals

- Let a user sign in with either Discord or Patreon
- Grant access if the user:
  - has one of the allowed Discord roles in the GameCult server, or
  - has an allowed Patreon entitlement
- Allow optional account linking so one human can attach both providers to one local account
- Keep queue ownership, job cancellation, and hosted limits tied to one local user/session model
- Avoid duplicated entitlement logic spread across unrelated login handlers

## Non-goals

- Turning Repixelizer into a general identity provider
- Requiring every Patreon supporter to join Discord
- Requiring every Discord member to have Patreon connected
- Complex billing state history or subscription analytics
- Multi-tenant auth machinery

## External providers

### Discord

Use Discord OAuth2 authorization code flow for login.

Minimum use:

- identify the user

Recommended access pattern:

- log the user in with Discord OAuth
- use the existing GameCult bot token server-side to inspect that user's guild membership and role ids

Why this path:

- role truth already lives in Discord
- the Patreon bot already manages supporter roles there
- role checks stay anchored to the same operational source of truth
- the web app does not need to ask the user for broader guild scopes unless we later decide it is necessary

If needed, we can still support the OAuth `guilds.members.read` path later. It is not the preferred first cut.

### Patreon

Use Patreon OAuth2 authorization code flow for login.

After login:

- fetch the logged-in user's identity and membership data
- inspect membership to the GameCult campaign
- inspect current entitled tiers or equivalent supporter status fields

This path exists specifically for the "I want access to Repixelizer and I do not care about Discord" crowd.

## Core model

### Local user

Repixelizer should have one internal user record that is independent of any provider.

Suggested shape:

- `users`
  - `id`
  - `created_at`
  - `last_seen_at`
  - `display_name`
  - `primary_email` nullable
  - `access_state` cached summary, optional

### Linked identities

Each local user may have zero or more linked external identities.

- `linked_identities`
  - `id`
  - `user_id`
  - `provider` enum: `discord`, `patreon`
  - `provider_user_id`
  - `username`
  - `access_token_encrypted`
  - `refresh_token_encrypted`
  - `token_expires_at`
  - `profile_json`
  - unique on `(provider, provider_user_id)`

Important invariant:

- one provider identity can belong to only one local user

### Entitlement snapshots

Do not make every page hit go out to Discord or Patreon. Cache the last evaluated entitlement state locally.

- `entitlement_snapshots`
  - `id`
  - `user_id`
  - `provider`
  - `evaluated_at`
  - `is_allowed`
  - `reason_code`
  - `reason_detail`
  - `raw_summary_json`

Examples:

- Discord provider says `is_allowed = true`, `reason_code = allowed_role`
- Patreon provider says `is_allowed = false`, `reason_code = no_active_membership`

### Sessions

Hosted GUI sessions should be local application sessions, not raw provider tokens sprayed into the browser.

- `sessions`
  - `id`
  - `user_id`
  - `created_at`
  - `last_seen_at`
  - `expires_at`
  - `access_granted`
  - `access_reason`

Store the session id in an HttpOnly secure cookie.

## Access policy

This is the heart of it.

Authentication answers:

- who is this user

Entitlement answers:

- should this user get through the gate

Those are different questions and should stay different in code.

### Provider-level entitlement checks

Implement two providers:

- `DiscordRoleEntitlementProvider`
- `PatreonEntitlementProvider`

Each provider returns something like:

```text
EntitlementResult {
  provider: "discord" | "patreon"
  allowed: boolean
  reason_code: string
  reason_detail: string
  checked_at: datetime
}
```

### Unified policy

Implement one access policy:

```text
access = discord_allowed || patreon_allowed
```

That is the only rule that should matter to the hosted demo.

Everything else is explanation and caching.

### Allowed sources

#### Discord allowed

Grant access if the user has at least one configured allowed role in the configured guild.

Examples:

- Patreon supporter role
- founder role
- moderator role
- special alpha-tester role

Keep the role list environment-configured, not hardcoded in some grimy branch.

#### Patreon allowed

Grant access if the user has at least one allowed Patreon entitlement.

Examples:

- any paid membership
- one or more specific tiers

Keep this configurable too.

## Login and linking flows

### Flow A: Login with Discord

1. User clicks `Log in with Discord`
2. Complete Discord OAuth
3. Find or create linked identity `(discord, discord_user_id)`
4. Attach to existing local user if already linked
5. Otherwise create a new local user and attach the identity
6. Evaluate entitlements
7. Create local session with access granted or denied

### Flow B: Login with Patreon

1. User clicks `Log in with Patreon`
2. Complete Patreon OAuth
3. Find or create linked identity `(patreon, patreon_user_id)`
4. Attach to existing local user if already linked
5. Otherwise create a new local user and attach the identity
6. Evaluate entitlements
7. Create local session with access granted or denied

### Flow C: Link second provider

Once logged in, the user may choose `Link Discord` or `Link Patreon`.

Rules:

- linking requires an authenticated local session
- linking a provider already attached to another local user is rejected
- after linking, re-run unified entitlement evaluation

This turns "Patreon-only supporter later joins Discord" and "Discord member later subscribes on Patreon" into boring account maintenance instead of support tickets from hell.

## Queue and job integration

The hosted queue should attach jobs to local sessions and local users, not directly to provider ids.

Suggested additions:

- `jobs.user_id`
- `jobs.session_id`
- `jobs.entitlement_checked_at`

Rules:

- only authenticated sessions with `access_granted = true` may enqueue jobs
- queue position and active job state are scoped to the local user session
- heartbeat and cancel behavior stay exactly as already designed

If a user loses entitlement mid-session:

- do not allow new jobs
- queued jobs may be canceled on next entitlement refresh
- active job policy can be either:
  - let current job finish, block future jobs
  - or cancel active job immediately

Recommended first cut:

- let current active job finish
- reject future submissions
- cancel queued-but-not-started jobs after entitlement loss

That is less spiteful and avoids turning transient API issues into ruined runs.

## Entitlement refresh strategy

Do not ask Discord and Patreon on every request like a lunatic.

Use a cached snapshot with refresh triggers.

Refresh on:

- successful login
- provider link
- explicit `refresh access` action
- first protected request after snapshot TTL expires

Recommended TTL:

- 10 to 15 minutes

### Discord refresh

Fetch guild membership and roles using the bot token and configured guild id.

If the user is not in the guild, the Discord entitlement is false.

### Patreon refresh

Refresh via Patreon API using the stored Patreon identity and tokens.

If the membership lookup fails because tokens are stale, attempt refresh.

If refresh fails, mark Patreon entitlement unknown or false depending on the failure mode.

Recommended behavior:

- provider outage should not instantly hard-lock already active users if a recent positive snapshot exists
- use a short grace window for transient provider failures

## Configuration

Suggested environment variables:

- `REPIXELIZER_AUTH_ENABLED=1`
- `REPIXELIZER_SESSION_SECRET=...`
- `REPIXELIZER_BASE_URL=https://repixelizer.gamecult.org`

- `REPIXELIZER_DISCORD_CLIENT_ID=...`
- `REPIXELIZER_DISCORD_CLIENT_SECRET=...`
- `REPIXELIZER_DISCORD_BOT_TOKEN=...`
- `REPIXELIZER_DISCORD_GUILD_ID=...`
- `REPIXELIZER_DISCORD_ALLOWED_ROLE_IDS=role1,role2,role3`

- `REPIXELIZER_PATREON_CLIENT_ID=...`
- `REPIXELIZER_PATREON_CLIENT_SECRET=...`
- `REPIXELIZER_PATREON_CAMPAIGN_ID=...`
- `REPIXELIZER_PATREON_ALLOWED_TIER_IDS=tier1,tier2`

- `REPIXELIZER_ENTITLEMENT_CACHE_TTL_SECONDS=900`
- `REPIXELIZER_PROVIDER_FAILURE_GRACE_SECONDS=3600`

## UI behavior

The hosted landing state should show:

- `Log in with Discord`
- `Log in with Patreon`

If access is denied after login:

- say why in plain English
- show what qualifies
- offer linking the other provider if the user is logged in locally

Examples:

- `Your Discord account is signed in, but you do not have one of the allowed GameCult roles.`
- `Your Patreon account is signed in, but no active Repixelizer-eligible membership was found.`
- `Already a supporter on the other platform? Link it here.`

Once allowed:

- the queue UI and hosted GUI behave normally

## Security notes

- Use authorization code flow for both providers
- Use `state` properly for both OAuth flows
- Store provider tokens server-side only
- Encrypt stored refresh/access tokens at rest
- Use secure, HttpOnly session cookies
- Keep local session invalidation separate from provider logout
- Audit log link/unlink, login, entitlement refresh, and submission denial events

## Recommended implementation order

### Phase 1: Local auth scaffolding

- add local user, linked identity, session, and entitlement snapshot models
- add session cookie handling
- add protected-session middleware/helpers

### Phase 2: Discord login and entitlement

- implement Discord OAuth login
- implement bot-backed guild role entitlement check
- gate the hosted queue behind Discord-based access

### Phase 3: Patreon login and entitlement

- implement Patreon OAuth login
- implement Patreon membership/tier entitlement check
- unify access policy across both providers

### Phase 4: Identity linking

- allow linking the second provider
- re-run entitlement evaluation on link/unlink

### Phase 5: Operational hardening

- add entitlement refresh TTLs and provider-failure grace behavior
- add UI for denied states and manual refresh
- add audit logs and admin diagnostics

## Invariants

Keep these sacred:

- login provider is not the same thing as entitlement provider
- the queue only knows local users and local sessions
- access is granted if either provider says yes
- provider-specific logic stays isolated behind entitlement providers
- linking identities must never merge two existing local users implicitly

If we violate those rules, the auth system will become a damp little pit of edge cases almost immediately.
