# Repixelizer on Idunn: cut map

**Ruling and defaults, 2026-09-30 (Self).** Operator: "Repixelizer is essential, it's our one actually shipped
SaaS despite its humble purpose." That is the ruling to give it an Idunn deploy path. Self has taken the map's
recommendations as defaults, and the operator may overrule any of them:
- Q1: merge the deployed codex branch into `main` and deploy `main`. This matches the operator's ruling for
  Ghostlight: "I'd rather be on main".
- Q2: port signed presence to `cultnet-py` in CultLib. That is a library capability any Python consumer would
  expect.
- Q3: use the pinned-interpreter layout. There is no Idunn change unless the torch probe fails.
- Q4: do not declare Heimdall as a dependency.
- Q5: flip when the old unit is idle, then soak for 48 h.

Status: cut map, no cuts landed. There is no separate target document. The
**Ends** section below holds the ends. The rest of this document holds the
means. Written 2026-09-30 by Imagination from source at repixelizer `main`
`9e8c8f6` and `origin/codex/repixelizer-eve-surface` `01dfab9`, gamecult-ops
`3d091c5`, Idunn `08d346d`, CultLib `42ba0f08`, plus read-only probes of
Yggdrasil (`ssh ygg`) and the public site. No `.cc` store was read. No env value
was printed; only key names were listed.

Operator, verbatim: "Repixelizer is essential, it's our one actually shipped
SaaS despite its humble purpose."

Rulings: none yet. See **Operator questions**.

Open: Q1 to Q5 below.

Follow-ups outside this migration:
- **StreamPixels exposes `/cultnet/snapshot` to the internet.** A public
  `POST https://streampixels.gamecult.org/cultnet/snapshot` with an empty body
  returned `400`, which means it reached the handler. The vhost's catch-all
  `location /` proxies it. Answering a real challenge makes the runtime sign and
  publish a fresh `active` presence to Odin
  (`CultLib packages/cultnet-ts/src/idunn-runtime-authority.ts:306`). The
  StreamPixels vhost should deny `/cultnet/`. This is PLAUSIBLE, not
  CONFIRMED: no signed request was sent. Cut 6 does not repeat the gap for
  Repixelizer.
- `Idunn docs/migration.md:283-293` describes Repixelizer as having "no CultLib
  dependency". Production has one: cultcache-py, cultmesh-py and cultnet-py via
  `repixelizer-cultlib.tar`. The deployed branch imports all three in
  `src/repixelizer/cultlib_support.py`. Cut 5 corrects that paragraph.

---

## Ends

1. `repixelizer.gamecult.org` is served by an Idunn-admitted generation: a
   sealed release built from an admitted ref, a signed runtime identity, a
   stable route, and two separate brakes. `idunn up repixelizer` is the only
   path that changes the running body.
2. No second lifecycle owner survives. The hand-written `repixelizer-gui.service`,
   the gamecult-ops deploy and check scripts, the Compose entry and the
   `trace-idunn-rudp.ps1` restart all go. So does the retired-generation
   `idunn.daemon_health` RUDP publication in the app.
3. The public product does not go dark during the move. The public hostname,
   TLS, path routing, upload and submit limits, SSE streaming, Heimdall login
   (Discord and Patreon), and the queue behaviour are unchanged from a user's
   point of view.
4. Secrets never reach a printable surface. Repixelizer holds no application
   secret today (see Current mechanism). Its only secret-file is the runtime
   presence identity. If a secret is added later, it goes in
   `[workload.secret_files]` and is never plain environment
   (`Idunn docs/secret-files-cut.md`).
5. Rollback after the public flip is one nginx reload for the whole soak
   window. After retirement, rollback goes through Idunn's retained release.

**Not a consumer, and not in scope:** Heimdall's own deployment (Heimdall stays
under its hand-written unit and is reached through its public URL), the Eve
native or Android lowerings, the solver, and any change to the Patreon or
Discord entitlement policy (Heimdall owns it,
`Heimdall/src/app-profiles.ts`).

---

## Current mechanism (verified 2026-09-30)

| Surface | Fact | Evidence |
|---|---|---|
| Host | Yggdrasil (hostname `yggdrasil-candidate`), Debian, `/usr/bin/python3` 3.13.5, 1.6 TB free on `/` | `ssh ygg` |
| Unit | `repixelizer-gui.service`, enabled, active since 2026-08-21 10:51 UTC, 93 MB RSS. Runs `/srv/repixelizer/.venv/bin/python /srv/repixelizer/app/scripts/run_gui.py --host 127.0.0.1 --port 8765` as `repixelizer`, with `EnvironmentFile=/srv/repixelizer/env/service.env` | `systemctl status`. The unit is byte-identical to `gamecult-ops/systemd/repixelizer-gui.service` |
| Running code | **`01dfab9`, on `codex/repixelizer-eve-surface`, not `main`.** Deployed 2026-06-16 by a workstation tarball (`repoRoot=E:\Projects\repixelizer`). CultLib is at `d595f5a5` as a separate tarball | `/srv/repixelizer/deployment-manifest.txt` |
| Branch drift | `main` and the deployed branch diverged at `1e1a344`. `main` has 1 commit the branch lacks (`9e8c8f6`, dead-code removal). The branch has 4 commits `main` lacks: the Eve surface, `verse_state.py`, `cultlib_support.py`, and the msgpack dependency. That is 3,081 lines across 20 files | `git diff --stat main origin/codex/...` |
| Venv | `/srv/repixelizer/.venv`, 1007 MB (CPU torch). Built on the host from an unpinned `pip install` | `ops scripts/deploy-repixelizer-gui.sh:113-116` |
| State | `/srv/repixelizer/cultcache/repixelizer.service.cc` (38 KB, rewritten continuously). It holds a queue, auth projection, Eve surface and `idunn.daemon_health` witness, and it is rebuildable. There are also 4 orphan `.tmp` files from 2026-08-20. The upload spool is at `/srv/repixelizer/spool`. Job queue state is in-process only | `ls`. Branch file `verse_state.py:374-395, 440-560` |
| Env keys | `REPIXELIZER_*` limits, `REPIXELIZER_SPOOL_DIR`, `GC_ACCESS_MODE/REQUIRED/PROTECT_QUEUE/HEIMDALL_BASE_URL/APP_PUBLIC_BASE_URL/ALLOWED_PROVIDERS`, `REPIXELIZER_ACCESS_DISCORD_GUILD_ID`, `..._DISCORD_ALLOWED_ROLE_IDS`, `REPIXELIZER_ACCESS_PATREON_TIER_TITLE`, `GC_ACCESS_CULTCACHE_PATH`, `GC_ACCESS_IDUNN_DAEMON`, `GC_ACCESS_IDUNN_HEALTH_CONTRACT`. There are 3 backup copies (`service.env.before-*`) | `cut -d= -f1`, key names only |
| Secrets | **None.** Auth verifies Heimdall JWTs against the public JWKS and refreshes through `/v1/apps/repixelizer/sessions/refresh` with the user's refresh cookie. No shared app secret is read. Payment is Patreon tier membership, which Heimdall resolves as an entitlement. Repixelizer holds no payment credential | `src/repixelizer/access.py:49-54, 487-575, 1030-1033`. Grep for `secret` finds nothing |
| Proxy | `/etc/nginx/sites-enabled/repixelizer.gamecult.org.conf` is byte-identical to `gamecult-ops/nginx/repixelizer.gamecult.org.conf`. TLS comes from Let's Encrypt (`certbot.timer` is active). It rate-limits `POST /api/jobs` to 6/min with a burst of 4, sets `client_max_body_size 5m`, and sets SSE `proxy_buffering off; proxy_read_timeout 3600s`. It proxies `/` and everything else to `127.0.0.1:8765` | `diff`, IDENTICAL |
| Public | `/` 200, `/app/` 303 (to the login), `/api/health` 200, `/api/config` 200 (heimdall mode, discord and patreon providers), `/api/queue` 401 | `curl` from the workstation |
| Traffic | The last 30 days of journal show **zero `/api/jobs` requests**, 12 `POST /api/auth/heimdall/start` (2026-09-17), and a few session checks. Most requests are scanners probing `/.env`. The journal reaches back at least to the 2026-08-21 restart | `journalctl -u repixelizer-gui --since -30d` |
| Idunn today | `idunn-yggdrasil.service` is active. Bindings exist for odin, ghostlight, heimdall, raven-muninn, and streampixels-service and -web. StreamPixels web and service are admitted, and their vhost already points at stable routes `8832` and `8833`. Odin continuity shows repeated failed restarts ("workload identity changed during native observation"). There is **no repixelizer binding** | `idunn status`, `ls /etc/gamecult/idunn/bindings` |
| Legacy actuator | No repixelizer arm in sudoers and no `/srv/odin/deploy-manifests/repixelizer`. Both were already retired. `deploy-repixelizer-gui.sh` refuses to run without `IDUNN_COMMAND_AUTHORITY=idunn-daemon`, so it is an actuator body with no caller | `sudoers.d` grep. `ops scripts/deploy-repixelizer-gui.sh:9-12`. `ops scripts/idunn/idunn-deployment-targets.ps1:196-208` (`Status = "blocked"`) |
| Retired health path | The app still publishes `idunn.daemon_health` over RUDP when `GC_ACCESS_IDUNN_RUDP_HEALTH` is set. That key is **absent** on the host, so the path is dormant. It belongs to the retired generation: the current `idunn serve` has no health ingress (`Idunn docs/migration.md:127-130`) | Branch file `verse_state.py:23-26, 276-305, 391-392, 638-697` |

The current Idunn contract has five requirements. They come from the admitted
StreamPixels and Ghostlight targets and from `Idunn/src`.

- **Recipe** (`gamecult.idunn.target_declaration.v1`) goes in the repo:
  steps, artifacts, `[service]`, health contract, state slots, provides and
  dependencies. It names no host paths.
- **Binding** (`gamecult.idunn.operator_binding.v2`) goes in gamecult-ops as
  `idunn/yggdrasil/bindings/<target>.toml.in`, rendered to
  `/etc/gamecult/idunn/bindings/<target>.toml`. It holds the runner images
  pinned by digest, the workload roots, `[workload.secret_files]`,
  `[runtime_identity]`, `[route]`, two different brake stores, rollout and
  placement.
- **The workload runs as a `systemd-run` transient unit.** The settings are
  `DynamicUser`, `PrivatePIDs`, `PrivateTmp`, `ProtectSystem=strict` and a
  read-only release (`Idunn src/drivers.rs:3615-3700`).
- **The release is sealed.** Every artifact destination is a single filename
  (`src/deployment.rs:1837-1844`). The service executable must be a regular
  file at the release root (`src/drivers.rs:3294-3297`). Hardening sets every
  file to `0444` except artifacts flagged `executable`, which get `0555`
  (`src/drivers.rs:6653-6692`). Every symlink must be relative, with no `..`,
  and must stay inside the release (`src/drivers.rs:6694-6718`).
- **Health is signed runtime presence**, `gamecult.runtime_presence_health.v2`.
  The process reads Idunn's parent-only `OpenFile=` descriptors
  (`LISTEN_FDS`, `LISTEN_PID`, and PID 1 in its namespace). It publishes signed
  `warming` and then `active` to Odin over CultNet RUDP (`10.77.0.1:17871`). It
  answers Idunn's signed `POST /cultnet/snapshot` challenge on the stable route
  with an explicit `Content-Length`, because chunked responses are refused. The
  reference implementation is TypeScript only:
  `CultLib packages/cultnet-ts/src/idunn-runtime-authority.ts` (643 lines),
  `idunn-odin-presence-publisher.ts` (244) and `runtime-presence-health.ts`
  (261). **`cultnet-py` has none of it.**

---

## Authority map (end state)

- **Owner:** Idunn owns Repixelizer's release, incarnation, continuity and
  stable-route membership (`127.0.0.1:8834`). Repixelizer owns its job queue,
  its auth-attempt table and its rebuildable Verse witness. Heimdall owns
  identity and entitlement. The operator owns the nginx vhost: TLS, hostname,
  path routing, rate limits and the `/cultnet/` deny.
- **Inputs:** the admitted ref of `GameCult/repixelizer`, the pinned
  `vendor/CultLib` gitlink, a pinned CPython external input, a hash-locked
  dependency lock, the binding, and the enrolled presence identity.
- **Outputs:** the stable HTTP route; signed presence to Odin; the
  `repixelizer.service.cc` witness in the state slot.
- **Derived state:** the witness is rebuildable and not authoritative. The job
  queue and auth attempts are in-process, so they are lost on restart, as they
  are today.
- **Forbidden writers:** `repixelizer-gui.service`, `scripts/run_gui.py` as a
  production entry, `deploy-repixelizer-gui.sh`, `check-repixelizer-gui.sh`,
  the Compose `repixelizer` service, `trace-idunn-rudp.ps1`, host `pip install`,
  workstation tarballs, and the `GC_ACCESS_IDUNN_*` / `idunn.daemon_health` RUDP
  publisher.
- **Shared paths:** first deploy, redeploy, continuity restart after a crash,
  and host reboot all go through `idunn up` or Idunn continuity. No path starts
  the process any other way.
- **Deletion line:** Cuts 3 and 5 delete before they add (listed per cut).
  Cut 8 removes the host unit and its files after the soak.

---

## Cuts

Order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Cut 2 is a CultLib foundation cut and
runs its own Eureka loop, with Soul passes and parity vectors. Cuts 1, 3 and 4
are repixelizer work. Cut 5 is gamecult-ops. Cuts 6 to 8 are host work on
Yggdrasil and need the operator's hands where noted. Heavy builds and tests run
on Yggdrasil (Idunn's runner, or `tools/stopgap/ygg-verify.sh`), not on
Starfire.

### Cut 0. Capture (read-only, no edits)

The table above is the capture, with one addition before any cut. On the
legacy venv, run the CLI on one small fixture from `tests/fixtures` and record
the output PNG's SHA-256, the torch version, and the wall time. Run it as the
`repixelizer` user, writing to `/tmp`, with no env file sourced. Cut 4's
acceptance step compares against this output. If the CPU torch version differs,
flag a mismatch rather than failing on it.

### Cut 1. Land the deployed branch on `main` (depends on Q1)

- **Repo/branch:** repixelizer `main` ← merge `origin/codex/repixelizer-eve-surface`
  (`01dfab9`). Merge base `1e1a344`. `main` adds `9e8c8f6`.
- **First:** run `pytest` on the branch head and on the merge result, on
  Yggdrasil.
- **Deletes first:** none. This cut only makes the ref honest
  (`Idunn docs/migration.md:137-145`, "Fix the ref before you carry it
  forward").
- **Verification:** the merge commit's tree contains `verse_state.py`,
  `eve_surface.py` and `cultlib_support.py`. `9e8c8f6`'s removals survive.
  Tests are green. After merging, delete the `codex/repixelizer-eve-surface`
  branch on origin so there is one ref.

### Cut 2. CultLib: Idunn runtime presence for Python (depends on Q2)

- **Repo/branch:** CultLib, a branch from `main` in its own worktree. It
  touches `packages/cultnet-py`, and possibly `packages/cultcache-py` for the
  private-store envelope reader.
- **Why here:** doctrine puts contracts in CultLib, and the operator's CultLib
  consumer rule applies: a Python service is a reasonable consumer. Without
  this cut Repixelizer cannot reach Warming, so no Python target on the swarm
  can be admitted.
- **Adds (smallest surface):** a port of the TS module set, with the same names
  in Python style:
  - load authority from the environment. That means `LISTEN_FDS`,
    `LISTEN_FDNAMES` and `LISTEN_PID` with the PID 1 namespace rule, read by
    fd number, not by reopening `/proc/self/fd`. It also means
    `GAMECULT_IDUNN_RUNTIME_BUNDLE`, `GAMECULT_IDUNN_CANDIDATE_BIND` and the
    optional `GAMECULT_IDUNN_PROCESS_WRITE_LEASE`;
  - Expected and activation readers (CultCache v1 snapshots);
  - the service-identity private-store reader. This is **not** the CultCache
    snapshot shape: it is a MessagePack list of one positional envelope, as
    described in `ops runbooks/streampixels-idunn-v2.md`, "The authority files
    use two distinct wire shapes";
  - the ed25519 dual signer (provider and activation). The signing messages and
    the 24-slot presence encoding follow `runtime-presence-health.ts:83-261`;
  - a presence publisher over the existing cultnet-py RUDP client;
  - a framework-free snapshot-challenge responder. It takes request bytes and
    returns `(status, content_type, body_bytes)`. The app mounts it.

  Use the `cryptography` package, which already arrives through `PyJWT[crypto]`,
  for ed25519. Add no new crypto dependency.
- **Sequencing hazard:** the cultnet-py RUDP ack cuts are landing now (see
  CultLib `c45edf58`, Python batch 2). Branch after they merge, or rebase onto
  them. Do not publish presence over an RUDP client that is mid-surgery.
- **Verification:**
  - parity: byte-identical presence encodings and signatures against
    `cultnet-rs/examples/runtime_presence_vector.rs` and the cultnet-ts tests,
    using fixed keys and fixed clocks;
  - private-store fixtures produced by `idunn-provision` or the Rust writer,
    not hand-built (see the scar in `Idunn docs/migration.md:417-446`);
  - a fake-descriptor test for the PID 1 rule and for the exact-PID rule in
    descendants;
  - negative test: a signer whose Expected names another target refuses to
    sign.
- **Soul:** attack parity first, then descriptor handling, then publisher
  retry and ordering. The StreamPixels attempt logged "runtime presence
  publisher sequence was reordered".

### Cut 3. Repixelizer: one production entry, no retired health, vendored CultLib

- **Repo/branch:** repixelizer, a branch from `main` after Cut 1. Depends on
  Cut 2 being merged in CultLib.
- **Deletes first:**
  - `verse_state.py`: the `idunn.daemon_health` document and record, the RUDP
    health publisher and every `GC_ACCESS_IDUNN_*` read. Line references are
    against `01dfab9`: `23-26`, `77-93`, `280-305` (the idunn fields),
    `363-365`, `384-392`, `526-560` (the transport-profile idunn fields),
    `631-634`, `638-706`. The transport and cut-line strings that describe the
    old generation go too.
  - `cultlib_support.py:19-38`: the sibling-checkout `sys.path` hack
    (`../CultLib/packages/...`, `E:\Projects\CultLib`). CultLib becomes an
    installed dependency from the gitlink.
  - The `build_health_payload()` fields that name Idunn daemon, contract and
    RUDP. `/api/health` returns app health only.
- **Adds:**
  - `vendor/CultLib` as a git submodule, pinned to the CultLib commit
    containing Cut 2 (same pattern as StreamPixels).
  - A hash-locked lock (`uv.lock`, or `requirements.lock` with `--hash`). It
    covers the app, the CPU torch wheel from `https://download.pytorch.org/whl/cpu`,
    msgpack, and the vendored `cultcache-py`, `cultmesh-py` and `cultnet-py`
    installed from `vendor/CultLib/packages/*`. Pin torch to the version
    recorded in Cut 0 unless there is a reason to move.
  - `src/repixelizer/idunn_serve.py`, the production entry (`python -m
    repixelizer.idunn_serve --state-root <binding>`). In order it: loads
    authority (Cut 2); publishes `warming`; waits for the write lease; builds
    the FastAPI app; mounts `POST /cultnet/snapshot` on the Cut 2 responder
    with an explicit `Content-Length` (FastAPI `Response(bytes)` sets it);
    binds uvicorn to `GAMECULT_IDUNN_CANDIDATE_BIND` with **one worker**, since
    queue state is in-process; waits for `/api/health`; publishes `active`; and
    sends a 10 s heartbeat that publishes `degraded` on a failing health probe.
    It mirrors `StreamPixels/deployment/idunn/runtime-presence.mjs`.
  - Verse witness path: `<state-root>/service/repixelizer.service.cc`. It is
    written only after the write lease is held.
  - The spool: remove the env override, so the default
    `tempfile.gettempdir()/repixelizer-gui-spool` lands in the unit's
    `PrivateTmp` (`gui.py:186-190`).
- **Keeps:** `scripts/run_gui.py` and `run_gui.ps1`, as dev-only local
  launchers. Their Windows port-killing logic is not a production path.
  `scripts/run_hosted_gui.ps1` is also dev-only.
- **Verification:**
  - tests: `idunn_serve` refuses to start without the authority descriptors,
    and is not usable as a dev launcher;
  - the snapshot route returns MessagePack with `Content-Length` and no
    `Transfer-Encoding`;
  - negative greps, none of which may match anywhere in `src/`:
    - `rg -n "GC_ACCESS_IDUNN|idunn\.daemon_health|IDUNN_HEALTH_RUDP" src`
    - `rg -n "parents\[2\]|E:\\\\\\\\Projects" src/repixelizer/cultlib_support.py`

### Cut 4. Repixelizer: the recipe and the Python artifact (depends on Q3)

- **Repo/branch:** repixelizer, same branch as Cut 3 or the next one.
- **First, a layout proof (probe, not code).** In the pinned runner image on
  Yggdrasil, unpack the chosen python-build-standalone `install_only` tarball.
  Run `readelf -d bin/python3.13`. If the RUNPATH is `$ORIGIN/../lib`, relocate
  it with `patchelf --set-rpath '$ORIGIN/lib'`. Copy the binary to
  `<release>/python` and the tree to `<release>/lib`, apply `0444` to
  everything except `python`, and run
  `./python -I -B -c "import sys, torch; print(sys.prefix)"`. `sys.prefix` must
  equal the release root, and imports must work from `0444` shared objects.
  Record the result in this map before writing the recipe. If the proof fails,
  stop: that is Q3's fork, not a reason to add a wrapper.
- **Adds `deployment/idunn/recipe.toml`:**
  - `target = "repixelizer"`, `source_stamp_environment = "REPIXELIZER_BUILD_COMMIT"`,
    `required_gitlinks = ["vendor/CultLib"]`.
  - `[[external_inputs]]` for the CPython tarball, URL plus `sha256`, on the
    runner `python-build`.
  - Steps:
    - prepare: unpack the interpreter into `release/`, patch its RUNPATH, and
      install the lock with `--require-hashes --no-deps` into
      `release/lib/python3.13/site-packages`, followed by the app and the three
      vendored CultLib packages;
    - test: `pytest -q` (scoped the way StreamPixels scopes it);
    - build: `compileall` over `release/lib`, because the release is read-only
      at runtime and `-B` stops `.pyc` writes;
    - acceptance: `release/python -I -B -m repixelizer.cli` on the Cut 0
      fixture. This proves a real conversion from the sealed layout before
      promotion.
  - `[[artifacts]]`: `python` (runner-output `release/python`,
    `executable = true`) and `lib` (runner-output `release/lib`).
  - `[service]`:
    - `executable_artifact = "python"`
    - `required_adjacent_artifacts = ["lib"]`
    - `arguments = ["-I", "-B", "-m", "repixelizer.idunn_serve", "--state-root", {binding state_root}]`
    - `transport = "http"`, `route_required = true`
    - `required_environment` names every `REPIXELIZER_*` and `GC_ACCESS_*` key
      the binding supplies, plus `GAMECULT_IDUNN_CANDIDATE_BIND`,
      `GAMECULT_IDUNN_RUNTIME_BUNDLE`, `GAMECULT_IDUNN_PROCESS_WRITE_LEASE` and
      `REPIXELIZER_ODIN_CULTMESH_RUDP`.
  - `[service.health] contract = "gamecult.runtime_presence_health.v2"`.
  - `[state]` with one slot: `id = "verse-witness"`,
    `relative_path = "service/repixelizer.service.cc"`, `kind = "cultcache-file"`,
    `writer = "process-bound-single-writer"`, `recovery = "rebuildable"`,
    `startup = "create-or-open-after-write-lease"`. Ghostlight's
    `mesh-projection` slot has the same shape.
  - `[[provides]] capability = "repixelizer.web"`, `schema = "http.v1"`.
  - `[[dependencies]]`: `odin.verse-rendezvous`, `shared-infrastructure`,
    `before-promotion`. **Heimdall is not declared** (see Q4).
- **Verification:**
  - builds: `idunn validate --recipe` in repixelizer CI, and on Yggdrasil
    `idunn validate --recipe ... --binding ...` once Cut 5's template is
    rendered;
  - release size under 2 GiB, and 2 retained releases fit;
  - negative: no symlink in `release/` is absolute or contains `..` (Idunn
    enforces this; test it in the build step so the failure is legible).

### Cut 5. gamecult-ops: binding template, runbook, deletions

- **Repo/branch:** gamecult-ops `main`. The worktree has **uncommitted edits to
  `scripts/idunn/idunn-deployment-targets.ps1` and `systemd/voidbot.service`**
  that belong to someone else. Stage explicit paths only, and coordinate the
  catalog edit with whoever owns those.
- **Deletes first:**
  - `scripts/deploy-repixelizer-gui.sh`
  - `scripts/check-repixelizer-gui.sh`
  - `repixelizer-service.env.example`
  - `compose/yggdrasil-apps.yaml:67-82` (the `repixelizer` service; Compose is
    not live on the host: no `/srv/compose/yggdrasil-apps.yaml`, no container)
  - `scripts/trace-idunn-rudp.ps1:11` (the Compose restart of repixelizer).
    Delete the whole script if repixelizer was its only subject.
  - `scripts/README.md:48-49`
  - `runbooks/repixelizer-demo-deploy.md`. Replace it with the new runbook
    below.
  - `systemd/repixelizer-gui.service`. The host copy stays until Cut 8. The
    repo copy is a second definition of a lifecycle owner the moment Idunn owns
    the target, and no path deploys it: its deploy script refuses without Idunn.
  - `nginx/repixelizer.gamecult.org.conf` is **kept and edited** (Cut 7). It is
    the operator-owned vhost.
- **Adds:**
  - `idunn/yggdrasil/bindings/repixelizer.toml.in`:
    - `target = "repixelizer"`, `profiles = ["full-gamecult"]`.
    - `[repository]`: origin `https://github.com/GameCult/repixelizer.git`,
      `refs/heads/main`, `minimum_revision = "PROVISIONED_REPIXELIZER_MINIMUM_REVISION"`,
      checkout `/var/lib/gamecult/idunn/sources/repixelizer`,
      `recipe_path = "deployment/idunn/recipe.toml"`.
    - `[repository.gitlinks."vendor/CultLib"]`.
    - `[runners.python-build]`:
      - `driver = "docker"`, an image pinned by digest (Hands picks a
        Debian-bookworm Python image; `uv` if Q3 uses it),
        `user = "65532:65532"`;
      - `affordances = ["source-read", "artifact-write", "build-cache"]`;
      - `cache_root = "/var/lib/gamecult/idunn/cache/repixelizer-python"`;
      - `allowed_programs` limited to what the steps call;
      - `network_profile = "bridge"`, 4096 MiB, 300% CPU.
    - `[workload]`:
      - `driver = "systemd-transient"`, `state_group = "repixelizer-state"`,
        `unit_prefix = "idunn-repixelizer"`;
      - `release_root = "/srv/repixelizer-idunn/releases"`,
        `state_root = "/var/lib/gamecult/repixelizer"`,
        `runtime_root = "/etc/gamecult/repixelizer/runtime"`;
      - `network = "host-private"`, `hardening = "strict"`;
      - `memory_mebibytes = 4096`, `cpu_quota_percent = 200`. The legacy unit
        has no cap. Record its peak in Cut 0 if a job can be driven; otherwise
        start at 4096.
    - `[workload.environment]` carries the current non-secret values. Their
      values are already committed in the old runbook. Discord guild and role
      IDs are also already in `ghostlight.toml.in`. The values:
      - `REPIXELIZER_HOSTED_DEMO`, the limits, `REPIXELIZER_PHASE_FIELD_PREVIEW_STRIDE`;
      - `GC_ACCESS_MODE=heimdall`, `GC_ACCESS_REQUIRED=1`, `GC_ACCESS_PROTECT_QUEUE=1`;
      - `GC_ACCESS_HEIMDALL_BASE_URL`, `GC_ACCESS_APP_PUBLIC_BASE_URL`,
        `GC_ACCESS_ALLOWED_PROVIDERS`;
      - `REPIXELIZER_ACCESS_DISCORD_GUILD_ID`,
        `REPIXELIZER_ACCESS_DISCORD_ALLOWED_ROLE_IDS`,
        `REPIXELIZER_ACCESS_PATREON_TIER_TITLE`;
      - `REPIXELIZER_ODIN_CULTMESH_RUDP = "10.77.0.1:17871"`.

      Before rendering, Hands **must diff the live key set** against this list.
      `GC_ACCESS_ALLOWED_PROVIDERS` in the old runbook says `discord`, but the
      live `/api/config` advertises discord and patreon. Carry the live value.
      It can be read on the host as a root-only operation that prints the
      single non-secret key.
    - `[workload.argument_bindings] state_root = "/var/lib/gamecult/repixelizer"`.
    - `[workload.secret_files] GAMECULT_RUNTIME_PRESENCE_IDENTITY =
      "/etc/gamecult/repixelizer/runtime-presence-identity.cc"`. That is the
      only entry. No other secret exists.
    - `[runtime_identity]`: `runtime_id = "repixelizer-yggdrasil"`,
      `expected_signer_identity_id = "PROVISIONED_REPIXELIZER_SIGNER_ID"`,
      `trust_anchor_store = "/etc/gamecult/trust/repixelizer-presence-anchor.cc"`.
    - `[route]`: `driver = "nginx-stream-tcp"`, `route_id = "repixelizer-http"`,
      `stable_endpoint = "http://127.0.0.1:8834"`, `private_host = "127.0.0.1"`,
      `private_port_start = 18861`, `private_port_end = 18869` (all free as of
      Cut 0), `config_path = "/etc/nginx/idunn-stream-routes/repixelizer.conf"`,
      `reload_unit = "nginx.service"`.
    - `[process_write_lease] record_path = "/etc/gamecult/repixelizer/runtime/process-write-lease.cc"`.
    - `[brakes]`: `.../repixelizer-deployment-brake.cc` and
      `.../repixelizer-lifecycle-brake.cc`.
    - `[rollout] strategy = "candidate-then-promote"`, `drain_seconds = 30`,
      `retain_releases = 2`.
    - `[placement] desired_replicas = 1`, `nodes = ["yggdrasil"]`.
  - `runbooks/repixelizer-idunn.md`. It follows the structure of
    `streampixels-idunn-v2.md`, but there is no credential-split section: only
    the identity is enrolled.
  - `scripts/idunn/idunn-deployment-targets.ps1:196-208`: change
    `yggdrasil-repixelizer` from `blocked` to Idunn-enforced, naming the
    binding. Tunnel: `scripts/start-yggdrasil-tunnel.ps1:20` and
    `runbooks/yggdrasil-ssh-tunnel.md:85` switch from `8765` to `8834`.
  - `inventory.md:862-873`: rewrite the Repixelizer host block.
  - Idunn `docs/migration.md:283-293`: replace the stale "no CultLib
    dependency" paragraph with a pointer to this map. That is an Idunn repo
    commit.
- **Verification:**
  - `rg -n -i "repixelizer-gui|deploy-repixelizer|check-repixelizer|8765" F:\Projects\gamecult-ops`
    may match only history docs (`docs/repo-census-2026-09/**`) and the vhost
    before Cut 7.

### Cut 6. Yggdrasil: preprovision, enrol, admit while legacy serves

Host work. The operator runs it, or Hands runs it under the operator's
deploy-instruction authorization. It stops on holds and wedged transactions.

1. Preprovision. The paths follow the `migration.md:448-478` table and the
   StreamPixels runbook shapes:
   - `groupadd --system repixelizer-state`
   - `/var/lib/gamecult/repixelizer`: `root:repixelizer-state 2770`
   - `/etc/gamecult/repixelizer`: `root:root 0711`
   - `/etc/gamecult/repixelizer/runtime`: `root:repixelizer-state 2750`
   - `/srv/repixelizer-idunn/releases`: `root:root 0755`
   - `/var/lib/gamecult/idunn/cache/repixelizer-python`:
     `65532:65532 0700`. Its parent must stay root-owned.
   - Add the three roots (runtime, state, release) to
     `idunn-yggdrasil.service` `ReadWritePaths` (`migration.md:474-478`). Run
     `daemon-reload`, then restart **Idunn only**. That is Idunn's own
     lifecycle, not Repixelizer's.
2. Enrol the identity. Run `idunn-provision enroll-provider-health-identity`,
   `export-provider-health-public-anchor` and `provider-health-identity-id`, as
   the StreamPixels runbook does. The private store is `root:root 0400`,
   `nlink` 1.
3. Render the template with the pushed `main` revision (Cuts 1, 3 and 4) and
   the signer ID. Install it at `/etc/gamecult/idunn/bindings/repixelizer.toml`,
   `root:root 0644`. Run `idunn validate --recipe ... --binding ...`.
4. Engage the deployment brake (`deployment-brake-engage`,
   `runtime-id repixelizer-yggdrasil`). Run `idunn up repixelizer --no-wait`.
   The legacy unit keeps serving on `8765` throughout. There is no port or
   state collision: the new state root, spool (`PrivateTmp`) and port are all
   distinct. Poll until the transaction waits at the brake. Release the brake
   for that exact runtime, release and transaction only, with a short expiry.
   **The legacy unit is not stopped here.** Repixelizer has no database and no
   exclusive external writer, unlike StreamPixels migration 013.
5. Poll until the transaction reaches terminal `Admitted`: signed `warming`,
   then `active`, then the route challenge passes, then promotion to `8834`.
   `Committing` is not admission. If it stalls in `Committing` or `Warming`
   with Odin continuity failing (as on 2026-09-28 for StreamPixels, and as the
   repeated Odin continuity failures in today's status suggest can happen),
   stop and route it as an Odin/Idunn defect. Do not widen trust or flip nginx.
6. Host probes against `127.0.0.1:8834`: `/` 200, `/api/health` 200,
   `/api/config` byte-equal to legacy's `/api/config` apart from any field that
   names the old generation, and `/app/` 303. Run a signed snapshot challenge
   three times in a row, and it must succeed each time.

### Cut 7. Flip the public route and verify externally

1. Edit the vhost in gamecult-ops and on the host (`nginx/repixelizer.gamecult.org.conf`):
   - `upstream repixelizer_gui { server 127.0.0.1:8834; }`
   - add `location ^~ /cultnet/ { return 404; }` so the signed challenge is
     loopback-only;
   - keep everything else, including the rate limit, body size, SSE settings
     and TLS lines certbot manages.
2. Check the legacy journal (last 2 minutes) for no `POST /api/jobs` and no
   open `/api/jobs/*/events` (Q5). Then run `nginx -t` and
   `systemctl reload nginx`.
3. External checks from the workstation:
   - `/`: 200;
   - `/app/`: 303 to login;
   - `/api/health` and `/api/config`: 200;
   - `/api/queue`: 401;
   - `POST /cultnet/snapshot`: 404;
   - an oversize upload is refused with 413 at nginx;
   - `POST /api/jobs` beyond the rate limit gets 503/429.
4. Check that a real conversion works. This needs a Heimdall-authenticated
   browser, so **the operator does it**: log in with Discord or Patreon, submit
   the Cut 0 fixture, watch live progress over SSE, and download the output.
   Compare its SHA-256 with Cut 0 and with the Cut 4 acceptance output. In
   parallel, confirm on the host that the job ran in the Idunn unit (its
   journal) and not in `repixelizer-gui` (whose journal stays quiet).
5. **Rollback (during the soak):** revert the upstream to `8765` and reload
   nginx. The legacy unit is still running, so that is the whole rollback.

### Cut 8. Retire the legacy body (after the soak, see Q5)

- `systemctl disable --now repixelizer-gui.service`, then remove
  `/etc/systemd/system/repixelizer-gui.service` and run `daemon-reload`.
- Archive `/srv/repixelizer` as a tarball under root's backup location. That
  covers the env files and their three `.before-*` backups, the source and
  CultLib tarballs, the venv, the old witness and its 4 orphan `.tmp` files,
  and the spool. Then remove the tree. The `repixelizer` system user is left in
  place unless the operator wants it gone. It owns nothing afterwards.
- Negative checks:
  - `systemctl list-unit-files | grep repixelizer-gui` is empty;
  - `ss -ltn '( sport = :8765 )'` is empty;
  - nothing under `/srv` references `repixelizer/.venv`;
  - `idunn status` shows `repixelizer` admitted;
  - kill the Idunn workload's main PID once. Idunn continuity restarts the
    same admitted release, and nothing else does.
- **Rollback after retirement:** an Idunn release decision on the retained
  previous release (`retain_releases = 2`). There is no legacy path.

---

## Operator questions

- **Q1. Which ref does Idunn admit?**
  - A: merge `codex/repixelizer-eve-surface` into `main` (Cut 1) and admit
    `main`.
  - B: admit the codex branch as it stands.
  - C: admit `main` as it stands, dropping the Eve surface and Verse witness
    that production runs today.

  **Recommended: A.** Production has run the branch for 3.5 months. C would
  regress the live product. B launders a branch pin into a typed contract
  (`migration.md:137-145`).

- **Q2. How does a Python workload prove signed presence?**
  - A: port the Idunn runtime authority and presence publisher into CultLib
    `cultnet-py` (Cut 2), with parity against Rust and TS.
  - B: make the release executable a small Node or Rust launcher that holds the
    identity, publishes presence, and supervises Python as a child.

  **Recommended: A.** Under B the signed identity attests a launcher, not the
  process serving traffic. The launcher would be PID 1 and hold the
  descriptors. It would publish `active` from a probe of its child, which is a
  compensator that relabels a proxy as the owner. A is a CultLib foundation
  cut of about 1,100 lines of TS to port, and it gives every future Python
  service the path.

- **Q3. What is a Python release?**
  - A: a pinned CPython (python-build-standalone as a sha-pinned external
    input), laid out as a root `python` executable plus a `lib/` tree, with
    RUNPATH patched and dependencies installed from a hash lock. This needs no
    Idunn change and depends on Cut 4's layout proof.
  - B: extend Idunn so a directory artifact can name an entry executable inside
    it, which would allow an unmodified interpreter tree. That is an Idunn
    contract change.
  - C: use the host's `/usr/bin/python3`. It is not sealed, and apt would
    change the release underneath Idunn. It cannot be expressed in the release
    model anyway.

  **Recommended: A**, with B recorded as the answer if the layout proof fails
  or a second interpreted target appears. A works in the recipe alone. B is
  the more general shape but spends Idunn surface on one target today.

- **Q4. Does the recipe declare Heimdall as a dependency?**
  - A: no. Heimdall stays an external HTTPS dependency, reached by public URL,
    as it is today.
  - B: yes. Declare `heimdall.*` `before-promotion`.

  **Recommended: A.** Heimdall is not Idunn-managed. Every `idunn up heimdall`
  in `idunn status` has failed, and the bindings README says Ghostlight's
  equivalent declaration cannot be satisfied. B would block Repixelizer on
  Heimdall's migration. Revisit when Heimdall is admitted.

- **Q5. Cutover window and soak length.**
  - A: flip at any time the legacy journal shows no job in flight. Soak for 48
    hours with legacy running but out of route, then Cut 8.
  - B: announce a maintenance window.

  **Recommended: A.** The flip is a graceful nginx reload. The journal shows
  zero job submissions in 30 days. The only seam is a job that straddles the
  flip: its SSE stream stays on the old worker, but browser heartbeats
  (`POST /api/jobs/{id}/heartbeat`, `gui.py:1272-1283`) would reach the new
  process, get a 404, and the legacy job would be cancelled as stale after
  30 s. An OAuth login started before the flip also fails its callback, because
  auth attempts are in-process (`access.py:482`), and the user retries. Neither
  justifies downtime. The check in step 2 of Cut 7 avoids both.

---

## Verification summary (what "done" means)

- `idunn status` shows `repixelizer` in terminal `Admitted`. Its unit is the
  only process listening for Repixelizer, and the old unit file is gone.
- Public: `/`, `/app/`, `/api/health`, `/api/config` and `/api/queue` return
  the same codes as Cut 0. `/cultnet/snapshot` returns 404 publicly.
- An operator-authenticated conversion of the Cut 0 fixture completes through
  the public site, with live SSE, and its output matches Cut 0 and the Cut 4
  acceptance hash (or a recorded torch-version difference explains it).
- A continuity restart (killing the workload PID) comes back as the same
  release, driven only by Idunn.
- Negative greps from Cuts 3 and 5 are clean. No `GC_ACCESS_IDUNN_*` key
  exists in the binding or the app.

## Subtraction ledger (estimate)

| Cut | Removed | Added | Targets/formats |
|---|---|---|---|
| 1 | 0 | merge only | none |
| 2 | 0 | ~900-1,200 lines of Python plus parity tests (CultLib) | a new public cultnet-py surface: Idunn runtime presence |
| 3 | ~150 (the idunn health path in `verse_state.py`, and `cultlib_support.py`'s path hack) | ~150 (`idunn_serve.py`), plus the lock and submodule | retires the `idunn.daemon_health` RUDP publication |
| 4 | 0 | ~90-line recipe | one Idunn target, one runner |
| 5 | ~330 (deploy and check scripts, env example, Compose entry, trace script line, old runbook, repo unit) | ~110 (binding template) plus a ~150-line runbook | one retired lifecycle definition set |
| 8 | host: 1 unit, a ~1 GB venv, tarballs | none | legacy body retired |

Net: the machine gains one CultLib capability that any Python service can use,
and loses a second lifecycle owner, a host-built unpinned venv, a
workstation-tarball deploy path, and a dead health generation.
