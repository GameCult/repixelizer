from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urljoin

import jwt


_CURRENT_ACCESS_SUBJECT: contextvars.ContextVar["AccessSubject | None"] = contextvars.ContextVar(
    "repixelizer_current_access_subject",
    default=None,
)

_PERMISSIVE_CAPABILITIES = frozenset(
    {
        "app_access",
        "queue_submit",
        "job_read_own",
        "job_cancel_own",
        "admin_access",
    }
)

_HEIMDALL_PROVIDER_LABELS = {
    "discord": "Discord",
    "patreon": "Patreon",
    "github": "GitHub",
    "twitch": "Twitch",
    "youtube": "YouTube",
}

_DEFAULT_REPIXELIZER_PROVIDERS = ("discord", "patreon")
_CLOCK_SKEW_SECONDS = 30
_AUTH_START_ENDPOINT_TEMPLATE = "/v1/oauth/{provider}/start"
_AUTH_CALLBACK_PATH = "/api/auth/heimdall/callback"
_AUTH_START_PATH = "/api/auth/heimdall/start"
_AUTH_ATTEMPT_PATH_TEMPLATE = "/api/auth/attempts/{attemptId}"
_AUTH_LOGOUT_PATH = "/api/auth/logout"


def _env_text(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _trim_trailing_slash(value: str | None) -> str | None:
    if value is None:
        return None
    return value.rstrip("/")


def _parse_capabilities(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _json_headers() -> dict[str, str]:
    return {
        "accept": "application/json",
        "content-type": "application/json",
    }


def _read_json_response(response) -> Any:
    raw = response.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _fetch_json(url: str, *, timeout_seconds: float) -> Any:
    request = urllib.request.Request(url, headers={"accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return _read_json_response(response)


def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> Any:
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, headers=_json_headers(), method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return _read_json_response(response)


@dataclass(frozen=True, slots=True)
class AccessProvider:
    slug: str
    label: str

    def to_public_json(self) -> dict[str, str]:
        return {
            "slug": self.slug,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class AccessSubject:
    account_id: str | None = None
    session_id: str | None = None
    access_revision: int | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    display_name: str | None = None
    auth_mode: str = "off"
    claims: dict[str, Any] = field(default_factory=dict)

    @property
    def is_authenticated(self) -> bool:
        return self.account_id is not None or self.session_id is not None

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def owns_job(self, *, account_id: str | None, session_id: str | None) -> bool:
        if account_id is not None and self.account_id is not None and account_id == self.account_id:
            return True
        if session_id is not None and self.session_id is not None and session_id == self.session_id:
            return True
        return False

    def to_public_json(self) -> dict[str, Any]:
        return {
            "authenticated": self.is_authenticated,
            "accountId": self.account_id,
            "sessionId": self.session_id,
            "accessRevision": self.access_revision,
            "displayName": self.display_name,
            "capabilities": sorted(self.capabilities),
            "authMode": self.auth_mode,
        }


@dataclass(frozen=True, slots=True)
class AccessRuntimeConfig:
    mode: str
    app_slug: str
    required: bool
    protect_queue: bool
    login_url: str | None
    logout_url: str | None
    session_cookie_name: str
    session_cookie_secure: bool
    session_cookie_samesite: str
    session_cookie_domain: str | None
    providers: tuple[AccessProvider, ...] = ()
    heimdall_base_url: str | None = None
    heimdall_issuer: str | None = None
    heimdall_jwks_url: str | None = None
    app_public_base_url: str | None = None
    start_endpoint: str | None = None
    auth_attempt_ttl_seconds: int = 600
    http_timeout_seconds: float = 10.0
    jwks_cache_seconds: int = 300
    discord_access_guild_id: str | None = None
    discord_access_role_ids: tuple[str, ...] = ()
    patreon_required_tier_title: str | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def callback_url(self) -> str | None:
        if self.app_public_base_url is None:
            return None
        return f"{self.app_public_base_url}{_AUTH_CALLBACK_PATH}"

    @property
    def default_return_to(self) -> str | None:
        if self.app_public_base_url is None:
            return None
        return f"{self.app_public_base_url}/app/"

    def to_public_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "mode": self.mode,
            "required": self.required,
            "protectQueue": self.protect_queue,
            "appSlug": self.app_slug,
            "loginUrl": self.login_url,
            "logoutUrl": self.logout_url,
        }
        if self.mode == "heimdall":
            payload.update(
                {
                    "providers": [provider.to_public_json() for provider in self.providers],
                    "startEndpoint": self.start_endpoint,
                    "attemptEndpointTemplate": _AUTH_ATTEMPT_PATH_TEMPLATE,
                }
            )
        return payload


@dataclass(slots=True)
class AuthAttempt:
    attempt_id: str
    provider: str
    return_to: str
    created_at: float
    expires_at: float
    status: str = "pending"
    authorization_url: str | None = None
    state_expires_at: str | None = None
    error: str | None = None
    error_description: str | None = None
    access_token: str | None = None
    subject: AccessSubject | None = None

    def sync_status(self, *, now: float) -> None:
        if self.status not in {"pending", "succeeded"}:
            return
        if now >= self.expires_at:
            self.status = "expired"
            self.error = "auth_attempt_expired"
            self.error_description = "This sign-in attempt expired before the local session was adopted."
            self.access_token = None
            self.subject = None

    def to_public_json(self, *, now: float) -> dict[str, Any]:
        self.sync_status(now=now)
        payload: dict[str, Any] = {
            "attemptId": self.attempt_id,
            "provider": self.provider,
            "status": self.status,
        }
        if self.status == "pending":
            payload["returnTo"] = self.return_to
        if self.status == "succeeded" and self.subject is not None:
            payload["subject"] = self.subject.to_public_json()
            payload["returnTo"] = self.return_to
        if self.status in {"failed", "expired"}:
            payload["error"] = self.error
            payload["errorDescription"] = self.error_description
        return payload


@dataclass(frozen=True, slots=True)
class AdoptedAuthSession:
    access_token: str
    subject: AccessSubject
    return_to: str
    expires_at: float

    def to_public_json(self) -> dict[str, Any]:
        return {
            "status": "authenticated",
            "returnTo": self.return_to,
            "subject": self.subject.to_public_json(),
        }


class AccessDenied(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


class AccessOperationError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


class HeimdallVerifier:
    def __init__(
        self,
        *,
        issuer: str,
        app_slug: str,
        jwks_url: str,
        timeout_seconds: float,
        cache_ttl_seconds: int,
    ) -> None:
        self.issuer = issuer
        self.app_slug = app_slug
        self.jwks_url = jwks_url
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = max(1, cache_ttl_seconds)
        self._lock = threading.Lock()
        self._cached_keys: dict[str, Any] = {}
        self._cache_expires_at = 0.0

    def verify(self, token: str, *, now: float | None = None) -> dict[str, Any]:
        if not token:
            raise AccessOperationError(400, "Heimdall callback did not include an access token.")
        issued_at = time.time() if now is None else now

        try:
            header = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError as exc:
            raise AccessOperationError(400, "Heimdall callback carried a malformed access token.") from exc

        key_id = header.get("kid")
        algorithm = header.get("alg")
        if algorithm != "EdDSA" or not isinstance(key_id, str) or not key_id.strip():
            raise AccessOperationError(400, "Heimdall access token used an unsupported signing header.")

        key = self._get_key(key_id, now=issued_at)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["EdDSA"],
                audience=self.app_slug,
                issuer=self.issuer,
                options={
                    "require": [
                        "iss",
                        "aud",
                        "sub",
                        "sid",
                        "iat",
                        "nbf",
                        "exp",
                        "typ",
                        "account_id",
                        "access_revision",
                        "app",
                        "facts",
                        "capabilities",
                        "identities",
                    ],
                },
                leeway=_CLOCK_SKEW_SECONDS,
            )
        except jwt.InvalidTokenError as exc:
            raise AccessOperationError(400, f"Heimdall access token failed local verification: {exc}") from exc

        if claims.get("typ") != "heimdall_access":
            raise AccessOperationError(400, "Heimdall access token is not an access claim.")
        app_claim = claims.get("app")
        if not isinstance(app_claim, dict):
            raise AccessOperationError(400, "Heimdall access token did not include a valid app claim.")
        if app_claim.get("slug") != self.app_slug:
            raise AccessOperationError(400, "Heimdall access token targeted a different app.")
        if not isinstance(app_claim.get("profile_version"), str):
            raise AccessOperationError(400, "Heimdall access token is missing the app profile version.")
        for field_name in ("facts", "capabilities", "identities"):
            if not isinstance(claims.get(field_name), list):
                raise AccessOperationError(400, f"Heimdall access token field '{field_name}' is malformed.")
        if claims.get("display_name") is not None and not isinstance(claims.get("display_name"), str):
            raise AccessOperationError(400, "Heimdall access token has a malformed display name.")
        if not isinstance(claims.get("access_revision"), int):
            raise AccessOperationError(400, "Heimdall access token has a malformed access revision.")
        return claims

    def _get_key(self, key_id: str, *, now: float) -> Any:
        with self._lock:
            if not self._cached_keys or now >= self._cache_expires_at:
                self._refresh_keys(now=now)
            key = self._cached_keys.get(key_id)
            if key is not None:
                return key
            self._refresh_keys(now=now)
            key = self._cached_keys.get(key_id)
            if key is None:
                raise AccessOperationError(400, f"Heimdall access token used unknown signing key '{key_id}'.")
            return key

    def _refresh_keys(self, *, now: float) -> None:
        try:
            payload = _fetch_json(self.jwks_url, timeout_seconds=self.timeout_seconds)
        except urllib.error.HTTPError as exc:
            raise AccessOperationError(502, f"Failed to fetch Heimdall JWKS: HTTP {exc.code}.") from exc
        except OSError as exc:
            raise AccessOperationError(502, f"Failed to fetch Heimdall JWKS: {exc}.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("keys"), list):
            raise AccessOperationError(502, "Heimdall JWKS response was malformed.")

        key_map: dict[str, Any] = {}
        for entry in payload["keys"]:
            if not isinstance(entry, dict):
                continue
            key_id = entry.get("kid")
            if not isinstance(key_id, str) or not key_id.strip():
                continue
            try:
                key_map[key_id] = jwt.PyJWK.from_dict(entry).key
            except jwt.InvalidKeyError:
                continue
        if not key_map:
            raise AccessOperationError(502, "Heimdall JWKS did not expose any usable signing keys.")
        self._cached_keys = key_map
        self._cache_expires_at = now + float(self.cache_ttl_seconds)


@contextlib.contextmanager
def bind_access_subject(subject: AccessSubject) -> Iterator[None]:
    token = _CURRENT_ACCESS_SUBJECT.set(subject)
    try:
        yield
    finally:
        _CURRENT_ACCESS_SUBJECT.reset(token)


class AccessController:
    def __init__(self, runtime: AccessRuntimeConfig, verifier: HeimdallVerifier | None = None) -> None:
        self.runtime = runtime
        self._verifier = verifier
        self._attempts: dict[str, AuthAttempt] = {}
        self._attempt_lock = threading.Lock()

    @classmethod
    def from_env(cls, *, hosted_demo: bool) -> "AccessController":
        mode = (_env_text("GC_ACCESS_MODE") or "off").strip().lower()
        if mode not in {"off", "trusted-header", "heimdall"}:
            mode = "off"
        required_default = hosted_demo if mode != "off" else False
        app_slug = _env_text("GC_ACCESS_APP_SLUG") or "repixelizer"

        providers: tuple[AccessProvider, ...] = ()
        heimdall_base_url = None
        heimdall_issuer = None
        heimdall_jwks_url = None
        app_public_base_url = None
        start_endpoint = None
        discord_access_guild_id = None
        discord_access_role_ids: tuple[str, ...] = ()
        patreon_required_tier_title = None
        login_url = _env_text("GC_ACCESS_LOGIN_URL")
        logout_url = _env_text("GC_ACCESS_LOGOUT_URL")
        verifier: HeimdallVerifier | None = None
        timeout_seconds = _env_float("GC_ACCESS_HTTP_TIMEOUT_SECONDS", 10.0)
        attempt_ttl_seconds = max(60, _env_int("GC_ACCESS_AUTH_ATTEMPT_TTL_SECONDS", 900))
        jwks_cache_seconds = max(60, _env_int("GC_ACCESS_JWKS_CACHE_SECONDS", 300))
        session_cookie_name = _env_text("GC_ACCESS_SESSION_COOKIE_NAME") or "gc_access_token"
        session_cookie_samesite = (_env_text("GC_ACCESS_SESSION_COOKIE_SAMESITE") or "lax").lower()
        if session_cookie_samesite not in {"lax", "strict", "none"}:
            session_cookie_samesite = "lax"
        session_cookie_domain = _env_text("GC_ACCESS_SESSION_COOKIE_DOMAIN")
        session_cookie_secure = _env_flag("GC_ACCESS_SESSION_COOKIE_SECURE", True)

        if mode == "heimdall":
            heimdall_base_url = _trim_trailing_slash(_env_text("GC_ACCESS_HEIMDALL_BASE_URL"))
            if heimdall_base_url is None:
                raise RuntimeError("GC_ACCESS_HEIMDALL_BASE_URL is required when GC_ACCESS_MODE=heimdall.")
            heimdall_issuer = _trim_trailing_slash(_env_text("GC_ACCESS_HEIMDALL_ISSUER")) or heimdall_base_url
            heimdall_jwks_url = _env_text("GC_ACCESS_HEIMDALL_JWKS_URL") or f"{heimdall_base_url}/.well-known/jwks.json"
            app_public_base_url = _trim_trailing_slash(_env_text("GC_ACCESS_APP_PUBLIC_BASE_URL"))
            if app_public_base_url is None:
                raise RuntimeError("GC_ACCESS_APP_PUBLIC_BASE_URL is required when GC_ACCESS_MODE=heimdall.")
            session_cookie_secure = _env_flag(
                "GC_ACCESS_SESSION_COOKIE_SECURE",
                app_public_base_url.startswith("https://"),
            )
            configured_providers = _parse_csv(_env_text("GC_ACCESS_ALLOWED_PROVIDERS"))
            if not configured_providers and app_slug == "repixelizer":
                configured_providers = _DEFAULT_REPIXELIZER_PROVIDERS
            providers = tuple(
                AccessProvider(slug=provider, label=_HEIMDALL_PROVIDER_LABELS.get(provider, provider.title()))
                for provider in configured_providers
            )
            if not providers:
                raise RuntimeError("GC_ACCESS_ALLOWED_PROVIDERS must name at least one provider in Heimdall mode.")
            login_url = login_url or "/"
            logout_url = logout_url or _AUTH_LOGOUT_PATH
            start_endpoint = _AUTH_START_PATH
            discord_access_guild_id = _env_text("REPIXELIZER_ACCESS_DISCORD_GUILD_ID")
            discord_access_role_ids = tuple(_parse_csv(_env_text("REPIXELIZER_ACCESS_DISCORD_ALLOWED_ROLE_IDS")))
            patreon_required_tier_title = _env_text("REPIXELIZER_ACCESS_PATREON_TIER_TITLE") or "Inner Sanctum"
            verifier = HeimdallVerifier(
                issuer=heimdall_issuer,
                app_slug=app_slug,
                jwks_url=heimdall_jwks_url,
                timeout_seconds=timeout_seconds,
                cache_ttl_seconds=jwks_cache_seconds,
            )

        runtime = AccessRuntimeConfig(
            mode=mode,
            app_slug=app_slug,
            required=_env_flag("GC_ACCESS_REQUIRED", required_default),
            protect_queue=_env_flag("GC_ACCESS_PROTECT_QUEUE", False),
            login_url=login_url,
            logout_url=logout_url,
            session_cookie_name=session_cookie_name,
            session_cookie_secure=session_cookie_secure,
            session_cookie_samesite=session_cookie_samesite,
            session_cookie_domain=session_cookie_domain,
            providers=providers,
            heimdall_base_url=heimdall_base_url,
            heimdall_issuer=heimdall_issuer,
            heimdall_jwks_url=heimdall_jwks_url,
            app_public_base_url=app_public_base_url,
            start_endpoint=start_endpoint,
            auth_attempt_ttl_seconds=attempt_ttl_seconds,
            http_timeout_seconds=timeout_seconds,
            jwks_cache_seconds=jwks_cache_seconds,
            discord_access_guild_id=discord_access_guild_id,
            discord_access_role_ids=discord_access_role_ids,
            patreon_required_tier_title=patreon_required_tier_title,
        )
        return cls(runtime, verifier=verifier)

    def public_payload(self) -> dict[str, Any]:
        return self.runtime.to_public_json()

    def current_subject(self) -> AccessSubject:
        subject = _CURRENT_ACCESS_SUBJECT.get()
        if subject is not None:
            return subject
        if self.runtime.mode == "off":
            return AccessSubject(
                capabilities=_PERMISSIVE_CAPABILITIES,
                auth_mode=self.runtime.mode,
            )
        return AccessSubject(auth_mode=self.runtime.mode)

    def peek_request_subject(self, request: Any) -> AccessSubject:
        return self._subject_from_request(request, allow_anonymous=True)

    def require_request_capability(self, request: Any, capability: str) -> AccessSubject:
        subject = self._subject_from_request(request, allow_anonymous=not self.runtime.required)
        self._require_capability(subject, capability)
        return subject

    def require_current_capability(self, capability: str) -> AccessSubject:
        subject = self.current_subject()
        self._require_capability(subject, capability)
        return subject

    def require_current_job_access(self, job: Any, capability: str) -> AccessSubject:
        subject = self.require_current_capability(capability)
        self.require_subject_job_access(subject, job)
        return subject

    def require_subject_job_access(self, subject: AccessSubject, job: Any) -> None:
        if self.runtime.mode == "off":
            return
        account_id = getattr(job, "account_id", None)
        session_id = getattr(job, "session_id", None)
        if account_id is None and session_id is None:
            raise AccessDenied(403, "Job has no local ownership metadata to validate against.")
        if not subject.owns_job(account_id=account_id, session_id=session_id):
            raise AccessDenied(403, "That job belongs to a different local account or session.")

    def app_gate_redirect_url(self) -> str:
        return self.runtime.login_url or "/"

    def start_auth_attempt(self, provider: str) -> dict[str, Any]:
        self._require_heimdall_mode()
        provider_slug = provider.strip().lower()
        allowed_providers = {entry.slug for entry in self.runtime.providers}
        if provider_slug not in allowed_providers:
            raise AccessOperationError(400, f"Provider '{provider_slug}' is not enabled for this hosted demo.")

        attempt_id = uuid.uuid4().hex
        now = time.time()
        return_to = self.runtime.default_return_to
        callback_url = self.runtime.callback_url
        heimdall_base_url = self.runtime.heimdall_base_url
        if return_to is None or callback_url is None or heimdall_base_url is None:
            raise AccessOperationError(503, "Hosted Heimdall auth is misconfigured on this Repixelizer node.")

        attempt = AuthAttempt(
            attempt_id=attempt_id,
            provider=provider_slug,
            return_to=return_to,
            created_at=now,
            expires_at=now + float(self.runtime.auth_attempt_ttl_seconds),
        )
        with self._attempt_lock:
            self._prune_attempts_locked(now=now)
            self._attempts[attempt_id] = attempt

        request_payload = {
            "appSlug": self.runtime.app_slug,
            "mode": "sign_in",
            "returnTo": return_to,
            "handoff": {
                "kind": "backend_callback",
                "attemptId": attempt_id,
                "callbackUrl": callback_url,
            },
        }
        if (
            provider_slug == "discord"
            and self.runtime.discord_access_guild_id
            and self.runtime.discord_access_role_ids
        ):
            request_payload["entitlementPolicy"] = {
                "kind": "discord_role_access",
                "guildId": self.runtime.discord_access_guild_id,
                "allowedRoleIds": list(self.runtime.discord_access_role_ids),
            }
        elif provider_slug == "patreon":
            request_payload["entitlementPolicy"] = {
                "kind": "patreon_membership_access",
                "requiredTierTitle": self.runtime.patreon_required_tier_title or "Inner Sanctum",
            }
        start_url = urljoin(f"{heimdall_base_url}/", _AUTH_START_ENDPOINT_TEMPLATE.format(provider=provider_slug).lstrip("/"))
        try:
            response_payload = _post_json(
                start_url,
                request_payload,
                timeout_seconds=self.runtime.http_timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            self._forget_attempt(attempt_id)
            detail = self._http_error_detail(exc)
            raise AccessOperationError(exc.code, detail) from exc
        except OSError as exc:
            self._forget_attempt(attempt_id)
            raise AccessOperationError(502, f"Failed to contact Heimdall: {exc}.") from exc

        if not isinstance(response_payload, dict):
            self._forget_attempt(attempt_id)
            raise AccessOperationError(502, "Heimdall start response was malformed.")
        authorization_url = response_payload.get("authorizationUrl")
        if not isinstance(authorization_url, str) or not authorization_url.strip():
            self._forget_attempt(attempt_id)
            raise AccessOperationError(502, "Heimdall start response did not include an authorization URL.")

        with self._attempt_lock:
            stored = self._attempts.get(attempt_id)
            if stored is not None:
                stored.authorization_url = authorization_url
                state_expires_at = response_payload.get("stateExpiresAt")
                if isinstance(state_expires_at, str):
                    stored.state_expires_at = state_expires_at

        return {
            "attemptId": attempt_id,
            "provider": provider_slug,
            "authorizationUrl": authorization_url,
            "returnTo": return_to,
            "statusEndpoint": _AUTH_ATTEMPT_PATH_TEMPLATE.format(attemptId=attempt_id),
        }

    def get_auth_attempt_status(self, attempt_id: str) -> dict[str, Any]:
        with self._attempt_lock:
            self._prune_attempts_locked(now=time.time())
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise AccessOperationError(404, "Unknown auth attempt.")
            return attempt.to_public_json(now=time.time())

    def adopt_auth_attempt(self, attempt_id: str) -> AdoptedAuthSession:
        with self._attempt_lock:
            self._prune_attempts_locked(now=time.time())
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise AccessOperationError(404, "Unknown auth attempt.")
            attempt.sync_status(now=time.time())
            if attempt.status == "pending":
                raise AccessOperationError(409, "That sign-in attempt has not finished yet.")
            if attempt.status in {"failed", "expired"}:
                raise AccessOperationError(409, attempt.error_description or "That sign-in attempt did not succeed.")
            if attempt.access_token is None or attempt.subject is None:
                raise AccessOperationError(500, "The sign-in attempt finished without an adoptable session.")
            claims = attempt.subject.claims
            expires_at = float(claims.get("exp", int(time.time())))
            return AdoptedAuthSession(
                access_token=attempt.access_token,
                subject=attempt.subject,
                return_to=attempt.return_to,
                expires_at=expires_at,
            )

    def receive_backend_handoff(self, payload: Any) -> None:
        self._require_heimdall_mode()
        if not isinstance(payload, dict):
            raise AccessOperationError(400, "Heimdall callback payload was not valid JSON.")

        attempt_id = payload.get("attemptId")
        provider = payload.get("provider")
        app_slug = payload.get("appSlug")
        status = payload.get("status")
        if payload.get("source") != "heimdall" or payload.get("kind") != "oauth_result":
            raise AccessOperationError(400, "Callback payload was not identified as a Heimdall oauth result.")
        if payload.get("handoffKind") != "backend_callback":
            raise AccessOperationError(400, "Callback payload used the wrong handoff kind.")
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise AccessOperationError(400, "Heimdall callback did not include an attempt id.")
        if not isinstance(provider, str) or not provider.strip():
            raise AccessOperationError(400, "Heimdall callback did not include a provider slug.")
        if app_slug != self.runtime.app_slug:
            raise AccessOperationError(400, "Heimdall callback targeted the wrong app slug.")
        if status not in {"success", "error"}:
            raise AccessOperationError(400, "Heimdall callback used an unknown status.")

        with self._attempt_lock:
            self._prune_attempts_locked(now=time.time())
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise AccessOperationError(404, "Unknown auth attempt.")
            if attempt.provider != provider:
                raise AccessOperationError(409, "Heimdall callback provider did not match the waiting auth attempt.")

        if status == "error":
            error_code = payload.get("error")
            error_description = payload.get("errorDescription")
            with self._attempt_lock:
                attempt = self._attempts.get(attempt_id)
                if attempt is not None:
                    attempt.status = "failed"
                    attempt.error = error_code if isinstance(error_code, str) else "oauth_callback_failed"
                    attempt.error_description = (
                        error_description
                        if isinstance(error_description, str) and error_description.strip()
                        else "Heimdall reported an upstream auth failure."
                    )
                    attempt.access_token = None
                    attempt.subject = None
            return

        access_token = payload.get("accessToken")
        if not isinstance(access_token, str) or not access_token.strip():
            raise AccessOperationError(400, "Heimdall callback succeeded without an access token.")
        claims = self._verify_access_token(access_token)
        subject = self._subject_from_claims(claims)
        if "app_access" not in subject.capabilities:
            with self._attempt_lock:
                attempt = self._attempts.get(attempt_id)
                if attempt is not None:
                    attempt.status = "failed"
                    attempt.error = "access_denied"
                    attempt.error_description = (
                        "Heimdall authenticated the account, but it did not grant Repixelizer app access."
                    )
                    attempt.access_token = None
                    attempt.subject = None
            return

        session_payload = payload.get("session")
        if not isinstance(session_payload, dict):
            raise AccessOperationError(400, "Heimdall callback did not include session metadata.")
        session_id = session_payload.get("sessionId")
        account_id = session_payload.get("accountId")
        access_revision = session_payload.get("accessRevision")
        if session_id != subject.session_id or account_id != subject.account_id or access_revision != subject.access_revision:
            raise AccessOperationError(400, "Heimdall callback session metadata did not match the access token.")

        expires_at = float(claims["exp"])
        with self._attempt_lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is not None:
                attempt.status = "succeeded"
                attempt.error = None
                attempt.error_description = None
                attempt.access_token = access_token
                attempt.subject = subject
                attempt.expires_at = expires_at

    def attach_session_cookie(self, response: Any, session: AdoptedAuthSession) -> None:
        max_age = max(0, int(session.expires_at - time.time()))
        response.set_cookie(
            key=self.runtime.session_cookie_name,
            value=session.access_token,
            max_age=max_age,
            httponly=True,
            secure=self.runtime.session_cookie_secure,
            samesite=self.runtime.session_cookie_samesite,
            path="/",
            domain=self.runtime.session_cookie_domain,
        )

    def clear_session_cookie(self, response: Any) -> None:
        response.delete_cookie(
            key=self.runtime.session_cookie_name,
            path="/",
            domain=self.runtime.session_cookie_domain,
            secure=self.runtime.session_cookie_secure,
            httponly=True,
            samesite=self.runtime.session_cookie_samesite,
        )

    def _require_capability(self, subject: AccessSubject, capability: str) -> None:
        if self.runtime.mode == "off":
            return
        if not subject.is_authenticated and self.runtime.required:
            raise AccessDenied(401, "Sign-in required.")
        if not subject.has_capability(capability):
            raise AccessDenied(403, f"Missing required capability '{capability}'.")

    def _subject_from_request(self, request: Any, *, allow_anonymous: bool) -> AccessSubject:
        if self.runtime.mode == "off":
            return AccessSubject(
                capabilities=_PERMISSIVE_CAPABILITIES,
                auth_mode=self.runtime.mode,
            )
        if self.runtime.mode == "trusted-header":
            return self._trusted_header_subject(request, allow_anonymous=allow_anonymous)
        if self.runtime.mode == "heimdall":
            return self._heimdall_subject(request, allow_anonymous=allow_anonymous)
        return AccessSubject(auth_mode=self.runtime.mode)

    def _trusted_header_subject(self, request: Any, *, allow_anonymous: bool) -> AccessSubject:
        headers = request.headers
        account_id = headers.get("x-gc-account-id")
        session_id = headers.get("x-gc-session-id")
        access_revision_raw = headers.get("x-gc-access-revision")
        display_name = headers.get("x-gc-display-name")
        capabilities = _parse_capabilities(headers.get("x-gc-capabilities"))
        access_revision: int | None = None
        if access_revision_raw is not None:
            try:
                access_revision = int(access_revision_raw)
            except ValueError:
                access_revision = None
        if account_id is None and session_id is None:
            if allow_anonymous:
                return AccessSubject(auth_mode=self.runtime.mode)
            raise AccessDenied(401, "Sign-in required.")
        return AccessSubject(
            account_id=account_id,
            session_id=session_id,
            access_revision=access_revision,
            capabilities=capabilities,
            display_name=display_name,
            auth_mode=self.runtime.mode,
        )

    def _heimdall_subject(self, request: Any, *, allow_anonymous: bool) -> AccessSubject:
        token = request.cookies.get(self.runtime.session_cookie_name)
        if token is None or not token.strip():
            if allow_anonymous:
                return AccessSubject(auth_mode=self.runtime.mode)
            raise AccessDenied(401, "Sign-in required.")
        try:
            claims = self._verify_access_token(token)
        except AccessOperationError:
            if allow_anonymous:
                return AccessSubject(auth_mode=self.runtime.mode)
            raise AccessDenied(401, "Sign-in required.") from None
        return self._subject_from_claims(claims)

    def _subject_from_claims(self, claims: dict[str, Any]) -> AccessSubject:
        return AccessSubject(
            account_id=str(claims["account_id"]),
            session_id=str(claims["sid"]),
            access_revision=int(claims["access_revision"]),
            capabilities=frozenset(str(entry) for entry in claims.get("capabilities", [])),
            display_name=str(claims["display_name"]) if claims.get("display_name") is not None else None,
            auth_mode="heimdall",
            claims=claims,
        )

    def _verify_access_token(self, token: str) -> dict[str, Any]:
        if self._verifier is None:
            raise AccessOperationError(503, "Heimdall verification is not configured on this node.")
        return self._verifier.verify(token)

    def _require_heimdall_mode(self) -> None:
        if self.runtime.mode != "heimdall":
            raise AccessOperationError(404, "Heimdall auth is not enabled in this runtime mode.")

    def _forget_attempt(self, attempt_id: str) -> None:
        with self._attempt_lock:
            self._attempts.pop(attempt_id, None)

    def _prune_attempts_locked(self, *, now: float) -> None:
        stale_ids = []
        for attempt_id, attempt in self._attempts.items():
            attempt.sync_status(now=now)
            if attempt.status == "expired" and now - attempt.expires_at > 60.0:
                stale_ids.append(attempt_id)
        for attempt_id in stale_ids:
            self._attempts.pop(attempt_id, None)

    def _http_error_detail(self, exc: urllib.error.HTTPError) -> str:
        try:
            payload = _read_json_response(exc)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("errorDescription") or payload.get("message")
            if isinstance(detail, str) and detail.strip():
                return detail
        return f"Heimdall rejected the auth start request with HTTP {exc.code}."
