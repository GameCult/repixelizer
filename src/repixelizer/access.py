from __future__ import annotations

import contextlib
import contextvars
import os
from dataclasses import dataclass, field
from typing import Any, Iterator


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


def _parse_capabilities(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


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

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def to_public_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "required": self.required,
            "protectQueue": self.protect_queue,
            "appSlug": self.app_slug,
            "loginUrl": self.login_url,
            "logoutUrl": self.logout_url,
        }


class AccessDenied(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


@contextlib.contextmanager
def bind_access_subject(subject: AccessSubject) -> Iterator[None]:
    token = _CURRENT_ACCESS_SUBJECT.set(subject)
    try:
        yield
    finally:
        _CURRENT_ACCESS_SUBJECT.reset(token)


class AccessController:
    def __init__(self, runtime: AccessRuntimeConfig) -> None:
        self.runtime = runtime

    @classmethod
    def from_env(cls, *, hosted_demo: bool) -> "AccessController":
        mode = (_env_text("GC_ACCESS_MODE") or "off").strip().lower()
        if mode not in {"off", "trusted-header"}:
            mode = "off"
        required_default = hosted_demo if mode != "off" else False
        runtime = AccessRuntimeConfig(
            mode=mode,
            app_slug=_env_text("GC_ACCESS_APP_SLUG") or "repixelizer",
            required=_env_flag("GC_ACCESS_REQUIRED", required_default),
            protect_queue=_env_flag("GC_ACCESS_PROTECT_QUEUE", False),
            login_url=_env_text("GC_ACCESS_LOGIN_URL"),
            logout_url=_env_text("GC_ACCESS_LOGOUT_URL"),
        )
        return cls(runtime)

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
