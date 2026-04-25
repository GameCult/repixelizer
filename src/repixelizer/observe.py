from __future__ import annotations

from typing import Any, Protocol


class PipelineObserver(Protocol):
    def __call__(self, event: str, payload: dict[str, Any]) -> None: ...


class PipelineCancelled(RuntimeError):
    """Raised when a cooperative pipeline cancellation request is observed."""


def emit_observer(observer: PipelineObserver | None, event: str, /, **payload: Any) -> None:
    if observer is None:
        return
    observer(event, payload)


def observer_attribute(observer: PipelineObserver | None, name: str, default: object = None) -> object:
    if observer is None:
        return default
    direct = getattr(observer, name, None)
    if direct is not None:
        return direct
    owner = getattr(observer, "__self__", None)
    if owner is not None:
        owner_value = getattr(owner, name, None)
        if owner_value is not None:
            return owner_value
    return default


def check_observer_cancelled(observer: PipelineObserver | None) -> None:
    if observer is None:
        return
    checker = observer_attribute(observer, "check_cancelled", None)
    if callable(checker) and checker():
        message = observer_attribute(observer, "cancellation_message", "Pipeline canceled.")
        raise PipelineCancelled(str(message))
