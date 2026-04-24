from __future__ import annotations

from typing import Any, Protocol


class PipelineObserver(Protocol):
    def __call__(self, event: str, payload: dict[str, Any]) -> None: ...


def emit_observer(observer: PipelineObserver | None, event: str, /, **payload: Any) -> None:
    if observer is None:
        return
    observer(event, payload)
