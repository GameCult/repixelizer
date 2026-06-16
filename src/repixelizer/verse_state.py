from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

from .access import AccessController
from .cultlib_support import CultLibBindings, load_cultlib

if TYPE_CHECKING:
    from .gui import HostedDemoConfig


LOGGER = logging.getLogger(__name__)

CULTNET_RUDP_PROTOCOL_ID = "cultnet.transport.rudp.v0"
IDUNN_HEALTH_RUDP_CONNECTION_ID = 0x1D0D_0001
IDUNN_HEALTH_CHANNEL_ID = "idunn.daemon_health"
DEFAULT_IDUNN_HEALTH_CONTRACT = "repixelizer.cultnet-rudp-service-health"
DEFAULT_CULTCACHE_FILENAME = "repixelizer.service.cc"
PUBLISH_INTERVAL_SECONDS = 5.0
MAX_EVENT_RECORDS_PER_JOB = 32
MAX_EVENT_LIST_ITEMS = 12
MAX_EVENT_DICT_ITEMS = 12
IMAGEISH_KEYS = {
    "cleanedImage",
    "diagnostics",
    "displacementImage",
    "edgeMapImage",
    "heatmapImage",
    "inference",
    "latticeImage",
    "lossHistory",
    "outputImage",
    "palette",
    "runSummary",
    "samplingOverlayImage",
    "signalImage",
    "sourceImage",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@lru_cache(maxsize=1)
def _cultlib() -> CultLibBindings:
    return load_cultlib()


def _define_database_entry_type(*args: Any, **kwargs: Any) -> Any:
    return _cultlib().define_database_entry_type(*args, **kwargs)


def _normalize_public_base_url(value: str | None) -> str:
    if value and value.strip():
        return value.rstrip("/")
    return "https://repixelizer.gamecult.org"


def _default_cultcache_path(hosted_config: HostedDemoConfig) -> Path:
    return hosted_config.spool_dir.parent / "cultcache" / DEFAULT_CULTCACHE_FILENAME


def _health_document():
    return _define_database_entry_type(
        "idunn.daemon_health",
        [
            ("daemon_id", 0),
            ("state", 1),
            ("detail", 2),
            ("observed_at", 3),
            ("health_contract", 4),
            ("publication_source", 5, ""),
            ("transport", 6, ""),
        ],
        schema_id="idunn.daemon_health",
        schema_name="idunn.daemon_health",
        schema_version="idunn.daemon_health.v1",
    )


def _provider_advertisement_document():
    return _define_database_entry_type(
        "gamecult.eve.provider_advertisement",
        [("value", 0)],
        schema_id="gamecult.eve.provider_advertisement.v1",
        schema_name="gamecult.eve.provider_advertisement",
        schema_version="gamecult.eve.provider_advertisement.v1",
    )


def _eve_surface_document():
    return _define_database_entry_type(
        "gamecult.eve.surface_state",
        [
            ("provider_id", 0),
            ("title", 1),
            ("version", 2),
            ("updated_at", 3),
            ("surface", 4),
        ],
        schema_id="gamecult.eve.surface_state.v1",
        schema_name="gamecult.eve.surface_state",
        schema_version="gamecult.eve.surface_state.v1",
    )


def _runtime_config_document():
    return _define_database_entry_type(
        "repixelizer.runtime_config",
        [
            ("daemon_id", 0),
            ("hosted_demo", 1),
            ("queue_capacity", 2),
            ("show_queue_panel", 3),
            ("max_upload_bytes", 4),
            ("max_input_dimension", 5),
            ("max_output_dimension", 6),
            ("default_steps", 7),
            ("max_steps", 8),
            ("heartbeat_interval_seconds", 9),
            ("stale_after_seconds", 10),
            ("spool_path", 11),
            ("public_base_url", 12),
        ],
        schema_id="repixelizer.runtime_config",
        schema_name="repixelizer.runtime_config",
        schema_version="repixelizer.runtime_config.v0",
    )


def _auth_projection_document():
    return _define_database_entry_type(
        "repixelizer.auth_projection",
        [
            ("daemon_id", 0),
            ("enabled", 1),
            ("mode", 2),
            ("required", 3),
            ("protect_queue", 4),
            ("providers", 5),
            ("updated_at", 6),
        ],
        schema_id="repixelizer.auth_projection",
        schema_name="repixelizer.auth_projection",
        schema_version="repixelizer.auth_projection.v0",
    )


def _queue_snapshot_document():
    return _define_database_entry_type(
        "repixelizer.queue_snapshot",
        [
            ("daemon_id", 0),
            ("queue_depth", 1),
            ("waiting_count", 2),
            ("queue_capacity", 3),
            ("has_active_job", 4),
            ("active_status", 5, None),
            ("active_job_id", 6, None),
            ("queued_job_ids", 7),
            ("updated_at", 8),
        ],
        schema_id="repixelizer.queue_snapshot",
        schema_name="repixelizer.queue_snapshot",
        schema_version="repixelizer.queue_snapshot.v0",
    )


def _job_document():
    return _define_database_entry_type(
        "repixelizer.job",
        [
            ("job_id", 0),
            ("status", 1),
            ("filename", 2),
            ("created_at", 3),
            ("running_at", 4, None),
            ("last_heartbeat_at", 5),
            ("queue_position", 6, None),
            ("queue_depth", 7),
            ("waiting_count", 8),
            ("event_count", 9),
            ("error", 10, None),
            ("account_id", 11, None),
            ("session_id", 12, None),
            ("access_revision", 13, None),
            ("display_name", 14, None),
            ("auth_mode", 15, None),
            ("identity_provider", 16, None),
            ("target_size", 17, None),
            ("target_width", 18, None),
            ("target_height", 19, None),
            ("steps", 20, None),
            ("spool_path", 21),
            ("updated_at", 22),
        ],
        schema_id="repixelizer.job",
        schema_name="repixelizer.job",
        schema_version="repixelizer.job.v0",
    )


def _job_event_document():
    return _define_database_entry_type(
        "repixelizer.job_event",
        [
            ("job_id", 0),
            ("event_id", 1),
            ("event_name", 2),
            ("timestamp", 3),
            ("payload_summary", 4),
            ("updated_at", 5),
        ],
        schema_id="repixelizer.job_event",
        schema_name="repixelizer.job_event",
        schema_version="repixelizer.job_event.v0",
    )


def _command_boundary_document():
    return _define_database_entry_type(
        "repixelizer.command_boundary",
        [
            ("daemon_id", 0),
            ("boundary_id", 1),
            ("updated_at", 2),
            ("owner", 3),
            ("deploy_authority", 4),
            ("health_authority", 5),
            ("queue_authority", 6),
            ("compatibility_witnesses", 7),
            ("forbidden_writers", 8),
        ],
        schema_id="repixelizer.command_boundary.v1",
        schema_name="repixelizer.command_boundary",
        schema_version="repixelizer.command_boundary.v1",
    )


def _transport_profile_document():
    return _define_database_entry_type(
        "repixelizer.transport_profile",
        [
            ("daemon_id", 0),
            ("profile_id", 1),
            ("updated_at", 2),
            ("target_transport", 3),
            ("current_transport", 4),
            ("health_transport", 5),
            ("state_transport", 6),
            ("renderer_transport", 7),
            ("health_contract", 8),
            ("cut_line", 9),
        ],
        schema_id="repixelizer.transport_profile.v1",
        schema_name="repixelizer.transport_profile",
        schema_version="repixelizer.transport_profile.v1",
    )


@dataclass(frozen=True)
class RepixelizerVerseRuntimeConfig:
    enabled: bool
    daemon_id: str
    cultcache_path: Path
    idunn_rudp_health: str | None
    idunn_health_contract: str
    public_base_url: str
    workspace_root: Path

    @classmethod
    def from_env(cls, hosted_config: HostedDemoConfig) -> "RepixelizerVerseRuntimeConfig":
        hosted_default_daemon = "yggdrasil-repixelizer" if hosted_config.hosted_demo else "repixelizer"
        daemon_id = os.getenv("GC_ACCESS_IDUNN_DAEMON", hosted_default_daemon).strip() or hosted_default_daemon
        cultcache_path = Path(os.getenv("GC_ACCESS_CULTCACHE_PATH", str(_default_cultcache_path(hosted_config)))).expanduser()
        idunn_rudp_health = os.getenv("GC_ACCESS_IDUNN_RUDP_HEALTH")
        if idunn_rudp_health is not None:
            idunn_rudp_health = idunn_rudp_health.strip() or None
        idunn_health_contract = os.getenv("GC_ACCESS_IDUNN_HEALTH_CONTRACT", DEFAULT_IDUNN_HEALTH_CONTRACT).strip()
        enabled = (
            os.getenv("REPIXELIZER_VERSE_RUNTIME", "").strip().lower() in {"1", "true", "yes", "on"}
            or "GC_ACCESS_CULTCACHE_PATH" in os.environ
            or "GC_ACCESS_IDUNN_RUDP_HEALTH" in os.environ
            or "GC_ACCESS_IDUNN_DAEMON" in os.environ
        )
        return cls(
            enabled=enabled,
            daemon_id=daemon_id,
            cultcache_path=cultcache_path,
            idunn_rudp_health=idunn_rudp_health,
            idunn_health_contract=idunn_health_contract,
            public_base_url=_normalize_public_base_url(os.getenv("GC_ACCESS_APP_PUBLIC_BASE_URL")),
            workspace_root=_repo_root(),
        )


@dataclass(frozen=True)
class RepixelizerRuntimePulse:
    updated_at: str
    queue: dict[str, Any]
    jobs: list[dict[str, Any]]
    auth_payload: dict[str, Any]


class RepixelizerVerseRuntime:
    def __init__(
        self,
        runtime_config: RepixelizerVerseRuntimeConfig,
        hosted_config: HostedDemoConfig,
        access_controller: AccessController,
        manager: Any,
    ) -> None:
        self.runtime_config = runtime_config
        self.hosted_config = hosted_config
        self.access_controller = access_controller
        self.manager = manager
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._publish_lock = Lock()
        self._documents = self._build_documents() if runtime_config.enabled else ()
        self._document_by_type = {document.type: document for document in self._documents}

    @property
    def enabled(self) -> bool:
        return self.runtime_config.enabled

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.publish_once()
        self._stop_event.clear()
        self._thread = Thread(target=self._loop, name="repixelizer-verse-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def build_health_payload(self) -> dict[str, Any]:
        queue_summary = self.manager.get_queue_summary()
        auth_payload = self.access_controller.public_payload()
        return {
            "status": "ok",
            "verseRuntime": "enabled" if self.enabled else "disabled",
            "cultCachePath": str(self.runtime_config.cultcache_path) if self.enabled else None,
            "idunnDaemon": self.runtime_config.daemon_id,
            "idunnHealthContract": self.runtime_config.idunn_health_contract,
            "idunnRudpHealth": self.runtime_config.idunn_rudp_health,
            "queue": queue_summary,
            "auth": {
                "enabled": bool(auth_payload.get("enabled")),
                "mode": auth_payload.get("mode"),
                "required": bool(auth_payload.get("required")),
            },
        }

    def publish_once(self) -> None:
        if not self.enabled:
            return
        with self._publish_lock:
            pulse = self._build_pulse()
            final_path = self.runtime_config.cultcache_path
            temp_path = final_path.with_suffix(final_path.suffix + f".tmp-{os.getpid()}-{int(time.time() * 1000)}")
            try:
                cache = self._build_cache(temp_path)
                records = self._build_records(pulse)
                health_envelope = None
                for document, key, value in records:
                    cache.put(document, key, value)
                    if document.type == "idunn.daemon_health":
                        health_envelope = cache.get_required_envelope(document, key)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.replace(final_path)
                if health_envelope is not None and self.runtime_config.idunn_rudp_health:
                    self._publish_idunn_health(health_envelope.key, health_envelope.stored_at, health_envelope.payload)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    temp_path.unlink()

    def _loop(self) -> None:
        while not self._stop_event.wait(PUBLISH_INTERVAL_SECONDS):
            try:
                self.publish_once()
            except Exception:
                LOGGER.exception("Repixelizer verse runtime publish failed.")

    def _build_documents(self) -> tuple[Any, ...]:
        return (
            _provider_advertisement_document(),
            _eve_surface_document(),
            _runtime_config_document(),
            _auth_projection_document(),
            _queue_snapshot_document(),
            _job_document(),
            _job_event_document(),
            _command_boundary_document(),
            _transport_profile_document(),
            _health_document(),
        )

    def _build_cache(self, store_path: Path) -> Any:
        bindings = _cultlib()
        builder = bindings.CultCache.builder()
        for document in self._documents:
            builder.register_document_type(document)
        builder.add_generic_store(bindings.SingleFileMessagePackBackingStore(store_path))
        cache = builder.build()
        cache.pull_all_backing_stores()
        return cache

    def _document(self, document_type: str) -> Any:
        return self._document_by_type[document_type]

    def _build_pulse(self) -> RepixelizerRuntimePulse:
        snapshot = self.manager.snapshot_runtime_state(max_events_per_job=MAX_EVENT_RECORDS_PER_JOB)
        return RepixelizerRuntimePulse(
            updated_at=_utc_now_iso(),
            queue=snapshot["queue"],
            jobs=snapshot["jobs"],
            auth_payload=self.access_controller.public_payload(),
        )

    def _build_records(self, pulse: RepixelizerRuntimePulse) -> list[tuple[Any, str, Any]]:
        from .eve_surface import build_repixelizer_eve_surface

        queue = pulse.queue
        provider_advertisement = self._build_provider_advertisement(pulse)
        eve_surface = build_repixelizer_eve_surface(
            updated_at=pulse.updated_at,
            public_base_url=self.runtime_config.public_base_url,
            config=self.hosted_config,
            access_controller=self.access_controller,
            queue_summary={
                "queueDepth": queue["queueDepth"],
                "waitingCount": queue["waitingCount"],
                "queueCapacity": queue["queueCapacity"],
                "hasActiveJob": queue["hasActiveJob"],
                "activeStatus": queue["activeStatus"],
            },
        )
        records: list[tuple[Any, str, Any]] = [
            (self._document("gamecult.eve.provider_advertisement"), "repixelizer", {"value": provider_advertisement}),
            (
                self._document("gamecult.eve.surface_state"),
                "repixelizer",
                {
                    "provider_id": "repixelizer",
                    "title": "Repixelizer",
                    "version": int(eve_surface.get("version", 1)),
                    "updated_at": pulse.updated_at,
                    "surface": eve_surface,
                },
            ),
            (
                self._document("repixelizer.runtime_config"),
                self.runtime_config.daemon_id,
                {
                    "daemon_id": self.runtime_config.daemon_id,
                    "hosted_demo": self.hosted_config.hosted_demo,
                    "queue_capacity": self.hosted_config.queue_capacity,
                    "show_queue_panel": self.hosted_config.show_queue_panel,
                    "max_upload_bytes": self.hosted_config.max_upload_bytes,
                    "max_input_dimension": self.hosted_config.max_input_dimension,
                    "max_output_dimension": self.hosted_config.max_output_dimension,
                    "default_steps": self.hosted_config.default_steps,
                    "max_steps": self.hosted_config.max_steps,
                    "heartbeat_interval_seconds": self.hosted_config.heartbeat_interval_seconds,
                    "stale_after_seconds": self.hosted_config.stale_after_seconds,
                    "spool_path": str(self.hosted_config.spool_dir),
                    "public_base_url": self.runtime_config.public_base_url,
                },
            ),
            (
                self._document("repixelizer.auth_projection"),
                self.runtime_config.daemon_id,
                {
                    "daemon_id": self.runtime_config.daemon_id,
                    "enabled": bool(pulse.auth_payload.get("enabled")),
                    "mode": pulse.auth_payload.get("mode"),
                    "required": bool(pulse.auth_payload.get("required")),
                    "protect_queue": bool(pulse.auth_payload.get("protectQueue")),
                    "providers": pulse.auth_payload.get("providers", []),
                    "updated_at": pulse.updated_at,
                },
            ),
            (
                self._document("repixelizer.queue_snapshot"),
                self.runtime_config.daemon_id,
                {
                    "daemon_id": self.runtime_config.daemon_id,
                    "queue_depth": queue["queueDepth"],
                    "waiting_count": queue["waitingCount"],
                    "queue_capacity": queue["queueCapacity"],
                    "has_active_job": queue["hasActiveJob"],
                    "active_status": queue["activeStatus"],
                    "active_job_id": queue["activeJobId"],
                    "queued_job_ids": queue["queuedJobIds"],
                    "updated_at": pulse.updated_at,
                },
            ),
            (
                self._document("repixelizer.command_boundary"),
                "repixelizer",
                {
                    "daemon_id": self.runtime_config.daemon_id,
                    "boundary_id": "repixelizer",
                    "updated_at": pulse.updated_at,
                    "owner": "Repixelizer GUI runtime",
                    "deploy_authority": "idunn.yggdrasil-source-app.deploy",
                    "health_authority": "repixelizer.gui->idunn.daemon_health",
                    "queue_authority": "repixelizer.gui.GuiJobManager",
                    "compatibility_witnesses": [
                        "repixelizer-gui.service",
                        "GET /api/health",
                        "GET /api/config",
                        "nginx /app/ and /api/health host routing",
                    ],
                    "forbidden_writers": [
                        "Idunn and Odin may observe Repixelizer daemon truth but do not mutate queue ownership.",
                        "The browser app, nginx, and systemd do not own queue or daemon health truth.",
                    ],
                },
            ),
            (
                self._document("repixelizer.transport_profile"),
                "repixelizer",
                {
                    "daemon_id": self.runtime_config.daemon_id,
                    "profile_id": "repixelizer",
                    "updated_at": pulse.updated_at,
                    "target_transport": CULTNET_RUDP_PROTOCOL_ID,
                    "current_transport": self._current_transport(),
                    "health_transport": CULTNET_RUDP_PROTOCOL_ID if self.runtime_config.idunn_rudp_health else "cultcache-store",
                    "state_transport": "cultcache.store.v1",
                    "renderer_transport": "browser-http-lowering",
                    "health_contract": self.runtime_config.idunn_health_contract,
                    "cut_line": self._transport_cut_line(),
                },
            ),
            (
                self._document("idunn.daemon_health"),
                self.runtime_config.daemon_id,
                self._build_daemon_health_record(pulse),
            ),
        ]
        for job in pulse.jobs:
            records.append(
                (
                    self._document("repixelizer.job"),
                    job["jobId"],
                    {
                        "job_id": job["jobId"],
                        "status": job["status"],
                        "filename": job["filename"],
                        "created_at": job["createdAt"],
                        "running_at": job["runningAt"],
                        "last_heartbeat_at": job["lastHeartbeatAt"],
                        "queue_position": job["queuePosition"],
                        "queue_depth": job["queueDepth"],
                        "waiting_count": job["waitingCount"],
                        "event_count": job["eventCount"],
                        "error": job["error"],
                        "account_id": job["accountId"],
                        "session_id": job["sessionId"],
                        "access_revision": job["accessRevision"],
                        "display_name": job["displayName"],
                        "auth_mode": job["authMode"],
                        "identity_provider": job["identityProvider"],
                        "target_size": job["options"].get("target_size"),
                        "target_width": job["options"].get("target_width"),
                        "target_height": job["options"].get("target_height"),
                        "steps": job["options"].get("steps"),
                        "spool_path": job["spoolPath"],
                        "updated_at": pulse.updated_at,
                    },
                )
            )
            for event in job["events"]:
                records.append(
                    (
                        self._document("repixelizer.job_event"),
                        f"{job['jobId']}:{event['id']}",
                        {
                            "job_id": job["jobId"],
                            "event_id": event["id"],
                            "event_name": event["event"],
                            "timestamp": event["timestamp"],
                            "payload_summary": _summarize_event_payload(event["payload"]),
                            "updated_at": pulse.updated_at,
                        },
                    )
                )
        return records

    def _build_provider_advertisement(self, pulse: RepixelizerRuntimePulse) -> dict[str, Any]:
        from .witness import build_provider_advertisement

        witness_path = _display_witness_path(self.runtime_config.workspace_root, self.runtime_config.cultcache_path)
        advertisement = build_provider_advertisement(
            updated_at=pulse.updated_at,
            public_base_url=self.runtime_config.public_base_url,
            cc_witness_path=witness_path,
            config=self.hosted_config,
            access_controller=self.access_controller,
        )
        witness = dict(advertisement["witnesses"][0])
        witness["kind"] = "cc-export-path-live"
        witness["freshness"] = {"state": "fresh", "updatedAt": pulse.updated_at}
        advertisement["witnesses"] = [witness]
        advertisement["status"] = "daemon_live"
        advertisement["runtime"] = {
            **advertisement["runtime"],
            "cultCachePath": str(self.runtime_config.cultcache_path),
            "idunnDaemon": self.runtime_config.daemon_id,
            "idunnHealthContract": self.runtime_config.idunn_health_contract,
            "idunnRudpHealth": self.runtime_config.idunn_rudp_health,
            "healthTransport": CULTNET_RUDP_PROTOCOL_ID if self.runtime_config.idunn_rudp_health else "cultcache-store",
        }
        return advertisement

    def _build_daemon_health_record(self, pulse: RepixelizerRuntimePulse) -> dict[str, Any]:
        queue = pulse.queue
        auth_mode = pulse.auth_payload.get("mode") or "off"
        health_transport = "CultNet/RUDP" if self.runtime_config.idunn_rudp_health else "CultCache witness only"
        active = queue["activeStatus"] or "idle"
        detail = (
            f"Repixelizer GUI runtime active; hostedDemo={self.hosted_config.hosted_demo}; "
            f"queue={queue['queueDepth']}/{queue['queueCapacity']} waiting={queue['waitingCount']}; "
            f"active={active}; auth={auth_mode}; healthTransport={health_transport}"
        )
        return {
            "daemon_id": self.runtime_config.daemon_id,
            "state": "active",
            "detail": detail,
            "observed_at": pulse.updated_at,
            "health_contract": self.runtime_config.idunn_health_contract,
            "publication_source": "daemon-published" if self.runtime_config.idunn_rudp_health else "daemon-published-cultcache",
            "transport": CULTNET_RUDP_PROTOCOL_ID if self.runtime_config.idunn_rudp_health else "cultcache-store",
        }

    def _publish_idunn_health(self, key: str, stored_at: str, payload: bytes) -> None:
        endpoint = self.runtime_config.idunn_rudp_health
        if not endpoint:
            return
        bindings = _cultlib()
        client = bindings.CultMesh.create_rudp_client(
            runtime_id=self.runtime_config.daemon_id,
            connection_id=IDUNN_HEALTH_RUDP_CONNECTION_ID,
            endpoint=f"rudp://{endpoint}",
            bind_host="0.0.0.0",
        )
        try:
            client.socket.settimeout(0.2)
            client.connect()
            deadline = time.time() + 2.0
            while time.time() < deadline and not client.connected:
                client.poll_resends()
                client.receive_once()
                time.sleep(0.02)
            if not client.connected:
                raise RuntimeError(f"Idunn RUDP health connect timed out for {endpoint}")
            message = bindings.document_put_raw(
                message_id=f"{self.runtime_config.daemon_id}-{int(time.time() * 1000)}",
                key=key,
                schema_id="idunn.daemon_health",
                stored_at=stored_at,
                payload=payload,
                source_runtime_id=self.runtime_config.daemon_id,
            )
            client.send(IDUNN_HEALTH_CHANNEL_ID, message.to_bytes())
            settle_deadline = time.time() + 1.0
            while time.time() < settle_deadline:
                client.poll_resends()
                client.receive_once()
                time.sleep(0.02)
        finally:
            with contextlib.suppress(Exception):
                client.disconnect()
            client.close()

    def _current_transport(self) -> str:
        if self.runtime_config.idunn_rudp_health:
            return "daemon-owned-cultcache-witness + daemon-published-rudp-health + compatibility.ssh-systemd-http"
        return "daemon-owned-cultcache-witness + compatibility.ssh-systemd-http"

    def _transport_cut_line(self) -> str:
        if self.runtime_config.idunn_rudp_health:
            return "Repixelizer now publishes daemon-owned witness state and Idunn health from the live GUI runtime; systemd, /api/health, /api/config, and nginx routing are deployment/debug witnesses only once Odin and Idunn consume the typed surfaces."
        return "Repixelizer now publishes a daemon-owned witness store from the live GUI runtime; the remaining transport debt is Idunn health publication over CultNet/RUDP."


def _display_witness_path(workspace_root: Path, cultcache_path: Path) -> str:
    with contextlib.suppress(ValueError):
        return str(cultcache_path.relative_to(workspace_root)).replace("\\", "/")
    with contextlib.suppress(ValueError):
        return os.path.relpath(cultcache_path, workspace_root).replace("\\", "/")
    return str(cultcache_path)


def _summarize_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if key in IMAGEISH_KEYS or "Image" in key or key.endswith("Image"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
            continue
        if isinstance(value, list):
            scalar_values = [item for item in value[:MAX_EVENT_LIST_ITEMS] if isinstance(item, (str, int, float, bool)) or item is None]
            if scalar_values:
                summary[key] = scalar_values
            continue
        if isinstance(value, dict):
            compact: dict[str, Any] = {}
            for inner_key, inner_value in list(value.items())[:MAX_EVENT_DICT_ITEMS]:
                if inner_key in IMAGEISH_KEYS or "Image" in inner_key or inner_key.endswith("Image"):
                    continue
                if isinstance(inner_value, (str, int, float, bool)) or inner_value is None:
                    compact[inner_key] = inner_value
            if compact:
                summary[key] = compact
    return summary
