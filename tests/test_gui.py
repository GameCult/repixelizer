from __future__ import annotations

import importlib.util
import io
import asyncio
import base64
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.datastructures import UploadFile
from fastapi import HTTPException

import repixelizer.inference as inference_module
import repixelizer.pipeline as pipeline_module
from repixelizer.access import AccessSubject, bind_access_subject
from repixelizer.observe import PipelineCancelled
from repixelizer.pipeline import run_pipeline_rgba
from repixelizer.synthetic import fake_pixelize, make_emblem
from repixelizer.params import SolverHyperParams
from repixelizer.types import InferenceCandidate, InferenceResult, PhaseFieldSourceAnalysis, SolverArtifacts


async def _get_response(
    app,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> tuple[int, dict[str, str], bytes]:
    messages: list[dict[str, object]] = []
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    raw_headers = []
    if headers:
        raw_headers.extend(
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in headers.items()
        )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    status = int(start["status"])
    headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in start.get("headers", [])
    }
    return status, headers, body


def _png_bytes(*, width: int = 4, height: int = 4) -> bytes:
    image = Image.new("RGBA", (width, height), (12, 34, 56, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _heimdall_signing_material() -> tuple[Ed25519PrivateKey, dict[str, str], str]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    key_id = "test-ed25519-1"
    jwk = {
        "kty": "OKP",
        "crv": "Ed25519",
        "x": _b64url(public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)),
        "use": "sig",
        "alg": "EdDSA",
        "kid": key_id,
    }
    return private_key, jwk, key_id


def _heimdall_access_token(
    private_key: Ed25519PrivateKey,
    *,
    key_id: str,
    issuer: str = "https://heimdall.gamecult.org",
    app_slug: str = "repixelizer",
    account_id: str = "acct-1",
    session_id: str = "sess-1",
    access_revision: int = 4,
    display_name: str = "Meta",
    capabilities: list[str] | None = None,
    facts: list[str] | None = None,
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": issuer,
            "aud": app_slug,
            "sub": account_id,
            "sid": session_id,
            "jti": "jti-test-1",
            "iat": now,
            "nbf": now,
            "exp": now + 3600,
            "typ": "heimdall_access",
            "account_id": account_id,
            "access_revision": access_revision,
            "display_name": display_name,
            "app": {
                "slug": app_slug,
                "profile_version": "test-profile-v1",
            },
            "facts": facts or ["entitlement.app_access"],
            "capabilities": capabilities or ["app_access", "queue_submit", "job_read_own", "job_cancel_own"],
            "identities": [
                {
                    "provider": "discord",
                    "providerUserId": "123456789",
                    "username": "meta",
                }
            ],
        },
        private_key,
        algorithm="EdDSA",
        headers={"kid": key_id},
    )


def _route_endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"Missing route {method} {path}")


def _response_json(response) -> dict[str, object]:
    import json

    return json.loads(response.body.decode("utf-8"))


def test_run_pipeline_rgba_emits_observer_events_for_gui() -> None:
    source = make_emblem(20, 20)
    fake = fake_pixelize(source, upscale=8, blur_radius=0.45, seed=3)
    events: list[str] = []

    def observer(event: str, payload) -> None:
        del payload
        events.append(event)

    run_pipeline_rgba(
        fake,
        target_width=20,
        target_height=20,
        steps=2,
        device="cpu",
        enable_candidate_rerank=False,
        observer=observer,
    )

    assert events[0] == "source_loaded"
    assert events.count("stage_started") == 6
    assert events.index("inference_candidates_ready") < events.index("analysis_completed")
    assert events.index("analysis_completed") < events.index("candidate_selection_completed")
    assert events.index("candidate_selection_completed") < events.index("phase_field_prepared")
    assert "phase_field_prepared" in events
    assert "phase_field_initial" in events
    assert events.count("phase_field_step") == 2
    assert events[-4:] == ["cleanup_completed", "stage_started", "palette_completed", "pipeline_completed"]


def test_run_pipeline_rgba_respects_phase_field_preview_stride() -> None:
    source = make_emblem(12, 12)
    fake = fake_pixelize(source, upscale=6, blur_radius=0.35, seed=5)

    class PreviewObserver:
        phase_field_preview_stride = 2

        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def __call__(self, event: str, payload: dict[str, object]) -> None:
            self.events.append((event, payload))

    observer = PreviewObserver()

    run_pipeline_rgba(
        fake,
        target_width=12,
        target_height=12,
        steps=5,
        device="cpu",
        enable_candidate_rerank=False,
        observer=observer,
    )

    phase_steps = [payload["step"] for event, payload in observer.events if event == "phase_field_step"]
    assert phase_steps == [2, 4]
    final_payload = next(payload for event, payload in observer.events if event == "phase_field_final")
    assert final_payload["step"] == 5


def test_repo_gui_runner_dispatches_to_gui_main(monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_gui.py"
    spec = importlib.util.spec_from_file_location("repixelizer_run_gui_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    called = {}

    def fake_gui_main(*, host: str, port: int, reload: bool) -> int:
        called["host"] = host
        called["port"] = port
        called["reload"] = reload
        return 0

    monkeypatch.setattr(module.os, "getpid", lambda: 4242)
    monkeypatch.setattr(module.sys, "executable", r"E:\Projects\repixelizer\.venv\Scripts\python.exe")
    monkeypatch.setattr("repixelizer.gui.main", fake_gui_main)
    exit_code = module.main(["--host", "127.0.0.1", "--port", "8123", "--reload"])
    assert exit_code == 0
    assert called == {"host": "127.0.0.1", "port": 8123, "reload": True}
    output = capsys.readouterr().out
    assert "Repixelizer GUI" in output
    assert "PID: 4242" in output
    assert r"Python: E:\Projects\repixelizer\.venv\Scripts\python.exe" in output
    assert "Bind: http://127.0.0.1:8123" in output
    assert "Open: http://127.0.0.1:8123/app/" in output
    assert "Health: http://127.0.0.1:8123/api/health" in output
    assert "Reload: on" in output


def test_repo_gui_runner_queue_ui_override_sets_env(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_gui.py"
    spec = importlib.util.spec_from_file_location("repixelizer_run_gui_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.delenv("REPIXELIZER_SHOW_QUEUE_PANEL", raising=False)
    monkeypatch.setattr(module, "_reclaim_stale_gui_port", lambda host, port, repo_root: None)
    monkeypatch.setattr("repixelizer.gui.main", lambda **_: 0)

    exit_code = module.main(["--queue-ui", "show"])

    assert exit_code == 0
    assert os.environ["REPIXELIZER_SHOW_QUEUE_PANEL"] == "1"


def test_repo_gui_runner_reclaims_stale_gui_port_before_launch(monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_gui.py"
    spec = importlib.util.spec_from_file_location("repixelizer_run_gui_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    called = {}

    def fake_reclaim(host: str, port: int, repo_root: Path) -> str | None:
        called["reclaim"] = (host, port, repo_root)
        return "Reclaimed stale Repixelizer GUI process 4242 on port 8000."

    def fake_gui_main(*, host: str, port: int, reload: bool) -> int:
        called["gui"] = {"host": host, "port": port, "reload": reload}
        return 0

    monkeypatch.setattr(module.os, "getpid", lambda: 9001)
    monkeypatch.setattr(module, "_reclaim_stale_gui_port", fake_reclaim)
    monkeypatch.setattr("repixelizer.gui.main", fake_gui_main)

    exit_code = module.main(["--host", "127.0.0.1", "--port", "8000"])

    assert exit_code == 0
    assert called["gui"] == {"host": "127.0.0.1", "port": 8000, "reload": False}
    assert called["reclaim"] == ("127.0.0.1", 8000, repo_root)
    output = capsys.readouterr().out
    assert "Reclaimed stale Repixelizer GUI process 4242 on port 8000." in output
    assert "PID: 9001" in output
    assert "Open: http://127.0.0.1:8000/app/" in output


def test_repo_gui_runner_formats_browser_url_for_wildcard_host(monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_gui.py"
    spec = importlib.util.spec_from_file_location("repixelizer_run_gui_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module.os, "getpid", lambda: 77)
    monkeypatch.setattr(module, "_reclaim_stale_gui_port", lambda host, port, repo_root: None)
    monkeypatch.setattr("repixelizer.gui.main", lambda **_: 0)

    exit_code = module.main(["--host", "0.0.0.0", "--port", "8765"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Bind: http://0.0.0.0:8765" in output
    assert "Open: http://127.0.0.1:8765/app/" in output
    assert "Health: http://127.0.0.1:8765/api/health" in output


def test_repo_gui_runner_refuses_unrelated_port_owner(monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_gui.py"
    spec = importlib.util.spec_from_file_location("repixelizer_run_gui_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "_reclaim_stale_gui_port",
        lambda host, port, repo_root: (_ for _ in ()).throw(
            RuntimeError("Port 8000 is already in use by PID 9999. Refusing to kill an unrelated process.")
        ),
    )

    exit_code = module.main(["--port", "8000"])

    assert exit_code == 1
    assert "Refusing to kill an unrelated process" in capsys.readouterr().err


def test_create_job_upload_field_validates_without_forward_ref_errors() -> None:
    from repixelizer.gui import create_app

    image = Image.new("RGBA", (1, 1), (12, 34, 56, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    app = create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/api/jobs")
    image_field = next(field for field in route.dependant.body_params if field.name == "image")
    upload = UploadFile(filename="tiny.png", file=io.BytesIO(buffer.getvalue()))
    value, errors = image_field.validate(upload, {}, loc=("body", "image"))

    assert errors == []
    assert value.filename == "tiny.png"


def test_gui_static_assets_disable_browser_caching() -> None:
    from repixelizer.gui import create_app

    app = create_app()
    html_status, html_headers, html_body = asyncio.run(_get_response(app, "/app/"))
    js_status, js_headers, js_body = asyncio.run(_get_response(app, "/app/app.js"))
    css_status, _css_headers, css_body = asyncio.run(_get_response(app, "/app/styles.css"))

    assert html_status == 200
    assert js_status == 200
    assert css_status == 200
    assert html_headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert js_headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert html_headers["pragma"] == "no-cache"
    assert js_headers["pragma"] == "no-cache"
    html_text = html_body.decode("utf-8")
    js_text = js_body.decode("utf-8")
    css_text = css_body.decode("utf-8")
    assert "./styles.css?v=" in html_text
    assert "./app.js?v=" in html_text
    assert 'id="deviceField" class="field" hidden' in html_text
    assert 'id="stripBackgroundField" class="toggle-row ui-surface ui-frame" hidden' in html_text
    assert "skipRerankInput" not in html_text
    assert "Skip candidate rerank" not in html_text
    assert '["skip_candidate_rerank", "false"]' in js_text
    assert "skipRerankInput" not in js_text
    assert "[hidden]" in css_text
    assert "display: none !important" in css_text


def test_gui_hosted_root_serves_landing_page(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))

    app = create_app()
    status, headers, body = asyncio.run(_get_response(app, "/"))
    html = body.decode("utf-8")

    assert status == 200
    assert headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert "Repixelizer hosted demo" in html
    assert "Force fake pixel art back onto a real grid, then inspect each step with a full solver view." in html
    assert 'href="/app/"' in html
    assert 'href="/privacy"' in html
    assert 'href="/terms"' in html
    assert "Login with Discord" not in html


def test_gui_serves_public_legal_pages(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))

    app = create_app()
    privacy_status, privacy_headers, privacy_body = asyncio.run(_get_response(app, "/privacy"))
    terms_status, terms_headers, terms_body = asyncio.run(_get_response(app, "/terms"))
    privacy_head_status, _privacy_head_headers, _privacy_head_body = asyncio.run(
        _get_response(app, "/privacy", method="HEAD")
    )
    terms_head_status, _terms_head_headers, _terms_head_body = asyncio.run(
        _get_response(app, "/terms", method="HEAD")
    )
    privacy_html = privacy_body.decode("utf-8")
    terms_html = terms_body.decode("utf-8")

    assert privacy_status == 200
    assert "text/html" in privacy_headers["content-type"]
    assert "Privacy Policy" in privacy_html
    assert "meta@gamecult.org" in privacy_html
    assert terms_status == 200
    assert "text/html" in terms_headers["content-type"]
    assert "Terms of Service" in terms_html
    assert "meta@gamecult.org" in terms_html
    assert privacy_head_status == 200
    assert terms_head_status == 200


def test_gui_hosted_root_renders_heimdall_login_buttons(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "heimdall")
    monkeypatch.setenv("GC_ACCESS_HEIMDALL_BASE_URL", "https://heimdall.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_APP_PUBLIC_BASE_URL", "https://repixelizer.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_ALLOWED_PROVIDERS", "discord,patreon")
    monkeypatch.setenv("REPIXELIZER_ACCESS_DISCORD_GUILD_ID", "gamecult-guild")
    monkeypatch.setenv("REPIXELIZER_ACCESS_DISCORD_ALLOWED_ROLE_IDS", "role-repixelizer,role-patreon")

    app = create_app()
    status, _headers, body = asyncio.run(_get_response(app, "/"))
    html = body.decode("utf-8")

    assert status == 200
    assert "Login with Discord" in html
    assert "Login with Patreon" in html
    assert "Sign in to the demo" not in html
    assert "Already authenticated as" not in html
    assert 'href="#auth"' not in html
    assert "/api/auth/heimdall/start" in html
    assert "Hosted access will be gated through Heimdall once the auth pass lands." not in html


def test_gui_hosted_app_redirects_to_landing_when_auth_required(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "trusted-header")
    monkeypatch.setenv("GC_ACCESS_REQUIRED", "1")

    app = create_app()
    status, headers, _body = asyncio.run(_get_response(app, "/app/"))

    assert status in {302, 303, 307}
    assert headers["location"] == "/"


def test_gui_heimdall_mode_redirects_to_landing_without_session(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "heimdall")
    monkeypatch.setenv("GC_ACCESS_HEIMDALL_BASE_URL", "https://heimdall.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_APP_PUBLIC_BASE_URL", "https://repixelizer.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_ALLOWED_PROVIDERS", "discord,patreon")

    app = create_app()
    status, headers, _body = asyncio.run(_get_response(app, "/app/"))

    assert status in {302, 303, 307}
    assert headers["location"] == "/"


def test_gui_heimdall_mode_serves_static_assets_without_session(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "heimdall")
    monkeypatch.setenv("GC_ACCESS_HEIMDALL_BASE_URL", "https://heimdall.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_APP_PUBLIC_BASE_URL", "https://repixelizer.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_ALLOWED_PROVIDERS", "discord")

    app = create_app()
    css_status, css_headers, css_body = asyncio.run(_get_response(app, "/app/styles.css"))
    logo_status, logo_headers, logo_body = asyncio.run(
        _get_response(app, "/app/logos/repixelizer-logo-monogram.svg")
    )

    assert css_status == 200
    assert "text/css" in css_headers["content-type"]
    assert b"body" in css_body
    assert logo_status == 200
    assert "image/svg+xml" in logo_headers["content-type"]
    assert b"<svg" in logo_body


def test_gui_api_jobs_denies_unauthenticated_request_when_auth_required(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "trusted-header")
    monkeypatch.setenv("GC_ACCESS_REQUIRED", "1")

    app = create_app()
    status, _headers, body = asyncio.run(_get_response(app, "/api/jobs", method="POST"))
    payload = _response_json(type("Response", (), {"body": body})())

    assert status == 401
    assert payload["detail"] == "Sign-in required."


def test_gui_local_root_keeps_redirect_to_app(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.delenv("REPIXELIZER_HOSTED_DEMO", raising=False)
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))

    app = create_app()
    status, headers, _body = asyncio.run(_get_response(app, "/"))

    assert status in {302, 307}
    assert headers["location"] == "/app/"
    assert headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"


def test_infer_autocorr_lattice_emits_candidate_progress_events(monkeypatch) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    fake_estimate = inference_module.AutocorrEstimate(
        best_lag=4.0,
        best_score=0.9,
        candidate_lags=(4.0,),
        candidate_scores=(0.9,),
    )
    monkeypatch.setattr(inference_module, "_require_torch", lambda: (_FakeTorch(), object()))
    monkeypatch.setattr(inference_module, "_estimate_lattice_autocorr_details", lambda rgba: (fake_estimate, fake_estimate))
    monkeypatch.setattr(
        inference_module,
        "_hint_target_sizes_from_autocorr",
        lambda width, height, *, autocorr_x_estimate, autocorr_y_estimate, shared_prior: [],
    )
    monkeypatch.setattr(inference_module, "_estimate_lattice_prior_details", lambda rgba: (4.0, 4.0, 0.5))
    monkeypatch.setattr(
        inference_module,
        "_candidate_dims_from_autocorr_hints",
        lambda *args, **kwargs: [(10, 8), (12, 10)],
    )

    def fake_score_size_candidate(
        rgba,
        *,
        target_width,
        target_height,
        prior_cell_x,
        prior_cell_y,
        prior_reliability,
        device,
    ):
        del rgba, prior_cell_x, prior_cell_y, prior_reliability, device
        return [
            InferenceCandidate(
                target_width=target_width,
                target_height=target_height,
                score=0.9 if target_width == 12 else 0.6,
                breakdown={},
            )
        ]

    monkeypatch.setattr(inference_module, "_score_size_candidate", fake_score_size_candidate)
    monkeypatch.setattr(inference_module, "_top_candidates_by_size", lambda candidates, limit: candidates[:limit])
    monkeypatch.setattr(
        inference_module,
        "_rerank_size_candidates_with_source_evidence",
        lambda rgba, candidates: sorted(candidates, key=lambda candidate: candidate.score, reverse=True),
    )

    events: list[tuple[str, dict[str, object]]] = []

    def observer(event: str, payload: dict[str, object]) -> None:
        events.append((event, payload))

    result = inference_module.infer_autocorr_lattice(np.zeros((16, 20, 4), dtype=np.uint8), observer=observer)

    assert result.target_width == 12
    assert [event for event, _payload in events] == [
        "lattice_inference_started",
        "lattice_inference_progress",
        "lattice_inference_progress",
    ]
    assert events[0][1]["candidate_count"] == 2
    assert events[1][1]["completed_candidates"] == 1
    assert events[2][1]["completed_candidates"] == 2
    assert events[2][1]["target_width"] == 12


def test_candidate_rerank_emits_candidate_progress_events(monkeypatch) -> None:
    inference = InferenceResult(
        target_width=16,
        target_height=16,
        confidence=0.0,
        top_candidates=[
            InferenceCandidate(target_width=16, target_height=16, score=0.75, breakdown={}),
            InferenceCandidate(target_width=18, target_height=18, score=0.73, breakdown={}),
        ],
    )
    analysis = PhaseFieldSourceAnalysis(edge_map=np.zeros((4, 4), dtype=np.float32))

    def fake_run_reconstruction(source, *, observer=None, inference, **kwargs):
        del source, kwargs
        if observer is not None:
            observer("phase_field_initial", {"step": 0, "total_steps": 2})
            observer("phase_field_step", {"step": 1, "total_steps": 2, "loss": 0.25})
            observer("phase_field_step", {"step": 2, "total_steps": 2, "loss": 0.125})
        target_rgba = np.zeros((inference.target_height, inference.target_width, 4), dtype=np.uint8)
        artifacts = SolverArtifacts(
            target_rgba=target_rgba,
            uv_field=np.zeros((inference.target_height, inference.target_width, 2), dtype=np.float32),
            signal_strength=np.zeros((inference.target_height, inference.target_width), dtype=np.float32),
            initial_rgba=target_rgba.copy(),
            loss_history=[0.25, 0.125],
        )
        return artifacts, {}

    monkeypatch.setattr(pipeline_module, "_run_reconstruction", fake_run_reconstruction)
    monkeypatch.setattr(pipeline_module, "source_lattice_consistency_breakdown", lambda *args, **kwargs: {"score": 0.8})
    monkeypatch.setattr(
        pipeline_module,
        "nearest_resize",
        lambda rgba, width, height: np.zeros((height, width, 4), dtype=np.uint8),
    )
    monkeypatch.setattr(pipeline_module, "foreground_edge_position_error", lambda preview, source: 0.12)
    monkeypatch.setattr(pipeline_module, "foreground_stroke_wobble_error", lambda preview, source: 0.08)
    monkeypatch.setattr(pipeline_module, "foreground_edge_concentration", lambda rgba: 0.9)

    events: list[tuple[str, dict[str, object]]] = []

    def observer(event: str, payload: dict[str, object]) -> None:
        events.append((event, payload))

    pipeline_module._select_candidate_with_reconstruction(
        np.zeros((16, 16, 4), dtype=np.uint8),
        inference,
        steps=2,
        seed=1,
        device="cpu",
        solver_params=SolverHyperParams(candidate_rerank_preview_steps=2, candidate_rerank_confidence_threshold=1.0),
        observer=observer,
    )

    event_names = [event for event, _payload in events]
    assert event_names[0] == "candidate_rerank_started"
    assert event_names.count("candidate_rerank_candidate_started") == 2
    assert event_names.count("candidate_rerank_candidate_step") == 6
    assert event_names.count("candidate_rerank_candidate_completed") == 2
    first_step_payload = next(payload for event, payload in events if event == "candidate_rerank_candidate_step")
    assert first_step_payload["candidate_index"] == 1
    assert first_step_payload["total_steps"] == 2


def test_gui_hosted_config_endpoint_exposes_demo_limits(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))

    app = create_app()
    payload = _response_json(_route_endpoint(app, "/api/config", "GET")())

    assert payload["hostedDemo"] is True
    assert payload["limits"]["maxUploadBytes"] == 2 * 1_048_576
    assert payload["limits"]["maxOutputDimension"] == 512
    assert payload["limits"]["defaultSteps"] == 32
    assert payload["ui"]["showDeviceControl"] is False
    assert payload["ui"]["showStripBackgroundControl"] is False
    assert payload["ui"]["showQueuePanel"] is True
    assert payload["auth"]["enabled"] is False
    assert payload["auth"]["mode"] == "off"


def test_gui_hosted_config_endpoint_exposes_heimdall_auth_surface(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "heimdall")
    monkeypatch.setenv("GC_ACCESS_HEIMDALL_BASE_URL", "https://heimdall.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_APP_PUBLIC_BASE_URL", "https://repixelizer.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_ALLOWED_PROVIDERS", "discord,patreon")

    app = create_app()
    payload = _response_json(_route_endpoint(app, "/api/config", "GET")())

    assert payload["auth"]["enabled"] is True
    assert payload["auth"]["mode"] == "heimdall"
    assert payload["auth"]["loginUrl"] == "/"
    assert payload["auth"]["logoutUrl"] == "/api/auth/logout"
    assert payload["auth"]["startEndpoint"] == "/api/auth/heimdall/start"
    assert payload["auth"]["providers"] == [
        {"slug": "discord", "label": "Discord"},
        {"slug": "patreon", "label": "Patreon"},
    ]


def test_gui_auth_session_endpoint_reports_anonymous_subject_by_default(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))

    app = create_app()
    status, _headers, body = asyncio.run(_get_response(app, "/api/auth/session"))
    payload = _response_json(type("Response", (), {"body": body})())

    assert status == 200
    assert payload["auth"]["enabled"] is False
    assert payload["subject"]["authenticated"] is False
    assert payload["subject"]["capabilities"] == sorted(
        ["admin_access", "app_access", "job_cancel_own", "job_read_own", "queue_submit"]
    )


def test_gui_heimdall_callback_flow_adopts_local_cookie_session(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    private_key, jwk, key_id = _heimdall_signing_material()
    start_calls: list[tuple[str, dict[str, object], float]] = []

    def fake_post_json(url: str, payload: dict[str, object], *, timeout_seconds: float):
        start_calls.append((url, payload, timeout_seconds))
        return {
            "authorizationUrl": "https://discord.com/oauth2/authorize?state=test-state",
            "stateExpiresAt": "2026-04-27T12:15:00.000Z",
        }

    def fake_fetch_json(url: str, *, timeout_seconds: float):
        del timeout_seconds
        assert url == "https://heimdall.gamecult.org/.well-known/jwks.json"
        return {"keys": [jwk]}

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "heimdall")
    monkeypatch.setenv("GC_ACCESS_HEIMDALL_BASE_URL", "https://heimdall.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_APP_PUBLIC_BASE_URL", "https://repixelizer.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_ALLOWED_PROVIDERS", "discord,patreon")
    monkeypatch.setenv("REPIXELIZER_ACCESS_DISCORD_GUILD_ID", "gamecult-guild")
    monkeypatch.setenv("REPIXELIZER_ACCESS_DISCORD_ALLOWED_ROLE_IDS", "role-repixelizer,role-patreon")
    monkeypatch.setattr("repixelizer.access._post_json", fake_post_json)
    monkeypatch.setattr("repixelizer.access._fetch_json", fake_fetch_json)

    app = create_app()

    start_status, _start_headers, start_body = asyncio.run(
        _get_response(
            app,
            "/api/auth/heimdall/start",
            method="POST",
            headers={"content-type": "application/json"},
            body=_json_bytes({"provider": "discord"}),
        )
    )
    start_payload = _response_json(type("Response", (), {"body": start_body})())
    attempt_id = start_payload["attemptId"]

    assert start_status == 201
    assert start_payload["authorizationUrl"] == "https://discord.com/oauth2/authorize?state=test-state"
    assert start_calls
    assert start_calls[0][0] == "https://heimdall.gamecult.org/v1/oauth/discord/start"
    assert start_calls[0][1]["handoff"]["callbackUrl"] == "https://repixelizer.gamecult.org/api/auth/heimdall/callback"
    assert start_calls[0][1]["entitlementPolicy"] == {
        "kind": "discord_role_access",
        "guildId": "gamecult-guild",
        "allowedRoleIds": ["role-repixelizer", "role-patreon"],
    }

    pending_status, _pending_headers, pending_body = asyncio.run(_get_response(app, f"/api/auth/attempts/{attempt_id}"))
    pending_payload = _response_json(type("Response", (), {"body": pending_body})())
    assert pending_status == 200
    assert pending_payload["status"] == "pending"

    access_token = _heimdall_access_token(private_key, key_id=key_id)
    callback_payload = {
        "source": "heimdall",
        "kind": "oauth_result",
        "handoffKind": "backend_callback",
        "attemptId": attempt_id,
        "status": "success",
        "provider": "discord",
        "appSlug": "repixelizer",
        "mode": "sign_in",
        "returnTo": "https://repixelizer.gamecult.org/app/",
        "account": {
            "id": "acct-1",
            "displayName": "Meta",
        },
        "session": {
            "accountId": "acct-1",
            "sessionId": "sess-1",
            "appSlug": "repixelizer",
            "accessRevision": 4,
            "expiresAt": "2026-04-27T13:00:00.000Z",
        },
        "accessToken": access_token,
        "claimSet": {},
        "verification": {
            "issuer": "https://heimdall.gamecult.org",
            "jwksUri": "https://heimdall.gamecult.org/.well-known/jwks.json",
            "alg": "EdDSA",
            "kid": key_id,
        },
        "sharedCapabilities": ["app_access", "queue_submit"],
        "hybridCapabilities": [],
        "entitlements": {
            "facts": ["entitlement.app_access"],
            "snapshots": [],
        },
    }
    callback_status, _callback_headers, _callback_body = asyncio.run(
        _get_response(
            app,
            "/api/auth/heimdall/callback",
            method="POST",
            headers={"content-type": "application/json"},
            body=_json_bytes(callback_payload),
        )
    )
    assert callback_status == 204

    completed_status, _completed_headers, completed_body = asyncio.run(_get_response(app, f"/api/auth/attempts/{attempt_id}"))
    completed_payload = _response_json(type("Response", (), {"body": completed_body})())
    assert completed_status == 200
    assert completed_payload["status"] == "succeeded"
    assert completed_payload["subject"]["accountId"] == "acct-1"

    adopt_status, adopt_headers, adopt_body = asyncio.run(
        _get_response(app, f"/api/auth/attempts/{attempt_id}/adopt", method="POST")
    )
    adopt_payload = _response_json(type("Response", (), {"body": adopt_body})())
    assert adopt_status == 200
    assert adopt_payload["status"] == "authenticated"
    assert "set-cookie" in adopt_headers
    assert "gc_access_token=" in adopt_headers["set-cookie"]
    cookie_header = adopt_headers["set-cookie"].split(";", 1)[0]

    session_status, _session_headers, session_body = asyncio.run(
        _get_response(app, "/api/auth/session", headers={"cookie": cookie_header})
    )
    session_payload = _response_json(type("Response", (), {"body": session_body})())
    assert session_status == 200
    assert session_payload["subject"]["authenticated"] is True
    assert session_payload["subject"]["accountId"] == "acct-1"
    assert "app_access" in session_payload["subject"]["capabilities"]

    app_status, app_headers, _app_body = asyncio.run(_get_response(app, "/app/", headers={"cookie": cookie_header}))
    assert app_status == 200
    assert app_headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"


def test_gui_heimdall_patreon_auth_start_sends_membership_policy(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    start_calls: list[tuple[str, dict[str, object], float]] = []

    def fake_post_json(url: str, payload: dict[str, object], *, timeout_seconds: float):
        start_calls.append((url, payload, timeout_seconds))
        return {
            "authorizationUrl": "https://www.patreon.com/oauth2/authorize?state=test-state",
            "stateExpiresAt": "2026-04-27T12:15:00.000Z",
        }

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "heimdall")
    monkeypatch.setenv("GC_ACCESS_HEIMDALL_BASE_URL", "https://heimdall.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_APP_PUBLIC_BASE_URL", "https://repixelizer.gamecult.org")
    monkeypatch.setenv("GC_ACCESS_ALLOWED_PROVIDERS", "discord,patreon")
    monkeypatch.setenv("REPIXELIZER_ACCESS_PATREON_TIER_TITLE", "Inner Sanctum")
    monkeypatch.setattr("repixelizer.access._post_json", fake_post_json)

    app = create_app()

    start_status, _start_headers, start_body = asyncio.run(
        _get_response(
            app,
            "/api/auth/heimdall/start",
            method="POST",
            headers={"content-type": "application/json"},
            body=_json_bytes({"provider": "patreon"}),
        )
    )
    start_payload = _response_json(type("Response", (), {"body": start_body})())

    assert start_status == 201
    assert start_payload["authorizationUrl"] == "https://www.patreon.com/oauth2/authorize?state=test-state"
    assert start_calls
    assert start_calls[0][0] == "https://heimdall.gamecult.org/v1/oauth/patreon/start"
    assert start_calls[0][1]["entitlementPolicy"] == {
        "kind": "patreon_membership_access",
        "requiredTierTitle": "Inner Sanctum",
    }


def test_hosted_job_options_use_autocorr_and_force_rerank() -> None:
    from repixelizer.gui import HostedDemoConfig, _normalize_job_options

    config = HostedDemoConfig(
        hosted_demo=True,
        show_queue_panel=True,
        max_upload_bytes=2 * 1_048_576,
        max_input_dimension=2048,
        max_output_dimension=512,
        default_steps=32,
        max_steps=48,
        queue_capacity=10,
        heartbeat_interval_seconds=10,
        stale_after_seconds=30,
        phase_field_preview_stride=4,
        spool_dir=Path("spool"),
    )

    options = _normalize_job_options(
        config,
        source_width=1254,
        source_height=1254,
        target_size=None,
        target_width=None,
        target_height=None,
        steps=None,
        seed=7,
        device="auto",
        strip_background=False,
        skip_candidate_rerank=True,
    )

    assert options["target_size"] is None
    assert options["lattice_inference_mode"] == "autocorr"
    assert options["max_inferred_target_size"] == 512
    assert options["skip_candidate_rerank"] is False


def test_gui_job_routes_enforce_bound_subject_ownership(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "trusted-header")
    monkeypatch.setenv("GC_ACCESS_REQUIRED", "1")

    app = create_app()
    create_job = _route_endpoint(app, "/api/jobs", "POST")
    get_job = _route_endpoint(app, "/api/jobs/{job_id}", "GET")

    subject = AccessSubject(
        account_id="acct-1",
        session_id="sess-1",
        access_revision=4,
        capabilities=frozenset({"app_access", "queue_submit"}),
        auth_mode="trusted-header",
    )
    with bind_access_subject(subject):
        created = asyncio.run(
            create_job(
                image=UploadFile(filename="tiny.png", file=io.BytesIO(_png_bytes())),
                target_size=None,
                target_width=None,
                target_height=None,
                steps=None,
                seed=7,
                device="auto",
                strip_background=False,
                skip_candidate_rerank=False,
            )
        )
    assert created.status_code == 200
    job_id = _response_json(created)["jobId"]

    with bind_access_subject(subject):
        owned = get_job(job_id)
    assert owned.status_code == 200

    intruder = AccessSubject(
        account_id="acct-2",
        session_id="sess-2",
        access_revision=8,
        capabilities=frozenset({"app_access"}),
        auth_mode="trusted-header",
    )
    with bind_access_subject(intruder):
        with pytest.raises(HTTPException) as excinfo:
            get_job(job_id)
    assert excinfo.value.status_code == 403
    assert "different local account or session" in str(excinfo.value.detail)


def test_gui_job_logs_authenticated_actor_metadata(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("GC_ACCESS_MODE", "trusted-header")
    monkeypatch.setenv("GC_ACCESS_REQUIRED", "1")

    app = create_app()
    create_job = _route_endpoint(app, "/api/jobs", "POST")

    subject = AccessSubject(
        account_id="acct-1",
        session_id="sess-1",
        access_revision=4,
        display_name="Meta",
        capabilities=frozenset({"app_access", "queue_submit"}),
        auth_mode="heimdall",
        claims={
            "identities": [
                {
                    "provider": "discord",
                    "providerUserId": "123456789",
                }
            ]
        },
    )

    caplog.set_level(logging.INFO, logger="repixelizer.gui")
    with bind_access_subject(subject):
        created = asyncio.run(
            create_job(
                image=UploadFile(filename="tiny.png", file=io.BytesIO(_png_bytes())),
                target_size=None,
                target_width=None,
                target_height=None,
                steps=None,
                seed=7,
                device="auto",
                strip_background=False,
                skip_candidate_rerank=False,
            )
        )

    assert created.status_code == 200
    queued_messages = [record.getMessage() for record in caplog.records if "repixelizer_job_queued" in record.getMessage()]
    assert queued_messages
    assert 'actor={"accountId":"acct-1","authMode":"heimdall","displayName":"Meta","provider":"discord","sessionId":"sess-1"}' in queued_messages[0]


def test_gui_queue_panel_defaults_off_for_local_runs_and_can_be_forced(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.delenv("REPIXELIZER_HOSTED_DEMO", raising=False)
    monkeypatch.delenv("REPIXELIZER_SHOW_QUEUE_PANEL", raising=False)
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))

    app = create_app()
    payload = _response_json(_route_endpoint(app, "/api/config", "GET")())
    assert payload["hostedDemo"] is False
    assert payload["ui"]["showQueuePanel"] is False

    monkeypatch.setenv("REPIXELIZER_SHOW_QUEUE_PANEL", "1")
    app = create_app()
    payload = _response_json(_route_endpoint(app, "/api/config", "GET")())
    assert payload["ui"]["showQueuePanel"] is True


def test_gui_queue_rejects_eleventh_waiting_job(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    release = threading.Event()

    def fake_run_pipeline_rgba(*args, **kwargs):
        observer = kwargs.get("observer")
        if observer is not None:
            observer("stage_started", {"stage": "solver", "label": "Solver", "detail": "Working."})
        release.wait(timeout=2.0)
        if observer is not None and observer.__self__.check_cancelled():  # type: ignore[attr-defined]
            raise PipelineCancelled(observer.__self__.cancellation_message)  # type: ignore[attr-defined]

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_QUEUE_CAPACITY", "10")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setattr("repixelizer.gui.run_pipeline_rgba", fake_run_pipeline_rgba)

    upload = _png_bytes()
    app = create_app()
    create_job = _route_endpoint(app, "/api/jobs", "POST")
    accepted = []
    for _ in range(11):
        accepted.append(
            asyncio.run(
                create_job(
                    image=UploadFile(filename="tiny.png", file=io.BytesIO(upload)),
                    target_size=None,
                    target_width=None,
                    target_height=None,
                    steps=None,
                    seed=7,
                    device="auto",
                    strip_background=False,
                    skip_candidate_rerank=False,
                )
            )
        )
    rejected = None
    try:
        asyncio.run(
            create_job(
                image=UploadFile(filename="tiny.png", file=io.BytesIO(upload)),
                target_size=None,
                target_width=None,
                target_height=None,
                steps=None,
                seed=7,
                device="auto",
                strip_background=False,
                skip_candidate_rerank=False,
            )
        )
    except Exception as exc:
        rejected = exc
    release.set()

    assert all(response.status_code == 200 for response in accepted)
    assert rejected is not None
    assert "Queue is full" in getattr(rejected, "detail", str(rejected))


def test_gui_hosted_jobs_dispatch_direct_autocorr_pipeline(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    release = threading.Event()
    calls: list[dict[str, object]] = []

    def fake_run_pipeline_rgba(*args, **kwargs):
        calls.append(kwargs)
        release.wait(timeout=2.0)

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setattr("repixelizer.gui.run_pipeline_rgba", fake_run_pipeline_rgba)

    upload = _png_bytes()
    app = create_app()
    create_job = _route_endpoint(app, "/api/jobs", "POST")
    created = asyncio.run(
        create_job(
            image=UploadFile(filename="tiny.png", file=io.BytesIO(upload)),
            target_size=None,
            target_width=None,
            target_height=None,
            steps=None,
            seed=7,
            device="auto",
            strip_background=False,
            skip_candidate_rerank=False,
        )
    )
    deadline = time.time() + 2.0
    while not calls and time.time() < deadline:
        time.sleep(0.05)
    release.set()

    assert created.status_code == 200
    assert calls
    assert calls[0]["lattice_inference_mode"] == "autocorr"
    assert calls[0]["max_inferred_target_size"] == 512
    assert calls[0]["enable_candidate_rerank"] is True
    assert calls[0]["target_size"] is None


def test_gui_canceling_queued_job_cleans_spool_file(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    release = threading.Event()

    def fake_run_pipeline_rgba(*args, **kwargs):
        release.wait(timeout=2.0)

    spool_dir = tmp_path / "spool"
    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(spool_dir))
    monkeypatch.setattr("repixelizer.gui.run_pipeline_rgba", fake_run_pipeline_rgba)

    upload = _png_bytes()
    app = create_app()
    create_job = _route_endpoint(app, "/api/jobs", "POST")
    cancel_job = _route_endpoint(app, "/api/jobs/{job_id}", "DELETE")
    queue_endpoint = _route_endpoint(app, "/api/queue", "GET")
    first = asyncio.run(
        create_job(
            image=UploadFile(filename="first.png", file=io.BytesIO(upload)),
            target_size=None,
            target_width=None,
            target_height=None,
            steps=None,
            seed=7,
            device="auto",
            strip_background=False,
            skip_candidate_rerank=False,
        )
    )
    second = asyncio.run(
        create_job(
            image=UploadFile(filename="second.png", file=io.BytesIO(upload)),
            target_size=None,
            target_width=None,
            target_height=None,
            steps=None,
            seed=7,
            device="auto",
            strip_background=False,
            skip_candidate_rerank=False,
        )
    )
    assert first.status_code == 200
    assert second.status_code == 200
    queued_job_id = _response_json(second)["jobId"]
    before_cancel = sorted(spool_dir.iterdir())
    canceled = cancel_job(queued_job_id)
    queue_summary = _response_json(queue_endpoint())
    after_cancel = sorted(spool_dir.iterdir())
    release.set()

    assert len(before_cancel) == 2
    assert canceled.status_code == 200
    assert _response_json(canceled)["status"] == "canceled"
    assert queue_summary["waitingCount"] == 0
    assert len(after_cancel) == 1


def test_gui_running_job_is_canceled_after_stale_heartbeat(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    def fake_run_pipeline_rgba(*args, **kwargs):
        observer = kwargs.get("observer")
        assert observer is not None
        observer("stage_started", {"stage": "solver", "label": "Solver", "detail": "Working."})
        for _ in range(80):
            owner = observer.__self__  # type: ignore[attr-defined]
            if owner.check_cancelled():
                raise PipelineCancelled(owner.cancellation_message)
            time.sleep(0.05)

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_HEARTBEAT_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("REPIXELIZER_STALE_AFTER_SECONDS", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setattr("repixelizer.gui.run_pipeline_rgba", fake_run_pipeline_rgba)

    app = create_app()
    create_job = _route_endpoint(app, "/api/jobs", "POST")
    get_job = _route_endpoint(app, "/api/jobs/{job_id}", "GET")
    created = asyncio.run(
        create_job(
            image=UploadFile(filename="tiny.png", file=io.BytesIO(_png_bytes())),
            target_size=None,
            target_width=None,
            target_height=None,
            steps=None,
            seed=7,
            device="auto",
            strip_background=False,
            skip_candidate_rerank=False,
        )
    )
    assert created.status_code == 200
    job_id = _response_json(created)["jobId"]
    deadline = time.time() + 4.0
    latest = None
    while time.time() < deadline:
        latest = get_job(job_id)
        if _response_json(latest)["status"] == "canceled":
            break
        time.sleep(0.1)

    assert latest is not None
    assert latest.status_code == 200
    assert _response_json(latest)["status"] == "canceled"


def test_gui_rejects_oversized_upload_and_output(monkeypatch, tmp_path: Path) -> None:
    from repixelizer.gui import create_app

    monkeypatch.setenv("REPIXELIZER_HOSTED_DEMO", "1")
    monkeypatch.setenv("REPIXELIZER_SPOOL_DIR", str(tmp_path / "spool"))

    monkeypatch.setenv("REPIXELIZER_MAX_UPLOAD_BYTES", "20")
    app = create_app()
    create_job = _route_endpoint(app, "/api/jobs", "POST")
    too_large_upload = None
    try:
        too_large_upload = asyncio.run(
            create_job(
                image=UploadFile(filename="tiny.png", file=io.BytesIO(_png_bytes())),
                target_size=None,
                target_width=None,
                target_height=None,
                steps=None,
                seed=7,
                device="auto",
                strip_background=False,
                skip_candidate_rerank=False,
            )
        )
    except Exception as exc:
        too_large_upload = exc

    monkeypatch.setenv("REPIXELIZER_MAX_UPLOAD_BYTES", "1048576")
    monkeypatch.setenv("REPIXELIZER_MAX_OUTPUT_DIMENSION", "8")
    app = create_app()
    create_job = _route_endpoint(app, "/api/jobs", "POST")
    explicit_big_output = None
    try:
        explicit_big_output = asyncio.run(
            create_job(
                image=UploadFile(filename="tiny.png", file=io.BytesIO(_png_bytes(width=4, height=4))),
                target_size=None,
                target_width=16,
                target_height=None,
                steps=None,
                seed=7,
                device="auto",
                strip_background=False,
                skip_candidate_rerank=False,
            )
        )
    except Exception as exc:
        explicit_big_output = exc

    assert too_large_upload is not None
    assert "too large" in getattr(too_large_upload, "detail", str(too_large_upload)).lower()
    assert explicit_big_output is not None
    assert "exceeds the maximum hosted output dimension" in getattr(
        explicit_big_output, "detail", str(explicit_big_output)
    )
