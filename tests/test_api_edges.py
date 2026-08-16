"""Edge and failure paths in the API layer.

These are the branches that only execute when something goes wrong — limits
hit, files oversized, callers hostile. They are the least exercised in normal
use and the most important to get right, because each one is a place where the
server could leak internals, hang, or lose data.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.api import routes
from backend.config import Settings, resolve_port
from backend.core.types import Track, TrackKind
from backend.main import _FRONTEND_DIR, create_app
from tests.conftest import make_harmonic_tone


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _wav(audio, sample_rate=44_100, subtype="PCM_24"):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


# =============================================================================
# Upload limits
# =============================================================================


def test_oversized_upload_is_rejected_with_413(client, monkeypatch):
    """Rejected while streaming, not after buffering the whole thing."""
    monkeypatch.setattr(routes.settings, "max_upload_bytes", 2048)
    response = client.post(
        "/api/analyze", files={"file": ("big.wav", b"\x00" * 100_000, "audio/wav")}
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_oversized_upload_leaves_no_file_behind(client, monkeypatch, tmp_path):
    """A rejected upload must not litter the session directory."""
    monkeypatch.setattr(routes.settings, "session_dir", tmp_path)
    monkeypatch.setattr(routes.settings, "max_upload_bytes", 1024)
    client.post("/api/analyze", files={"file": ("big.wav", b"\x00" * 50_000, "audio/wav")})
    assert list(tmp_path.glob("upload_*")) == []


def test_rate_limit_returns_429(client, monkeypatch):
    """The one expensive endpoint is protected against a refresh loop."""
    monkeypatch.setattr(routes, "_RATE_LIMIT_REQUESTS", 3)
    monkeypatch.setattr(routes, "_rate_history", {})

    payload = _wav(make_harmonic_tone(220.0, 3, duration_s=0.2))
    statuses = [
        client.post("/api/analyze", files={"file": ("a.wav", payload, "audio/wav")}).status_code
        for _ in range(6)
    ]
    assert 429 in statuses, f"rate limit never fired: {statuses}"
    assert statuses[0] == 200, "the first request should always be allowed"


def test_rate_limit_window_expires(monkeypatch):
    """Old requests fall out of the sliding window rather than counting forever."""
    monkeypatch.setattr(routes, "_RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(routes, "_RATE_LIMIT_WINDOW_S", 0.05)
    monkeypatch.setattr(routes, "_rate_history", {})

    routes._check_rate_limit("someone")
    routes._check_rate_limit("someone")
    with pytest.raises(Exception):
        routes._check_rate_limit("someone")

    import time

    time.sleep(0.08)
    routes._check_rate_limit("someone")  # window has slid; allowed again


# =============================================================================
# Session cleanup — must be surgical
# =============================================================================


def test_cleanup_removes_only_its_own_artifacts(monkeypatch, tmp_path):
    """The launcher's log lives alongside these; it must survive.

    An earlier version rmtree'd the whole directory, deleting the log that the
    crash dialog exists to display.
    """
    monkeypatch.setattr(routes.settings, "session_dir", tmp_path)

    (tmp_path / "upload_abc").write_bytes(b"x")
    (tmp_path / "fp_export_xyz").mkdir()
    (tmp_path / "fp_export_xyz" / "t.wav").write_bytes(b"x")
    (tmp_path / "server.log").write_text("important diagnostics")
    (tmp_path / "server.pid").write_text("1234")

    routes.cleanup_sessions()

    assert not (tmp_path / "upload_abc").exists()
    assert not (tmp_path / "fp_export_xyz").exists()
    assert (tmp_path / "server.log").read_text() == "important diagnostics"
    assert (tmp_path / "server.pid").exists()


def test_cleanup_on_a_missing_directory_is_harmless(monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "session_dir", tmp_path / "gone")
    routes.cleanup_sessions()  # must not raise


# =============================================================================
# Result lookup helpers
# =============================================================================


def test_requesting_tracks_of_a_running_job_is_409(client, monkeypatch):
    """Asking for results too early is a conflict, not a 404 or a hang."""
    from backend.api.jobs import InMemoryJobStore

    store = InMemoryJobStore(ttl_seconds=60)
    monkeypatch.setattr(routes, "_jobs", store)
    store.create("running-job")

    assert client.get("/api/jobs/running-job/tracks").status_code == 409


def test_requesting_tracks_of_a_failed_job_reports_the_reason(client, monkeypatch):
    from backend.api.jobs import InMemoryJobStore

    store = InMemoryJobStore(ttl_seconds=60)
    monkeypatch.setattr(routes, "_jobs", store)
    store.create("bad-job")
    store.fail("bad-job", "That audio is too long.")

    response = client.get("/api/jobs/bad-job/tracks")
    assert response.status_code == 400
    assert response.json()["detail"] == "That audio is too long."


def test_websocket_on_unknown_job_reports_expiry_then_closes(client):
    """A socket for a purged job must say so rather than hang open."""
    with client.websocket_connect("/api/jobs/nonexistent/events") as socket:
        frame = socket.receive_json()
    assert frame["state"] == "failed"
    assert "expired" in frame["message"].lower()


def test_audio_of_an_unrendered_track_is_409(client, monkeypatch):
    """A track with no audio is a conflict, not a crash."""
    from backend.api.jobs import InMemoryJobStore
    from backend.core.types import AnalysisParams, AnalysisResult

    store = InMemoryJobStore(ttl_seconds=60)
    monkeypatch.setattr(routes, "_jobs", store)
    store.create("j")
    bare = Track(0, TrackKind.PARTIAL, "no audio", ())
    store.complete(
        "j",
        AnalysisResult(44_100, 1.0, [bare], AnalysisParams(), np.zeros(10, dtype=np.float32)),
    )
    assert client.get("/api/jobs/j/tracks/0/audio").status_code == 409


# =============================================================================
# App construction
# =============================================================================


def test_frontend_directory_exists():
    """A relative path here would silently 404 the whole UI under the .app."""
    assert _FRONTEND_DIR.is_dir()
    assert (_FRONTEND_DIR / "index.html").is_file()
    assert _FRONTEND_DIR.is_absolute()


def test_app_builds_twice_independently():
    """create_app is a factory, not a singleton — tests rely on that."""
    assert create_app() is not create_app()


def test_settings_reject_unknown_keys():
    with pytest.raises(Exception):
        Settings(not_a_real_setting=1)


def test_resolve_port_prefers_the_requested_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert resolve_port("127.0.0.1", free) == free
