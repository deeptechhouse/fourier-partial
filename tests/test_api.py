"""Tests for the HTTP layer — which is also the security boundary.

Everything reaching these routes came from outside and is hostile until proven
otherwise, so most of what is asserted here is about *rejection*: oversized
uploads, files that lie about their format, jobs that do not exist, exports
that make no sense. The happy path is one test; the rest is the boundary
holding.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _wav_bytes(duration_s: float = 0.6, freq_hz: float = 440.0, sample_rate: int = 44_100):
    """A real, valid WAV file in memory."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV", subtype="PCM_24")
    return buffer.getvalue()


def _analyse(client, data: bytes, filename: str = "tone.wav"):
    """Upload and wait for the job to reach a terminal state."""
    response = client.post("/api/analyze", files={"file": (filename, data, "audio/wav")})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    for _ in range(400):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in ("complete", "failed"):
            return job_id, status
    raise AssertionError("analysis did not finish in time")


# =============================================================================
# Liveness and static serving
# =============================================================================


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_frontend_is_served(client):
    """The UI must actually be reachable, not just present on disk.

    This is the check that catches a relative-path mount breaking when the app
    is started from a different working directory -- which is exactly what the
    .app launcher does.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "fourier" in response.text.lower()


# =============================================================================
# Security headers (design document section 13.3)
# =============================================================================


def test_security_headers_present(client):
    headers = client.get("/api/health").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"


def test_csp_is_restrictive(client):
    csp = client.get("/api/health").headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    # The progress socket needs an explicit ws: source; without it the bar
    # silently never moves in browsers that do not fold ws into 'self'.
    assert "ws://127.0.0.1:*" in csp
    # No wildcard anywhere -- that would defeat the whole header.
    assert "*" not in csp.replace("ws://127.0.0.1:*", "").replace("ws://localhost:*", "")


# =============================================================================
# Upload validation
# =============================================================================


def test_empty_upload_is_rejected(client):
    response = client.post("/api/analyze", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_text_file_renamed_as_wav_is_rejected(client):
    """The extension is a claim by the uploader; magic bytes are evidence."""
    fake = b"this is not audio, it is a text file wearing a .wav costume" * 20
    _, status = _analyse(client, fake, filename="liar.wav")
    assert status["state"] == "failed"
    assert status["message"]
    # The message must be safe to show a user: no paths, no stack frames.
    assert "Traceback" not in status["message"]
    assert "/Users/" not in status["message"]


def test_oversized_upload_is_rejected(client, monkeypatch):
    """Rejected on the way in, rather than after buffering the whole thing."""
    from backend.api import routes

    monkeypatch.setattr(routes.settings, "max_upload_bytes", 1024)
    response = client.post(
        "/api/analyze", files={"file": ("big.wav", _wav_bytes(1.0), "audio/wav")}
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_audio_over_the_duration_ceiling_is_rejected(client, monkeypatch):
    from backend.api import routes

    monkeypatch.setattr(routes._loader, "_max_duration", 0.2)
    _, status = _analyse(client, _wav_bytes(1.0))
    assert status["state"] == "failed"
    assert "too long" in (status["message"] or "").lower()


# =============================================================================
# Unknown jobs and tracks
# =============================================================================


def test_unknown_job_status_is_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_unknown_job_tracks_is_404(client):
    assert client.get("/api/jobs/does-not-exist/tracks").status_code == 404


def test_unknown_track_is_404(client):
    job_id, status = _analyse(client, _wav_bytes())
    assert status["state"] == "complete"
    assert client.get(f"/api/jobs/{job_id}/tracks/9999/audio").status_code == 404


def test_bad_export_format_is_rejected(client):
    """The format is constrained by pattern, so anything else is a 422."""
    job_id, _ = _analyse(client, _wav_bytes())
    response = client.get(f"/api/jobs/{job_id}/tracks/0/export?format=exe")
    assert response.status_code == 422


# =============================================================================
# The happy path, end to end
# =============================================================================


def test_full_analysis_round_trip(client):
    job_id, status = _analyse(client, _wav_bytes())
    assert status["state"] == "complete", status

    summary = client.get(f"/api/jobs/{job_id}/tracks").json()
    assert summary["reconstruction_exact"] is True
    assert summary["sample_rate"] == 44_100
    assert len(summary["tracks"]) >= 2  # at least one pitched track plus residual

    kinds = {t["kind"] for t in summary["tracks"]}
    assert "residual" in kinds

    pitched = [t for t in summary["tracks"] if t["kind"] != "residual"]
    assert pitched[0]["trajectory"], "a pitched track must carry a display trajectory"
    assert pitched[0]["mean_frequency_hz"] == pytest.approx(440.0, abs=2.0)


def test_track_audio_sums_back_to_the_original(client):
    """THE INVARIANT, measured on the bytes the browser actually receives.

    Not on internal arrays -- on the WAV payloads served over HTTP. If the
    audio endpoint ever quantised to 16-bit PCM to save bandwidth, this is the
    test that would catch it.
    """
    source = _wav_bytes(0.8)
    job_id, status = _analyse(client, source)
    assert status["state"] == "complete"

    summary = client.get(f"/api/jobs/{job_id}/tracks").json()
    mixed = None
    for track in summary["tracks"]:
        raw = client.get(f"/api/jobs/{job_id}/tracks/{track['track_id']}/audio").content
        samples, _ = sf.read(io.BytesIO(raw), dtype="float64")
        mixed = samples if mixed is None else mixed + samples

    original, _ = sf.read(io.BytesIO(source), dtype="float64")
    assert np.max(np.abs(mixed - original)) < 1e-6


def test_wav_export_downloads(client):
    job_id, _ = _analyse(client, _wav_bytes())
    response = client.get(f"/api/jobs/{job_id}/tracks/0/export?format=wav")
    assert response.status_code == 200
    assert len(response.content) > 1000
    samples, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert rate == 44_100
    assert np.max(np.abs(samples)) > 0.0


def test_midi_export_of_a_pitched_track(client):
    import mido

    job_id, _ = _analyse(client, _wav_bytes())
    summary = client.get(f"/api/jobs/{job_id}/tracks").json()
    pitched = next(t for t in summary["tracks"] if t["exportable_as_midi"])

    response = client.get(
        f"/api/jobs/{job_id}/tracks/{pitched['track_id']}/export?format=mid"
    )
    assert response.status_code == 200

    midi = mido.MidiFile(file=io.BytesIO(response.content))
    messages = list(midi)
    assert any(m.type == "note_on" for m in messages)
    # The RPN bend-range block must be present, or every glide clips silently.
    assert any(m.type == "control_change" and m.control == 6 for m in messages)


def test_midi_export_of_the_residual_is_refused(client):
    """Noise has no notes. Writing it as MIDI would invent information."""
    job_id, _ = _analyse(client, _wav_bytes())
    summary = client.get(f"/api/jobs/{job_id}/tracks").json()
    residual = next(t for t in summary["tracks"] if t["kind"] == "residual")

    response = client.get(
        f"/api/jobs/{job_id}/tracks/{residual['track_id']}/export?format=mid"
    )
    assert response.status_code == 400
    assert "noise" in response.json()["detail"].lower()


# =============================================================================
# Progress socket
# =============================================================================


def test_progress_socket_streams_to_completion(client):
    """The socket must exist and reach a terminal state.

    Regression guard: bare uvicorn ships no WebSocket implementation, so this
    endpoint returned 404 in production while the frontend's polling fallback
    hid the failure completely.
    """
    response = client.post(
        "/api/analyze", files={"file": ("t.wav", _wav_bytes(), "audio/wav")}
    )
    job_id = response.json()["job_id"]

    with client.websocket_connect(f"/api/jobs/{job_id}/events") as socket:
        states = []
        for _ in range(200):
            frame = socket.receive_json()
            states.append(frame["state"])
            assert 0.0 <= frame["progress"] <= 1.0
            if frame["state"] in ("complete", "failed"):
                break
    assert states[-1] == "complete", states


def test_progress_socket_for_unknown_job_reports_expiry(client):
    with client.websocket_connect("/api/jobs/nope/events") as socket:
        frame = socket.receive_json()
    assert frame["state"] == "failed"
    assert "expired" in frame["message"].lower()
