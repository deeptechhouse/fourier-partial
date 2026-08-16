"""Tests for the entry point and the exporters' failure paths.

`main()` matters more than its size suggests: FourierPartial.app depends on it
choosing a free port and announcing which one on stdout. If that contract
breaks, the app launches a server the launcher cannot find, waits 30 seconds,
and shows an error dialog — with the server running fine the whole time.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from backend.config import is_port_available, resolve_port
from backend.core.errors import ExportError, UnsupportedExportError
from backend.core.types import Track, TrackKind
from backend.export.wav_exporter import WavExporter


# =============================================================================
# Port resolution — the contract the .app launcher depends on
# =============================================================================


def test_free_port_is_reported_available():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert is_port_available("127.0.0.1", free_port)


def test_bound_port_is_reported_unavailable():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        assert not is_port_available("127.0.0.1", port)


def test_resolve_port_steps_past_a_busy_port():
    """The behaviour that makes running two copies at once work."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        busy = held.getsockname()[1]
        chosen = resolve_port("127.0.0.1", busy)
        assert chosen != busy
        assert chosen > busy


def test_resolve_port_gives_up_rather_than_looping_forever():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        busy = held.getsockname()[1]
        with pytest.raises(RuntimeError, match="No free port"):
            resolve_port("127.0.0.1", busy, max_attempts=1)


# =============================================================================
# main() — startup and the port announcement
# =============================================================================


def test_main_announces_the_port_it_actually_chose(monkeypatch, capsys):
    """The launcher greps stdout for this line. It is a real interface.

    Deliberately asserted on the exact prefix rather than "some URL appears":
    a well-meaning tidy-up that changed the format would break the .app in a
    way no other test would notice.
    """
    import backend.main as main_module

    started = {}

    def fake_run(app, host, port, log_level):
        started["host"] = host
        started["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr(main_module, "resolve_port", lambda host, port: 9999)

    main_module.main()

    printed = capsys.readouterr().out
    assert "FOURIER_PARTIAL_URL=http://" in printed
    assert ":9999" in printed
    assert started["port"] == 9999


def test_main_announces_the_fallback_port_not_the_preferred_one(monkeypatch, capsys):
    """When 8420 is busy the announced URL must be the port really in use.

    The launcher explicitly refuses to guess 8420, because the most likely
    squatter is another copy of this app — which answers /api/health happily
    and would send the user to the wrong server.
    """
    import backend.main as main_module

    monkeypatch.setattr("uvicorn.run", lambda **kwargs: None)
    monkeypatch.setattr("uvicorn.run", lambda app, host, port, log_level: None)
    monkeypatch.setattr(main_module, "resolve_port", lambda host, port: port + 7)

    main_module.main()

    printed = capsys.readouterr().out
    assert f":{main_module.settings.port + 7}" in printed
    assert f"FOURIER_PARTIAL_URL=http://{main_module.settings.host}:" in printed


def test_create_app_survives_a_missing_frontend(monkeypatch, caplog):
    """The API must still work if the UI directory is absent.

    A packaging mistake should degrade to "no interface" rather than "no
    server", so the error is diagnosable from the log rather than from a
    process that refuses to start.
    """
    from pathlib import Path

    import backend.main as main_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main_module, "_FRONTEND_DIR", Path("/nonexistent/frontend"))
    app = main_module.create_app()

    with caplog.at_level("ERROR"), TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
    assert any("frontend directory missing" in record.message for record in caplog.records)


# =============================================================================
# WavExporter failure paths
# =============================================================================


def test_unknown_wav_subtype_is_rejected_at_construction():
    """Fail at configuration time, not after the user waits for an analysis."""
    with pytest.raises(ExportError, match="unknown WAV subtype"):
        WavExporter(subtype="NOT_A_REAL_SUBTYPE")


def test_exporting_a_track_with_no_audio_is_refused():
    exporter = WavExporter()
    track = Track(0, TrackKind.PARTIAL, "silent")
    assert not exporter.supports(track)
    with pytest.raises(UnsupportedExportError):
        exporter.export(track, __import__("pathlib").Path("/tmp/never_written.wav"), 44_100)


def test_export_wraps_libsndfile_failures(tmp_path):
    """Third-party errors are wrapped so the API layer's one catch holds.

    An unwrapped OSError would sail past `except FourierPartialError` in the
    route handler and reach the client as a 500 with a stack trace.
    """
    exporter = WavExporter()
    track = Track(0, TrackKind.PARTIAL, "tone")
    track.attach_audio(np.zeros(128, dtype=np.float32))

    # A directory where a file should go: libsndfile cannot write this.
    destination = tmp_path / "blocked"
    destination.mkdir()

    with pytest.raises(ExportError):
        exporter.export(track, destination, 44_100)
