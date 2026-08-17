"""Adjustable analysis settings, and re-analysing without re-uploading.

The property that matters most here is the one that is easiest to get wrong:
**only changed settings may be sent**. The server picks cheaper settings for
long audio because at phrase settings a ten-minute file needs ~6.8 GB. If a
client sent every value it holds, the phrase defaults would all count as
explicit overrides and would silently drag a long recording back to settings
that cannot analyse it — the user would have moved one slider and broken the
file. Several tests below exist purely to pin that.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.api.settings_schema import GROUPS, SETTABLE, clamp, schema
from backend.core.types import AnalysisParams
from backend.main import create_app

SR = 44_100


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _audio(duration_s: float = 1.2) -> bytes:
    """Three plucked notes plus a little noise — enough to have structure."""
    n = int(duration_s * SR)
    y = np.zeros(n)
    for onset in np.arange(0, duration_s - 0.35, 0.4):
        i0 = int(onset * SR)
        i1 = min(n, i0 + int(0.35 * SR))
        t = np.arange(i1 - i0) / SR
        for k in range(1, 7):
            y[i0:i1] += np.exp(-4 * t) * np.sin(2 * np.pi * 220 * k * t) / k
    y += 0.03 * np.random.default_rng(3).standard_normal(n)
    y = (0.7 * y / np.max(np.abs(y))).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, y, SR, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _analyse(client, settings=None, duration_s=1.2):
    files = {"file": ("a.wav", _audio(duration_s), "audio/wav")}
    data = {"settings": __import__("json").dumps(settings)} if settings is not None else None
    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    for _ in range(900):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert status["state"] == "complete", status
    return job_id, client.get(f"/api/jobs/{job_id}/tracks").json()


# =============================================================================
# The schema — the single source of truth the UI builds itself from
# =============================================================================


def test_schema_is_served(client):
    data = client.get("/api/settings/schema").json()
    assert data["groups"]
    names = [s["name"] for g in data["groups"] for s in g["settings"]]
    assert len(names) == len(set(names)), "a setting is declared twice"


def test_every_declared_setting_is_a_real_field():
    """A schema entry naming a field that does not exist would be a dead slider."""
    fields = {f for f in AnalysisParams().__dataclass_fields__}
    for name in SETTABLE:
        assert name in fields, f"{name} is offered but is not an AnalysisParams field"


def test_every_default_matches_the_dataclass():
    """The whole reason the schema is served rather than hardcoded in JS.

    If these drift, the panel's "reset to defaults" restores values the
    analyser no longer uses, and nothing complains.
    """
    params = AnalysisParams()
    for group in schema()["groups"]:
        for setting in group["settings"]:
            assert setting["default"] == getattr(params, setting["name"]), setting["name"]


def test_every_default_lies_inside_its_own_range():
    for group in schema()["groups"]:
        for s in group["settings"]:
            assert s["min"] <= s["default"] <= s["max"], f"{s['name']} default is out of range"


def test_every_setting_explains_itself():
    """A slider with no explanation is a slider nobody will touch."""
    for group in GROUPS:
        for setting in group.settings:
            assert setting.help.strip(), f"{setting.name} has no help"
            assert setting.higher.strip() and setting.lower.strip(), (
                f"{setting.name} does not say what moving it does"
            )


def test_clamp_holds_values_inside_range():
    assert clamp("peak_threshold_db", -500) == -90
    assert clamp("peak_threshold_db", 0) == -20
    assert clamp("max_tracks", 99999) == 300
    # An unknown name passes through rather than raising; it is ignored later.
    assert clamp("not_a_setting", 42) == 42


# =============================================================================
# Settings actually reach the analysis
# =============================================================================


def test_threshold_changes_how_many_partials_are_found(client):
    """The most direct sensitivity dial, measured where it actually shows.

    Track COUNT is the wrong thing to assert on: grouping consolidates, so a
    clean recording can yield the same four tracks from very different numbers
    of partials. The partial count is where the setting is visible.
    """
    _, strict = _analyse(client, {"peak_threshold_db": -35})
    _, loose = _analyse(client, {"peak_threshold_db": -75})

    strict_partials = sum(t["partial_count"] for t in strict["tracks"])
    loose_partials = sum(t["partial_count"] for t in loose["tracks"])
    assert loose_partials > strict_partials, (
        f"a lower threshold should find more: {loose_partials} vs {strict_partials}"
    )


def test_minimum_duration_discards_short_partials(client):
    _, short = _analyse(client, {"min_partial_duration_ms": 20})
    _, long_only = _analyse(client, {"min_partial_duration_ms": 300})
    assert sum(t["partial_count"] for t in long_only["tracks"]) < sum(
        t["partial_count"] for t in short["tracks"]
    )


def test_energy_cull_changes_how_many_tracks_are_shown(client):
    _, culled = _analyse(client, {"min_track_energy_share": 0.05})
    _, everything = _analyse(client, {"min_track_energy_share": 0.0})
    assert len(everything["tracks"]) > len(culled["tracks"])


def test_max_tracks_is_a_ceiling(client):
    _, capped = _analyse(client, {"min_track_energy_share": 0.0, "max_tracks": 5})
    # Plus the residual, which is added after the ceiling applies.
    assert len(capped["tracks"]) <= 6


def test_settings_do_not_break_the_invariant(client):
    """Whatever the settings, the tracks must still sum to the original."""
    for override in (
        {"peak_threshold_db": -85},
        {"window_size": 1024},
        {"min_partial_duration_ms": 300},
        {"min_track_energy_share": 0.0},
        {"max_freq_drift_cents": 250},
    ):
        _, summary = _analyse(client, override)
        assert summary["reconstruction_exact"] is True, override


# =============================================================================
# THE IMPORTANT ONE: untouched settings must not override the duration tier
# =============================================================================


def test_omitted_settings_leave_the_duration_tier_intact():
    """Sending nothing must give exactly the tier the server would have chosen.

    This is what lets the panel exist at all. If a client sent its phrase
    defaults for a ten-minute file, every one would count as an override and
    the file would be analysed at settings that need gigabytes.
    """
    from backend.api.routes import _params_for

    for duration in (10.0, 120.0, 600.0):
        assert _params_for(duration, None) == AnalysisParams.for_duration(duration)
        assert _params_for(duration, {}) == AnalysisParams.for_duration(duration)
        from backend.api.routes import _clean_settings

        assert _params_for(duration, _clean_settings({})) == AnalysisParams.for_duration(
            duration
        )


def test_an_override_wins_but_only_for_itself():
    """One changed setting must not drag the rest back to phrase defaults."""
    from backend.api.routes import _params_for

    from backend.api.routes import _clean_settings

    tier = AnalysisParams.for_duration(600.0)
    tuned = _params_for(600.0, _clean_settings({"peak_threshold_db": -70}))

    assert tuned.peak_threshold_db == -70
    # Everything else still comes from the long-form tier.
    assert tuned.hop_size == tier.hop_size
    assert tuned.zero_pad_factor == tier.zero_pad_factor
    assert tuned.max_tracks == tier.max_tracks


def test_out_of_range_values_are_clamped_not_rejected():
    """A slider is not a validation boundary; the server holds the line.

    Cleaning happens at request time and tiering in the worker, so this
    composes the two exactly as the route does.
    """
    from backend.api.routes import _clean_settings, _params_for

    params = _params_for(
        10.0, _clean_settings({"peak_threshold_db": -9999, "max_tracks": 99999})
    )
    assert params.peak_threshold_db == -90
    assert params.max_tracks == 300


def test_unknown_settings_are_ignored():
    """A stale field from an older client must not fail the analysis."""
    from backend.api.routes import _clean_settings, _params_for

    cleaned = _clean_settings({"not_a_real_setting": 5})
    assert cleaned == {}
    assert _params_for(10.0, cleaned) == AnalysisParams.for_duration(10.0)


def test_integer_fields_stay_integers():
    """A float window size would carry the wrong type into sample arithmetic."""
    from backend.api.routes import _clean_settings, _params_for

    params = _params_for(10.0, _clean_settings({"window_size": 2048.0, "max_tracks": 12.7}))
    assert isinstance(params.window_size, int) and params.window_size == 2048
    assert isinstance(params.max_tracks, int) and params.max_tracks == 13


def test_impossible_combinations_are_refused(client):
    """Per-field ranges cannot catch a hop larger than its window."""
    response = client.post(
        "/api/analyze",
        files={"file": ("a.wav", _audio(0.5), "audio/wav")},
        data={"settings": '{"window_size": 512, "hop_size": 4096}'},
    )
    assert response.status_code == 400
    assert "hop_size" in response.json()["detail"]


def test_malformed_settings_are_rejected_clearly(client):
    for payload, expect in [("not json", "JSON"), ('["a"]', "object")]:
        response = client.post(
            "/api/analyze",
            files={"file": ("a.wav", _audio(0.5), "audio/wav")},
            data={"settings": payload},
        )
        assert response.status_code == 400
        assert expect.lower() in response.json()["detail"].lower()


def test_non_numeric_setting_value_is_rejected(client):
    response = client.post(
        "/api/analyze",
        files={"file": ("a.wav", _audio(0.5), "audio/wav")},
        data={"settings": '{"peak_threshold_db": "loud"}'},
    )
    assert response.status_code == 400


# =============================================================================
# Re-analysis without re-uploading
# =============================================================================


def test_reanalyze_uses_the_stored_audio(client):
    """No re-upload: the completed result still holds the decoded samples."""
    job_id, first = _analyse(client)

    response = client.post(
        f"/api/jobs/{job_id}/reanalyze", json={"settings": {"min_track_energy_share": 0.0}}
    )
    assert response.status_code == 200
    new_job = response.json()["job_id"]
    assert new_job != job_id, "re-analysis must be a new job"

    for _ in range(900):
        status = client.get(f"/api/jobs/{new_job}").json()
        if status["state"] in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert status["state"] == "complete", status

    second = client.get(f"/api/jobs/{new_job}/tracks").json()
    assert len(second["tracks"]) > len(first["tracks"])
    assert second["duration_s"] == pytest.approx(first["duration_s"], abs=0.01)
    assert second["reconstruction_exact"] is True


def test_reanalyze_leaves_the_original_alone(client):
    """Backing out of a change should not mean rebuilding the old result."""
    job_id, first = _analyse(client)
    response = client.post(
        f"/api/jobs/{job_id}/reanalyze", json={"settings": {"peak_threshold_db": -35}}
    )
    assert response.status_code == 200

    still_there = client.get(f"/api/jobs/{job_id}/tracks")
    assert still_there.status_code == 200
    assert len(still_there.json()["tracks"]) == len(first["tracks"])


def test_reanalyze_of_unknown_job_is_404(client):
    assert client.post("/api/jobs/nope/reanalyze", json={"settings": {}}).status_code == 404


def test_reanalyze_with_no_settings_is_allowed(client):
    """Harmless, and simpler than special-casing an empty payload."""
    job_id, _ = _analyse(client)
    assert client.post(f"/api/jobs/{job_id}/reanalyze", json={"settings": {}}).status_code == 200


# =============================================================================
# Settings that clash with the duration tier
# =============================================================================


def test_a_window_smaller_than_the_tier_hop_is_a_real_clash():
    """The one validation that cannot happen at request time.

    Per-field ranges cannot catch it and neither can checking against the
    defaults, because it only exists once the duration is known: a 1024-sample
    window is perfectly legal, and so is the long-form tier's 2048-sample hop,
    but together they are not. The audio has to be decoded before this can be
    detected at all.
    """
    from backend.api.routes import _clean_settings, _params_for

    cleaned = _clean_settings({"window_size": 1024})
    # Fine for a phrase, where the tier hop is 512.
    assert _params_for(10.0, cleaned).window_size == 1024
    # Not fine for ten minutes, where the tier hop is 2048.
    with pytest.raises(ValueError, match="hop_size"):
        _params_for(600.0, cleaned)


def test_a_clash_fails_the_job_with_something_actionable(client, monkeypatch):
    """It must reach the user as a sentence, not as a crash.

    Forced rather than waited for: reproducing it honestly needs ten minutes of
    audio, and the branch under test is the reporting, not the arithmetic.
    """
    from backend.api import routes

    job_id, _ = _analyse(client)

    def clashing(duration_s, overrides):
        raise ValueError("hop_size must be in (0, window_size], got 2048")

    monkeypatch.setattr(routes, "_params_for", clashing)

    response = client.post(
        f"/api/jobs/{job_id}/reanalyze", json={"settings": {"window_size": 1024}}
    )
    assert response.status_code == 200  # the job is accepted, then fails
    new_job = response.json()["job_id"]

    for _ in range(400):
        status = client.get(f"/api/jobs/{new_job}").json()
        if status["state"] in ("complete", "failed"):
            break
        time.sleep(0.05)

    assert status["state"] == "failed"
    assert "do not work together" in status["message"]
    assert "hop_size" in status["message"]
    # Still user-safe: no traceback, no paths.
    assert "Traceback" not in status["message"]
    assert "/Users/" not in status["message"]


def test_a_crashing_reanalysis_reports_safely(client, monkeypatch):
    """An unexpected failure must not leak internals into the browser."""
    from backend.api import routes

    job_id, _ = _analyse(client)

    class Exploding:
        def analyze(self, *a, **k):
            raise RuntimeError("/Users/secret/path blew up")

        def verify(self, *a, **k):
            raise RuntimeError("unreachable")

    monkeypatch.setattr(routes, "_pipeline", Exploding())

    new_job = client.post(f"/api/jobs/{job_id}/reanalyze", json={"settings": {}}).json()["job_id"]
    for _ in range(400):
        status = client.get(f"/api/jobs/{new_job}").json()
        if status["state"] in ("complete", "failed"):
            break
        time.sleep(0.05)

    assert status["state"] == "failed"
    assert "/Users/" not in status["message"]
    assert "RuntimeError" not in status["message"]


def test_boolean_is_not_accepted_as_a_number():
    """True is an int in Python, and would silently become 1."""
    from backend.api.routes import _clean_settings
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        _clean_settings({"max_tracks": True})
