"""The mixdown endpoint — the invariant paying rent in the browser.

Playing everything never needed one buffer per track; it needed one buffer,
because the tracks sum to the original exactly. At ten minutes that is the
difference between ~106 MB in the tab and several gigabytes.

These tests pin the property that makes the shortcut legitimate: with nothing
excluded, the mixdown must be the upload, byte for byte. If that ever stops
being true the optimisation becomes a lie and playback stops matching the file.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.core.types import pitch_label
from backend.main import create_app
from tests.conftest import make_harmonic_tone


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


# A length that deliberately does NOT divide evenly into whole seconds.
#
# The first version of these tests used exactly 1.0 s — 44100 samples — and
# passed while the endpoint was broken for every real recording. The accumulator
# was sized as `int(duration_s * sample_rate)`, and at exactly 1.0 s that round
# trip is exact, so the off-by-one never appeared. On a real file it did:
# 941056 / 44100 * 44100 == 941055.9999999999, which truncates one sample short
# and makes the whole request raise. Every mute and solo returned HTTP 500.
#
# 0.483 s is 21300.3 samples — an awkward number on purpose. Fixtures that
# divide evenly are exactly the ones that hide arithmetic like this.
AWKWARD_DURATION_S = 0.483

# Sample counts for which `n / 44100 * 44100` does NOT come back as n in
# float64. These are not arbitrary awkward numbers — they are specifically the
# ones that break the arithmetic this endpoint used to do. Verified by
# test_the_float_round_trip_this_guards_really_does_lose_a_sample below.
FLOAT_HOSTILE_LENGTHS = (44_109, 44_127, 44_145, 88_218)


@pytest.fixture(scope="module")
def analysed(client):
    """One analysis of a known signal, reused by the tests below."""
    original = make_harmonic_tone(220.0, 6, duration_s=AWKWARD_DURATION_S)
    buffer = io.BytesIO()
    # FLOAT so the decoded upload is bit-identical to what we generated, which
    # is what lets the byte-identity assertion below be exact rather than close.
    sf.write(buffer, original, 44_100, format="WAV", subtype="FLOAT")

    response = client.post(
        "/api/analyze", files={"file": ("t.wav", buffer.getvalue(), "audio/wav")}
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    for _ in range(600):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert status["state"] == "complete", status

    summary = client.get(f"/api/jobs/{job_id}/tracks").json()
    return job_id, summary, original


def _wav(response) -> np.ndarray:
    audio, _ = sf.read(io.BytesIO(response.content), dtype="float64")
    return audio


# =============================================================================
# The core property
# =============================================================================


def test_mixdown_with_nothing_excluded_is_the_original(client, analysed):
    """Byte-identical, not merely close.

    This is what makes "play everything = one buffer" correct rather than an
    approximation the user might hear.
    """
    job_id, _, original = analysed
    response = client.get(f"/api/jobs/{job_id}/mixdown")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert np.array_equal(_wav(response), original.astype(np.float64))


def test_mixdown_excluding_everything_is_silence(client, analysed):
    """Removing every track must leave nothing, since they sum to the whole."""
    job_id, summary, _ = analysed
    ids = ",".join(str(t["track_id"]) for t in summary["tracks"])
    audio = _wav(client.get(f"/api/jobs/{job_id}/mixdown?exclude={ids}"))
    assert np.max(np.abs(audio)) < 1e-6


def test_excluding_a_track_removes_exactly_that_track(client, analysed):
    """mixdown(exclude=k) + track k == original.

    The arithmetic the browser relies on when a user mutes one row: it holds
    the rest as a single buffer plus the muted track, and unmuting must land
    back on the original rather than near it.
    """
    job_id, summary, original = analysed
    track_id = summary["tracks"][0]["track_id"]

    rest = _wav(client.get(f"/api/jobs/{job_id}/mixdown?exclude={track_id}"))
    single = _wav(client.get(f"/api/jobs/{job_id}/tracks/{track_id}/audio"))

    assert np.max(np.abs((rest + single) - original.astype(np.float64))) < 1e-6


def test_mixdown_is_one_buffer_not_many(client, analysed):
    """Sanity on the point of the endpoint: its size is one track's worth.

    If this ever returned something proportional to the track count, the
    memory problem it exists to solve would be back.
    """
    job_id, summary, _ = analysed
    mixdown_bytes = len(client.get(f"/api/jobs/{job_id}/mixdown").content)
    track_bytes = len(
        client.get(f"/api/jobs/{job_id}/tracks/{summary['tracks'][0]['track_id']}/audio").content
    )
    assert mixdown_bytes == pytest.approx(track_bytes, rel=0.01)
    assert len(summary["tracks"]) > 1, "fixture should produce several tracks"


# =============================================================================
# Input handling
# =============================================================================


def test_unknown_excluded_ids_are_ignored(client, analysed):
    """A stale id from a client that has moved on must not fail playback."""
    job_id, _, original = analysed
    audio = _wav(client.get(f"/api/jobs/{job_id}/mixdown?exclude=9999"))
    assert np.max(np.abs(audio - original.astype(np.float64))) < 1e-6


def test_blank_and_untidy_exclude_lists_are_tolerated(client, analysed):
    job_id, _, original = analysed
    for value in ("", " ", ",", " , ,"):
        audio = _wav(client.get(f"/api/jobs/{job_id}/mixdown?exclude={value}"))
        assert np.max(np.abs(audio - original.astype(np.float64))) < 1e-6


def test_non_numeric_exclude_is_rejected(client, analysed):
    job_id, _, _ = analysed
    response = client.get(f"/api/jobs/{job_id}/mixdown?exclude=abc")
    assert response.status_code == 400
    assert "whole numbers" in response.json()["detail"]


def test_mixdown_of_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope/mixdown").status_code == 404


def test_the_float_round_trip_this_guards_really_does_lose_a_sample():
    """Establish the hazard exists before testing that we survive it.

    `len(audio) / sample_rate * sample_rate` is not always `len(audio)` in
    float64. It fails for about 8% of lengths — 163,655 of the first two
    million — so roughly one real recording in twelve. The file that exposed
    this in production was 941,056 samples, and it is in that set.

    Without this test the sweep below could quietly stop reproducing anything
    (say, if someone "tidied" the lengths into round numbers) and keep passing.
    """
    sample_rate = 44_100
    for n in FLOAT_HOSTILE_LENGTHS:
        assert int((n / sample_rate) * sample_rate) != n, f"{n} no longer round-trips wrong"
    assert int((941_056 / sample_rate) * sample_rate) == 941_055


@pytest.mark.parametrize("n_samples", FLOAT_HOSTILE_LENGTHS)
def test_exclude_works_at_lengths_that_break_the_float_round_trip(client, n_samples):
    """The regression that shipped: exclude 500'd on ~1 file in 12.

    Sizing the accumulator by recomputing samples from a float duration loses a
    sample whenever the round trip does not land exactly, and `mixed += audio`
    then raises ValueError. Every mute and solo on such a file returned HTTP
    500, and the client reverted the intent it could not honour — so the user
    saw their buttons unselecting themselves.

    These lengths are chosen precisely because they FAIL that round trip. Note
    the array is built to an exact sample count rather than from a duration:
    going through `duration_s` would re-introduce the very rounding under test.
    """
    t = np.arange(n_samples) / 44_100
    original = np.zeros(n_samples, dtype=np.float64)
    for harmonic in range(1, 5):
        original += np.sin(2 * np.pi * 220.0 * harmonic * t) / harmonic
    original = (0.6 * original / np.max(np.abs(original))).astype(np.float32)
    assert len(original) == n_samples

    buffer = io.BytesIO()
    sf.write(buffer, original, 44_100, format="WAV", subtype="FLOAT")

    response = client.post(
        "/api/analyze", files={"file": ("t.wav", buffer.getvalue(), "audio/wav")}
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    for _ in range(600):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert status["state"] == "complete", status

    summary = client.get(f"/api/jobs/{job_id}/tracks").json()
    ids = [t["track_id"] for t in summary["tracks"]]

    # Excluding one track must succeed, and must still add back up.
    single = client.get(f"/api/jobs/{job_id}/mixdown?exclude={ids[0]}")
    assert single.status_code == 200, f"{duration_s}s: {single.status_code} {single.text[:200]}"

    rest = _wav(single)
    first = _wav(client.get(f"/api/jobs/{job_id}/tracks/{ids[0]}/audio"))
    assert len(rest) == len(original), f"{duration_s}s: length changed"
    assert np.max(np.abs((rest + first) - original.astype(np.float64))) < 1e-6

    # And excluding everything must be silence, not a crash.
    everything = client.get(f"/api/jobs/{job_id}/mixdown?exclude={','.join(map(str, ids))}")
    assert everything.status_code == 200
    assert np.max(np.abs(_wav(everything))) < 1e-6


# =============================================================================
# Pitch grouping key
# =============================================================================


def test_tracks_carry_a_bare_pitch_label(client, analysed):
    _, summary, _ = analysed
    pitched = [t for t in summary["tracks"] if t["kind"] != "residual"]
    assert pitched, "expected at least one pitched track"
    for track in pitched:
        assert track["pitch"], "pitched tracks need a grouping key"
        # The grouping key must be the note alone — no cents, or repeats of the
        # same note would land in different groups.
        assert "c" not in track["pitch"][1:], track["pitch"]


def test_residual_has_no_pitch(client, analysed):
    _, summary, _ = analysed
    residual = next(t for t in summary["tracks"] if t["kind"] == "residual")
    assert residual["pitch"] == ""


@pytest.mark.parametrize(
    "hz,expected",
    [(440.0, "A4"), (261.63, "C4"), (220.0, "A3"), (880.0, "A5"), (443.0, "A4"), (437.0, "A4")],
)
def test_pitch_label_ignores_cent_deviation(hz, expected):
    """443 and 437 Hz are both A4 — a few cents apart is the same note."""
    assert pitch_label(hz) == expected


def test_pitch_label_handles_nonsense_frequency():
    assert pitch_label(0.0) == "unpitched"
    assert pitch_label(-5.0) == "unpitched"
