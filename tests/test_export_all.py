"""Bulk export — every track in one download.

The thing under test is mostly honesty. A bulk export is where files get
dropped quietly: the residual cannot become MIDI, and a single multi-channel
MIDI file has fifteen usable channels where an analysis routinely has more
tracks. An archive containing fewer files than the user has rows on screen
looks complete while being wrong, so most of what is asserted here is that the
omissions are declared — in the manifest, and in the response headers.
"""

from __future__ import annotations

import io
import time
import zipfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.core.errors import ExportError
from backend.core.types import AnalysisParams, AnalysisResult, Track, TrackKind
from backend.export.archive import (
    ArchiveFormat,
    build_archive,
    estimate_bytes,
)
from backend.main import create_app
from tests.conftest import make_harmonic_tone

SR = 44_100


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def job(client):
    """One analysis with several pitched tracks plus a residual."""
    audio = make_harmonic_tone(196.0, 6, duration_s=0.9)
    audio = (audio + 0.05 * np.random.default_rng(4).standard_normal(len(audio))).astype(
        np.float32
    )
    buffer = io.BytesIO()
    sf.write(buffer, audio, SR, format="WAV", subtype="FLOAT")

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

    return job_id, client.get(f"/api/jobs/{job_id}/tracks").json()


def _zip(response) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(response.content))


# =============================================================================
# WAV: everything, including the residual
# =============================================================================


def test_wav_export_contains_one_file_per_track(client, job):
    job_id, summary = job
    response = client.get(f"/api/jobs/{job_id}/export?format=wav")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    archive = _zip(response)
    wavs = [n for n in archive.namelist() if n.endswith(".wav")]
    assert len(wavs) == len(summary["tracks"]), "every track should export as WAV"
    assert response.headers["X-Export-Skipped"] == "0"


def test_wav_export_includes_the_residual(client, job):
    """The residual is audio like any other track — only MIDI can't take it."""
    job_id, summary = job
    residual = next(t for t in summary["tracks"] if t["kind"] == "residual")
    names = _zip(client.get(f"/api/jobs/{job_id}/export?format=wav")).namelist()
    assert any(f"track{residual['track_id']:03d}" in n for n in names)


def test_exported_wavs_are_real_audio(client, job):
    job_id, _ = job
    archive = _zip(client.get(f"/api/jobs/{job_id}/export?format=wav"))
    name = next(n for n in archive.namelist() if n.endswith(".wav"))

    audio, rate = sf.read(io.BytesIO(archive.read(name)), dtype="float32")
    assert rate == SR
    assert len(audio) > 0


def test_exported_wavs_still_sum_to_the_original(client, job):
    """The claim the manifest makes, actually checked.

    24-bit rounds each sample, so this is 'inaudibly close' rather than
    bit-exact — which is precisely what the manifest says, and why it says it.
    """
    job_id, _ = job
    archive = _zip(client.get(f"/api/jobs/{job_id}/export?format=wav"))

    mixed = None
    for name in (n for n in archive.namelist() if n.endswith(".wav")):
        audio, _ = sf.read(io.BytesIO(archive.read(name)), dtype="float64")
        mixed = audio if mixed is None else mixed + audio

    original, _ = sf.read(
        io.BytesIO(client.get(f"/api/jobs/{job_id}/mixdown").content), dtype="float64"
    )
    # 24-bit resolution is ~1.2e-7; allow a few LSBs across a sum of tracks.
    assert np.max(np.abs(mixed - original)) < 1e-5


# =============================================================================
# MIDI: the residual cannot come, and must be declared missing
# =============================================================================


def test_midi_export_skips_the_residual_and_says_so(client, job):
    job_id, summary = job
    response = client.get(f"/api/jobs/{job_id}/export?format=mid")

    assert response.status_code == 200
    archive = _zip(response)
    mids = [n for n in archive.namelist() if n.endswith(".mid")]

    exportable = [t for t in summary["tracks"] if t["exportable_as_midi"]]
    assert len(mids) == len(exportable)
    assert int(response.headers["X-Export-Skipped"]) >= 1, "the residual must be reported skipped"

    manifest = archive.read("MANIFEST.txt").decode()
    assert "NOT INCLUDED" in manifest
    assert "residual" in manifest.lower()


def test_exported_midi_files_are_real_midi(client, job):
    job_id, _ = job
    archive = _zip(client.get(f"/api/jobs/{job_id}/export?format=mid"))
    name = next(n for n in archive.namelist() if n.endswith(".mid"))
    assert archive.read(name)[:4] == b"MThd"


def test_single_midi_returns_one_file_not_an_archive(client, job):
    job_id, _ = job
    response = client.get(f"/api/jobs/{job_id}/export?format=mid-single")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/midi"
    assert response.content[:4] == b"MThd"


# =============================================================================
# The manifest — the part that stops a partial export looking complete
# =============================================================================


def test_every_archive_carries_a_manifest(client, job):
    job_id, _ = job
    for fmt in ("wav", "mid"):
        archive = _zip(client.get(f"/api/jobs/{job_id}/export?format={fmt}"))
        assert "MANIFEST.txt" in archive.namelist(), f"{fmt} archive has no manifest"


def test_manifest_lists_every_included_file(client, job):
    job_id, _ = job
    archive = _zip(client.get(f"/api/jobs/{job_id}/export?format=wav"))
    manifest = archive.read("MANIFEST.txt").decode()
    for name in (n for n in archive.namelist() if n.endswith(".wav")):
        assert name in manifest, f"{name} is in the archive but not in the manifest"


def test_wav_manifest_states_the_reconstruction_property(client, job):
    """Someone unpacking forty WAVs cannot guess that they sum to the original."""
    job_id, _ = job
    manifest = _zip(client.get(f"/api/jobs/{job_id}/export?format=wav")).read(
        "MANIFEST.txt"
    ).decode()
    assert "SUM BACK TO THE ORIGINAL" in manifest
    # ...and is honest that 24-bit makes it inexact.
    assert "24-bit" in manifest


def test_midi_manifest_warns_about_the_bend_range(client, job):
    """The classic silent failure: an instrument ignoring RPN 0 flattens glides."""
    job_id, _ = job
    manifest = _zip(client.get(f"/api/jobs/{job_id}/export?format=mid")).read(
        "MANIFEST.txt"
    ).decode()
    assert "RPN 0" in manifest
    assert "lossy" in manifest.lower()


# =============================================================================
# Filenames
# =============================================================================


def test_filenames_are_unique_even_when_labels_collide(client, job):
    """Two occurrences of one note produce identical labels.

    Without the track-id prefix they would collide inside the zip, and the
    archive would silently contain fewer files than tracks — the exact failure
    this feature exists to avoid.
    """
    job_id, _ = job
    names = [n for n in _zip(client.get(f"/api/jobs/{job_id}/export?format=wav")).namelist()]
    assert len(names) == len(set(names))


def test_filenames_have_no_path_separators_or_oddities(client, job):
    job_id, _ = job
    for name in _zip(client.get(f"/api/jobs/{job_id}/export?format=wav")).namelist():
        assert "/" not in name and "\\" not in name, f"{name} could escape the archive"
        assert ".." not in name


# =============================================================================
# Estimates, errors, and cleanup
# =============================================================================


def test_estimate_reports_every_format(client, job):
    job_id, summary = job
    data = client.get(f"/api/jobs/{job_id}/export/estimate").json()

    assert data["track_count"] == len(summary["tracks"])
    assert set(data["formats"]) == {"wav", "wav-tones", "mid", "mid-single"}
    assert data["formats"]["wav"]["bytes"] > data["formats"]["mid"]["bytes"]
    assert data["formats"]["wav"]["files"] == len(summary["tracks"])
    # Tones-only is exactly one file smaller: the residual.
    assert data["formats"]["wav-tones"]["files"] == len(summary["tracks"]) - 1
    assert data["formats"]["wav-tones"]["bytes"] < data["formats"]["wav"]["bytes"]


# =============================================================================
# WAV without the residual — the stems case
# =============================================================================


def test_wav_tones_excludes_the_residual(client, job):
    """The whole point of the format: stems, without the noise track."""
    job_id, summary = job
    response = client.get(f"/api/jobs/{job_id}/export?format=wav-tones")

    assert response.status_code == 200
    archive = _zip(response)
    wavs = [n for n in archive.namelist() if n.endswith(".wav")]

    assert len(wavs) == len(summary["tracks"]) - 1
    residual = next(t for t in summary["tracks"] if t["kind"] == "residual")
    assert not any(f"track{residual['track_id']:03d}" in n for n in wavs)
    assert response.headers["X-Export-Skipped"] == "1"


def test_wav_tones_says_the_omission_was_deliberate(client, job):
    """A missing track needs a reason, or it reads as a lost track.

    "Left out on purpose" and "failed to write" produce the same-looking
    archive, so the manifest has to distinguish them.
    """
    job_id, _ = job
    manifest = _zip(client.get(f"/api/jobs/{job_id}/export?format=wav-tones")).read(
        "MANIFEST.txt"
    ).decode()

    assert "NOT INCLUDED" in manifest
    assert "on purpose" in manifest.lower()
    # And it must say how to get it, rather than just that it is absent.
    assert "wav" in manifest.lower()


def test_wav_tones_files_are_playable_audio(client, job):
    job_id, _ = job
    archive = _zip(client.get(f"/api/jobs/{job_id}/export?format=wav-tones"))
    name = next(n for n in archive.namelist() if n.endswith(".wav"))
    audio, rate = sf.read(io.BytesIO(archive.read(name)), dtype="float32")
    assert rate == SR and len(audio) > 0


def test_wav_tones_does_not_sum_to_the_original(client, job):
    """It cannot, and the difference should be the residual exactly.

    Worth pinning: someone who exports tones-only and finds the sum quiet or
    thin has not hit a bug — they removed the noise on purpose, and this is the
    arithmetic that says by how much.
    """
    job_id, summary = job
    archive = _zip(client.get(f"/api/jobs/{job_id}/export?format=wav-tones"))

    mixed = None
    for name in (n for n in archive.namelist() if n.endswith(".wav")):
        audio, _ = sf.read(io.BytesIO(archive.read(name)), dtype="float64")
        mixed = audio if mixed is None else mixed + audio

    residual_id = next(t["track_id"] for t in summary["tracks"] if t["kind"] == "residual")
    residual, _ = sf.read(
        io.BytesIO(client.get(f"/api/jobs/{job_id}/tracks/{residual_id}/audio").content),
        dtype="float64",
    )
    original, _ = sf.read(
        io.BytesIO(client.get(f"/api/jobs/{job_id}/mixdown").content), dtype="float64"
    )

    assert np.max(np.abs((mixed + residual) - original)) < 1e-5


def test_estimate_is_not_wildly_wrong(client, job):
    """A warning is only useful if the number is roughly right."""
    job_id, _ = job
    predicted = client.get(f"/api/jobs/{job_id}/export/estimate").json()["formats"]["wav"]["bytes"]
    actual = len(client.get(f"/api/jobs/{job_id}/export?format=wav").content)
    # Deliberately an over-estimate; it must not be an under-estimate.
    assert predicted >= actual * 0.9
    assert predicted < actual * 3


def test_unknown_format_is_rejected(client, job):
    job_id, _ = job
    assert client.get(f"/api/jobs/{job_id}/export?format=flac").status_code == 422
    assert client.get(f"/api/jobs/{job_id}/export?format=../etc").status_code == 422


def test_export_of_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope/export?format=wav").status_code == 404
    assert client.get("/api/jobs/nope/export/estimate").status_code == 404


def test_midi_export_of_a_residual_only_analysis_fails_cleanly(tmp_path):
    """An empty archive would be worse than an error.

    A clip too short to analyse produces nothing but a residual, so a MIDI
    export of it has no possible contents. Handing back a zip containing only a
    manifest saying "nothing here" is a worse answer than saying so.
    """
    audio = np.zeros(1024, dtype=np.float32)
    residual = Track(0, TrackKind.RESIDUAL, "Residual", ())
    residual.attach_audio(audio)
    result = AnalysisResult(SR, len(audio) / SR, [residual], AnalysisParams(), audio)

    with pytest.raises(ExportError):
        build_archive(result, ArchiveFormat.MIDI, tmp_path / "x.zip", tmp_path / "parts")

    # ...but WAV works, because the residual is perfectly good audio.
    report = build_archive(result, ArchiveFormat.WAV, tmp_path / "y.zip", tmp_path / "parts")
    assert len(report.included) == 1


def test_build_leaves_no_scratch_files_behind(tmp_path):
    """Forty tracks of ten-minute audio must not leave 3 GB of parts on disk."""
    audio = make_harmonic_tone(220.0, 4, duration_s=0.4)
    tracks = []
    for i in range(3):
        t = Track(i, TrackKind.PARTIAL, f"note{i}", ())
        t.attach_audio(audio)
        tracks.append(t)
    result = AnalysisResult(SR, len(audio) / SR, tracks, AnalysisParams(), audio * 3)

    parts = tmp_path / "parts"
    build_archive(result, ArchiveFormat.WAV, tmp_path / "a.zip", parts)

    leftovers = list(parts.glob("*"))
    assert leftovers == [], f"scratch files survived: {leftovers}"


def test_estimate_scales_with_duration_and_track_count():
    """The estimate has to track reality or the warning is noise."""
    short = make_harmonic_tone(220.0, 2, duration_s=0.2)
    long_audio = make_harmonic_tone(220.0, 2, duration_s=0.8)

    def result_for(audio, n_tracks):
        tracks = []
        for i in range(n_tracks):
            t = Track(i, TrackKind.PARTIAL, f"n{i}", ())
            t.attach_audio(audio)
            tracks.append(t)
        return AnalysisResult(SR, len(audio) / SR, tracks, AnalysisParams(), audio)

    a = estimate_bytes(result_for(short, 2), ArchiveFormat.WAV)
    b = estimate_bytes(result_for(long_audio, 2), ArchiveFormat.WAV)
    c = estimate_bytes(result_for(short, 4), ArchiveFormat.WAV)

    assert b == pytest.approx(a * 4, rel=0.05), "should scale with duration"
    assert c == pytest.approx(a * 2, rel=0.05), "should scale with track count"
