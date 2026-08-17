"""Record out: bounce the tracks that are currently audible.

WHAT IS BEING PROVEN
--------------------
One thing, and it is arithmetic rather than opinion: the file that comes back
is the sum of exactly the tracks that were asked for. Not approximately, not
"sounds right" — sample-for-sample, to the precision the format allows.

That matters because record-out is the feature with no visible failure mode.
A missing track in a zip is obvious the moment you open it. A missing track in
a bounce is a mix that is merely quieter than it should be, and the user finds
out in a session, days later, with no way to tell whether the app or their ears
were wrong.

The reconstruction invariant makes the strongest version of this testable:
because `residual := original - sum(partials)`, summing *every* track must give
back the original. So the test below does not compare the bounce to a
separately-computed expectation that could share a bug with the code — it
compares it to the input file.
"""

from __future__ import annotations

import io
import time
import wave

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.api.routes import _jobs
from backend.export.archive import ArchiveFormat
from backend.main import create_app
from tests.conftest import make_harmonic_tone

SR = 44_100


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def analysed_job(client):
    """One completed analysis, returned with the server-side result object.

    The result itself, not the JSON summary: every assertion here compares
    audio to audio, and the comparison is only meaningful against the same
    arrays the exporter is summing.
    """
    # A chord, not a single tone. A subset is only a meaningful test if there
    # is something to leave out: one harmonic tone groups into one pitched
    # track plus a residual, and "two of two tracks" is not a subset at all.
    # Three fundamentals a fifth and an octave apart group into three.
    audio = sum(make_harmonic_tone(f, 5, duration_s=0.9) for f in (196.0, 294.0, 392.0))
    audio = (
        audio / 3.0 + 0.02 * np.random.default_rng(11).standard_normal(len(audio))
    ).astype(np.float32)
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

    return job_id, _jobs.get(job_id)


def _read_wav(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV response body to float samples in [-1, 1]."""
    data, rate = sf.read(io.BytesIO(payload), dtype="float64", always_2d=False)
    return np.asarray(data), rate


class TestSelection:
    """The `only=` parameter, which every format honours."""

    def test_bounce_of_everything_reconstructs_the_original(self, client, analysed_job):
        """The whole point, stated as strongly as it can be stated.

        Every track selected must sum back to the input. If any track were
        dropped, double-counted, or mis-scaled, this fails — and it cannot pass
        by accident, because the reference is the user's own file.
        """
        job_id, result = analysed_job
        ids = ",".join(str(t.track_id) for t in result.tracks)

        response = client.get(
            f"/api/jobs/{job_id}/export", params={"format": "wav-mix", "only": ids}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")

        mixed, rate = _read_wav(response.content)
        assert rate == result.sample_rate
        assert mixed.shape[0] == result.original.shape[0]

        # 24-bit PCM quantises at 2^-23; anything above that is a real error in
        # the summing, not a rounding artefact of the format.
        assert np.max(np.abs(mixed - result.original)) < 1e-4

    def test_bounce_of_one_track_is_that_track(self, client, analysed_job):
        """Soloing a single track and recording out must give that track back."""
        job_id, result = analysed_job
        track = next(t for t in result.tracks if t.audio is not None)

        response = client.get(
            f"/api/jobs/{job_id}/export",
            params={"format": "wav-mix", "only": str(track.track_id)},
        )
        assert response.status_code == 200

        mixed, _ = _read_wav(response.content)
        assert np.max(np.abs(mixed - track.audio)) < 1e-4

    def test_bounce_of_a_subset_is_the_sum_of_that_subset(self, client, analysed_job):
        """A partial selection sums only what was named.

        The failure this catches is an off-by-one in the id filter: selecting
        two tracks and receiving three is inaudible unless you go looking.
        """
        job_id, result = analysed_job
        audible = [t for t in result.tracks if t.audio is not None]
        if len(audible) < 3:
            pytest.skip("fixture did not produce enough tracks to leave one out")
        # All but one, so the selection is always a *proper* subset. Taking a
        # fixed count instead would silently become "everything" the moment the
        # analyser grouped the fixture differently -- which is exactly what
        # happened the first time this was written.
        chosen = audible[:-1]

        ids = ",".join(str(t.track_id) for t in chosen)
        response = client.get(
            f"/api/jobs/{job_id}/export", params={"format": "wav-mix", "only": ids}
        )
        assert response.status_code == 200

        mixed, _ = _read_wav(response.content)
        expected = np.sum([t.audio for t in chosen], axis=0)
        assert np.max(np.abs(mixed - expected)) < 1e-4

        # And it is genuinely a subset -- not silently the whole mix.
        assert np.max(np.abs(mixed - result.original)) > 1e-4

    def test_selection_narrows_the_zip_formats_too(self, client, analysed_job):
        """`only=` is not exclusive to wav-mix.

        "Record out the soloed tracks as separate files" is the same request
        with a different container, so it goes through the same parameter
        rather than a parallel code path that could drift from it.
        """
        import zipfile

        job_id, result = analysed_job
        chosen = [t for t in result.tracks if t.audio is not None][:2]
        ids = ",".join(str(t.track_id) for t in chosen)

        response = client.get(
            f"/api/jobs/{job_id}/export", params={"format": "wav", "only": ids}
        )
        assert response.status_code == 200

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            wavs = [n for n in archive.namelist() if n.lower().endswith(".wav")]
        assert len(wavs) == len(chosen)

    def test_no_selection_still_means_everything(self, client, analysed_job):
        """Omitting the parameter must not be read as "select nothing".

        An empty selection and an absent one are one character apart on the
        wire and opposite in meaning, so the distinction is pinned here.
        """
        job_id, result = analysed_job
        response = client.get(f"/api/jobs/{job_id}/export", params={"format": "wav-mix"})
        assert response.status_code == 200

        mixed, _ = _read_wav(response.content)
        assert np.max(np.abs(mixed - result.original)) < 1e-4


class TestRejections:
    """What happens when the request cannot be honoured."""

    def test_selecting_nothing_that_exists_is_an_error_not_silence(
        self, client, analysed_job
    ):
        """Bouncing an empty selection must fail loudly.

        Writing a valid, silent WAV would be the worst outcome: the download
        succeeds, the file opens, and it contains nothing.
        """
        job_id, _ = analysed_job
        response = client.get(
            f"/api/jobs/{job_id}/export", params={"format": "wav-mix", "only": "99999"}
        )
        assert response.status_code >= 400
        assert "audible" in response.text.lower()

    def test_non_numeric_ids_are_rejected(self, client, analysed_job):
        job_id, _ = analysed_job
        response = client.get(
            f"/api/jobs/{job_id}/export",
            params={"format": "wav-mix", "only": "3,../../etc/passwd"},
        )
        assert response.status_code == 400


class TestFormatMetadata:
    """The new format must behave like a single file everywhere it is asked."""

    def test_wav_mix_is_a_single_wav(self):
        fmt = ArchiveFormat.WAV_MIX
        assert not fmt.is_archive
        assert fmt.is_wav
        assert fmt.extension == ".wav"

    def test_estimate_reports_one_file_whatever_the_track_count(
        self, client, analysed_job
    ):
        """A bounce is one file however many tracks went into it.

        Worth asserting because the estimate for every other WAV format scales
        with the track count, and inheriting that formula here would quote a
        forty-times-too-large download.
        """
        job_id, result = analysed_job
        data = client.get(f"/api/jobs/{job_id}/export/estimate").json()
        entry = data["formats"]["wav-mix"]

        assert entry["files"] == 1
        # 3 bytes per sample at 24-bit, within a header's worth.
        assert abs(entry["bytes"] - result.original.shape[0] * 3) < 1024

    def test_the_download_is_named_and_attached(self, client, analysed_job):
        """Content-Disposition decides whether this saves or navigates away."""
        job_id, _ = analysed_job
        response = client.get(
            f"/api/jobs/{job_id}/export", params={"format": "wav-mix"}
        )
        disposition = response.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert ".wav" in disposition

    def test_the_file_is_a_playable_wav(self, client, analysed_job):
        """Decodable by a plain stdlib reader, not only by soundfile.

        The bounce is going into someone's DAW. `wave` is the least forgiving
        reader available, so if it opens the file, a DAW will.
        """
        job_id, result = analysed_job
        response = client.get(
            f"/api/jobs/{job_id}/export", params={"format": "wav-mix"}
        )
        with wave.open(io.BytesIO(response.content)) as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == result.sample_rate
            assert handle.getsampwidth() == 3  # 24-bit
            assert handle.getnframes() == result.original.shape[0]
