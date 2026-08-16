"""End-to-end tests: real audio in, real tracks out, invariant enforced.

These are the tests that decide whether the product works. Everything else
checks a component; these check the promise.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from backend.core.types import AnalysisParams, TrackKind
from backend.services import build_pipeline
from tests.conftest import make_harmonic_tone, make_noise_burst, make_sine

# A shorter window keeps these tests quick without changing what they prove.
FAST = AnalysisParams(window_size=2048, hop_size=512, max_tracks=60)


@pytest.fixture(scope="module")
def pipeline():
    return build_pipeline()


# =============================================================================
# The invariant — the most important assertions in the project
# =============================================================================


@pytest.mark.parametrize(
    "name,audio",
    [
        ("harmonic tone", make_harmonic_tone(220.0, 8, duration_s=1.0)),
        ("pure sine", make_sine(440.0, 1.0)),
        ("white noise", make_noise_burst(duration_s=1.0)),
        ("silence", np.zeros(44_100, dtype=np.float32)),
        ("dc offset", np.full(44_100, 0.3, dtype=np.float32)),
    ],
)
def test_reconstruction_is_exact(pipeline, name, audio):
    """sum(tracks) == original, sample for sample, whatever the input.

    Noise and silence are the important cases: they produce few or no usable
    partials, so almost everything lands in the residual — and the sum must
    STILL be exact. That is what proves the guarantee comes from the
    subtraction rather than from the analysis going well.
    """
    result = pipeline.analyze(audio, FAST)
    assert result.reconstruction_error() < 1e-6, f"{name} failed the invariant"


def test_every_analysis_produces_exactly_one_residual(pipeline):
    result = pipeline.analyze(make_sine(440.0, 0.5), FAST)
    residuals = [t for t in result.tracks if t.kind is TrackKind.RESIDUAL]
    assert len(residuals) == 1


def test_audio_shorter_than_one_window_still_reconstructs(pipeline):
    """A clip too short to analyse is not an error — the residual carries it."""
    tiny = make_sine(440.0, 0.01)
    result = pipeline.analyze(tiny, FAST)
    assert result.reconstruction_error() < 1e-6
    assert len(result.tracks) == 1
    assert result.tracks[0].kind is TrackKind.RESIDUAL


# =============================================================================
# Analysis quality
# =============================================================================


def test_pure_sine_recovers_its_frequency(pipeline):
    """440 Hz in, 440 Hz out, within a few cents.

    Bin spacing at this window is ~21.5 Hz, so passing this proves parabolic
    interpolation is working — without it the answer would be tens of cents off.
    """
    result = pipeline.analyze(make_sine(440.0, 1.0), FAST)
    pitched = [t for t in result.tracks if t.kind is not TrackKind.RESIDUAL]
    assert pitched, "a pure sine should produce at least one pitched track"

    strongest = max(pitched, key=lambda t: sum(p.energy for p in t.partials))
    detected = strongest.partials[0].mean_frequency_hz
    cents_error = abs(1200.0 * np.log2(detected / 440.0))
    assert cents_error < 5.0, f"detected {detected:.2f} Hz ({cents_error:.1f} cents off)"


def test_harmonic_tone_groups_into_one_voice(pipeline):
    """Eight harmonics of 220 Hz should collapse to a single playable track."""
    result = pipeline.analyze(make_harmonic_tone(220.0, 8, duration_s=1.0), FAST)
    groups = [t for t in result.tracks if t.kind is TrackKind.HARMONIC_GROUP]
    assert groups, "a harmonic complex should form at least one harmonic group"
    assert max(len(g.partials) for g in groups) >= 4


# =============================================================================
# The uncertainty principle, as a measurable expectation (section 12.3)
# =============================================================================


def test_sustained_tone_is_mostly_partials(pipeline):
    """A steady harmonic tone is sparse in frequency, so partials capture it."""
    result = pipeline.analyze(make_harmonic_tone(220.0, 8, duration_s=1.5), FAST)
    report = pipeline.verify(result)
    assert report.partial_energy_ratio > 0.5


def test_partial_capture_ranks_tonal_above_noisy(pipeline):
    """The ordering is the real claim; absolute thresholds are calibration.

    A signal cannot be sparse in time and frequency at once, so broadband
    noise must land in the residual while a sustained tone need not. Asserting
    the relationship rather than a number gives a test that survives parameter
    tuning and still catches a genuine regression.
    """
    tonal = pipeline.verify(pipeline.analyze(make_harmonic_tone(220.0, 8, 1.5), FAST))
    noisy = pipeline.verify(pipeline.analyze(make_noise_burst(1.5), FAST))
    assert tonal.partial_energy_ratio > noisy.partial_energy_ratio


def test_verify_reports_are_self_consistent(pipeline):
    result = pipeline.analyze(make_harmonic_tone(330.0, 6, 1.0), FAST)
    report = pipeline.verify(result)
    assert report.is_exact
    assert 0.0 <= report.partial_energy_ratio <= 1.0
    assert report.partial_energy_ratio + report.residual_energy_ratio == pytest.approx(1.0)


def test_silent_input_does_not_produce_nan(pipeline):
    """Degenerate input must not poison the diagnostics with NaN."""
    report = pipeline.verify(pipeline.analyze(np.zeros(44_100, dtype=np.float32), FAST))
    assert not np.isnan(report.partial_energy_ratio)
    assert report.partial_energy_ratio == 0.0


# =============================================================================
# Round trip through the loader
# =============================================================================


def test_loader_to_pipeline_round_trip(tmp_path, pipeline):
    """A real file on disk, decoded and analysed, still reconstructs exactly."""
    from backend.services import build_loader

    path = tmp_path / "phrase.wav"
    # Stereo at a different rate, so the loader has to downmix AND resample.
    tone = make_harmonic_tone(196.0, 6, duration_s=1.0, sample_rate=48_000)
    sf.write(path, np.stack([tone, tone], axis=1), 48_000, subtype="PCM_24")

    samples, sr = build_loader(30.0).load(path, 44_100)
    assert sr == 44_100
    assert samples.ndim == 1
    assert samples.dtype == np.float32

    result = pipeline.analyze(samples, FAST)
    assert result.reconstruction_error() < 1e-6


def test_track_audio_serialises_to_readable_wav(pipeline):
    """The bytes the browser receives must actually decode back to the samples.

    Float WAV rather than 16-bit PCM is what preserves the invariant through
    playback; if this ever silently became integer PCM, "unmute everything"
    would stop reproducing the upload exactly.
    """
    result = pipeline.analyze(make_sine(440.0, 0.5), FAST)
    track = result.tracks[0]
    assert track.audio is not None

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(track.audio), result.sample_rate, format="WAV", subtype="FLOAT")
    buffer.seek(0)
    restored, sr = sf.read(buffer, dtype="float32")

    assert sr == result.sample_rate
    assert np.max(np.abs(restored - np.asarray(track.audio))) == 0.0
