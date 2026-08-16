"""The invariant test: the tracks must sum back to the original, exactly.

This is the most important test in the project. Everything else is quality;
this is correctness. If a user solos every track and hears something other
than what they uploaded, the product's central claim is false.

These were written against the contract during Phase 0 and went live in Phase
2. Complementary coverage (loader round-trip, WAV serialisation, grouping
behaviour) lives in tests/test_end_to_end.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.types import AnalysisParams, TrackKind
from backend.services import build_pipeline
from tests.conftest import make_harmonic_tone, make_noise_burst, make_sine

# A shorter window than the default keeps these fast without weakening what
# they prove -- the invariant does not depend on window size.
PARAMS = AnalysisParams(window_size=2048, hop_size=512, max_tracks=60)


@pytest.fixture(scope="module")
def pipeline():
    return build_pipeline()


def _analyze(pipeline, audio, params: AnalysisParams | None = None):
    """Run the full pipeline."""
    return pipeline.analyze(audio, params or PARAMS)


def _verify(pipeline, audio, params: AnalysisParams | None = None):
    """Run the pipeline and return its reconstruction report."""
    return pipeline.verify(_analyze(pipeline, audio, params))


# =============================================================================
# Exactness
# =============================================================================


def test_residual_makes_reconstruction_exact(pipeline):
    """sum(tracks) == original, sample for sample.

    Not 'close'. Not '-60 dB error'. Exact, because the residual is defined as
    the arithmetic difference rather than estimated.
    """
    result = _analyze(pipeline, make_harmonic_tone(220.0, 8, duration_s=2.0))
    assert result.reconstruction_error() < 1e-6


def test_reconstruction_exact_even_for_pathological_input(pipeline):
    """The invariant must hold for input the tracker handles badly.

    Noise produces few usable partials, so nearly everything lands in the
    residual -- and the sum must still be exact. This is the test that proves
    the guarantee comes from the subtraction, not from good analysis.
    """
    result = _analyze(pipeline, make_noise_burst(duration_s=1.0))
    assert result.reconstruction_error() < 1e-6


def test_reconstruction_exact_for_silence(pipeline):
    """Degenerate input must not produce NaNs or a missing residual."""
    result = _analyze(pipeline, np.zeros(44_100, dtype=np.float32))
    assert result.reconstruction_error() < 1e-6


def test_every_analysis_produces_exactly_one_residual(pipeline):
    result = _analyze(pipeline, make_sine(440.0, 1.0))
    residuals = [t for t in result.tracks if t.kind is TrackKind.RESIDUAL]
    assert len(residuals) == 1


# =============================================================================
# The uncertainty principle, as a measurable expectation
# =============================================================================


def test_sustained_tone_is_mostly_partials(pipeline):
    """A steady harmonic tone is sparse in frequency, so partials capture it."""
    report = _verify(pipeline, make_harmonic_tone(220.0, 8, duration_s=2.0))
    assert report.partial_energy_ratio > 0.85


def test_noise_is_mostly_residual(pipeline):
    """Noise cannot be sparse in frequency, so it must land in the residual.

    This is not a shortcoming of the tracker. A signal cannot be concentrated
    in both time and frequency at once, so broadband content has nowhere else
    to go.
    """
    report = _verify(pipeline, make_noise_burst(duration_s=1.0))
    assert report.residual_energy_ratio > 0.60


def test_partial_capture_ranks_tonal_above_noisy(pipeline):
    """The ordering is the real claim; the absolute thresholds are calibration."""
    tonal = _verify(pipeline, make_harmonic_tone(220.0, 8, duration_s=2.0))
    noisy = _verify(pipeline, make_noise_burst(duration_s=2.0))
    assert tonal.partial_energy_ratio > noisy.partial_energy_ratio


# =============================================================================
# Single-partial accuracy
# =============================================================================


def test_pure_sine_recovers_its_frequency(pipeline):
    """A 440 Hz sine must come back as one partial at 440 Hz, within a few cents.

    Bin spacing here is ~21.5 Hz, so passing this proves the parabolic
    interpolation is doing its job -- without it the answer would be off by
    tens of cents.
    """
    result = _analyze(pipeline, make_sine(440.0, 1.0))
    pitched = [t for t in result.tracks if t.kind is not TrackKind.RESIDUAL]
    assert len(pitched) >= 1

    strongest = max(pitched, key=lambda t: sum(p.energy for p in t.partials))
    detected = strongest.partials[0].mean_frequency_hz
    cents_error = abs(1200.0 * np.log2(detected / 440.0))
    assert cents_error < 5.0


def test_pure_sine_produces_exactly_one_pitched_track(pipeline):
    """One tone in, one track out.

    Before the window change (see tests/test_windowing.py) this produced
    eleven pitched tracks: one real, ten window sidelobes wearing note names.
    """
    result = _analyze(pipeline, make_sine(440.0, 1.0))
    pitched = [t for t in result.tracks if t.kind is not TrackKind.RESIDUAL]
    assert len(pitched) == 1
