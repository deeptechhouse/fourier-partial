"""Long-audio support: streamed spectrogram and duration-adaptive settings.

The 30-second ceiling was not a policy choice, it was a memory wall. The full
spectrogram costs ~11 MB per second of audio at phrase settings, so ten minutes
needed ~6.8 GB before a single partial had been tracked. Two changes lifted it:
streaming the spectrogram, and spending less per frame on long input.

These tests pin both, plus the property that matters most: the streamed path
and the all-at-once path must produce *identical* numbers. They contain
duplicated maths — deliberately, because the phase convention is subtle — and
duplicated maths drifts unless something checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.analysis.stft import (
    DEFAULT_BLOCK_FRAMES,
    compute_stft,
    frame_count,
    iter_stft_blocks,
)
from backend.core.types import AnalysisParams
from backend.services import StftPeakAnalyzer

SR = 44_100


def _signal(duration_s: float, seed: int = 0):
    """A tone plus noise — enough spectral variety to be a real test."""
    t = np.arange(int(duration_s * SR)) / SR
    audio = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 997 * t)
    audio += 0.05 * np.random.default_rng(seed).standard_normal(len(t))
    return audio.astype(np.float32)


# =============================================================================
# The streamed path must equal the all-at-once path
# =============================================================================


@pytest.mark.parametrize("block_frames", [1, 7, 64, DEFAULT_BLOCK_FRAMES, 100_000])
def test_streaming_stft_matches_compute_stft_exactly(block_frames):
    """Bit-for-bit agreement, at every block size including degenerate ones.

    Block size must be an implementation detail with no observable effect. A
    block of 1 and a block larger than the whole file must both reproduce the
    monolithic result exactly.
    """
    params = AnalysisParams(window_size=1024, hop_size=256, zero_pad_factor=2)
    audio = _signal(0.5)

    expected_mag, expected_phase = compute_stft(audio, params)

    got_mag = np.zeros_like(expected_mag)
    got_phase = np.zeros_like(expected_phase)
    seen = 0
    for first, mag_block, phase_block in iter_stft_blocks(audio, params, block_frames):
        got_mag[first : first + mag_block.shape[0]] = mag_block
        got_phase[first : first + phase_block.shape[0]] = phase_block
        seen += mag_block.shape[0]

    assert seen == expected_mag.shape[0], "streaming visited a different number of frames"
    assert np.array_equal(got_mag, expected_mag), "magnitudes diverged"
    assert np.array_equal(got_phase, expected_phase), "phases diverged"


def test_streaming_covers_every_frame_exactly_once():
    params = AnalysisParams(window_size=1024, hop_size=256)
    audio = _signal(0.4)
    expected = frame_count(len(audio), params)

    visited = []
    for first, mag_block, _ in iter_stft_blocks(audio, params, block_frames=13):
        visited.extend(range(first, first + mag_block.shape[0]))

    assert visited == list(range(expected))


def test_streaming_yields_nothing_for_audio_shorter_than_one_window():
    params = AnalysisParams(window_size=4096, hop_size=512)
    assert list(iter_stft_blocks(np.zeros(100, dtype=np.float32), params)) == []


def test_streaming_rejects_stereo():
    """Silently flattening stereo would interleave channels into nonsense."""
    from backend.core.errors import InvalidParameterError

    params = AnalysisParams(window_size=1024, hop_size=256)
    stereo = np.zeros((1000, 2), dtype=np.float32)
    with pytest.raises(InvalidParameterError):
        list(iter_stft_blocks(stereo, params))


def test_analyzer_produces_the_same_peaks_as_the_monolithic_path():
    """The streaming analyzer must not change what the pipeline sees."""
    from backend.analysis.peaks import find_all_peaks

    params = AnalysisParams(window_size=1024, hop_size=256, zero_pad_factor=2)
    audio = _signal(0.4)

    magnitude, phase = compute_stft(audio, params)
    expected = find_all_peaks(magnitude, phase, params)
    got = StftPeakAnalyzer().find_peaks(audio, params)

    assert len(got) == len(expected)
    for frame_index, (a, b) in enumerate(zip(got, expected, strict=True)):
        assert len(a) == len(b), f"frame {frame_index} peak count differs"
        for p, q in zip(a, b, strict=True):
            assert p.frequency_hz == q.frequency_hz
            assert p.amplitude == q.amplitude
            assert p.frame_index == q.frame_index


# =============================================================================
# Duration-adaptive settings
# =============================================================================


def test_short_audio_keeps_full_detail():
    """A phrase must be unaffected — this is the case the tool is built for."""
    assert AnalysisParams.for_duration(10.0) == AnalysisParams()
    assert AnalysisParams.for_duration(45.0) == AnalysisParams()


def test_settings_get_cheaper_as_duration_grows():
    """Monotonic in the direction that matters: never more work per second."""
    short = AnalysisParams.for_duration(10.0)
    medium = AnalysisParams.for_duration(120.0)
    long_form = AnalysisParams.for_duration(600.0)

    assert short.hop_size < medium.hop_size < long_form.hop_size
    assert short.zero_pad_factor >= medium.zero_pad_factor >= long_form.zero_pad_factor
    assert short.frame_rate_hz > medium.frame_rate_hz > long_form.frame_rate_hz


def test_ten_minutes_stays_within_a_workable_frame_budget():
    """The concrete thing the tiers exist to bound.

    At phrase settings ten minutes is ~51,700 frames and ~6.8 GB of
    spectrogram. Both numbers have to come down for the file to be analysable,
    and this asserts the first one actually did.
    """
    params = AnalysisParams.for_duration(600.0)
    frames = frame_count(int(600 * SR), params)
    assert frames < 15_000, f"{frames} frames is too many for a ten-minute file"

    bins = params.window_size * params.zero_pad_factor // 2 + 1
    spectrogram_gb = frames * bins * 8 * 2 / 1e9
    # Streaming means this is never allocated at once, but if the per-frame
    # cost ever crept back up the streamed working set would grow with it.
    assert spectrogram_gb < 1.0, f"per-frame cost implies {spectrogram_gb:.1f} GB"


def test_overrides_win_over_the_duration_tier():
    params = AnalysisParams.for_duration(600.0, hop_size=256)
    assert params.hop_size == 256


def test_long_form_still_resolves_a_musical_interval():
    """Coarser settings must not make the analysis useless.

    The long-form tier drops zero-padding entirely, so it is worth checking
    that a real musical interval still separates — otherwise the tier would
    be trading away the thing the tool is for.
    """
    from backend.analysis.peaks import find_all_peaks

    params = AnalysisParams.for_duration(600.0)
    t = np.arange(int(1.0 * SR)) / SR
    # A perfect fifth, 440 and 660 Hz.
    audio = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 660 * t)).astype(
        np.float32
    )

    magnitude, phase = compute_stft(audio, params)
    frames = find_all_peaks(magnitude, phase, params)
    middle = sorted(p.frequency_hz for p in frames[len(frames) // 2])

    assert len(middle) >= 2
    assert middle[0] == pytest.approx(440.0, abs=6.0)
    assert middle[1] == pytest.approx(660.0, abs=6.0)


# =============================================================================
# The invariant survives the new code paths
# =============================================================================


def test_reconstruction_stays_exact_at_long_form_settings():
    """Streaming and coarser settings must not weaken the central promise."""
    from backend.services import build_pipeline

    params = AnalysisParams.for_duration(600.0)
    result = build_pipeline().analyze(_signal(3.0), params)
    assert result.reconstruction_error() < 1e-6


def test_reconstruction_error_does_not_stack_every_track():
    """Guards the accumulator fix.

    `np.stack` over N tracks allocates N copies of the whole file at once; at
    ten minutes that is ~106 MB per track. This checks the result is right for
    many tracks, which is the case that would have blown up.
    """
    from backend.core.types import AnalysisResult, Track, TrackKind

    n = 4096
    original = np.random.default_rng(0).standard_normal(n).astype(np.float32)

    tracks = []
    accumulated = np.zeros(n, dtype=np.float64)
    for i in range(12):
        chunk = (original / 24.0).astype(np.float32)
        track = Track(i, TrackKind.PARTIAL, f"t{i}", ())
        track.attach_audio(chunk)
        accumulated += chunk
        tracks.append(track)

    residual = Track(99, TrackKind.RESIDUAL, "Residual", ())
    residual.attach_audio((original.astype(np.float64) - accumulated).astype(np.float32))
    tracks.append(residual)

    result = AnalysisResult(SR, n / SR, tracks, AnalysisParams(), original)
    assert result.reconstruction_error() < 1e-6
