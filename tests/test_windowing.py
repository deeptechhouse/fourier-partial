"""Regression tests for the two STFT conventions that are easy to get wrong.

Both bugs guarded here were real, both shipped in the first implementation,
and both are the kind that produce plausible-looking output rather than a
crash — which is exactly why they need tests rather than vigilance.

  1. PHASE REFERENCE. The DFT reports phase relative to sample 0 of its input,
     but frames are timestamped at the window CENTRE. Without zero-phase
     windowing (and the cosine/sine convention fix) every rendered partial
     comes out anti-phase: right frequency, right amplitude, wrong sign. The
     residual then doubles the tone instead of cancelling it.

  2. SIDELOBE LEVEL. The peak threshold is -60 dB relative. A window whose
     sidelobes sit above that turns every loud partial into a small chord of
     phantom partials, each with a convincing note name.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.analysis.peaks import find_all_peaks
from backend.analysis.stft import compute_stft, frame_time, make_window
from backend.core.types import AnalysisParams

SR = 44_100
PARAMS = AnalysisParams(window_size=2048, hop_size=512)


def _sine(freq_hz: float, amplitude: float = 0.5, duration_s: float = 1.0):
    t = np.arange(int(duration_s * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


# =============================================================================
# 1. Phase reference
# =============================================================================


@pytest.mark.parametrize("freq_hz", [220.0, 440.0, 1000.0, 3000.0])
def test_phase_reconstructs_the_centre_sample(freq_hz):
    """(magnitude, phase) alone must rebuild the sample at the frame centre.

    This is the tightest possible statement of the convention: if you can take
    the peak bin's magnitude and phase and land on the actual waveform value
    at `frame_time`, then the renderer can too.

    Delete the `+ pi/2` in compute_stft, or the zero-phase roll, and this
    fails immediately.
    """
    amplitude = 0.5
    signal = _sine(freq_hz, amplitude)
    magnitude, phase = compute_stft(signal, PARAMS)

    frame = 20
    peak_bin = int(np.argmax(magnitude[frame]))
    reconstructed = magnitude[frame][peak_bin] * np.sin(phase[frame][peak_bin])

    centre_sample = int(round(frame_time(frame, PARAMS) * SR))
    actual = float(signal[centre_sample])

    assert reconstructed == pytest.approx(actual, abs=0.02)


def test_rendered_partial_is_in_phase_with_the_original():
    """The end-to-end statement of the same thing: correlation must be positive.

    A correlation near -1 means the analysis is perfect and the sign is
    inverted — the exact bug this guards. Asserting > 0.95 rather than
    != -1 keeps the test meaningful rather than merely non-regressive.
    """
    from backend.analysis.tracking import track_partials
    from backend.synthesis.oscillator_bank import render_partial

    signal = _sine(440.0)
    magnitude, phase = compute_stft(signal, PARAMS)
    partials = track_partials(find_all_peaks(magnitude, phase, PARAMS), PARAMS)
    assert partials, "a pure sine must produce at least one partial"

    strongest = max(partials, key=lambda p: p.energy)
    rendered = render_partial(strongest, PARAMS, len(signal))

    # Compare the steady middle, away from the fade at either end.
    a = signal[10_000:30_000].astype(np.float64)
    b = rendered[10_000:30_000].astype(np.float64)
    correlation = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))

    assert correlation > 0.95, f"rendered partial is out of phase (r={correlation:.4f})"


# =============================================================================
# 2. Sidelobe level
# =============================================================================


def test_window_sidelobes_are_below_the_peak_threshold():
    """The window must suppress sidelobes below the peak-picking floor.

    If this fails, the peak picker will admit window artefacts as tones. The
    threshold is the actual default, so the test tracks the real coupling
    between the two settings rather than a hardcoded guess.
    """
    window = make_window(PARAMS.window_size)
    padded = np.zeros(PARAMS.window_size * PARAMS.zero_pad_factor)
    padded[: PARAMS.window_size] = window
    spectrum = np.abs(np.fft.rfft(padded))

    # Exclude the main lobe. Blackman-Harris spans about +/-4 unpadded bins.
    main_lobe = 4 * PARAMS.zero_pad_factor
    outside = spectrum.copy()
    outside[:main_lobe] = 0.0
    sidelobe_db = 20 * np.log10(outside.max() / spectrum[0])

    assert sidelobe_db < PARAMS.peak_threshold_db, (
        f"first sidelobe at {sidelobe_db:.1f} dB is above the "
        f"{PARAMS.peak_threshold_db} dB peak threshold; phantom partials will appear"
    )


def test_pure_sine_yields_exactly_one_peak_per_frame():
    """The observable consequence: one tone in, one peak out.

    Before the window change this returned eleven — one real partial and ten
    sidelobe artefacts, each of which became a track with its own note name.
    """
    magnitude, phase = compute_stft(_sine(440.0), PARAMS)
    frames = find_all_peaks(magnitude, phase, PARAMS)

    middle = frames[len(frames) // 2]
    assert len(middle) == 1, f"expected 1 peak, got {len(middle)}: " + ", ".join(
        f"{p.frequency_hz:.0f}Hz" for p in middle
    )
    assert middle[0].frequency_hz == pytest.approx(440.0, abs=1.0)
    assert middle[0].amplitude == pytest.approx(0.5, rel=0.05)


def test_two_tones_a_major_third_apart_still_resolve():
    """The cost of the wider main lobe, bounded by a test.

    Blackman-Harris trades resolution for sidelobe suppression, so it is worth
    pinning what it can still separate. 440 and 554 Hz (a major third) must
    remain two distinct peaks — if this ever fails, the window is too wide for
    musical material and the trade has gone too far.
    """
    signal = (_sine(440.0, 0.5) + _sine(554.37, 0.5)).astype(np.float32)
    magnitude, phase = compute_stft(signal, PARAMS)
    frames = find_all_peaks(magnitude, phase, PARAMS)

    middle = frames[len(frames) // 2]
    found = sorted(p.frequency_hz for p in middle)
    assert len(found) >= 2, f"expected 2 resolved tones, got {found}"
    assert found[0] == pytest.approx(440.0, abs=3.0)
    assert found[1] == pytest.approx(554.37, abs=3.0)
