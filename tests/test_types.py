"""Tests for core value objects and entities.

These cover the parts of the system that are already implemented, so the
scaffold has a working, passing suite from day one rather than a directory of
empty files that nobody trusts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.core.types import (
    A4_HZ,
    AnalysisParams,
    Partial,
    PartialFrame,
    Peak,
    Track,
    TrackKind,
    hz_to_midi,
    midi_to_hz,
    note_name,
)

# --- AnalysisParams ----------------------------------------------------------


def test_default_params_are_valid():
    params = AnalysisParams()
    assert params.sample_rate == 44_100
    assert params.window_size == 4096


def test_window_size_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        AnalysisParams(window_size=4000)


def test_hop_size_may_not_exceed_window_size():
    with pytest.raises(ValueError, match="hop_size"):
        AnalysisParams(window_size=1024, hop_size=2048)


def test_params_are_immutable():
    params = AnalysisParams()
    with pytest.raises(AttributeError):
        params.window_size = 8192  # type: ignore[misc]


def test_uncertainty_tradeoff_is_visible_in_derived_properties():
    """Doubling the window halves frequency resolution but doubles window time.

    Their product is invariant. That invariance IS the uncertainty principle,
    and asserting it here keeps anyone from "optimising" it away.
    """
    small = AnalysisParams(window_size=2048)
    large = AnalysisParams(window_size=4096)

    assert large.frequency_resolution_hz == pytest.approx(small.frequency_resolution_hz / 2)
    assert large.window_duration_ms == pytest.approx(small.window_duration_ms * 2)

    product_small = small.frequency_resolution_hz * small.window_duration_ms
    product_large = large.frequency_resolution_hz * large.window_duration_ms
    assert product_small == pytest.approx(product_large)


# --- Peak --------------------------------------------------------------------


def test_peak_amplitude_db():
    assert Peak(0, 440.0, 1.0, 0.0).amplitude_db == pytest.approx(0.0)
    assert Peak(0, 440.0, 0.5, 0.0).amplitude_db == pytest.approx(-6.02, abs=0.01)


def test_silent_peak_is_negative_infinity_db():
    assert Peak(0, 440.0, 0.0, 0.0).amplitude_db == -math.inf


def test_cents_distance_is_ratio_based():
    """An octave is 1200 cents regardless of where on the spectrum it sits."""
    low = Peak(0, 100.0, 1.0, 0.0)
    assert Peak(0, 200.0, 1.0, 0.0).cents_from(low) == pytest.approx(1200.0)

    high = Peak(0, 5000.0, 1.0, 0.0)
    assert Peak(0, 10000.0, 1.0, 0.0).cents_from(high) == pytest.approx(1200.0)


# --- Partial -----------------------------------------------------------------


def test_partial_reports_duration(steady_partial: Partial):
    assert steady_partial.duration_s == pytest.approx(1.0, abs=0.02)


def test_partial_frames_are_defensively_copied(steady_partial: Partial):
    """Callers get a tuple, so they cannot mutate a completed trajectory."""
    frames = steady_partial.frames
    assert isinstance(frames, tuple)
    original_length = len(steady_partial)
    with pytest.raises(AttributeError):
        frames.append(PartialFrame(99.0, 100.0, 0.1, 0.0))  # type: ignore[attr-defined]
    assert len(steady_partial) == original_length


def test_frames_must_arrive_in_time_order():
    partial = Partial(0)
    partial.append_frame(PartialFrame(1.0, 440.0, 0.5, 0.0))
    with pytest.raises(ValueError, match="time-ordered"):
        partial.append_frame(PartialFrame(0.5, 440.0, 0.5, 0.0))


def test_mean_frequency_is_amplitude_weighted():
    """A loud 440 and a whisper-quiet 880 should average near 440, not 660."""
    partial = Partial(0)
    partial.append_frame(PartialFrame(0.0, 440.0, 1.0, 0.0))
    partial.append_frame(PartialFrame(0.1, 880.0, 0.001, 0.0))
    assert partial.mean_frequency_hz == pytest.approx(440.0, abs=1.0)


def test_empty_partial_is_harmless():
    partial = Partial(0)
    assert len(partial) == 0
    assert partial.duration_s == 0.0
    assert partial.mean_frequency_hz == 0.0
    assert partial.peak_amplitude == 0.0


# --- Track -------------------------------------------------------------------


def test_residual_track_is_not_midi_exportable():
    """Noise has no notes. Exporting it as MIDI would invent information."""
    residual = Track(0, TrackKind.RESIDUAL, "Residual")
    assert residual.is_exportable_as_midi is False


def test_pitched_track_is_midi_exportable(steady_partial: Partial):
    track = Track(1, TrackKind.PARTIAL, "A4", [steady_partial])
    assert track.is_exportable_as_midi is True


def test_empty_pitched_track_is_not_midi_exportable():
    assert Track(1, TrackKind.PARTIAL, "empty").is_exportable_as_midi is False


def test_attached_audio_is_read_only():
    """Rendered audio is handed out as a non-writeable view."""
    track = Track(0, TrackKind.PARTIAL, "x")
    track.attach_audio(np.zeros(16, dtype=np.float32))
    audio = track.audio
    assert audio is not None
    with pytest.raises(ValueError):
        audio[0] = 1.0


def test_audio_may_only_be_attached_once():
    track = Track(0, TrackKind.PARTIAL, "x")
    track.attach_audio(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="already has rendered audio"):
        track.attach_audio(np.zeros(4, dtype=np.float32))


# --- MIDI conversion ---------------------------------------------------------


def test_a440_is_midi_note_69():
    assert hz_to_midi(A4_HZ) == pytest.approx(69.0)


def test_midi_hz_roundtrip():
    for note in (21.0, 60.0, 69.0, 108.0):
        assert hz_to_midi(midi_to_hz(note)) == pytest.approx(note)


def test_hz_to_midi_is_fractional():
    """Fractional notes are the point — they become pitch bends, not rounding."""
    quarter_tone_sharp = A4_HZ * (2.0 ** (0.5 / 12.0))
    assert hz_to_midi(quarter_tone_sharp) == pytest.approx(69.5)


def test_hz_to_midi_rejects_nonpositive():
    with pytest.raises(ValueError):
        hz_to_midi(0.0)


def test_note_name_exact():
    assert note_name(A4_HZ) == "A4"
    assert note_name(midi_to_hz(60)) == "C4"


def test_note_name_reports_cent_deviation():
    slightly_sharp = A4_HZ * (2.0 ** (0.12 / 12.0))
    assert note_name(slightly_sharp) == "A4 +12c"
