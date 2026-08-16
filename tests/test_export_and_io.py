"""Exporter, loader, and config tests.

The MIDI cases here target the failure mode the design document calls the
classic one: a file that opens fine, imports fine, and plays at the wrong
pitch because the pitch-bend range was never widened. That cannot be caught by
"did it write a file" — it needs the bytes read back and inspected.
"""

from __future__ import annotations

import math

import mido
import numpy as np
import pytest
import soundfile as sf

from backend.audio_io.loader import SoundFileLoader, sniff_format
from backend.config import Settings, is_port_available, resolve_port
from backend.core.errors import (
    AudioTooLongError,
    EmptyAudioError,
    ExportError,
    UnsupportedAudioError,
    UnsupportedExportError,
)
from backend.core.types import Partial, PartialFrame, Track, TrackKind
from backend.export.midi_exporter import MidiExporter, bend_value, seconds_to_ticks
from backend.export.wav_exporter import WavExporter

SR = 44_100


def _glide(partial_id: int = 0, start_hz: float = 440.0, end_hz: float = 880.0, n: int = 60):
    """A partial sweeping an octave — the case that needs a wide bend range."""
    partial = Partial(partial_id)
    for i in range(n):
        partial.append_frame(
            PartialFrame(
                time_s=i * 512 / SR,
                frequency_hz=start_hz * (end_hz / start_hz) ** (i / (n - 1)),
                amplitude=0.5,
                phase_rad=0.0,
            )
        )
    return partial


def _steady(partial_id: int, hz: float, n: int = 40, amplitude: float = 0.5):
    partial = Partial(partial_id)
    for i in range(n):
        partial.append_frame(PartialFrame(i * 512 / SR, hz, amplitude, 0.0))
    return partial


def _track(partials, kind=TrackKind.PARTIAL, track_id: int = 0, with_audio: bool = True):
    track = Track(track_id, kind, "test", partials)
    if with_audio:
        track.attach_audio(np.zeros(SR, dtype=np.float32))
    return track


# =============================================================================
# MIDI: the pitch-bend range trap
# =============================================================================


def test_midi_emits_the_rpn_sequence_widening_the_bend_range(tmp_path):
    """RPN 0 must be set, or every glide clips at the +/-2 semitone default.

    This is the single most common way MIDI export ships silently broken.
    """
    path = MidiExporter().export(_track([_glide()]), tmp_path / "g.mid", SR)
    messages = list(mido.MidiFile(path))

    ccs = [(m.control, m.value) for m in messages if m.type == "control_change"]
    # RPN 0 selection, then data entry carrying the semitone count.
    assert (101, 0) in ccs, "missing RPN MSB — bend range was never selected"
    assert (100, 0) in ccs, "missing RPN LSB — bend range was never selected"
    assert (6, 48) in ccs, "data entry MSB should carry the 48-semitone range"


def test_midi_pitch_bend_tracks_the_glide_upward(tmp_path):
    path = MidiExporter().export(_track([_glide()]), tmp_path / "g.mid", SR)
    bends = [m.pitch for m in mido.MidiFile(path) if m.type == "pitchwheel"]

    assert len(bends) > 5, "a one-octave glide should emit many bend messages"
    assert bends == sorted(bends), "bend should rise monotonically with the glide"

    # The nominal note is the partial's MEAN frequency, not its first — for a
    # 440 -> 880 glide that is the geometric middle, ~622 Hz. So the bend runs
    # -600 to +600 cents rather than 0 to +1200. That is the better choice: it
    # halves the range a glide needs and keeps the note centred on the pitch a
    # listener would name, instead of pinning it to wherever the tone began.
    span_cents = 600.0
    full_scale_cents = 48 * 100
    assert bends[-1] == pytest.approx(8191 * span_cents / full_scale_cents, rel=0.1)
    assert bends[0] == pytest.approx(-8191 * span_cents / full_scale_cents, rel=0.15)


def test_midi_has_exactly_one_note_per_partial(tmp_path):
    path = MidiExporter().export(_track([_glide()]), tmp_path / "g.mid", SR)
    note_ons = [m for m in mido.MidiFile(path) if m.type == "note_on" and m.velocity > 0]
    assert len(note_ons) == 1


def test_midi_delta_times_are_non_negative(tmp_path):
    """Absolute times must be converted to deltas correctly.

    A negative delta is impossible in the MIDI format and means the interleave
    or the sort went wrong — the classic bug when two partials are merged.
    """
    track = _track([_steady(0, 220.0), _steady(1, 440.0), _steady(2, 660.0)])
    path = MidiExporter().export(track, tmp_path / "multi.mid", SR)
    for midi_track in mido.MidiFile(path).tracks:
        for message in midi_track:
            assert message.time >= 0


def test_midi_puts_each_partial_on_its_own_channel(tmp_path):
    """Pitch bend is per-channel, so shared channels would bend each other."""
    track = _track([_steady(0, 220.0), _steady(1, 445.0), _steady(2, 669.0)])
    path = MidiExporter().export(track, tmp_path / "multi.mid", SR)
    channels = {m.channel for m in mido.MidiFile(path) if m.type == "note_on"}
    assert len(channels) == 3, f"expected one channel per partial, got {channels}"
    assert 9 not in channels, "channel 9 is GM percussion and must be skipped"


def test_midi_refuses_the_residual(tmp_path):
    residual = _track([], kind=TrackKind.RESIDUAL)
    with pytest.raises(UnsupportedExportError):
        MidiExporter().export(residual, tmp_path / "r.mid", SR)


def test_midi_refuses_an_empty_track(tmp_path):
    with pytest.raises(UnsupportedExportError):
        MidiExporter().export(_track([]), tmp_path / "e.mid", SR)


def test_midi_exporter_rejects_an_unsendable_bend_range():
    """The range travels in one 7-bit data byte, so >127 cannot be expressed."""
    with pytest.raises(ExportError):
        MidiExporter(bend_range_semitones=200)


def test_bend_value_clamps_at_the_range_limit():
    """Beyond the configured range the bend saturates rather than wrapping."""
    assert bend_value(0.0, 48) == 0
    assert bend_value(4800.0, 48) == 8191
    assert bend_value(99_999.0, 48) == 8191, "must clamp, not overflow"
    assert bend_value(-99_999.0, 48) <= -8191


def test_seconds_to_ticks_is_monotonic():
    assert seconds_to_ticks(0.0) == 0
    assert seconds_to_ticks(1.0) > seconds_to_ticks(0.5) > 0


def _fragment(partial_id: int, start_s: float, end_s: float, hz: float, amplitude: float = 0.5):
    """A partial occupying an explicit time window — for channel-sharing cases."""
    partial = Partial(partial_id)
    n = 20
    for i in range(n):
        partial.append_frame(
            PartialFrame(start_s + i * (end_s - start_s) / (n - 1), hz, amplitude, 0.0)
        )
    return partial


def _note_stream(path):
    """(absolute_tick, type, (channel, note)) for every note event in the file."""
    stream = []
    for midi_track in mido.MidiFile(path).tracks:
        tick = 0
        for message in midi_track:
            tick += message.time
            if message.type in ("note_on", "note_off"):
                stream.append((tick, message.type, (message.channel, message.note)))
    return stream


def test_midi_never_sounds_one_channel_and_note_twice_at_once(tmp_path):
    """MIDI addresses a sounding note by (channel, note) and nothing else.

    A second note_on for a pair already sounding is not a second note: the next
    note_off ends both and the one after it is an orphan. Real analyses hit
    this — a vibrato tone's upper harmonics fragment into several partials that
    round to the same note and overlap — and a bundle puts every partial of a
    track on one channel, so it has to be merged rather than written out.
    """
    track = _track(
        [
            _fragment(0, 0.0, 0.6, 440.0),
            _fragment(1, 0.3, 0.9, 441.0),  # same note number, overlapping
        ]
    )
    path = MidiExporter().export_bundle([track], tmp_path / "overlap.mid", SR)

    depth: dict[tuple[int, int], int] = {}
    for tick, kind, key in _note_stream(path):
        depth[key] = depth.get(key, 0) + (1 if kind == "note_on" else -1)
        assert depth[key] in (0, 1), (
            f"note {key} reached sounding depth {depth[key]} at tick {tick}; "
            "MIDI cannot sound one note twice on one channel"
        )
    assert all(v == 0 for v in depth.values()), "every note must be released"


def test_midi_note_off_precedes_a_note_on_sharing_its_tick(tmp_path):
    """Insertion order is partial order, not time order.

    When one partial ends exactly where another begins on the same channel and
    note, a note_on emitted before the old note_off kills the note that just
    started. The later-starting partial is listed FIRST here on purpose — that
    is what makes insertion order the wrong answer.
    """
    track = _track(
        [
            _fragment(0, 0.5, 1.0, 440.0),  # listed first, but starts second
            _fragment(1, 0.0, 0.5, 440.0),
        ]
    )
    path = MidiExporter().export_bundle([track], tmp_path / "abut.mid", SR)

    stream = _note_stream(path)
    for index in range(len(stream) - 1):
        first, second = stream[index], stream[index + 1]
        if first[0] != second[0] or first[2] != second[2]:
            continue
        killed = first[1] == "note_on" and second[1] == "note_off"
        assert not killed, f"note {first[2]} is started and stopped on tick {first[0]}"
    assert len([s for s in stream if s[1] == "note_on"]) == 2, "both notes must survive"


def test_midi_bundle_reports_what_it_dropped(tmp_path):
    """More than 15 tracks cannot share one file; the omission must be visible."""
    tracks = [_track([_steady(i, 110.0 * (i + 1))], track_id=i) for i in range(20)]
    path = MidiExporter().export_bundle(tracks, tmp_path / "bundle.mid", SR)
    assert path.exists()
    midi = mido.MidiFile(path)
    channels = {m.channel for t in midi.tracks for m in t if m.type == "note_on"}
    assert len(channels) <= 15


# =============================================================================
# WAV export
# =============================================================================


def test_wav_export_round_trips(tmp_path):
    audio = (0.4 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    track = Track(0, TrackKind.PARTIAL, "t", ())
    track.attach_audio(audio)

    path = WavExporter("FLOAT").export(track, tmp_path / "out.wav", SR)
    restored, rate = sf.read(path, dtype="float32")

    assert rate == SR
    assert np.max(np.abs(restored - audio)) == 0.0, "FLOAT subtype must be bit-exact"


def test_wav_export_creates_missing_directories(tmp_path):
    track = Track(0, TrackKind.PARTIAL, "t", ())
    track.attach_audio(np.zeros(128, dtype=np.float32))
    path = WavExporter().export(track, tmp_path / "a" / "b" / "out.wav", SR)
    assert path.exists()


def test_wav_export_refuses_a_track_with_no_audio(tmp_path):
    with pytest.raises(UnsupportedExportError):
        WavExporter().export(Track(0, TrackKind.PARTIAL, "t", ()), tmp_path / "x.wav", SR)


def test_wav_exporter_rejects_an_unknown_subtype():
    with pytest.raises(ExportError):
        WavExporter("NOT_A_REAL_SUBTYPE")


# =============================================================================
# Loader — the untrusted-input boundary
# =============================================================================


def test_sniff_identifies_a_real_wav(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(1000, dtype=np.float32), SR)
    assert sniff_format(path) == "wav"


def test_sniff_rejects_a_renamed_text_file(tmp_path):
    path = tmp_path / "fake.wav"
    path.write_bytes(b"I am definitely a wav file, honest, trust me")
    with pytest.raises(UnsupportedAudioError):
        sniff_format(path)


def test_sniff_rejects_a_file_too_short_to_identify(tmp_path):
    path = tmp_path / "tiny.wav"
    path.write_bytes(b"RIF")
    with pytest.raises(UnsupportedAudioError):
        sniff_format(path)


def test_sniff_rejects_a_missing_file(tmp_path):
    with pytest.raises(UnsupportedAudioError):
        sniff_format(tmp_path / "nope.wav")


def test_sniff_rejects_riff_that_is_not_wave(tmp_path):
    """RIFF is a container — an AVI is RIFF too, and is not audio."""
    path = tmp_path / "video.wav"
    path.write_bytes(b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"\x00" * 32)
    with pytest.raises(UnsupportedAudioError):
        sniff_format(path)


def test_loader_downmixes_and_resamples(tmp_path):
    path = tmp_path / "stereo48.wav"
    t = np.arange(48_000 * 2) / 48_000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, np.stack([tone, tone], axis=1), 48_000, subtype="PCM_24")

    samples, rate = SoundFileLoader(30.0).load(path, SR)

    assert rate == SR
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert len(samples) == pytest.approx(2 * SR, abs=64)

    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    peak_hz = int(np.argmax(spectrum)) * SR / len(samples)
    assert peak_hz == pytest.approx(440.0, abs=3.0)


def test_loader_rejects_audio_over_the_duration_ceiling(tmp_path):
    path = tmp_path / "long.wav"
    audio = np.zeros(SR * 5, dtype=np.float32)
    audio[::100] = 0.5
    sf.write(path, audio, SR)
    with pytest.raises(AudioTooLongError):
        SoundFileLoader(max_duration_seconds=1.0).load(path, SR)


def test_loader_rejects_silence(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(SR, dtype=np.float32), SR)
    with pytest.raises(EmptyAudioError):
        SoundFileLoader(30.0).load(path, SR)


def test_loader_scales_down_hot_audio_but_not_quiet_audio(tmp_path):
    """Clipping-hot input is tamed; quiet input keeps its level.

    Normalising quiet audio would distort the energy-ratio diagnostics, which
    are how a user reads how the decomposition went.
    """
    hot = tmp_path / "hot.wav"
    sf.write(hot, (2.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)), SR, subtype="FLOAT")
    loud, _ = SoundFileLoader(30.0).load(hot, SR)
    assert np.max(np.abs(loud)) == pytest.approx(1.0, abs=1e-5)

    quiet_path = tmp_path / "quiet.wav"
    sf.write(quiet_path, (0.01 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)), SR, subtype="FLOAT")
    quiet, _ = SoundFileLoader(30.0).load(quiet_path, SR)
    assert np.max(np.abs(quiet)) == pytest.approx(0.01, abs=1e-3)


# =============================================================================
# Config
# =============================================================================


def test_wildcard_cors_origin_is_refused():
    """A wildcard would let any site drive this API through a visitor."""
    with pytest.raises(ValueError, match="[Ww]ildcard"):
        Settings(allowed_origins=["*"])


def test_duration_ceiling_must_be_positive():
    with pytest.raises(ValueError):
        Settings(max_duration_seconds=0.0)


def test_resolve_port_returns_the_preferred_port_when_free():
    settings = Settings()
    port = resolve_port(settings.host, 8765)
    assert port == 8765 or not is_port_available(settings.host, 8765)


def test_resolve_port_steps_past_a_busy_port():
    """Auto-selection beats failing with 'address already in use'."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        held.bind(("127.0.0.1", 0))
        busy = held.getsockname()[1]
        held.listen(1)
        assert resolve_port("127.0.0.1", busy) != busy


def test_resolve_port_gives_up_rather_than_looping_forever():
    import socket

    holders = []
    try:
        base = None
        for _ in range(3):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            holders.append(s)
        base = holders[0].getsockname()[1]
        # max_attempts=1 with the only candidate held must raise.
        with pytest.raises(RuntimeError):
            resolve_port("127.0.0.1", base, max_attempts=1)
    finally:
        for s in holders:
            s.close()


def test_derived_analysis_quantities_are_consistent():
    from backend.core.types import AnalysisParams

    params = AnalysisParams(window_size=4096, hop_size=512, sample_rate=44_100)
    assert params.frequency_resolution_hz == pytest.approx(44_100 / 4096)
    assert params.window_duration_ms == pytest.approx(1000 * 4096 / 44_100)
    assert params.frame_rate_hz == pytest.approx(44_100 / 512)
    # The uncertainty product is dimensionless and ~1 by construction.
    product = params.frequency_resolution_hz * (params.window_duration_ms / 1000)
    assert math.isclose(product, 1.0, rel_tol=1e-9)
