"""MIDI export — the lossy path, and the honest accounting of what is lost.

BE CLEAR WITH YOURSELF ABOUT THIS
---------------------------------
WAV export is lossless. MIDI export is not, and cannot be. A partial is a
continuous curve of frequency and amplitude; MIDI is a stream of discrete
note-on and note-off events on a 12-tone grid. Converting one to the other
throws information away. This module's job is to throw away as little as
possible and to be honest about the rest.

WHAT WE PRESERVE, AND HOW
-------------------------
    frequency  -> note number + pitch bend
    amplitude  -> note-on velocity (initial) + CC 11 expression (continuous)
    lifetime   -> note-on / note-off timing

PITCH BEND, AND THE RANGE TRAP
------------------------------
A partial that glides through a whole tone cannot be one MIDI note unless we
bend it. Pitch bend is 14-bit and relative, and its *range* is a per-channel
setting that defaults to +/-2 semitones. A partial sweeping an octave will
simply clip at the limit unless we first widen the range via RPN 0. So on
every channel we use, we emit the RPN sequence to set the bend range to
`PITCH_BEND_RANGE_SEMITONES` before anything else. Forgetting this is the
single most common way MIDI export silently comes out wrong.

THE 16-CHANNEL WALL
-------------------
Pitch bend applies to a whole MIDI channel, not to one note. Two tracks
sharing a channel would bend each other. MIDI has 16 channels and we routinely
have 200 tracks, so they cannot all coexist in one file.

Therefore the DEFAULT export is ONE FILE PER TRACK -- unambiguous, unlimited,
and every DAW imports it correctly. A bundled MPE-style multi-track file is
offered as a convenience for the 15 highest-energy tracks (channel 10 is
skipped: it is percussion by convention and would sound absurd).

STATUS: implemented (Phase 3).

HOW THE PIECES FIT TOGETHER (read this before editing)
------------------------------------------------------
    _partial_messages   -> one partial  -> [(absolute_seconds, Message), ...]
    _assemble_track     -> that list    -> a mido.MidiTrack with DELTA times
    export              -> one Track    -> a one-MidiTrack file
    export_bundle       -> many Tracks  -> a conductor track + one per Track

Two invariants run through all of it:

1. EVERYTHING IS BUILT IN ABSOLUTE SECONDS AND CONVERTED ONCE, AT THE END.
   mido stores `Message.time` as a *delta* from the previous message in the
   same track. Building deltas as you go is the classic way to get MIDI export
   wrong, because the moment you interleave two partials the deltas of one are
   invalidated by the other. So every helper below returns absolute times and
   only `_assemble_track` -- which sees the whole sorted stream at once -- ever
   computes a delta.

2. ORDER AT EQUAL TICKS IS INSERTION ORDER.
   `list.sort` in Python is stable, so messages sharing a tick come out in the
   order they were appended. That is load-bearing: the RPN block must precede
   the first note on its channel, and a partial's opening pitch bend must
   precede its note-on, or the note sounds at the wrong pitch for one buffer.
   With one deliberate exception -- a note-off is sorted ahead of anything else
   on its tick, because insertion order is partial order rather than time
   order, and a partial that begins exactly where another ends would otherwise
   have its note killed on the tick it started. See `_assemble_track`.

3. ONE (CHANNEL, NOTE) SOUNDS AT MOST ONCE AT A TIME.
   MIDI identifies a sounding note only by its channel and note number, so
   overlapping duplicates cannot be expressed and must be merged. Only reached
   where partials share a channel; `_assemble_track` handles and reports it.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import mido
import numpy as np
from numpy.typing import NDArray

from backend.core.errors import ExportError, UnsupportedExportError
from backend.core.types import Partial, Track, hz_to_midi, midi_to_hz

logger = logging.getLogger(__name__)

# ADJUSTABLE: pitch bend range, in semitones, set via RPN 0 on each channel.
# 48 (four octaves) covers essentially any partial glide. Some older hardware
# caps at 12 or 24; lower this if you are targeting such a device.
PITCH_BEND_RANGE_SEMITONES: int = 48

# MIDI pitch bend is 14-bit signed: -8192..+8191, centred at 0.
PITCH_BEND_MAX: int = 8191

# ADJUSTABLE: ticks per beat. Higher gives finer timing at larger file size.
TICKS_PER_BEAT: int = 960

# ADJUSTABLE: tempo assumed when converting seconds to ticks. Partials have no
# intrinsic tempo, so this is arbitrary -- 120 BPM keeps the numbers readable.
ASSUMED_TEMPO_BPM: float = 120.0

# ADJUSTABLE: how often to emit expression (CC 11) updates, in milliseconds.
# Denser tracks the amplitude envelope more faithfully; sparser produces
# smaller files that are easier to edit by hand.
EXPRESSION_INTERVAL_MS: float = 20.0

# ADJUSTABLE: partials quieter than this (relative to the track peak) are
# treated as silent, so a long fade-out does not become a long held note.
NOTE_OFF_THRESHOLD_DB: float = -48.0

# ADJUSTABLE: shortest note this exporter will write, in milliseconds. A
# partial can in principle be a single frame long, which would give a note-on
# and note-off on the same tick -- legal MIDI, but many DAWs discard a
# zero-length note outright, so the export would silently lose it.
MIN_NOTE_DURATION_MS: float = 10.0

# Channel 10 (1-indexed) is the General MIDI percussion channel.
_PERCUSSION_CHANNEL: int = 9  # 0-indexed

# The channels we are actually willing to use, in the order we hand them out.
# Fifteen of them, because the percussion channel is skipped: a pitched partial
# routed there plays as a drum hit on any GM-compatible instrument.
_USABLE_CHANNELS: tuple[int, ...] = tuple(ch for ch in range(16) if ch != _PERCUSSION_CHANNEL)

# MIDI controller numbers used below, named so the message-building code reads
# as intent rather than as magic numbers.
_CC_EXPRESSION: int = 11  # continuous loudness, rides on top of note velocity
_CC_DATA_ENTRY_MSB: int = 6  # RPN payload, coarse (here: semitones)
_CC_DATA_ENTRY_LSB: int = 38  # RPN payload, fine (here: cents)
_CC_RPN_LSB: int = 100  # selects which registered parameter we mean
_CC_RPN_MSB: int = 101  # ditto, high byte

# RPN 0 (MSB 0, LSB 0) is "pitch bend sensitivity". RPN 127/127 is the "null"
# parameter, parked on afterwards so a stray later data-entry CC cannot be
# mistaken for another edit to the bend range.
_RPN_PITCH_BEND_RANGE: tuple[int, int] = (0, 0)
_RPN_NULL: tuple[int, int] = (127, 127)

# Not adjustable -- MIDI data bytes are 7-bit.
_MIDI_MAX_VALUE: int = 127

# Guard against log2(0) when a trajectory contains a zero frequency. Any value
# this small is far outside the bend range anyway, so the result clamps.
_MIN_POSITIVE_HZ: float = 1e-9


class MidiExporter:
    """Writes a track as a MIDI file, mapping partials to pitch-bent notes."""

    def __init__(self, bend_range_semitones: int = PITCH_BEND_RANGE_SEMITONES) -> None:
        """Create an exporter with the given pitch bend range.

        Args:
            bend_range_semitones: Semitones of pitch bend to request via RPN 0.
                Must fit in a 7-bit MIDI data byte (1..127).

        Raises:
            ExportError: If the requested range cannot be expressed in MIDI.
        """
        # Caught here rather than at write time: the range is sent as a single
        # data-entry byte, so anything outside 1..127 is simply unsendable and
        # would blow up inside mido with a much less helpful message.
        if not 1 <= bend_range_semitones <= _MIDI_MAX_VALUE:
            raise ExportError(
                f"pitch bend range must be 1..{_MIDI_MAX_VALUE} semitones, "
                f"got {bend_range_semitones}"
            )
        self._bend_range = bend_range_semitones

    @property
    def bend_range_semitones(self) -> int:
        """The pitch bend range this exporter configures, in semitones."""
        return self._bend_range

    @property
    def file_extension(self) -> str:
        """Extension produced by this exporter."""
        return ".mid"

    def supports(self, track: Track) -> bool:
        """Whether MIDI is meaningful for this track.

        The residual is broadband noise. It has no pitch, so writing it as
        notes would be fabricating information rather than converting it.
        """
        return track.is_exportable_as_midi

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def export(self, track: Track, destination: Path, sample_rate: int) -> Path:
        """Write `track` as a single-track MIDI file.

        Every partial in the track gets its own MIDI channel, because pitch
        bend is a per-channel setting and partials that shared a channel would
        bend each other. Fifteen channels are available (percussion is
        skipped); a track with more partials than that reuses them, which is
        reported as a warning rather than silently accepted.

        Args:
            track: The track to write. Must be MIDI-exportable.
            destination: Path to write to. Parent directories are created.
            sample_rate: Unused for MIDI (times come from the trajectories,
                which are already in seconds), accepted to satisfy the
                `TrackExporter` protocol.

        Returns:
            The path actually written.

        Raises:
            UnsupportedExportError: For the residual, or an empty track.
            ExportError: If the file could not be written.
        """
        if not self.supports(track):
            raise UnsupportedExportError(
                f"track {track.track_id} ({track.kind.value}) has no pitch content to export"
            )

        partials = track.partials
        self._warn_on_channel_reuse(len(partials), track.label, "partials")

        # `timed` accumulates (absolute_seconds, message) pairs from every
        # partial. Nothing here is in delta time yet -- see invariant 1 in the
        # module docstring.
        timed: list[tuple[float, mido.Message | mido.MetaMessage]] = []

        # The tempo map. Partials have no tempo of their own, so this exists
        # purely to give ticks a defined meaning; see ASSUMED_TEMPO_BPM.
        timed.append((0.0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(ASSUMED_TEMPO_BPM))))
        timed.append((0.0, mido.MetaMessage("track_name", name=_safe_meta_text(track.label))))

        # THE RPN SEQUENCE. This is the trap the module docstring warns about:
        # without it the channel bends only +/-2 semitones and every glide
        # wider than a whole tone clips flat against the limit.
        for channel in self._channels_for(len(partials)):
            timed.extend(self._rpn_bend_range_messages(channel))

        track_peak = _track_peak_amplitude(partials)
        for index, partial in enumerate(partials):
            channel = _USABLE_CHANNELS[index % len(_USABLE_CHANNELS)]
            velocity = _velocity_for(partial, track_peak)
            timed.extend(self._partial_messages(partial, channel, velocity))

        # mido's default is a type-1 (multi-track) file. We keep that even for
        # the single-track case so both code paths below produce structurally
        # identical files; every DAW reads a type-1 file with one track.
        midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        midi.tracks.append(self._assemble_track(timed))
        return self._save(midi, destination)

    def export_bundle(self, tracks: list[Track], destination: Path, sample_rate: int) -> Path:
        """Write up to 15 tracks into one multi-channel MIDI file.

        Convenience only. Tracks beyond the channel limit are omitted, and the
        caller is expected to tell the user which ones -- silently dropping
        tracks would read as "everything exported" when it did not.

        Selection is by descending total energy, so what survives the cut is
        what the listener would most notice missing. Anything dropped -- both
        the not-exportable (residual, empty) and the over-the-limit -- is named
        in a logged warning.

        MULTI-PARTIAL CAVEAT: a bundle gives each *track* one channel, so a
        harmonic-group track's partials share a channel and their pitch bends
        interact -- the last bend written wins for all of them. That is
        unavoidable at 15 channels and is why per-track `export()` (one file
        each, one channel per partial) is the lossless-as-possible default.
        Affected tracks are named in a warning.

        Args:
            tracks: Candidate tracks, in any order.
            destination: Path to write to. Parent directories are created.
            sample_rate: Unused for MIDI; accepted for interface symmetry.

        Returns:
            The path actually written.

        Raises:
            UnsupportedExportError: If no supplied track is MIDI-exportable.
            ExportError: If the file could not be written.
        """
        skipped = [t for t in tracks if not self.supports(t)]
        candidates = [t for t in tracks if self.supports(t)]
        if not candidates:
            raise UnsupportedExportError(
                f"none of the {len(tracks)} supplied tracks has pitch content to export"
            )

        # Highest energy first: if we must drop tracks, drop the ones the
        # listener is least likely to miss.
        ranked = sorted(candidates, key=_track_energy, reverse=True)
        selected = ranked[: len(_USABLE_CHANNELS)]
        dropped = ranked[len(_USABLE_CHANNELS) :]

        if skipped:
            logger.warning(
                "MIDI bundle omitted %d track(s) with no pitch content: %s",
                len(skipped),
                _describe(skipped),
            )
        if dropped:
            logger.warning(
                "MIDI bundle omitted %d track(s) beyond the %d-channel limit "
                "(lowest energy first dropped): %s",
                len(dropped),
                len(_USABLE_CHANNELS),
                _describe(dropped),
            )

        midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)

        # Standard MIDI File type 1 convention: track 0 is the "conductor"
        # track and carries only the tempo map. Putting set_tempo in every
        # track is legal but confuses editors that show it as 15 tempo changes.
        conductor: list[tuple[float, mido.Message | mido.MetaMessage]] = [
            (0.0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(ASSUMED_TEMPO_BPM))),
            (0.0, mido.MetaMessage("track_name", name="Tempo")),
        ]
        midi.tracks.append(self._assemble_track(conductor))

        for index, track in enumerate(selected):
            channel = _USABLE_CHANNELS[index]
            partials = track.partials
            if len(partials) > 1:
                logger.warning(
                    "bundled track %s (%r) has %d partials sharing MIDI channel %d; "
                    "their pitch bends interact. Export this track on its own for "
                    "per-partial bend accuracy.",
                    track.track_id,
                    track.label,
                    len(partials),
                    channel,
                )

            timed: list[tuple[float, mido.Message | mido.MetaMessage]] = [
                (0.0, mido.MetaMessage("track_name", name=_safe_meta_text(track.label)))
            ]
            # Same RPN treatment as the single-track path: widen the bend range
            # on this channel before anything is played on it.
            timed.extend(self._rpn_bend_range_messages(channel))

            track_peak = _track_peak_amplitude(partials)
            for partial in partials:
                velocity = _velocity_for(partial, track_peak)
                timed.extend(self._partial_messages(partial, channel, velocity))

            midi.tracks.append(self._assemble_track(timed))

        return self._save(midi, destination)

    # -------------------------------------------------------------------------
    # Message construction
    # -------------------------------------------------------------------------

    def _channels_for(self, partial_count: int) -> list[int]:
        """Return the distinct channels needed for `partial_count` partials.

        Deduplicated because the RPN block must be sent once per channel, not
        once per partial: sending it again mid-file is harmless but bloats the
        file and makes the event list hard to read in a DAW.
        """
        needed = min(partial_count, len(_USABLE_CHANNELS))
        return list(_USABLE_CHANNELS[:needed])

    def _rpn_bend_range_messages(self, channel: int) -> list[tuple[float, mido.Message]]:
        """Build the RPN 0 sequence that widens this channel's pitch bend range.

        The order is fixed by the MIDI spec and all five messages matter:

            CC 101 = 0    select RPN, high byte  -> RPN 0 = pitch bend range
            CC 100 = 0    select RPN, low byte
            CC 6   = N    data entry MSB         -> N semitones
            CC 38  = 0    data entry LSB         -> plus 0 cents
            CC 101/100 = 127  RPN null           -> deselect, so nothing else
                                                   can accidentally edit it

        Everything is stamped at absolute time 0.0 so it lands before the first
        note (see invariant 2 in the module docstring).
        """
        rpn_msb, rpn_lsb = _RPN_PITCH_BEND_RANGE
        null_msb, null_lsb = _RPN_NULL
        values = (
            (_CC_RPN_MSB, rpn_msb),
            (_CC_RPN_LSB, rpn_lsb),
            (_CC_DATA_ENTRY_MSB, self._bend_range),
            (_CC_DATA_ENTRY_LSB, 0),
            (_CC_RPN_MSB, null_msb),
            (_CC_RPN_LSB, null_lsb),
        )
        return [
            (0.0, mido.Message("control_change", channel=channel, control=cc, value=value))
            for cc, value in values
        ]

    def _partial_messages(
        self, partial: Partial, channel: int, velocity: int
    ) -> list[tuple[float, mido.Message]]:
        """Turn one partial's trajectory into absolutely-timed MIDI messages.

        The shape of the output is always:

            pitchwheel, expression, note_on          (all at the start time)
            pitchwheel/expression pairs              (every EXPRESSION_INTERVAL_MS)
            note_off                                 (at the end time)

        The opening pitchwheel and expression deliberately precede the note-on:
        a synth reads its bend and expression state at the moment a note
        starts, so sending them afterwards makes the attack sound at the wrong
        pitch and the wrong loudness.

        Args:
            partial: The trajectory to convert.
            channel: MIDI channel to place it on.
            velocity: Note-on velocity, already scaled against the track peak.

        Returns:
            (absolute_seconds, message) pairs, in the order described above.
            Empty if the partial carries no usable pitch.
        """
        frames = partial.frames
        if not frames:
            return []

        mean_hz = partial.mean_frequency_hz
        if mean_hz <= 0.0:
            # Cannot name a note for a partial with no positive frequency.
            # Dropping it is right, but never quietly.
            logger.warning(
                "skipping partial %s: mean frequency %.3f Hz is not a pitch",
                partial.partial_id,
                mean_hz,
            )
            return []

        # THE NOMINAL NOTE. Rounding the amplitude-weighted mean gives the
        # nearest equal-tempered note; everything the trajectory does either
        # side of it is carried by pitch bend, so no microtonal detail is lost
        # (up to the configured bend range).
        note = max(0, min(_MIDI_MAX_VALUE, round(hz_to_midi(mean_hz))))
        nominal_hz = midi_to_hz(note)

        times, freqs, amps = self._sample_trajectory(partial)

        # Cents from the nominal note, vectorised. Cents rather than Hz for the
        # same reason the tracker uses them: the ear hears ratios, and pitch
        # bend is itself defined in semitones.
        safe_freqs = np.maximum(freqs, _MIN_POSITIVE_HZ)
        cents = 1200.0 * np.log2(safe_freqs / nominal_hz)

        peak = float(np.max(amps)) if amps.size else 0.0
        end_index = _note_off_index(amps, peak)

        messages: list[tuple[float, mido.Message]] = []
        last_bend: int | None = None
        last_expression: int | None = None

        for i in range(end_index + 1):
            bend = bend_value(float(cents[i]), self._bend_range)
            expression = _expression_for(float(amps[i]), peak)

            # Skip repeats. A steady partial would otherwise emit 50 identical
            # controller messages per second, which bloats the file and buries
            # the real gestures when a human opens it in a piano roll. Because
            # repeats are skipped rather than rewritten, the last message we
            # emitted always still describes the current state.
            if bend != last_bend:
                messages.append(
                    (float(times[i]), mido.Message("pitchwheel", channel=channel, pitch=bend))
                )
                last_bend = bend
            if expression != last_expression:
                messages.append(
                    (
                        float(times[i]),
                        mido.Message(
                            "control_change",
                            channel=channel,
                            control=_CC_EXPRESSION,
                            value=expression,
                        ),
                    )
                )
                last_expression = expression

            if i == 0:
                # Note-on goes after the first bend/expression pair but still
                # at the same absolute time; stable sorting preserves that.
                messages.append(
                    (
                        float(times[0]),
                        mido.Message("note_on", channel=channel, note=note, velocity=velocity),
                    )
                )

        # Enforce a minimum sounding length so a one-frame partial does not
        # become a zero-length note that DAWs quietly discard.
        minimum_end = float(times[0]) + MIN_NOTE_DURATION_MS / 1000.0
        note_off_time = max(float(times[end_index]), minimum_end)
        messages.append(
            (
                note_off_time,
                mido.Message("note_off", channel=channel, note=note, velocity=0),
            )
        )
        return messages

    def _sample_trajectory(
        self, partial: Partial
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Resample a partial onto an even grid every EXPRESSION_INTERVAL_MS.

        Analysis frames arrive at the STFT hop rate (86 Hz at the defaults),
        which is not the rate we want to write controllers at -- it is both
        denser than a MIDI file needs and tied to an analysis setting the user
        can change. Resampling decouples the two: change `hop_size` and the
        exported controller density stays put.

        Linear interpolation is used because the trajectory is already a
        piecewise-linear approximation of a curve; fitting anything fancier
        would invent detail the analysis never measured.

        Returns:
            (times_s, frequencies_hz, amplitudes) as float64 arrays of equal
            length, with the first sample exactly at the partial's start and
            the last exactly at its end.
        """
        frames = partial.frames
        frame_times = np.array([f.time_s for f in frames], dtype=np.float64)
        frame_freqs = np.array([f.frequency_hz for f in frames], dtype=np.float64)
        frame_amps = np.array([f.amplitude for f in frames], dtype=np.float64)

        start = float(frame_times[0])
        end = float(frame_times[-1])
        interval_s = EXPRESSION_INTERVAL_MS / 1000.0

        # `ceil` then `linspace` (rather than `arange`) so the endpoint is hit
        # exactly: the final controller value must describe where the partial
        # actually ended, not where a fixed step happened to land.
        steps = max(1, int(math.ceil((end - start) / interval_s))) if end > start else 1
        times = np.linspace(start, end, steps + 1, dtype=np.float64)

        if len(frames) == 1:
            # np.interp needs at least two points to interpolate between; a
            # single-frame partial is simply constant.
            freqs = np.full_like(times, frame_freqs[0])
            amps = np.full_like(times, frame_amps[0])
        else:
            freqs = np.interp(times, frame_times, frame_freqs)
            amps = np.interp(times, frame_times, frame_amps)
        return times, freqs, amps

    # -------------------------------------------------------------------------
    # Assembly and writing
    # -------------------------------------------------------------------------

    @staticmethod
    def _assemble_track(
        timed_messages: list[tuple[float, mido.Message | mido.MetaMessage]],
    ) -> mido.MidiTrack:
        """Convert absolutely-timed messages into a mido track of DELTA times.

        THIS IS THE STEP EVERYONE GETS WRONG. `Message.time` in mido is not an
        absolute timestamp; it is the gap since the previous message in the
        same track. Interleaving several partials in absolute time and then
        differencing once -- here, and only here -- is what keeps that correct.

        The sort is stable, so messages sharing a tick keep the order they were
        appended in. That is what puts the RPN block ahead of the first note
        and a partial's opening bend ahead of its note-on.

        TWO THINGS THE STABLE SORT ALONE DOES NOT GIVE US
        ------------------------------------------------
        Insertion order is *partial* order, not *time* order, so two rules have
        to be applied explicitly once the whole stream is visible:

        1. AT AN EQUAL TICK, A NOTE-OFF COMES FIRST. When one partial ends
           exactly where another begins on the same channel and note, plain
           insertion order can put the new note-on before the old note-off --
           and the note that just started is then killed on the same tick.
           Hence the secondary sort key.

        2. THE SAME NOTE CANNOT SOUND TWICE AT ONCE ON ONE CHANNEL. MIDI
           addresses a sounding note by (channel, note) and nothing else, so a
           second note-on for a pair already sounding is not a second note: the
           next note-off ends both, and the one after it is an orphan. Real
           analyses hit this -- a vibrato tone's upper harmonics routinely
           fragment into several partials that round to the same note and
           overlap -- and only where partials share a channel, i.e. an
           `export_bundle` track or an `export()` past the 15-partial wall.
           Those overlaps are merged into the note already playing, which is
           the only well-formed thing MIDI can express, and reported.
        """
        ticked = [(seconds_to_ticks(seconds), msg) for seconds, msg in timed_messages]
        # Rule 1: note-off sorts ahead of everything else sharing its tick. The
        # RPN block and the opening bend are unaffected -- they rank equal to
        # note_on, so insertion order still decides between them.
        ticked.sort(key=lambda pair: (pair[0], 0 if pair[1].type == "note_off" else 1))

        track = mido.MidiTrack()
        previous_tick = 0
        # Rule 2: how many times each (channel, note) is currently "on".
        sounding: dict[tuple[int, int], int] = {}
        merged = 0
        for tick, msg in ticked:
            if msg.type in ("note_on", "note_off"):
                key = (msg.channel, msg.note)
                depth = sounding.get(key, 0)
                if msg.type == "note_on":
                    sounding[key] = depth + 1
                    if depth > 0:
                        # Already sounding: fold this partial into the note that
                        # is playing rather than emit an unsoundable duplicate.
                        merged += 1
                        continue
                else:
                    sounding[key] = max(0, depth - 1)
                    if sounding[key] > 0:
                        # Another partial still holds this note; releasing here
                        # would cut it short. The final note-off does the work.
                        continue
            msg.time = tick - previous_tick
            previous_tick = tick
            track.append(msg)

        if merged:
            logger.warning(
                "%d note(s) landed on a channel and note already sounding and were merged "
                "into the note in progress; MIDI cannot sound one note twice on one "
                "channel. Export the track on its own to give every partial its own "
                "channel.",
                merged,
            )
        return track

    @staticmethod
    def _save(midi: mido.MidiFile, destination: Path) -> Path:
        """Create the parent directory, write the file, and return its path.

        Raises:
            ExportError: If the file could not be written.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            midi.save(str(destination))
        except (OSError, ValueError) as exc:
            # Same boundary discipline as the WAV exporter: third-party and OS
            # failures become our own error type so the API can catch one thing.
            logger.exception("MIDI export failed for %s", destination)
            raise ExportError(f"could not write MIDI to {destination}: {exc}") from exc
        return destination

    @staticmethod
    def _warn_on_channel_reuse(partial_count: int, label: str, what: str) -> None:
        """Warn when there are more partials than channels to isolate them on.

        Overflow partials are placed on already-used channels rather than
        dropped -- losing them outright would be worse -- but their pitch bends
        then fight with the partial they share a channel with. Saying so is the
        whole point of this method: the failure is real, so it must be visible.
        """
        limit = len(_USABLE_CHANNELS)
        if partial_count > limit:
            logger.warning(
                "track %r has %d %s but only %d MIDI channels are usable; %d partial(s) "
                "share channels and their pitch bends will interact. Consider exporting "
                "fewer partials, or accept the reduced bend accuracy.",
                label,
                partial_count,
                what,
                limit,
                partial_count - limit,
            )


# =============================================================================
# Module-level helpers
# =============================================================================


def bend_value(cents_deviation: float, bend_range_semitones: int) -> int:
    """Convert a pitch deviation in cents to a 14-bit MIDI pitch bend value.

    Args:
        cents_deviation: Signed distance from the note's nominal pitch.
        bend_range_semitones: The channel's configured bend range.

    Returns:
        A value in [-8192, 8191], clamped at the range limits.
    """
    fraction = cents_deviation / (bend_range_semitones * 100.0)
    return max(-PITCH_BEND_MAX - 1, min(PITCH_BEND_MAX, round(fraction * PITCH_BEND_MAX)))


def seconds_to_ticks(seconds: float) -> int:
    """Convert absolute time in seconds to MIDI ticks at the assumed tempo."""
    beats_per_second = ASSUMED_TEMPO_BPM / 60.0
    return round(seconds * beats_per_second * TICKS_PER_BEAT)


def _track_peak_amplitude(partials: tuple[Partial, ...]) -> float:
    """Loudest peak among a track's partials, used to normalise velocities.

    Normalising against the track rather than against each partial keeps the
    *relative* loudness of partials intact: a quiet upper harmonic should come
    out as a quiet note, not as a full-velocity one.
    """
    return max((p.peak_amplitude for p in partials), default=0.0)


def _track_energy(track: Track) -> float:
    """Total energy of a track, for ranking which tracks make it into a bundle."""
    return sum(p.energy for p in track.partials)


def _velocity_for(partial: Partial, track_peak: float) -> int:
    """Scale a partial's peak amplitude to a MIDI note-on velocity in 1..127.

    The floor is 1, not 0: velocity 0 is note-off in the MIDI wire format, so
    a note-on with velocity 0 would end a note instead of starting one -- a
    silent, baffling bug for anyone importing the file.
    """
    if track_peak <= 0.0:
        return 1
    scaled = round(_MIDI_MAX_VALUE * partial.peak_amplitude / track_peak)
    return max(1, min(_MIDI_MAX_VALUE, scaled))


def _expression_for(amplitude: float, peak_amplitude: float) -> int:
    """Scale an instantaneous amplitude to a CC 11 expression value in 0..127.

    Unlike velocity this may legitimately be 0 -- CC 11 is a continuous volume
    scaler, and 0 simply means "inaudible right now", not "stop the note".
    """
    if peak_amplitude <= 0.0:
        return 0
    scaled = int(round(_MIDI_MAX_VALUE * amplitude / peak_amplitude))
    return max(0, min(_MIDI_MAX_VALUE, scaled))


def _note_off_index(amplitudes: NDArray[np.float64], peak_amplitude: float) -> int:
    """Index of the sample where the note should stop sounding.

    A partial's amplitude envelope often has a very long, very quiet tail. Held
    as a MIDI note that tail becomes a note that overlaps everything after it,
    which looks wrong in a piano roll and, on a synth with its own envelope,
    sounds wrong too. So the note ends where the trajectory drops below
    `NOTE_OFF_THRESHOLD_DB` relative to its own peak.

    The scan starts at the peak, not at the beginning: a partial usually starts
    below the threshold while it is still attacking, and ending the note there
    would produce a note that stops before it has begun.

    Args:
        amplitudes: Resampled amplitude envelope.
        peak_amplitude: The maximum of that envelope.

    Returns:
        An index into `amplitudes`, always at least 1 where possible.
    """
    last = amplitudes.size - 1
    if last <= 0 or peak_amplitude <= 0.0:
        return max(last, 0)

    cutoff = peak_amplitude * (10.0 ** (NOTE_OFF_THRESHOLD_DB / 20.0))
    peak_index = int(np.argmax(amplitudes))
    below = np.flatnonzero(amplitudes[peak_index:] < cutoff)
    if below.size == 0:
        return last
    return min(last, max(1, peak_index + int(below[0])))


def _safe_meta_text(text: str) -> str:
    """Reduce a label to characters a MIDI meta event can carry.

    MIDI text events are bytes, and the de-facto encoding is Latin-1. A label
    containing anything outside it (a sharp sign, a curly quote) would raise
    when mido serialises the file, so unrepresentable characters are replaced
    rather than allowed to fail an export at the last step.
    """
    return text.encode("ascii", "replace").decode("ascii")


def _describe(tracks: list[Track]) -> str:
    """Render tracks as 'id:label' pairs for a log message.

    Named individually rather than counted, because "3 tracks were dropped" is
    not something a user can act on and "tracks 41, 47, 52 were dropped" is.
    """
    return ", ".join(f"{t.track_id}:{t.label!r}" for t in tracks)
