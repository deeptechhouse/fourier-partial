"""Core value objects and entities for fourier_partial.

READ THIS FIRST (for developers new to the project)
---------------------------------------------------
The whole system rests on one chain of ideas:

  1. A short window of audio can be described as a sum of sine waves.
     (This is the Fourier transform.)
  2. If we take many overlapping windows, each sine wave becomes a *trajectory*
     that drifts in pitch and loudness over time. We call that a PARTIAL.
  3. A partial, played on its own, is a single pure evolving tone. Play all the
     partials together and you get (almost) the original sound back.
  4. "Almost" is not good enough, so we define a RESIDUAL track as the literal
     time-domain subtraction:  residual = original - sum(partials).
     That makes the reconstruction EXACT by construction. See TDD section 7.

So: Peak -> PartialFrame -> Partial -> Track -> AnalysisResult.

Encapsulation note: value objects below are frozen (immutable) dataclasses.
Entities (Partial, Track, AnalysisResult) keep their state private and expose
read-only properties that return defensive copies, so no caller can reach in
and corrupt an analysis after it has been computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# ADJUSTABLE PARAMETERS — analysis defaults
# =============================================================================
# These are the knobs you will actually want to turn. Every one of them is a
# trade-off, and most of them are trading time resolution against frequency
# resolution -- a trade you cannot win, only choose (the uncertainty principle;
# see TDD section 2).
# =============================================================================


@dataclass(frozen=True, slots=True)
class AnalysisParams:
    """Tunable parameters for the analysis pipeline.

    Frozen so that a result can always report exactly which settings produced
    it. To analyse with different settings, build a new instance.
    """

    # --- Sample rate ---------------------------------------------------------
    # ADJUSTABLE: everything is resampled to this before analysis. 44100 keeps
    # the full audible band. Lowering it speeds analysis but loses top end.
    sample_rate: int = 44_100

    # --- STFT geometry -------------------------------------------------------
    # ADJUSTABLE: window length in samples. THIS IS THE MOST IMPORTANT KNOB.
    #   Larger  -> finer frequency detail, blurrier in time (smears transients)
    #   Smaller -> sharper transients, coarser pitch detail
    # At 44100 Hz, 4096 samples = 92.9 ms window ~= 10.8 Hz frequency resolution.
    window_size: int = 4096

    # ADJUSTABLE: hop between successive windows, in samples.
    # 512 gives 8x overlap -- smooth trajectories, 8x the compute of no overlap.
    hop_size: int = 512

    # ADJUSTABLE: zero-padding factor. Does NOT add real resolution, but it
    # interpolates the spectrum so peak-finding lands more accurately.
    zero_pad_factor: int = 4

    # --- Peak picking --------------------------------------------------------
    # ADJUSTABLE: ignore spectral peaks quieter than this, relative to the
    # loudest peak in the same frame. Raise it to get fewer, cleaner partials.
    peak_threshold_db: float = -60.0

    # ADJUSTABLE: maximum peaks considered per frame. Caps worst-case cost.
    max_peaks_per_frame: int = 120

    # --- Partial tracking ----------------------------------------------------
    # ADJUSTABLE: how far (in cents, 100 cents = 1 semitone) a peak may drift
    # between frames and still be considered the same partial continuing.
    # Too small -> tracks fragment on vibrato. Too large -> tracks jump pitches.
    max_freq_drift_cents: float = 60.0

    # ADJUSTABLE: a partial may vanish for this many frames and still resume
    # (helps with brief maskings). Set to 0 for strict, unforgiving tracking.
    max_gap_frames: int = 3

    # ADJUSTABLE: partials shorter than this are discarded (they fall into the
    # residual instead). Filters out spectral noise masquerading as tones.
    min_partial_duration_ms: float = 50.0

    # --- Grouping ------------------------------------------------------------
    # ADJUSTABLE: tolerance when deciding whether a partial is the Nth harmonic
    # of a candidate fundamental, in cents.
    harmonic_tolerance_cents: float = 50.0

    # ADJUSTABLE: hard ceiling on tracks returned to the UI. Beyond this the
    # quietest partials are folded into the residual. Keeps the interface sane.
    max_tracks: int = 200

    def __post_init__(self) -> None:
        """Validate invariants at construction, so bad settings fail loudly."""
        if self.window_size <= 0 or (self.window_size & (self.window_size - 1)) != 0:
            raise ValueError(f"window_size must be a positive power of two, got {self.window_size}")
        if not 0 < self.hop_size <= self.window_size:
            raise ValueError(f"hop_size must be in (0, window_size], got {self.hop_size}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.max_tracks <= 0:
            raise ValueError(f"max_tracks must be positive, got {self.max_tracks}")

    # --- Derived, human-meaningful quantities --------------------------------

    @property
    def window_duration_ms(self) -> float:
        """How much time one analysis window spans."""
        return 1000.0 * self.window_size / self.sample_rate

    @property
    def frequency_resolution_hz(self) -> float:
        """Smallest frequency difference two tones can have and stay separable.

        This is the frequency half of the uncertainty trade-off. Multiply it by
        `window_duration_ms / 1000` and you get roughly 1 -- that product is the
        thing you cannot shrink.
        """
        return self.sample_rate / self.window_size

    @property
    def frame_rate_hz(self) -> float:
        """How many analysis frames are produced per second."""
        return self.sample_rate / self.hop_size

    @classmethod
    def for_duration(cls, duration_s: float, **overrides: object) -> AnalysisParams:
        """Pick settings appropriate to how long the audio is.

        WHY LONG FILES NEED DIFFERENT SETTINGS
        --------------------------------------
        The defaults are tuned for a phrase, where detail is free. Applied to
        ten minutes they are not merely slow, they are the wrong trade:

          * hop 512 gives 86 analysis frames per second. Over ten minutes that
            is ~51,700 frames, and tracking has to reconcile every active
            partial against every peak in each one.
          * zero-padding 4x makes each frame's spectrum 8193 bins wide. That
            buys sub-bin peak accuracy which matters enormously for a single
            sustained note and very little for a whole piece of music.

        So for long input we spend less per frame and analyse fewer of them.
        The cost is real and worth stating plainly: coarser onset timing and
        slightly less precise pitch. The alternative is not "a better analysis
        of ten minutes" -- it is no analysis of ten minutes at all.

        Thresholds are deliberately coarse: three tiers, not a formula, because
        a continuous function of duration would make results change subtly and
        inexplicably as a user trims a file.

        Args:
            duration_s: Length of the audio about to be analysed.
            **overrides: Any field to pin regardless of duration.

        Returns:
            Settings for that length, with `overrides` applied last.
        """
        # ADJUSTABLE: the tier boundaries, in seconds, and what each tier costs.
        if duration_s <= 45.0:
            # Phrase. Full detail -- this is the case the tool was designed for.
            tuned: dict[str, object] = {}
        elif duration_s <= 180.0:
            # A minute or three. Halve the frame rate and the spectrum width;
            # roughly a quarter of the work for barely-noticeable loss.
            tuned = {"hop_size": 1024, "zero_pad_factor": 2, "max_peaks_per_frame": 80}
        else:
            # Long-form. Frame rate down to ~21/s and no zero-padding.
            # Parabolic interpolation still recovers sub-bin frequency, so
            # pitch stays usable; it is onset timing that gets coarser.
            tuned = {
                "hop_size": 2048,
                "zero_pad_factor": 1,
                "max_peaks_per_frame": 60,
                # FEWER tracks, not more, despite a long recording containing
                # more distinct material. Three independent limits bite here
                # and they all point the same way:
                #
                #   1. Rendering. Every track gets a buffer the length of the
                #      whole file. Measured on ten minutes: 300 tracks took
                #      5.7 minutes to render, 40 takes well under one.
                #   2. The browser. Each track is ~106 MB decoded at ten
                #      minutes, so even this many is a lot to hold; 300 would
                #      be ~32 GB and would simply crash the tab.
                #   3. The interface. Nobody operates 300 rows.
                #
                # The material that does not get its own row is not lost -- the
                # residual absorbs it exactly, so it still plays and still
                # exports. Only its claim to a separate row goes away, and the
                # user is told so in the warnings.
                "max_tracks": 40,
                # Longer minimum duration: at 21 frames/s a 50 ms partial is
                # one frame, which is noise, not a tone.
                "min_partial_duration_ms": 120.0,
            }

        return cls(**{**tuned, **overrides})  # type: ignore[arg-type]


# =============================================================================
# VALUE OBJECTS — immutable records produced by the pipeline
# =============================================================================


@dataclass(frozen=True, slots=True)
class Peak:
    """One spectral peak found in one STFT frame.

    Attributes:
        frame_index: Which analysis frame this came from.
        frequency_hz: Sub-bin-accurate frequency, from parabolic interpolation.
        amplitude: Linear (not dB) magnitude.
        phase_rad: Phase in radians, needed for click-free resynthesis.
    """

    frame_index: int
    frequency_hz: float
    amplitude: float
    phase_rad: float

    @property
    def amplitude_db(self) -> float:
        """Amplitude expressed in decibels; -inf for true silence."""
        return 20.0 * math.log10(self.amplitude) if self.amplitude > 0.0 else -math.inf

    def cents_from(self, other: Peak) -> float:
        """Pitch distance to another peak, in cents (100 cents = 1 semitone).

        Cents rather than Hz because a 10 Hz drift is enormous at 100 Hz and
        negligible at 5000 Hz; the ear works in ratios, so tracking should too.
        """
        if self.frequency_hz <= 0.0 or other.frequency_hz <= 0.0:
            return math.inf
        return 1200.0 * math.log2(self.frequency_hz / other.frequency_hz)


@dataclass(frozen=True, slots=True)
class PartialFrame:
    """A single point along a partial's trajectory through time."""

    time_s: float
    frequency_hz: float
    amplitude: float
    phase_rad: float


class TrackKind(Enum):
    """What sort of thing a track represents."""

    PARTIAL = "partial"
    """A single tracked sine wave -- one pure evolving tone."""

    HARMONIC_GROUP = "harmonic_group"
    """A fundamental plus its harmonics, bundled as one playable voice."""

    RESIDUAL = "residual"
    """Everything the partials did not capture: transients, breath, noise.

    Not a failure bucket. Sharp-in-time content provably cannot be sparse in
    frequency, so this material has nowhere else to go. See TDD section 2.
    """


# =============================================================================
# ENTITIES — private state, read-only access
# =============================================================================


class Partial:
    """One tracked sine wave: a trajectory of frequency and amplitude in time.

    Built up frame by frame by the tracker, then frozen in practice by only
    ever being read afterwards. Frames are kept private and handed out as an
    immutable tuple so a consumer cannot reorder or mutate the trajectory.
    """

    __slots__ = ("_partial_id", "_frames")

    def __init__(self, partial_id: int, frames: Sequence[PartialFrame] | None = None) -> None:
        """Create a partial, optionally pre-populated with frames."""
        self._partial_id = partial_id
        self._frames: list[PartialFrame] = list(frames) if frames else []

    # --- Mutation goes through methods that protect the invariant ------------

    def append_frame(self, frame: PartialFrame) -> None:
        """Extend the trajectory. Frames must arrive in ascending time order."""
        if self._frames and frame.time_s <= self._frames[-1].time_s:
            raise ValueError(
                f"frames must be strictly time-ordered; "
                f"got {frame.time_s}s after {self._frames[-1].time_s}s"
            )
        self._frames.append(frame)

    # --- Read-only access ----------------------------------------------------

    @property
    def partial_id(self) -> int:
        """Stable identifier for this partial within its analysis."""
        return self._partial_id

    @property
    def frames(self) -> tuple[PartialFrame, ...]:
        """The trajectory, as an immutable tuple (defensive copy)."""
        return tuple(self._frames)

    @property
    def start_time_s(self) -> float:
        """When this partial is born."""
        return self._frames[0].time_s if self._frames else 0.0

    @property
    def end_time_s(self) -> float:
        """When this partial dies."""
        return self._frames[-1].time_s if self._frames else 0.0

    @property
    def duration_s(self) -> float:
        """How long the partial sounds for."""
        return self.end_time_s - self.start_time_s

    @property
    def mean_frequency_hz(self) -> float:
        """Amplitude-weighted average frequency.

        Weighted, because the pitch you actually perceive is the pitch it held
        while it was loud -- a quiet dying tail should not drag the answer.
        """
        if not self._frames:
            return 0.0
        total_amp = sum(f.amplitude for f in self._frames)
        if total_amp <= 0.0:
            return sum(f.frequency_hz for f in self._frames) / len(self._frames)
        return sum(f.frequency_hz * f.amplitude for f in self._frames) / total_amp

    @property
    def peak_amplitude(self) -> float:
        """Loudest point on the trajectory."""
        return max((f.amplitude for f in self._frames), default=0.0)

    @property
    def energy(self) -> float:
        """Rough total energy; used for ranking which partials matter most."""
        return sum(f.amplitude**2 for f in self._frames)

    def __len__(self) -> int:
        """Number of frames in the trajectory."""
        return len(self._frames)

    def __repr__(self) -> str:
        """Developer-facing summary."""
        return (
            f"Partial(id={self._partial_id}, {self.mean_frequency_hz:.1f}Hz, "
            f"{self.duration_s:.3f}s, {len(self._frames)} frames)"
        )


class Track:
    """A playable, visualisable, exportable lane in the interface.

    A track wraps one or more partials (or, for the residual, no partials at
    all) together with the rendered audio that represents it.
    """

    __slots__ = ("_track_id", "_kind", "_label", "_partials", "_audio")

    def __init__(
        self,
        track_id: int,
        kind: TrackKind,
        label: str,
        partials: Sequence[Partial] | None = None,
        audio: NDArray[np.float32] | None = None,
    ) -> None:
        """Create a track. `audio` is the rendered mono/stereo buffer, if known."""
        self._track_id = track_id
        self._kind = kind
        self._label = label
        self._partials: tuple[Partial, ...] = tuple(partials) if partials else ()
        self._audio = audio

    # --- Read-only access ----------------------------------------------------

    @property
    def track_id(self) -> int:
        """Stable identifier within the analysis."""
        return self._track_id

    @property
    def kind(self) -> TrackKind:
        """Whether this is a partial, a harmonic group, or the residual."""
        return self._kind

    @property
    def label(self) -> str:
        """Human-readable name, e.g. 'A4 + harmonics' or 'Residual'."""
        return self._label

    @property
    def partials(self) -> tuple[Partial, ...]:
        """Constituent partials, immutable."""
        return self._partials

    @property
    def audio(self) -> NDArray[np.float32] | None:
        """Rendered audio, as a read-only view so callers cannot scribble on it."""
        if self._audio is None:
            return None
        view = self._audio.view()
        view.flags.writeable = False
        return view

    def attach_audio(self, audio: NDArray[np.float32]) -> None:
        """Attach rendered audio. Set once by the renderer; not for general use."""
        if self._audio is not None:
            raise ValueError(f"track {self._track_id} already has rendered audio")
        self._audio = audio

    @property
    def is_exportable_as_midi(self) -> bool:
        """Whether MIDI export is meaningful for this track.

        The residual is noise; it has no pitch, so writing it as MIDI notes
        would be inventing information that is not there.
        """
        return self._kind is not TrackKind.RESIDUAL and len(self._partials) > 0

    def __repr__(self) -> str:
        """Developer-facing summary."""
        return f"Track(id={self._track_id}, {self._kind.value}, {self._label!r})"


class AnalysisResult:
    """The complete output of one analysis: every track, plus provenance.

    Holds the invariant that makes the product's central promise true:
    summing every track's audio reproduces the original sample for sample.
    """

    __slots__ = ("_sample_rate", "_duration_s", "_tracks", "_params", "_original")

    def __init__(
        self,
        sample_rate: int,
        duration_s: float,
        tracks: Sequence[Track],
        params: AnalysisParams,
        original: NDArray[np.float32],
    ) -> None:
        """Assemble a result from rendered tracks and the source audio."""
        self._sample_rate = sample_rate
        self._duration_s = duration_s
        self._tracks: tuple[Track, ...] = tuple(tracks)
        self._params = params
        self._original = original

    @property
    def sample_rate(self) -> int:
        """Sample rate of every track in this result."""
        return self._sample_rate

    @property
    def duration_s(self) -> float:
        """Duration of the analysed audio."""
        return self._duration_s

    @property
    def tracks(self) -> tuple[Track, ...]:
        """All tracks, immutable."""
        return self._tracks

    @property
    def params(self) -> AnalysisParams:
        """The settings that produced this result."""
        return self._params

    @property
    def original(self) -> NDArray[np.float32]:
        """The analysed audio, as a read-only view.

        Exposed because the API serves it directly for "play everything": the
        tracks sum to it exactly, so re-summing forty buffers to reproduce a
        number we already hold would be both slower and less certain.

        Read-only for the same reason `Track.audio` is -- this array is the
        reference the reconstruction invariant is measured against, and a
        caller that scribbled on it would make the invariant unfalsifiable.
        """
        view = self._original.view()
        view.flags.writeable = False
        return view

    @property
    def residual_track(self) -> Track | None:
        """The residual track, if one was produced."""
        return next((t for t in self._tracks if t.kind is TrackKind.RESIDUAL), None)

    def reconstruction_error(self) -> float:
        """Max absolute sample error between sum(tracks) and the original.

        Should be 0.0 (or float32 rounding dust) because the residual is
        defined as the exact difference. If this is ever meaningfully non-zero,
        the renderer has a bug -- which is precisely why it is a CI assertion.
        """
        rendered = [t.audio for t in self._tracks if t.audio is not None]
        if not rendered:
            return math.inf

        # Accumulate in place rather than np.stack-ing every track together.
        # stack materialises n_tracks copies of the whole file at once: at ten
        # minutes that is ~106 MB per track, so a dozen tracks would allocate
        # over a gigabyte inside what looks like a read-only diagnostic.
        # Adding one track at a time costs a single accumulator instead.
        mixed = np.zeros(self._original.shape, dtype=np.float64)
        for audio in rendered:
            mixed += audio
        mixed -= self._original.astype(np.float64)
        return float(np.max(np.abs(mixed)))

    def __repr__(self) -> str:
        """Developer-facing summary."""
        return (
            f"AnalysisResult({self._duration_s:.2f}s @ {self._sample_rate}Hz, "
            f"{len(self._tracks)} tracks)"
        )


# =============================================================================
# MIDI note helpers
# =============================================================================

# ADJUSTABLE: reference tuning. Change to 432.0 and every note name shifts.
A4_HZ: float = 440.0
A4_MIDI_NOTE: int = 69

_NOTE_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)  # fmt: skip


def hz_to_midi(frequency_hz: float) -> float:
    """Convert a frequency to a fractional MIDI note number.

    Fractional on purpose: 60.5 means "halfway between C4 and C#4", which the
    MIDI exporter turns into note 60 plus a pitch-bend. Rounding here would
    throw away exactly the microtonal detail that makes partials interesting.
    """
    if frequency_hz <= 0.0:
        raise ValueError(f"frequency must be positive, got {frequency_hz}")
    return A4_MIDI_NOTE + 12.0 * math.log2(frequency_hz / A4_HZ)


def midi_to_hz(midi_note: float) -> float:
    """Convert a (possibly fractional) MIDI note number back to frequency."""
    return A4_HZ * (2.0 ** ((midi_note - A4_MIDI_NOTE) / 12.0))


def pitch_label(frequency_hz: float) -> str:
    """The note name alone, with no cent deviation: 'G4', not 'G4 -1c'.

    This is the grouping key the interface needs. A ten-minute recording
    surfaces the same pitch many times -- each occurrence is legitimately its
    own track, because they are separate notes -- but a flat list of forty rows
    reading "G4 -1c", "G4", "G4 -2c" is far less useful than one collapsible
    "G4" row that opens into its seven occurrences.

    Deliberately drops the cents: two performances of the same note that differ
    by a few cents are the same note, and keeping the deviation would split
    them into separate groups for no reason a musician would recognise.
    """
    if frequency_hz <= 0.0:
        return "unpitched"
    nearest = round(hz_to_midi(frequency_hz))
    return f"{_NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}"


def note_name(frequency_hz: float) -> str:
    """Human-readable note name with cent deviation, e.g. 'A4 +12c'."""
    midi = hz_to_midi(frequency_hz)
    nearest = round(midi)
    cents = round((midi - nearest) * 100.0)
    name = _NOTE_NAMES[nearest % 12]
    octave = nearest // 12 - 1
    if cents == 0:
        return f"{name}{octave}"
    return f"{name}{octave} {cents:+d}c"


@dataclass(frozen=True, slots=True)
class ReconstructionReport:
    """Diagnostic summary comparing tracks against the original.

    Used by tests and shown in the UI so a user can see, numerically, how much
    of their sound became tones and how much stayed noise.
    """

    max_sample_error: float
    partial_energy_ratio: float
    """Fraction of total energy captured by pitched partials (0.0 - 1.0).

    High for sustained, harmonic material. Low for percussive or noisy
    material -- not a defect, but the uncertainty principle showing up in
    your metrics.
    """
    partial_count: int
    residual_energy_ratio: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_exact(self) -> bool:
        """Whether reconstruction is exact to within float32 rounding."""
        return self.max_sample_error < 1e-6
