"""Resynthesis — turning partials back into sound.

WHAT THIS DOES
--------------
Each partial is a recipe: at time t, oscillate at frequency f(t) with
amplitude a(t). Rendering means running an oscillator that follows that
recipe. Do it for every partial in a track, sum them, and you have that
track's audio.

THE TWO THINGS THAT GO WRONG
----------------------------
1. CLICKS. If amplitude jumps between frames, you get a click at every frame
   boundary -- 86 clicks per second, which sounds like buzzing rather than
   music. Fix: interpolate amplitude linearly across the hop.

2. PHASE DISCONTINUITY. If you restart the oscillator's phase each frame, the
   waveform tears. Fix: integrate frequency continuously to get phase, so the
   oscillator never jumps. The classic McAulay-Quatieri approach fits a cubic
   polynomial to phase; linear frequency interpolation with accumulated phase
   is simpler, nearly as good, and much easier to get right. Start there.

THE INVARIANT THIS MODULE OWNS
------------------------------
After rendering, `build_residual` computes:

        residual = original - sum(rendered tracks)

which makes  sum(all tracks) == original  true by *arithmetic*, not by
approximation. Every track can be individually imperfect and the mix is still
sample-exact. This is the single most important property in the system; do not
replace the subtraction with a model, an estimate, or a filter.

STATUS: implemented (Phase 2).

HOW THE THREE FUNCTIONS FIT TOGETHER (read this before editing)
---------------------------------------------------------------
    render_partial  -> one sine trajectory  -> float32 buffer
    render_track    -> sum of its partials  -> float32 buffer
    build_residual  -> original minus every rendered track -> float32 buffer

Note the arithmetic discipline that runs through all three: every *sum* is
accumulated in float64 and cast to float32 exactly once, at the very end.
float32 carries about 7 significant decimal digits. Add several hundred
float32 buffers together in float32 and the rounding error of each addition
compounds into a faint, wideband hiss -- a noise floor you can actually hear
on a quiet passage, and one large enough to break the invariant test in
`tests/test_reconstruction.py`. Accumulating in float64 (about 16 digits)
keeps that error roughly a billion times smaller than the final float32
rounding step, so the cast at the end is the only error that survives.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from backend.core.errors import ReconstructionInvariantError
from backend.core.types import AnalysisParams, Partial, Track, TrackKind

# =============================================================================
# ADJUSTABLE PARAMETERS — resynthesis
# =============================================================================


# ADJUSTABLE: length of the raised-cosine fade applied at a partial's birth and
# at its death, in milliseconds.
#   Longer  -> gentler onsets, but the attack of a plucked/struck note softens
#              and very short partials lose most of their body to the fade.
#   Shorter -> crisper attacks, but below roughly 1 ms the amplitude step at
#              the edge becomes an audible click again.
# 3 ms (132 samples at 44100 Hz) is the usual compromise: well below the ear's
# ~10 ms transient-fusion window, so it is inaudible as a fade, yet long enough
# to bandlimit the edge discontinuity out of the audible range.
FADE_MS: float = 3.0

# Not adjustable -- this is mathematics, not taste. One full turn of the unit
# circle, the amount of phase a 1 Hz oscillator advances in one second.
_TWO_PI: float = 2.0 * np.pi

# Not adjustable: a raised-cosine window needs at least two samples to be a
# ramp rather than a single zero that would simply delete a sample.
_MIN_FADE_SAMPLES: int = 2


# =============================================================================
# PUBLIC API
# =============================================================================


def render_partial(
    partial: Partial, params: AnalysisParams, num_samples: int
) -> NDArray[np.float32]:
    """Render one partial to a mono buffer.

    Args:
        partial: The trajectory to synthesise.
        params: Sample rate and hop size.
        num_samples: Length of the output buffer, matching the original.

    Returns:
        Mono float32 audio, zero outside the partial's lifetime.

    Implementation sketch:
        for each consecutive pair of frames:
            interpolate frequency linearly across the hop
            interpolate amplitude linearly across the hop
            accumulate phase:  phase += 2*pi*f[n]/sample_rate
            write  amp[n] * sin(phase)

        Apply a short fade (a few ms) at birth and death so partials do not
        switch on and off abruptly.

    Implementation notes (what the code below actually does, and why):
        * A partial is a sequence of frames, and sound only exists *between*
          frames -- a frame on its own is an instant, and an instant has no
          duration. So we walk consecutive PAIRS of frames and fill the sample
          range each pair spans. A partial with fewer than two frames therefore
          renders as silence. That is correct, not a bug: the tracker's minimum
          duration filter should have removed such partials already.

        * PHASE IS CARRIED ACROSS SEGMENT BOUNDARIES. This is the subtle part.
          Each segment computes its phase as a running total (`np.cumsum`) of
          the per-sample phase increments, then starts from wherever the
          previous segment finished. If instead we reset phase to 0 (or to the
          frame's measured `phase_rad`) at every frame, the waveform would jump
          discontinuously ~86 times a second and you would hear a buzz rather
          than a tone. We seed the very first segment from the first frame's
          measured phase so the partial starts in step with the original audio.

        * Everything is vectorised per segment: numpy builds the whole ramp of
          frequencies, amplitudes and phases for the segment in one go. A
          Python `for` loop over individual samples would be roughly two orders
          of magnitude slower and no clearer.

        * Accumulate in float64, cast to float32 once at the end -- see the
          module docstring for why.
    """
    # Guard the degenerate cases first so the main loop can assume sane input.
    # `max(..., 0)` means a negative request yields an empty buffer rather than
    # a numpy error, which keeps callers from having to special-case it.
    length = max(int(num_samples), 0)
    output = np.zeros(length, dtype=np.float64)
    render_partial_into(output, partial, params)
    return output.astype(np.float32)


def render_partial_into(
    destination: NDArray[np.float64], partial: Partial, params: AnalysisParams
) -> None:
    """Add one partial's audio into `destination`, in place.

    This is where the rendering actually happens; `render_partial` is a thin
    wrapper that allocates a buffer and calls this.

    WHY IT WORKS ON A SLICE
    -----------------------
    A partial occupies the stretch of time between its first and last frame and
    is silent everywhere else. Rendering it into a whole-file buffer therefore
    spends almost all of its effort writing zeros over zeros. For a plucked
    note in a ten-minute recording that is twenty thousand useful samples out
    of twenty-six million.

    So we size a scratch buffer to the partial's own lifetime, render into
    that, fade its edges, and add it into `destination` at the right offset.
    Everything about the maths below is unchanged -- only the extent is.

    Args:
        destination: Float64 accumulator, modified in place. Must be at least
            as long as the audio being reconstructed.
        partial: The trajectory to synthesise.
        params: Sample rate.
    """
    length = destination.shape[0]

    # `partial.frames` hands back a defensive tuple copy; grab it ONCE. Calling
    # the property inside the loop would rebuild that tuple on every iteration.
    frames = partial.frames
    if length == 0 or len(frames) < 2:
        # No buffer to write into, or no segment to render (a single frame is
        # an instant with no duration). Silence is the right answer.
        return

    sample_rate_for_span = float(params.sample_rate)
    # The partial's whole extent. Frames are strictly time-ordered (enforced by
    # Partial.append_frame), so the first and last frames bound every segment.
    span_start = min(max(int(round(frames[0].time_s * sample_rate_for_span)), 0), length)
    span_stop = min(max(int(round(frames[-1].time_s * sample_rate_for_span)), 0), length)
    if span_stop <= span_start:
        return

    # The scratch buffer, sized to the partial rather than to the file.
    output = np.zeros(span_stop - span_start, dtype=np.float64)
    base = span_start

    sample_rate = float(params.sample_rate)

    # The running phase, in radians, carried from one segment to the next.
    # Seeded from the first frame's measured phase so the rendered partial
    # lines up with the original waveform instead of starting arbitrarily.
    phase_carry = float(frames[0].phase_rad)

    # Half-open bounds [active_start, active_stop) of everything we actually
    # wrote, so the fades below know where the partial is audible. Initialised
    # inverted so the first written segment overwrites both.
    active_start = span_stop
    active_stop = 0

    for start_frame, end_frame in zip(frames[:-1], frames[1:], strict=True):
        # Map frame times onto sample indices. `round` (not floor) keeps the
        # rendered onset within half a sample of the analysed onset. These are
        # the TRUE, unclipped bounds of the segment -- the clamped ones come
        # below, and the difference between the two matters (see `offset`).
        raw_i0 = int(round(start_frame.time_s * sample_rate))
        raw_i1 = int(round(end_frame.time_s * sample_rate))

        # How many samples the segment spans in total, whether or not they all
        # land inside the buffer. Every ramp below is measured against THIS,
        # never against the surviving portion.
        span = raw_i1 - raw_i0
        if span <= 0:
            # Two frames rounded to the same sample (possible with a very small
            # hop). There is no duration to integrate over, so there is nothing
            # to write and no phase to advance.
            continue

        start_frequency_hz = start_frame.frequency_hz
        frequency_delta_hz = end_frame.frequency_hz - start_frequency_hz

        # Clamp into the buffer. A partial whose trajectory runs past the end
        # of the audio (or starts before it) is legal input; we render only the
        # part that lands inside the buffer rather than raising.
        i0 = min(max(raw_i0, 0), length)
        i1 = min(max(raw_i1, 0), length)
        n = i1 - i0

        if n > 0:
            # How far into the segment the first WRITTEN sample sits. Non-zero
            # only when the segment starts before the buffer does.
            offset = i0 - raw_i0

            # Ramp from offset/span at the first written sample up to
            # (offset+n-1)/span at the last. Measured against the segment's
            # true span, so a segment clipped by either end of the buffer
            # renders the slice of the trajectory that actually belongs there
            # instead of squeezing the whole glide into the surviving samples.
            # Deliberately NOT reaching 1.0: the value 1.0 belongs to the first
            # sample of the NEXT segment, which starts exactly where this one
            # ends. That off-by-one is what makes frequency and amplitude
            # continuous across the whole partial rather than repeating a
            # sample at each seam.
            ramp = (offset + np.arange(n, dtype=np.float64)) / span

            # Linear interpolation of the two things the ear notices most.
            frequency_hz = start_frequency_hz + frequency_delta_hz * ramp
            amplitude = start_frame.amplitude + (end_frame.amplitude - start_frame.amplitude) * ramp

            # Phase is the running integral of frequency. In discrete time that
            # integral is a cumulative sum of per-sample increments, and the
            # increment for a frequency f is 2*pi*f/sample_rate radians.
            # Starting the sum from `phase_carry` is what stops the waveform
            # tearing.
            increments = _TWO_PI * frequency_hz / sample_rate

            # `phase_carry` is the phase at the segment's first sample, which
            # is `offset` samples before the first one we write. Advance it
            # across that skipped head so the seed lands on the sample it
            # actually describes.
            phase_at_i0 = phase_carry + _accrued_phase(
                start_frequency_hz, frequency_delta_hz, span, offset, sample_rate
            )

            # `phase_at_i0` is the phase AT this segment's first sample, so the
            # first written sample must come out as sin(phase_at_i0) exactly --
            # `phase_rad` is defined as the value you feed straight to `sin` at
            # `time_s` (see the zero-phase rotation in analysis/stft.py, pinned
            # by tests/test_windowing.py). `cumsum` includes increments[0] in
            # its first element, which would advance the whole partial by one
            # sample; subtracting it back puts the seed exactly where the
            # analysis says it belongs. One sample sounds like nothing, but it
            # is a full 2*pi*f/sample_rate of phase error -- inaudible at
            # 100 Hz and very nearly ANTI-PHASE at 15 kHz, which would make the
            # residual subtraction add energy instead of cancelling it.
            phase = (phase_at_i0 + np.cumsum(increments)) - increments[0]

            # Segments are contiguous and disjoint by construction (this
            # segment's i1 is the next segment's i0), so a plain assignment is
            # correct and avoids accumulating into stale values.
            # `- base` converts the absolute sample index into an index in the
            # scratch buffer, which starts at the partial's first frame rather
            # than at the start of the file.
            output[i0 - base : i1 - base] = amplitude * np.sin(phase)

            active_start = min(active_start, i0)
            active_stop = max(active_stop, i1)

        # Advance the carry across the WHOLE segment, written or not, so the
        # next segment picks up mid-cycle at the right point. Done in closed
        # form rather than from `phase[-1]` because the written region may be
        # only part of the segment (or none of it, when the trajectory is
        # temporarily outside the buffer) -- and a partial that re-enters the
        # buffer with a stale phase would be just as out of step as one that
        # never had a seed at all.
        phase_carry += _accrued_phase(
            start_frequency_hz, frequency_delta_hz, span, span, sample_rate
        )

    if active_stop <= active_start:
        # Every segment was clipped away; the scratch buffer is still all
        # zeros, so there is nothing to add.
        return

    # Soften the two hard edges where the partial switches on and off. The
    # fades belong to THIS partial, which is why they are applied to the
    # scratch buffer before it is mixed in -- fading the shared accumulator
    # would attenuate every partial that happens to overlap this one's edges.
    _apply_edge_fades(output, active_start - base, active_stop - base, params.sample_rate)

    # Mix in. `+=` rather than assignment: several partials of the same track
    # routinely overlap in time, and each must add to the others.
    destination[span_start:span_stop] += output


def render_track(track: Track, params: AnalysisParams, num_samples: int) -> NDArray[np.float32]:
    """Render every partial in a track and sum them.

    Args:
        track: The track whose partials should be synthesised.
        params: Sample rate and hop size, passed through to `render_partial`.
        num_samples: Length of the output buffer, matching the original.

    Returns:
        Mono float32 audio: the sum of every constituent partial. A track with
        no partials (the residual, before it is built) renders as silence.

    A harmonic-group track can hold dozens of partials, so the summation runs
    in a float64 accumulator and casts once at the end -- same discipline as
    everywhere else in this module.
    """
    length = max(int(num_samples), 0)

    # float64 accumulator: adding N float32 buffers in float32 compounds
    # rounding error N times over; in float64 that error stays ~1e-9 times
    # smaller than the single float32 cast we do at the end.
    mixed = np.zeros(length, dtype=np.float64)

    # `track.partials` is already an immutable tuple, so iterating it cannot be
    # disturbed by anything else holding a reference to the track.
    #
    # Accumulate in place rather than `mixed += render_partial(...)`. That
    # version allocated and zeroed a WHOLE-FILE buffer for every partial, no
    # matter how briefly the partial actually sounded. On a phrase it is merely
    # wasteful; on a ten-minute file it is fatal -- 26.5 million samples
    # allocated per partial, of which a plucked note might occupy twenty
    # thousand. `render_partial_into` allocates the partial's lifetime instead,
    # which for typical material is a reduction of two to three orders of
    # magnitude in both allocation and arithmetic.
    for partial in track.partials:
        render_partial_into(mixed, partial, params)

    return mixed.astype(np.float32)


def build_residual(original: NDArray[np.float32], rendered: list[Track], track_id: int) -> Track:
    """Construct the residual as an exact time-domain subtraction.

    Args:
        original: The source audio, mono float32.
        rendered: Every track that already has audio attached.
        track_id: Identifier to assign to the new residual track.

    Returns:
        A RESIDUAL-kind Track whose audio is `original - sum(rendered)`.

    Raises:
        ReconstructionInvariantError: If a rendered track's audio has a
            different shape from `original`. That can only happen if the
            renderer was called with a mismatched `num_samples`, and silently
            broadcasting or truncating would break the invariant in a way that
            is very hard to debug later -- so fail loudly, here, instead.

    This function is the reason `AnalysisResult.reconstruction_error()` should
    always return ~0. Accumulate in float64 and cast once at the end, so
    summing several hundred float32 buffers does not accrue rounding error
    that shows up as a noise floor.

    Why this is a subtraction and not an estimate:
        Partials are sinusoids, and sinusoids are the wrong shape for
        transients, breath and noise -- provably so, since content that is
        sharp in time cannot be sparse in frequency (TDD section 2). So there
        will always be leftovers. Defining the residual as literally
        `original - sum(partials)` means the leftovers are captured exactly,
        whatever they are, and `sum(all tracks) == original` holds as a matter
        of arithmetic. Model the residual instead -- a noise synthesiser, a
        filter bank -- and the guarantee degrades into an approximation.
    """
    if not rendered:
        return Track(track_id, TrackKind.RESIDUAL, "Residual", (), original.copy())

    # float64 accumulator, shaped like the original so the subtraction below
    # cannot silently broadcast.
    mixed = np.zeros_like(original, dtype=np.float64)

    for track in rendered:
        # `Track.audio` returns a READ-ONLY view. We only ever read from it --
        # writing would corrupt the track a user can still solo and export.
        audio = track.audio
        if audio is None:
            # A track without rendered audio contributes nothing. Skipping it
            # keeps the invariant true for the tracks that DO have audio, which
            # is exactly what AnalysisResult.reconstruction_error() checks.
            continue
        if audio.shape != original.shape:
            raise ReconstructionInvariantError(
                f"track {track.track_id} rendered {audio.shape} samples "
                f"but the original is {original.shape}"
            )
        # Reading `audio` (float32) into a float64 accumulator: numpy upcasts.
        mixed += audio

    # THE CENTRAL INVARIANT, in one line. Subtract in float64 so no information
    # is lost before the single, final cast to float32.
    residual = (original.astype(np.float64) - mixed).astype(np.float32)

    return Track(track_id, TrackKind.RESIDUAL, "Residual", (), residual)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _accrued_phase(
    start_frequency_hz: float,
    frequency_delta_hz: float,
    span: int,
    steps: int,
    sample_rate: float,
) -> float:
    """Phase accumulated over the first `steps` sample-steps of a linear glide.

    Args:
        start_frequency_hz: Frequency at the segment's first sample.
        frequency_delta_hz: End frequency minus start frequency.
        span: Total length of the segment in samples (must be > 0).
        steps: How many one-sample steps to integrate over, 0 <= steps <= span.
        sample_rate: Samples per second.

    Returns:
        Radians accrued, i.e. `sum(2*pi*f(k)/sample_rate for k in 1..steps)`.

    This is the same rectangular sum `np.cumsum` computes in `render_partial`,
    written in closed form so it costs O(1) instead of O(steps). It is used for
    the parts of a segment that fall OUTSIDE the output buffer, where building
    the array would be wasted work (and, for a trajectory that starts far
    before the buffer, potentially a very large allocation) even though the
    phase still has to travel through them. Using the same summation rule in
    both places is what keeps the waveform continuous across the join.

    Derivation: f(k) = f0 + df*k/span, so
        sum_{k=1..m} 2*pi*f(k)/sr
          = (2*pi/sr) * (m*f0 + (df/span) * m*(m+1)/2)
    """
    return (_TWO_PI / sample_rate) * (
        steps * start_frequency_hz + frequency_delta_hz * steps * (steps + 1) / (2.0 * span)
    )


def _apply_edge_fades(buffer: NDArray[np.float64], start: int, stop: int, sample_rate: int) -> None:
    """Multiply a raised-cosine fade into the birth and death of a partial.

    Args:
        buffer: float64 audio to modify IN PLACE. Private to this module, so
            in-place mutation is safe -- nothing outside holds a reference yet.
        start: First sample index of the partial's audible region.
        stop: One past the last sample index of the audible region.
        sample_rate: Samples per second, used to convert `FADE_MS` to samples.

    Why a raised cosine rather than a straight line? A linear fade is
    continuous in value but not in slope, and that kink still radiates a little
    high-frequency energy. The raised cosine 0.5*(1 - cos(x)) starts and ends
    with zero slope, so the join is smooth in both value and derivative and the
    edge is genuinely inaudible.

    The fade is clamped to at most HALF the audible length. Without that clamp
    a 5 ms partial given a 3 ms fade-in and a 3 ms fade-out would have
    overlapping fades and be almost entirely erased -- the cure would be worse
    than the click.
    """
    audible_length = stop - start

    # ADJUSTABLE via FADE_MS at the top of this module.
    fade_samples = int(round(FADE_MS * sample_rate / 1000.0))

    # Integer division: with fade_samples <= audible_length // 2 the two fades
    # can meet in the middle but never overlap.
    fade_samples = min(fade_samples, audible_length // 2)

    if fade_samples < _MIN_FADE_SAMPLES:
        # The partial is too short for a meaningful fade. Leaving the edges
        # sharp is better than deleting the only samples it has.
        return

    # linspace(0, pi) makes cos run 1 -> -1, so the window runs 0.0 -> 1.0,
    # hitting exactly 0.0 on the first sample and exactly 1.0 on the last.
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_samples)))

    buffer[start : start + fade_samples] *= fade_in
    # The fade-out is the same curve read backwards -- symmetric by design, so
    # a partial that is born and dies has the same shape at both ends.
    buffer[stop - fade_samples : stop] *= fade_in[::-1]
