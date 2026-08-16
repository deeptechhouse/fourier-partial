"""Partial tracking — turning snapshots into voices.

WHAT THIS DOES
--------------
Peak picking gives us, for each moment, a list of frequencies present. But a
list of frequencies per frame is not yet music. If a violin note slides from
440 Hz to 450 Hz over half a second, we currently have hundreds of unrelated
dots. We want one object that says "a tone that started at 440 and rose to
450". That object is a partial, and building it is *tracking*.

THE ALGORITHM (McAulay-Quatieri, 1986)
--------------------------------------
Frame by frame, match each new peak to an existing partial:

    BIRTH    a peak matches nothing  -> start a new partial
    CONTINUE a peak is close in pitch to a partial's last frame -> extend it
    DEATH    a partial goes unmatched too long -> close it out

"Close in pitch" is measured in cents rather than Hz, because a 10 Hz drift is
enormous at 100 Hz and inaudible at 5 kHz. The ear hears ratios; the tracker
should too.

WHY GAPS ARE TOLERATED
----------------------
A partial can be briefly masked by a louder neighbour and vanish for a frame
or two. Killing it immediately would shatter one long tone into a dozen
fragments, each too short to survive the minimum-duration filter -- so a real
sustained note would silently become residual noise. `max_gap_frames` buys
the tolerance that prevents that.

COST OF THE WHOLE THING
-----------------------
A 30-second clip at the defaults is ~2500 frames of up to 120 peaks each. The
work is therefore done strictly frame by frame: nothing here ever builds a
structure spanning the whole file. Within a frame, candidate pairs are pruned
by frequency (see `_peaks_within_drift`) so we never enumerate the full
active x peaks product either.

STATUS: implemented (Phase 1). Contract unchanged from the scaffold.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right

from backend.analysis.stft import frame_time
from backend.core.errors import InvalidParameterError
from backend.core.types import AnalysisParams, Partial, PartialFrame, Peak

# =============================================================================
# Numerical constants
# =============================================================================

# Slack applied to the frequency window used to *pre-select* candidate peaks.
# NOT a tuning knob -- the real tolerance is params.max_freq_drift_cents (an
# ADJUSTABLE parameter, defined in backend/core/types.py). This factor only
# widens the binary search by a floating-point hair so that a peak sitting
# exactly on the tolerance boundary can never be lost to rounding before
# `_cents_cost` gets to make the actual accept/reject decision.
_SEARCH_WINDOW_SLACK: float = 1e-9


class TrackingConfigError(InvalidParameterError, ValueError):
    """Raised when tracking parameters cannot produce sensible results.

    WHY IT INHERITS BOTH
    --------------------
    `backend/core/errors.py` states the rule the whole system depends on:
    every error raised here inherits from `FourierPartialError`, so the API
    layer can catch one type and be certain nothing escapes past it, and every
    error carries a `user_message` that is safe to send to a client. A bare
    `ValueError` satisfies neither -- it would sail straight through
    `except FourierPartialError` in `backend/api/routes.py`, and the handler
    that reaches for `exc.user_message` would find nothing there. Since this
    IS "a supplied analysis parameter is out of range", `InvalidParameterError`
    is exactly the right parent, and it is what the sibling analysis modules
    (`peaks.py`, `stft.py`) already raise for the same class of problem.

    `ValueError` is kept in the bases purely for backward compatibility, so
    any caller written against the original signature and catching
    `ValueError` keeps working.
    """


# =============================================================================
# INTERNAL STATE — one open partial, plus a cached view of its tail
# =============================================================================


class _ActivePartial:
    """A partial that is still open and may yet be continued by a new peak.

    WHY THIS CLASS EXISTS
    ---------------------
    On every frame the tracker needs exactly two facts about each open
    partial: where it last was in pitch, and when it was last matched. Both
    could be read back off the `Partial` itself -- but `Partial.frames` hands
    out a defensive *copy* of the entire trajectory (rightly so: nothing
    outside the partial should be able to mutate an analysis). Rebuilding a
    2500-element tuple once per candidate pair would turn tracking from a
    linear scan into a quadratic crawl.

    So this class keeps a two-field mirror of the *tail* of the trajectory.
    The mirror is private and the only way to move it is `extend()`, which
    appends to the underlying partial in the same breath -- so the cache
    cannot drift out of sync with the thing it is caching.
    """

    __slots__ = ("_partial", "_last_frame_index", "_last_frequency_hz")

    def __init__(self, partial: Partial, last_frame_index: int, last_frequency_hz: float) -> None:
        """Wrap a freshly-born partial together with its tail state.

        Args:
            partial: The partial being tracked. Must already hold its first
                frame, so that `last_frame_index` / `last_frequency_hz`
                describe something real.
            last_frame_index: Frame in which this partial was last matched.
            last_frequency_hz: Frequency of that last matched frame.
        """
        self._partial = partial
        self._last_frame_index = last_frame_index
        self._last_frequency_hz = last_frequency_hz

    # --- Read-only access ----------------------------------------------------

    @property
    def partial(self) -> Partial:
        """The partial being built. Callers read it; only `extend()` grows it."""
        return self._partial

    @property
    def last_frame_index(self) -> int:
        """Index of the most recent frame this partial was matched in."""
        return self._last_frame_index

    @property
    def last_frequency_hz(self) -> float:
        """Frequency at the tail of the trajectory -- what new peaks match against.

        Matching against the *last* frequency rather than, say, the mean is
        what lets a glissando stay a single partial: the target moves along
        with the tone instead of anchoring it to where it started.
        """
        return self._last_frequency_hz

    # --- The single mutation point -------------------------------------------

    def extend(self, frame: PartialFrame, frame_index: int) -> None:
        """Append a frame to the partial and move the cached tail with it.

        Args:
            frame: The new point on the trajectory. `Partial.append_frame`
                enforces strictly increasing `time_s`, and the caller
                guarantees it by matching each partial at most once per frame.
            frame_index: The analysis frame this point came from.
        """
        # Append FIRST: if the partial rejects the frame (time order violated,
        # which would mean a bug in the caller) we must not have already moved
        # the cache, or the two would disagree from then on.
        self._partial.append_frame(frame)
        self._last_frame_index = frame_index
        self._last_frequency_hz = frame.frequency_hz


# =============================================================================
# COST
# =============================================================================


def _cents_cost(peak_hz: float, reference_hz: float, max_drift_cents: float) -> float:
    """Absolute pitch distance in cents, or `inf` if it exceeds the tolerance.

    Args:
        peak_hz: Frequency of the candidate peak.
        reference_hz: Frequency the peak is being compared against (the tail
            of a partial's trajectory).
        max_drift_cents: Largest jump, in cents, still considered "the same
            tone continuing".

    Returns:
        `abs(cents)` when within tolerance, otherwise `math.inf`.

    Shared by the public `match_cost` and by the tracking loop, so there is
    exactly one definition of "how far apart are these two frequencies" in
    the module and no way for the two call sites to disagree.
    """
    # Guard: log2 of zero is -inf and of a negative number is undefined. A
    # non-positive frequency is not a tone at all (DC or a numerical accident),
    # so it can never continue a partial.
    if peak_hz <= 0.0 or reference_hz <= 0.0:
        return math.inf

    cents = abs(1200.0 * math.log2(peak_hz / reference_hz))
    # `inf` is not "a very bad match" -- it means "not a candidate", and the
    # caller uses that to drop the pair entirely rather than rank it last.
    return cents if cents <= max_drift_cents else math.inf


def match_cost(peak: Peak, partial: Partial, params: AnalysisParams) -> float:
    """Cost of assigning `peak` to continue `partial`. Lower is better.

    Args:
        peak: The candidate peak from the current frame.
        partial: An open partial that might continue through this peak.
        params: Supplies `max_freq_drift_cents`, the acceptance tolerance.

    Returns:
        Absolute pitch distance in cents, or `inf` if the peak is outside
        `params.max_freq_drift_cents` and therefore not a candidate at all.
        Also `inf` for an empty partial, which has no last frame to continue
        from, and for any non-positive frequency.

    Using pitch distance alone keeps this simple and predictable. An amplitude
    term can be added later if tracks are observed jumping between a loud and
    a quiet partial that happen to share a frequency.

    Note on performance: reading `partial.frames` copies the whole trajectory
    (that copy is deliberate -- see `Partial.frames`), so this function is
    O(len(partial)). It is the right entry point for tests and for alternative
    trackers, but `track_partials` deliberately does not call it in its inner
    loop; it uses `_ActivePartial`'s cached tail instead. Both go through
    `_cents_cost`, so they cannot disagree about what counts as a match.
    """
    frames = partial.frames
    if not frames:
        # Nothing to continue from. A partial with no frames has no position
        # in pitch, so no peak can be "close" to it.
        return math.inf
    return _cents_cost(peak.frequency_hz, frames[-1].frequency_hz, params.max_freq_drift_cents)


# =============================================================================
# CANDIDATE PRUNING
# =============================================================================


def _peaks_within_drift(
    sorted_frequencies: list[float], reference_hz: float, max_drift_cents: float
) -> tuple[int, int]:
    """Half-open index range of peaks that *could* be within the drift window.

    Args:
        sorted_frequencies: Frequencies of this frame's peaks, ascending.
        reference_hz: Frequency to search around.
        max_drift_cents: Tolerance in cents.

    Returns:
        `(lo, hi)` such that only `sorted_frequencies[lo:hi]` need testing.

    WHY: without this, every frame would test every open partial against every
    peak. In noisy material that is ~600 partials x 120 peaks = 72,000 pairs
    per frame, times 2500 frames -- far too slow in Python. Because "within N
    cents" is a *ratio* window around `reference_hz`, and the peaks are sorted
    by frequency, two binary searches reduce that to only the handful of peaks
    that stand any chance. The arithmetic is identical; the work is not.
    """
    if reference_hz <= 0.0 or not sorted_frequencies:
        return 0, 0
    # cents = 1200*log2(f_peak / f_ref), so |cents| <= D is exactly
    # f_ref / 2^(D/1200) <= f_peak <= f_ref * 2^(D/1200).
    ratio = 2.0 ** (max_drift_cents / 1200.0) * (1.0 + _SEARCH_WINDOW_SLACK)
    lo = bisect_left(sorted_frequencies, reference_hz / ratio)
    hi = bisect_right(sorted_frequencies, reference_hz * ratio)
    return lo, hi


# =============================================================================
# THE TRACKER
# =============================================================================


def track_partials(frames: list[list[Peak]], params: AnalysisParams) -> list[Partial]:
    """Link per-frame peaks into partials.

    Args:
        frames: Per-frame peak lists in ascending time order.
        params: Drift tolerance, gap tolerance, minimum duration.

    Returns:
        Partials that survived the minimum-duration filter, unordered.

    Raises:
        TrackingConfigError: If a tolerance is negative. A negative drift or
            gap allowance cannot mean anything -- it would silently disable
            tracking entirely and hand the caller an empty result that looks
            like "this audio had no tones in it".

    Implementation notes:
        - Greedy nearest-match in ascending cost order is sufficient and is
          what the original paper describes. Optimal assignment (Hungarian
          algorithm) is available if greedy proves inadequate, but measure
          before reaching for it -- it is markedly slower and the audible
          difference is usually nil.
        - Each peak may be claimed by at most one partial, and each partial
          may claim at most one peak per frame.
        - Partials shorter than `params.min_partial_duration_ms` are dropped
          here. They are not lost: because the residual is computed as an
          exact subtraction downstream, anything discarded reappears in the
          residual track automatically. Nothing ever falls out of the system.
    """
    # --- Step 0: reject settings that cannot mean anything -------------------
    if params.max_freq_drift_cents < 0.0:
        raise TrackingConfigError(
            f"max_freq_drift_cents must be >= 0, got {params.max_freq_drift_cents}"
        )
    if params.max_gap_frames < 0:
        raise TrackingConfigError(f"max_gap_frames must be >= 0, got {params.max_gap_frames}")
    if params.min_partial_duration_ms < 0.0:
        raise TrackingConfigError(
            f"min_partial_duration_ms must be >= 0, got {params.min_partial_duration_ms}"
        )

    max_drift_cents = float(params.max_freq_drift_cents)
    max_gap_frames = int(params.max_gap_frames)

    # `active` holds every partial still eligible for continuation; `finished`
    # collects the ones that have died. Keeping them in separate lists is what
    # bounds the per-frame work: a 30-second file may produce thousands of
    # partials in total, but only a few hundred are ever open at once.
    active: list[_ActivePartial] = []
    finished: list[Partial] = []

    # Monotonic across the whole run, incremented on every birth. Ids are NOT
    # contiguous in the returned list, because short partials are filtered out
    # at the end -- an id is a name, not an index.
    next_partial_id = 0

    for frame_index, peaks in enumerate(frames):
        # --- Step a: retire partials that have been silent too long ----------
        # A partial matched at frame `last` may still be resumed while it has
        # skipped no more than `max_gap_frames` frames. The most recent frame
        # it could have been matched in is `frame_index - 1`, so the oldest
        # still-resumable `last` is `frame_index - 1 - max_gap_frames`.
        # With max_gap_frames = 0 this collapses to "must have been matched in
        # the immediately preceding frame", i.e. strict, gapless tracking.
        oldest_resumable = frame_index - 1 - max_gap_frames
        if active:
            still_active: list[_ActivePartial] = []
            for candidate in active:
                if candidate.last_frame_index < oldest_resumable:
                    finished.append(candidate.partial)
                else:
                    still_active.append(candidate)
            active = still_active

        if not peaks:
            # No peaks to match or to be born from. Note we do NOT retire the
            # active partials here -- an empty frame is exactly the brief
            # masking that `max_gap_frames` exists to survive.
            continue

        # --- Step b: enumerate the plausible (partial, peak) pairs -----------
        # Peaks arrive sorted by descending amplitude (see peaks.py). Sort a
        # *view* of them by frequency so the drift window can be found by
        # binary search; `order` maps a position in that sorted view back to
        # the peak's original index, which stays the canonical identity used
        # for the "already claimed" bookkeeping and for birth order.
        order = sorted(range(len(peaks)), key=lambda index: peaks[index].frequency_hz)
        sorted_frequencies = [peaks[index].frequency_hz for index in order]

        # (cost, active index, peak index). All three entries are numbers, so
        # the plain tuple sort below is total and deterministic: ties in cost
        # break by older partial first, then by louder peak first.
        candidates: list[tuple[float, int, int]] = []
        for active_index, open_partial in enumerate(active):
            lo, hi = _peaks_within_drift(
                sorted_frequencies, open_partial.last_frequency_hz, max_drift_cents
            )
            for sorted_position in range(lo, hi):
                peak_index = order[sorted_position]
                cost = _cents_cost(
                    sorted_frequencies[sorted_position],
                    open_partial.last_frequency_hz,
                    max_drift_cents,
                )
                # `inf` means "not a candidate at all" -- drop it rather than
                # rank it, so it can never be greedily accepted.
                if cost < math.inf:
                    candidates.append((cost, active_index, peak_index))

        # --- Step c: greedy assignment, cheapest match first ------------------
        # Sorting by cost and taking pairs in order is the McAulay-Quatieri
        # matching rule: the most confident continuation gets first refusal on
        # its peak, and every later pair that wanted either half is skipped.
        candidates.sort()

        # bytearray rather than set(): membership is an O(1) index either way,
        # but this allocates once per frame and never rehashes.
        partial_claimed = bytearray(len(active))
        peak_claimed = bytearray(len(peaks))

        # --- Step d: extend the matched partials -----------------------------
        # One timestamp for the whole frame, computed once. `frame_time` uses
        # the window CENTRE, which is the honest answer to "when did this
        # happen" -- see stft.frame_time.
        time_s = frame_time(frame_index, params)

        for _cost, active_index, peak_index in candidates:
            if partial_claimed[active_index] or peak_claimed[peak_index]:
                continue
            partial_claimed[active_index] = 1
            peak_claimed[peak_index] = 1

            peak = peaks[peak_index]
            # Safe against Partial.append_frame's strict time ordering: each
            # partial is claimed at most once per frame (the flag above), and
            # `time_s` strictly increases with `frame_index`, so no partial can
            # ever be handed two frames with the same timestamp.
            active[active_index].extend(
                PartialFrame(
                    time_s=time_s,
                    frequency_hz=peak.frequency_hz,
                    amplitude=peak.amplitude,
                    phase_rad=peak.phase_rad,
                ),
                frame_index,
            )

        # --- Step e: every unclaimed peak is the birth of a new partial ------
        # Iterated in the ORIGINAL (descending-amplitude) peak order, so the
        # loudest new tone in a frame gets the lowest id. Purely cosmetic, but
        # it makes logs and debugging sessions much easier to read.
        for peak_index, peak in enumerate(peaks):
            if peak_claimed[peak_index]:
                continue
            newborn = Partial(next_partial_id)
            next_partial_id += 1
            newborn.append_frame(
                PartialFrame(
                    time_s=time_s,
                    frequency_hz=peak.frequency_hz,
                    amplitude=peak.amplitude,
                    phase_rad=peak.phase_rad,
                )
            )
            active.append(_ActivePartial(newborn, frame_index, peak.frequency_hz))

    # --- Step f: the audio ended, so everything still open dies here ---------
    # A partial that was sounding when the clip ran out is not an error; it is
    # a note that outlasted the recording. Close it where the audio stops.
    finished.extend(open_partial.partial for open_partial in active)

    # --- Step g: minimum-duration filter -------------------------------------
    # Spectral noise produces short-lived peaks that wander in and out of the
    # drift window; they track as real partials for two or three frames and
    # then vanish. Resynthesising them would add a fizzing, granular texture
    # that was not in the source. Dropping them costs nothing: the residual is
    # `original - sum(tracks)`, so every discarded partial reappears there
    # verbatim and the reconstruction stays exact.
    # Compared in milliseconds (duration * 1000) rather than by converting the
    # threshold to seconds, so the arithmetic matches the parameter's own units
    # exactly and no float division sits between the setting and the decision.
    min_duration_ms = float(params.min_partial_duration_ms)
    return [partial for partial in finished if partial.duration_s * 1000.0 >= min_duration_ms]
