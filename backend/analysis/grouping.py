"""Grouping — making hundreds of partials into something a human can use.

THE PROBLEM
-----------
A 20-second phrase routinely yields 300-800 partials. A flat list of 800 rows
is not an interface; it is a wall. Worse, individual partials are often not
what a user means by "a sound" -- the 3rd harmonic of a cello note is not a
thing anyone wants to solo. The note is.

THE APPROACH
------------
Estimate fundamentals, then gather each fundamental's harmonics with it:

    1. Rank partials by energy.
    2. For the strongest ungrouped partial, treat its frequency as a candidate
       fundamental f0.
    3. Claim any ungrouped partial whose frequency is within
       `harmonic_tolerance_cents` of n*f0 for small integer n.
    4. If enough partials were claimed, emit a HARMONIC_GROUP track. Otherwise
       emit the partial alone as a PARTIAL track.
    5. Repeat until every partial is grouped or the track ceiling is hit.

Partials beyond `max_tracks` are dropped from the track list -- and, because
the residual is an exact subtraction, they reappear in the residual rather
than vanishing. The user still hears them; they just are not separately
soloable.

WHY GREEDY, LOUDEST-FIRST
-------------------------
The strongest partial in a mix is overwhelmingly likely to BE a fundamental
(harmonics are almost always quieter than the tone that produced them), so
seeding from the loudest ungrouped partial gets the important notes right
first and spends its mistakes on the quiet material at the bottom of the list.
This is a heuristic, not a theorem: a partial that is genuinely the Nth
harmonic of a note whose fundamental was never tracked will seed a group of
its own. That reads to the user as "the analysis heard a higher note", which
is honest -- the fundamental really was not in the analysis.

KNOWN LIMITATION (worth understanding before you tune anything)
---------------------------------------------------------------
Harmonic series overlap. At 50 cents tolerance a high partial can legitimately
satisfy two different fundamentals at once -- 1777 Hz is within 17 cents of
8*220 Hz *and* is a perfectly good lone tone. First seed wins, because a
partial can only be claimed once. Tightening `harmonic_tolerance_cents`
narrows the overlap but also loses genuinely stretched harmonics (piano
strings are famously inharmonic), so there is no setting that is right for
every sound.

STATUS: implemented (Phase 2). Contract unchanged from the scaffold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from backend.core.types import AnalysisParams, Partial, Track, TrackKind, note_name

logger = logging.getLogger(__name__)

# ADJUSTABLE: how many harmonics above the fundamental to look for. Beyond
# about 16 the harmonics are usually too quiet to matter and too likely to
# collide with another note's partials.
MAX_HARMONIC_NUMBER: int = 16

# ADJUSTABLE: a candidate f0 must gather at least this many partials (itself
# included) to be considered a harmonic group rather than a lone tone.
MIN_PARTIALS_PER_GROUP: int = 3

# ADJUSTABLE: a track must carry at least this share of the LOUDEST track's
# energy to earn a row in the interface. 0.005 = half a percent of the loudest.
#
# WHY THIS EXISTS. Noise and transients throw off short-lived peaks that
# occasionally survive tracking and then cluster into plausible-looking
# "harmonic groups" -- things like "D#6 +38c + 45 harmonics" built from three
# frames of hiss. They are inaudible (rounding to 0.0% of energy) but they are
# not invisible: on a real recording they outnumbered the genuine notes five to
# one, turning a readable four-row workbench into eighteen rows of junk.
#
# Dropping them costs the user nothing. Everything discarded here reappears in
# the residual, because the residual is computed as an exact subtraction
# downstream -- so it is still audible, still exportable, just no longer
# pretending to be a note. See the design document, section 7.2.
#
# Raise it for a stricter, shorter track list; lower it to 0.0 to see
# absolutely everything the tracker found (useful when debugging the tracker,
# unpleasant when actually using the tool).
MIN_TRACK_ENERGY_SHARE: float = 0.005

# ADJUSTABLE: default pitch tolerance, in cents, when deciding whether a
# frequency is the Nth harmonic of a fundamental. 50 cents is a quarter tone --
# half a semitone either way -- which is wide enough to absorb the slight
# sharpness of real harmonics (piano strings stretch; so do most struck and
# plucked instruments) without swallowing a neighbouring note.
# NOTE: the pipeline passes `params.harmonic_tolerance_cents` explicitly, so
# this default only applies to direct callers and tests. Both live in the same
# units, so changing either in isolation is safe.
DEFAULT_HARMONIC_TOLERANCE_CENTS: float = 50.0

#: Label used when a set of partials has no usable pitch to name it by.
#: Not an error state: a partial whose frames are all silent has no meaningful
#: frequency, and inventing a note name for it would be a lie in the UI.
_UNPITCHED_LABEL: str = "unpitched"


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A partial plus the two facts grouping needs about it. Internal DTO.

    WHY THIS EXISTS
    ---------------
    `Partial.mean_frequency_hz` and `Partial.energy` both walk the partial's
    entire trajectory every time you read them. The grouping loop asks for the
    frequency of every ungrouped partial on every pass, so reading the property
    directly would re-walk thousands of frames per seed and turn a linear scan
    into a quadratic one. Computing both once up front costs one pass over the
    data and keeps the loop below trivial to read.

    Frozen because these are a snapshot: if anything could edit them, the
    cached numbers could silently disagree with the partial they describe.
    """

    partial: Partial
    mean_frequency_hz: float
    energy: float
    # Cached for the same reason as the two above: `Partial.frames` returns a
    # defensive tuple copy, so reading start/end inside the grouping loop would
    # rebuild the whole trajectory once per comparison.
    start_time_s: float
    end_time_s: float


# ADJUSTABLE: how much of the shorter partial's life must coincide with the
# seed's for the two to count as parts of the same note. 0.0 would accept a
# single shared instant; 1.0 would demand total containment, which real
# harmonics never manage (they decay at different rates, so upper partials die
# first). A quarter is forgiving enough for that while still separating two
# performances of the same pitch at different moments.
MIN_TEMPORAL_OVERLAP: float = 0.25


def _overlaps_in_time(
    candidate: _Candidate, seed: _Candidate, min_overlap: float = MIN_TEMPORAL_OVERLAP
) -> bool:
    """Whether two partials sound at the same time, enough to be one note.

    A note's harmonics start together and decay together. Two partials that
    never coincide cannot belong to the same note however cleanly their
    frequencies line up -- and on a long recording that distinction is the
    difference between "a voice" and "every occurrence of this pitch in the
    piece, stacked".

    The overlap is measured against the SHORTER of the two partials, because a
    long sustained fundamental and a brief upper harmonic genuinely are one
    note; requiring the overlap to be large relative to the longer one would
    reject exactly that case.
    """
    overlap = min(candidate.end_time_s, seed.end_time_s) - max(
        candidate.start_time_s, seed.start_time_s
    )
    if overlap <= 0.0:
        return False

    shorter = min(
        candidate.end_time_s - candidate.start_time_s,
        seed.end_time_s - seed.start_time_s,
    )
    # A zero-length partial (single frame) has no duration to compare against.
    # Any positive overlap is all it could ever manage, so accept it.
    if shorter <= 0.0:
        return True

    return (overlap / shorter) >= min_overlap


def harmonic_number(
    frequency_hz: float,
    fundamental_hz: float,
    tolerance_cents: float = DEFAULT_HARMONIC_TOLERANCE_CENTS,
) -> int | None:
    """Which harmonic of `fundamental_hz` is `frequency_hz`, if any.

    Args:
        frequency_hz: The frequency being tested.
        fundamental_hz: The candidate fundamental to test it against.
        tolerance_cents: How far from an exact integer multiple the frequency
            may sit and still count. In cents (100 cents = 1 semitone), because
            the ear hears ratios: 5 Hz of error is glaring at 100 Hz and
            inaudible at 5 kHz, so a tolerance in Hz would be far too strict up
            high and far too loose down low.

    Returns:
        The integer harmonic number, or None if the frequency is not close to
        any integer multiple within `MAX_HARMONIC_NUMBER`.

    Examples:
        >>> harmonic_number(660.0, 220.0)
        3
        >>> harmonic_number(700.0, 220.0) is None   # 102 cents sharp of 3*220
        True
    """
    # Guard: a non-positive frequency is not a tone (it is DC, or a numerical
    # accident). Dividing by a non-positive fundamental, or taking log2 of a
    # non-positive ratio, is meaningless -- so refuse rather than produce a
    # number that looks like an answer.
    if frequency_hz <= 0.0 or fundamental_hz <= 0.0:
        return None

    # Nearest integer multiple. Round rather than floor: we want the closest
    # candidate harmonic, and the tolerance test below is what actually decides
    # whether "closest" is close enough.
    n = round(frequency_hz / fundamental_hz)

    # n == 0 means the frequency is BELOW half the fundamental -- it cannot be
    # a harmonic of it (harmonics only go up). n > MAX_HARMONIC_NUMBER means we
    # are so far up the series that a coincidental match is more likely than a
    # real one; see the constant's comment.
    if n < 1 or n > MAX_HARMONIC_NUMBER:
        return None

    # Distance from the exact harmonic, in cents. `n * fundamental_hz` is
    # positive here (both factors are), so the log is always well defined.
    cents = abs(1200.0 * math.log2(frequency_hz / (n * fundamental_hz)))
    return n if cents <= tolerance_cents else None


def group_partials(partials: list[Partial], params: AnalysisParams) -> list[Track]:
    """Cluster partials into playable tracks.

    Args:
        partials: All surviving partials from tracking.
        params: Harmonic tolerance and the track ceiling.

    Returns:
        Tracks ordered by descending energy, so the most significant material
        appears at the top of the interface. Does NOT include the residual --
        that is the renderer's job, since it can only be computed once every
        other track has been rendered.

    Implementation notes:
        - Loudest-first greedy seeding, one pass. Each partial is claimed by at
          most one track, so the tracks partition (a subset of) the input and
          no partial is ever rendered twice.
        - Partials still ungrouped when `params.max_tracks` is reached are
          DROPPED from the returned list. They are NOT lost: the residual is
          computed downstream as `original - sum(tracks)`, so every dropped
          partial reappears there verbatim and the user still hears it. It
          simply is not separately soloable. `pipeline.verify()` raises a
          user-visible warning when this happens so the interface never
          silently claims to be showing everything.
    """
    # --- Step 1: snapshot the two numbers we need, sorted loudest first ------
    # Energy (sum of squared amplitudes) is the ranking that matters here: a
    # brief loud partial and a long quiet one can share a peak amplitude, but
    # the one carrying more energy is the one the listener would miss.
    candidates = sorted(
        (
            _Candidate(p, p.mean_frequency_hz, p.energy, p.start_time_s, p.end_time_s)
            for p in partials
        ),
        key=lambda candidate: candidate.energy,
        reverse=True,
    )

    ungrouped: list[_Candidate] = candidates
    tracks: list[Track] = []
    next_track_id = 0

    # --- Step 2: seed a group from the loudest partial still unclaimed -------
    while ungrouped and len(tracks) < params.max_tracks:
        seed = ungrouped.pop(0)
        fundamental_hz = seed.mean_frequency_hz

        # A partial with no positive mean frequency has no pitch to group
        # around and no note name to show. Emit no track at all -- the residual
        # absorbs it, exactly as it absorbs everything else we drop.
        if fundamental_hz <= 0.0:
            continue

        # --- Step 3: claim this fundamental's harmonics -----------------------
        # Single pass, partitioning `ungrouped` into "claimed by this seed" and
        # "still available". Rebuilding the list rather than deleting in place
        # keeps the surviving order (still descending by energy) intact, which
        # is what makes the next seed the loudest remaining partial.
        claimed: list[_Candidate] = []
        still_ungrouped: list[_Candidate] = []
        for candidate in ungrouped:
            # Calling the public scalar function rather than re-deriving the
            # rule with array maths keeps ONE definition of "is this a
            # harmonic" in the module, so a tuning change cannot make the
            # grouper and the exported helper disagree.
            n = harmonic_number(
                candidate.mean_frequency_hz,
                fundamental_hz,
                tolerance_cents=params.harmonic_tolerance_cents,
            )
            # Being a harmonic is necessary but NOT sufficient. The partials of
            # one note also sound AT THE SAME TIME as it -- that is what makes
            # them that note's harmonics rather than some other note's.
            #
            # Without this check the grouper merges every occurrence of a pitch
            # anywhere in the recording into a single "note". On a phrase you
            # never notice, because each pitch occurs once. On ten minutes of
            # music it is ruinous: a repeated note produced a track labelled
            # "G4 -46c + 11847 harmonics", which is not a voice, it is every G
            # in the piece stacked into one row.
            if n is not None and _overlaps_in_time(candidate, seed, params.min_temporal_overlap):
                claimed.append(candidate)
            else:
                still_ungrouped.append(candidate)

        # --- Step 4: name the result for what it actually is ------------------
        # One or two partials is not a "harmonic group" in any way a user would
        # recognise -- calling it one would promise structure that is not there.
        #
        # A near miss must not COST the user anything either. When the seed
        # gathers too few partials to be a group, the ones it did gather are
        # RELEASED back into `ungrouped` and the seed is emitted alone -- step 4
        # of the module docstring above, and TrackKind.PARTIAL ("a single
        # tracked sine wave"), both say a PARTIAL track holds exactly one
        # partial. Bundling the near miss in instead would strand it: the
        # interface gives a track one row and one trajectory (the lowest
        # partial), so anything else hiding inside a PARTIAL track can never be
        # seen, soloed, or exported on its own.
        #
        # Releasing cannot loop forever: `ungrouped` loses exactly one partial
        # (the popped seed) per iteration whichever branch is taken.
        members: list[Partial] = [seed.partial]
        if len(claimed) + 1 >= MIN_PARTIALS_PER_GROUP:
            kind = TrackKind.HARMONIC_GROUP
            members.extend(candidate.partial for candidate in claimed)
            ungrouped = still_ungrouped
        else:
            kind = TrackKind.PARTIAL

        tracks.append(Track(next_track_id, kind, label_for(members), members))
        next_track_id += 1

    # --- Step 5: present the loudest material first ---------------------------
    # A track's energy is the sum of its members', so a group of many modest
    # harmonics can outrank a single loud tone -- which is correct, because
    # that group is a whole note and the tone is one sliver of one.
    # Sorted here rather than during construction because a track's total is
    # only known once it has finished claiming members.
    tracks.sort(key=_track_energy, reverse=True)

    # --- Step 6: drop tracks too quiet to be worth a row ----------------------
    # See MIN_TRACK_ENERGY_SHARE for the full reasoning. In short: noise throws
    # off short-lived partials that cluster into convincing-looking but
    # inaudible "notes", and they crowd out the real ones in the interface.
    #
    # Nothing is lost by dropping them -- the residual is an exact subtraction,
    # so their sound is still in the mix and still exportable. Only their claim
    # to be a separate note goes away.
    # Measured against the LOUDEST track, not against the total.
    #
    # This looks like a detail and is actually the difference between the
    # filter working and the filter being a function of file length. A share of
    # the total does not mean the same thing at different durations: in a
    # four-note phrase each note carries ~25% of the energy, but in a
    # thousand-note piece each carries ~0.1% -- so a fixed share of the total
    # keeps everything in the first case and deletes everything in the second.
    # Measured on a ten-minute recording it culled every real note and left
    # 91% of the sound in the residual.
    #
    # Relative to the loudest track the question becomes "is this note
    # comparable to the most prominent one", which means the same thing however
    # long the recording is.
    loudest = max((_track_energy(track) for track in tracks), default=0.0)
    if loudest <= 0.0:
        # Silence, or partials with no energy at all. Nothing to rank, and
        # dividing by zero here would drop every track instead of keeping them.
        return tracks

    floor = loudest * params.min_track_energy_share
    audible = [track for track in tracks if _track_energy(track) >= floor]

    dropped = len(tracks) - len(audible)
    if dropped:
        logger.info(
            "folded %d track(s) below %.1f%% of the loudest track into the residual",
            dropped,
            100.0 * params.min_track_energy_share,
        )

    # Renumber so track ids stay contiguous from zero. The API exposes these
    # ids in URLs, and a gap-free sequence is easier to reason about when
    # reading a request log than one full of holes where junk used to be.
    return [
        Track(index, track.kind, track.label, track.partials)
        for index, track in enumerate(audible)
    ]


def label_for(track_partials: list[Partial]) -> str:
    """Build a human-readable label, e.g. 'A4 +12c + 6 harmonics'.

    Naming matters more than it looks: it is how a user finds the note they
    are hunting for in a list of two hundred.

    Args:
        track_partials: The partials making up one track. The LOWEST mean
            frequency among them is used as the nominal pitch, because that is
            the fundamental -- the pitch a listener would name the note by.
            Averaging the members instead would name a cello's open A somewhere
            up around its 3rd harmonic, which is not what anyone hears.

    Returns:
        The note name alone for a single partial, or the note name plus a count
        of the accompanying harmonics. `'unpitched'` when there is no usable
        pitch to name -- an empty track, or one whose partials are all silent.
    """
    # Guard: nothing to name. Returning a placeholder rather than raising keeps
    # a degenerate track visible in the UI instead of failing the whole
    # analysis over a cosmetic string.
    if not track_partials:
        return _UNPITCHED_LABEL

    lowest_hz = min(partial.mean_frequency_hz for partial in track_partials)

    # `note_name` raises on a non-positive frequency (rightly: log2(0) is -inf),
    # so check before calling rather than catching afterwards.
    if lowest_hz <= 0.0:
        return _UNPITCHED_LABEL

    name = note_name(lowest_hz)
    harmonic_count = len(track_partials) - 1
    if harmonic_count == 0:
        return name
    return f"{name} + {harmonic_count} harmonics"


def _track_energy(track: Track) -> float:
    """Total energy of a track's partials; the ordering key for the UI.

    Args:
        track: The track to weigh.

    Returns:
        Sum of every member partial's energy. Zero for a track with no partials
        (the residual, which this module never builds).
    """
    return sum(partial.energy for partial in track.partials)
