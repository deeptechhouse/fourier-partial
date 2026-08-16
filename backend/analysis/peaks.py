"""Spectral peak picking — finding where the energy actually sits.

WHAT THIS DOES
--------------
Each STFT frame is a spectrum: a curve of magnitude against frequency. A tone
in the sound shows up as a bump in that curve. This module finds those bumps.

THE SUB-BIN PROBLEM
-------------------
The FFT reports magnitudes only at discrete bins. A 440 Hz tone analysed with
10.8 Hz bins does not land on a bin -- it lands between two of them, and both
light up. Taking the taller bin would round 440 Hz to 431 or 442 Hz, an error
of tens of cents, which is audibly out of tune and which would make partial
tracking jitter.

The fix is parabolic interpolation: fit a parabola through the peak bin and
its two neighbours (in dB, where a windowed sinusoid's peak is very nearly
parabolic) and take the vertex. That typically recovers frequency to within a
few cents. It is three lines of arithmetic and it is the difference between a
tool that sounds in tune and one that does not.

STATUS: implemented (Phase 1). Contract unchanged from the scaffold.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from backend.analysis.stft import bin_to_hz
from backend.core.errors import InvalidParameterError
from backend.core.types import AnalysisParams, Peak

# =============================================================================
# ADJUSTABLE CONSTANTS
# =============================================================================

# ADJUSTABLE: the magnitude we treat as "indistinguishable from silence".
# Why it exists: log10(0) is -inf, which would poison every later arithmetic
# step (inf - inf = nan). Clamping to a tiny positive floor keeps the maths
# finite. -240 dBFS is ~40 dB below the noise floor of 24-bit audio, so no real
# musical content is ever affected. Raise it only if you want to silence very
# quiet frames outright.
_SILENCE_FLOOR: float = 1e-12

# Why 3: parabolic interpolation needs a bin plus a neighbour on each side.
# A spectrum shorter than that has no interpolatable interior at all.
_MIN_BINS_FOR_PEAK: int = 3


def parabolic_interpolate(
    left_db: float, centre_db: float, right_db: float
) -> tuple[float, float]:
    """Refine a peak's position and height from three dB samples.

    Args:
        left_db: Magnitude in dB one bin below the peak.
        centre_db: Magnitude in dB at the peak bin.
        right_db: Magnitude in dB one bin above the peak.

    Returns:
        A tuple of (bin_offset, peak_db) where `bin_offset` lies in
        (-0.5, 0.5) and is added to the integer bin index.

    The formula is the vertex of the parabola through the three points:

        offset = 0.5 * (left - right) / (left - 2*centre + right)

    A perfectly symmetric peak gives offset 0. A peak leaning right gives a
    positive offset.
    """
    denominator = left_db - 2.0 * centre_db + right_db
    # A flat or inverted neighbourhood has no meaningful vertex; refuse to
    # divide and report the bin centre unchanged.
    if abs(denominator) < 1e-12:
        return 0.0, centre_db
    offset = 0.5 * (left_db - right_db) / denominator
    peak_db = centre_db - 0.25 * (left_db - right_db) * offset
    return offset, peak_db


def find_frame_peaks(
    magnitude: NDArray[np.float64],
    phase: NDArray[np.float64],
    frame_index: int,
    params: AnalysisParams,
) -> list[Peak]:
    """Find the spectral peaks in a single frame.

    Args:
        magnitude: Magnitude spectrum for one frame, shape (n_bins,).
        phase: Matching phase spectrum, shape (n_bins,).
        frame_index: Index of this frame, stored on each returned Peak.
        params: Threshold and count limits.

    Returns:
        Peaks sorted by descending amplitude, at most
        `params.max_peaks_per_frame` of them.

    Implementation notes:
        - A bin is a candidate peak if it is strictly greater than both
          neighbours. Skip bin 0 and the last bin: they have only one
          neighbour, so the parabola is undefined.
        - Discard candidates below `params.peak_threshold_db` relative to the
          frame's loudest bin, not to full scale -- relative thresholding
          adapts to quiet passages instead of ignoring them entirely.
        - Also discard any bin at or below `_SILENCE_FLOOR` in absolute terms.
          A purely relative floor sinks below the floor's own clamp value in a
          frame quieter than about -180 dBFS, at which point the clamped shelf
          of sub-floor dust qualifies as "loud enough" and fills the frame with
          peaks that are not there.
        - Read phase at the *integer* peak bin. Interpolating phase is
          hazardous because it wraps; the integer-bin value is close enough
          for the renderer's purposes.

    Raises:
        InvalidParameterError: If `magnitude` and `phase` are not the same
            length. They come from one FFT, so a mismatch means the caller has
            paired unrelated spectra -- failing here is far cheaper to diagnose
            than an IndexError deep in the loop below.
    """
    # --- Step 0: reject spectra that are too short or entirely silent --------
    # A frame with fewer than three bins has no interior bin to interpolate,
    # and a silent frame has no meaningful "loudest peak" to threshold against
    # (every relative threshold would be measured from -inf).
    if magnitude.shape != phase.shape:
        raise InvalidParameterError(
            f"magnitude and phase must have the same shape; "
            f"got {magnitude.shape} and {phase.shape}"
        )

    n_bins = int(magnitude.shape[0])
    if n_bins < _MIN_BINS_FOR_PEAK:
        return []

    max_magnitude = float(np.max(magnitude))
    if max_magnitude <= _SILENCE_FLOOR:
        return []

    # --- Step 1: convert the whole frame to dB, once -------------------------
    # Peak picking, thresholding and interpolation all happen in dB:
    #   * thresholding, because loudness is perceived logarithmically, so a
    #     "-60 dB relative floor" means the same thing at every volume;
    #   * interpolation, because the top of a windowed sinusoid's main lobe is
    #     very nearly a parabola in dB but not in linear magnitude -- fitting
    #     in dB is what buys the sub-bin accuracy this module exists for.
    # np.maximum clamps away zeros before the log so no -inf can appear.
    magnitude_db: NDArray[np.float64] = 20.0 * np.log10(
        np.maximum(magnitude.astype(np.float64, copy=False), _SILENCE_FLOOR)
    )

    # --- Step 2: find local maxima (vectorised, no Python loop) --------------
    # `interior` is every bin except the first and last. A bin qualifies if it
    # is STRICTLY taller than the bin below and the bin above it. Strictness
    # matters: on a plateau (two equal neighbouring bins) neither bin wins, so
    # a flat region yields nothing rather than a run of duplicates a bin apart.
    # Compared in linear magnitude, which is monotonic with dB, so the answer
    # is identical -- but bins clamped to the silence floor tie in dB and would
    # otherwise create spurious plateaus.
    interior = magnitude[1:-1]
    is_local_max = (interior > magnitude[:-2]) & (interior > magnitude[2:])

    # --- Step 3: relative threshold ------------------------------------------
    # peak_threshold_db is NEGATIVE (default -60), so adding it to the frame's
    # loudest bin gives a floor *below* that maximum. Relative to the frame,
    # not to full scale, so a quiet passage still yields its own peaks instead
    # of being discarded wholesale.
    frame_max_db = float(np.max(magnitude_db))
    threshold_db = frame_max_db + float(params.peak_threshold_db)
    loud_enough = magnitude_db[1:-1] >= threshold_db

    # A RELATIVE floor alone is not enough. Every bin at or below _SILENCE_FLOOR
    # was clamped to the same dB value, so in a frame whose own maximum is very
    # near the floor (below about -180 dBFS, i.e. threshold_db <= the clamp
    # value) that entire clamped shelf passes `loud_enough`. Each wiggle in the
    # sub-floor numerical dust is then a strict local maximum in linear terms,
    # and the frame comes back full of peaks pinned at the clamp amplitude --
    # which also pushes the one real peak out of `max_peaks_per_frame`. Requiring
    # the bin itself to sit above the floor says the same thing the frame-level
    # silence test says, one bin at a time: what was clamped is silence, and
    # silence is not a tone.
    above_floor = interior > _SILENCE_FLOOR

    # +1 shifts indices from `interior` coordinates back to real bin numbers.
    candidate_bins = np.flatnonzero(is_local_max & loud_enough & above_floor) + 1

    # --- Step 4: refine each survivor to sub-bin accuracy --------------------
    peaks: list[Peak] = []
    for bin_index in candidate_bins.tolist():
        offset, peak_db = parabolic_interpolate(
            float(magnitude_db[bin_index - 1]),
            float(magnitude_db[bin_index]),
            float(magnitude_db[bin_index + 1]),
        )
        frequency_hz = bin_to_hz(bin_index + offset, params)

        # A peak at (or interpolated below) DC is not a tone anyone can hear or
        # track, and Peak.cents_from would return inf for it. Drop it.
        if frequency_hz <= 0.0:
            continue

        peaks.append(
            Peak(
                frame_index=frame_index,
                frequency_hz=frequency_hz,
                # Back from dB to the linear units the renderer synthesises in.
                # peak_db is the interpolated vertex height, so this is a better
                # amplitude estimate than the raw bin magnitude: the true peak
                # sits between bins, hence above both of them.
                amplitude=10.0 ** (peak_db / 20.0),
                # Integer bin on purpose -- see the implementation note above.
                phase_rad=float(phase[bin_index]),
            )
        )

    # --- Step 5: keep only the loudest few -----------------------------------
    # Descending amplitude, with ascending frequency as a tie-break purely so
    # the ordering is deterministic; the tracker downstream is easier to debug
    # when the same input always produces the same peak order.
    peaks.sort(key=lambda peak: (-peak.amplitude, peak.frequency_hz))

    # max(0, ...) guards against a negative limit, which would otherwise be
    # read as a from-the-end slice and silently keep the *quietest* peaks.
    limit = max(0, int(params.max_peaks_per_frame))
    return peaks[:limit]


def find_all_peaks(
    magnitude: NDArray[np.float64],
    phase: NDArray[np.float64],
    params: AnalysisParams,
) -> list[list[Peak]]:
    """Run `find_frame_peaks` across every frame.

    Args:
        magnitude: Magnitude spectrogram, shape (n_frames, n_bins), exactly as
            returned by `backend.analysis.stft.compute_stft`.
        phase: Matching phase spectrogram, same shape.
        params: Threshold and count limits, passed straight through.

    Returns:
        One peak list per frame, in ascending time order.

    Raises:
        InvalidParameterError: If the two spectrograms are not 2D or do not
            have the same shape. That combination cannot come from a single
            STFT, so continuing would silently pair magnitudes with unrelated
            phases and produce audible artefacts at resynthesis.
    """
    if magnitude.ndim != 2 or phase.ndim != 2:
        raise InvalidParameterError(
            f"magnitude and phase must be 2D (n_frames, n_bins); "
            f"got {magnitude.ndim}D and {phase.ndim}D"
        )
    if magnitude.shape != phase.shape:
        raise InvalidParameterError(
            f"magnitude and phase must have the same shape; "
            f"got {magnitude.shape} and {phase.shape}"
        )

    # A plain loop rather than a vectorised whole-spectrogram trick: each frame
    # produces a different NUMBER of peaks, so there is no rectangular array to
    # operate on. The per-frame work is already vectorised inside
    # find_frame_peaks, which is where the samples actually are.
    return [
        find_frame_peaks(magnitude[frame_index], phase[frame_index], frame_index, params)
        for frame_index in range(magnitude.shape[0])
    ]
