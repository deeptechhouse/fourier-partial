"""The composed analysis pipeline.

This module wires the stages together and is the only place that knows the
full sequence. Each stage is injected, so any one of them can be replaced
without touching the others -- swap in a different tracker, a different
grouper, a different renderer, and the pipeline is unchanged.

READING ORDER: `analyze()` is the whole story, top to bottom. Everything else
in this file exists to support it.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from backend.core.errors import ReconstructionInvariantError
from backend.core.interfaces import (
    PartialTracker,
    Renderer,
    SpectralAnalyzer,
    TrackGrouper,
)
from backend.core.types import (
    AnalysisParams,
    AnalysisResult,
    ReconstructionReport,
    Track,
    TrackKind,
)

logger = logging.getLogger(__name__)

#: Called with (fraction_complete, stage_name) so the API can stream progress.
ProgressCallback = Callable[[float, str], None]

# =============================================================================
# ADJUSTABLE — progress weights
# =============================================================================
# Roughly proportional to observed cost on a 20-second phrase. These only
# affect how smoothly the progress bar moves; getting them wrong makes the bar
# stutter but breaks nothing. Stage NAMES are written for the person waiting,
# not for the developer -- someone watching a progress bar wants to know what
# is happening to their sound.
_STAGES: tuple[tuple[str, float], ...] = (
    ("Finding tones", 0.30),
    ("Following tones through time", 0.25),
    ("Grouping harmonics", 0.05),
    ("Rendering tracks", 0.35),
    ("Finishing", 0.05),
)

# ADJUSTABLE: how exact "exact" has to be. The residual is defined as an exact
# subtraction, so the only error possible is float32 rounding on the final cast
# (~1e-7 for samples in [-1, 1]). Anything above this floor means a real bug.
_EXACTNESS_TOLERANCE: float = 1e-6


class PartialAnalysisPipeline:
    """Runs peaks -> tracking -> grouping -> rendering -> residual."""

    __slots__ = ("_analyzer", "_tracker", "_grouper", "_renderer")

    def __init__(
        self,
        analyzer: SpectralAnalyzer,
        tracker: PartialTracker,
        grouper: TrackGrouper,
        renderer: Renderer,
    ) -> None:
        """Compose a pipeline from its four stages."""
        self._analyzer = analyzer
        self._tracker = tracker
        self._grouper = grouper
        self._renderer = renderer

    # -- Progress plumbing ---------------------------------------------------

    @staticmethod
    def _report(
        on_progress: ProgressCallback | None, stage_index: int, within: float = 1.0
    ) -> None:
        """Emit cumulative progress for the given stage.

        `within` is how far through THIS stage we are (0..1), so a long stage
        can report intermediate motion instead of freezing the bar.
        """
        if on_progress is None:
            return
        done = sum(weight for _, weight in _STAGES[:stage_index])
        name, weight = _STAGES[stage_index]
        on_progress(min(1.0, done + weight * within), name)

    # -- The pipeline --------------------------------------------------------

    def analyze(
        self,
        samples: NDArray[np.float32],
        params: AnalysisParams,
        on_progress: ProgressCallback | None = None,
    ) -> AnalysisResult:
        """Decompose `samples` into tracks.

        Args:
            samples: Mono float32 audio, already loaded and resampled.
            params: Analysis settings.
            on_progress: Optional callback for streaming progress to a client.

        Returns:
            A complete AnalysisResult whose tracks sum back to `samples`
            exactly. See section 7 of the design document.
        """
        audio = np.asarray(samples, dtype=np.float32)
        num_samples = int(audio.size)
        duration_s = num_samples / params.sample_rate

        # --- Degenerate input: too short to analyse -------------------------
        # Not an error. A clip shorter than one analysis window simply has no
        # frames, so it has no partials -- and the residual then carries the
        # whole signal, which keeps the invariant true and lets the user still
        # hear (and export) what they uploaded.
        if num_samples < params.window_size:
            logger.info(
                "audio shorter than one window (%d < %d samples); residual only",
                num_samples,
                params.window_size,
            )
            residual = self._renderer.build_residual(audio, [], 0)
            self._report(on_progress, len(_STAGES) - 1)
            return AnalysisResult(params.sample_rate, duration_s, [residual], params, audio)

        # --- 1. Peaks -------------------------------------------------------
        # Every frame's spectrum, reduced to the frequencies where energy is
        # actually concentrated.
        self._report(on_progress, 0, 0.0)
        peak_frames = self._analyzer.find_peaks(audio, params)
        raw_peak_count = sum(len(frame) for frame in peak_frames)
        self._report(on_progress, 0)

        # --- 2. Tracking ----------------------------------------------------
        # Snapshots become trajectories: a tone that slides stays one object.
        self._report(on_progress, 1, 0.0)
        partials = self._tracker.track(peak_frames, params)
        self._report(on_progress, 1)
        logger.info(
            "%d peaks across %d frames -> %d partials",
            raw_peak_count,
            len(peak_frames),
            len(partials),
        )

        # --- 3. Grouping ----------------------------------------------------
        # Hundreds of partials become a list a human can operate.
        self._report(on_progress, 2, 0.0)
        tracks = list(self._grouper.group(partials, params))
        self._report(on_progress, 2)

        # --- 4. Rendering ---------------------------------------------------
        # Each track becomes audible. The most expensive stage by some margin.
        self._report(on_progress, 3, 0.0)
        self._renderer.render_tracks(tracks, params, num_samples)
        self._report(on_progress, 3)

        # --- 5. Residual ----------------------------------------------------
        # THE INVARIANT. Defined as a subtraction, so the sum of every track
        # reproduces the original sample for sample no matter how stages 1-4
        # performed. Never replace this with a model or a filter.
        self._report(on_progress, 4, 0.0)
        next_id = max((t.track_id for t in tracks), default=-1) + 1
        tracks.append(self._renderer.build_residual(audio, tracks, next_id))
        self._report(on_progress, 4)

        result = AnalysisResult(params.sample_rate, duration_s, tracks, params, audio)

        # Fail loudly rather than shipping a silently-wrong decomposition. This
        # cannot trip unless the renderer has an arithmetic bug -- which is
        # exactly why it is worth checking on every single analysis.
        error = result.reconstruction_error()
        if not math.isfinite(error) or error > _EXACTNESS_TOLERANCE:
            raise ReconstructionInvariantError(
                f"tracks do not sum to the original: max sample error {error:.3e} "
                f"exceeds tolerance {_EXACTNESS_TOLERANCE:.1e}"
            )

        return result

    # -- Diagnostics ---------------------------------------------------------

    def verify(self, result: AnalysisResult) -> ReconstructionReport:
        """Check the reconstruction invariant and summarise the decomposition.

        Returned to the user as well as asserted in tests, so the analysis is
        legible rather than magic: how much of their sound became tones, and
        how much stayed noise.
        """
        max_error = result.reconstruction_error()

        # Energy split. Sum of squares is a fine proxy for "how much of the
        # sound is this", and it is what the uncertainty-principle expectation
        # in section 12 is stated in terms of.
        partial_energy = 0.0
        residual_energy = 0.0
        partial_count = 0
        for track in result.tracks:
            audio = track.audio
            if audio is None:
                continue
            energy = float(np.sum(np.square(audio.astype(np.float64))))
            if track.kind is TrackKind.RESIDUAL:
                residual_energy += energy
            else:
                partial_energy += energy
                partial_count += len(track.partials)

        total = partial_energy + residual_energy
        # A totally silent upload has no energy to apportion. Report zeros
        # rather than dividing by zero and handing the UI a NaN.
        partial_ratio = partial_energy / total if total > 0.0 else 0.0
        residual_ratio = residual_energy / total if total > 0.0 else 0.0

        warnings: list[str] = []
        if max_error > _EXACTNESS_TOLERANCE:
            warnings.append(
                f"Reconstruction is not exact (error {max_error:.2e}). This is a bug."
            )
        if partial_count >= result.params.max_tracks:
            # Silently showing fewer tracks than were found would read as
            # "here is everything" when it is not. Say so.
            warnings.append(
                f"Hit the {result.params.max_tracks}-track ceiling. Quieter partials "
                f"were folded into the residual and cannot be soloed separately."
            )
        if partial_ratio < 0.25 and total > 0.0:
            # Not an error -- this is the uncertainty principle showing up in
            # the metrics. Percussive or noisy material genuinely cannot be
            # represented as a small set of tones. Explain rather than alarm.
            warnings.append(
                "Most of this sound is noise or transients rather than steady tones, "
                "so it is concentrated in the residual track. That is expected for "
                "percussive material."
            )

        return ReconstructionReport(
            max_sample_error=max_error,
            partial_energy_ratio=partial_ratio,
            partial_count=partial_count,
            residual_energy_ratio=residual_ratio,
            warnings=tuple(warnings),
        )


def downsample_trajectory(
    track: Track, max_points: int = 400
) -> list[tuple[float, float, float]]:
    """Reduce a track's partials to a display-resolution trajectory.

    The UI draws these into a row a few hundred pixels wide, so sending full
    resolution would be megabytes of data the browser immediately throws away.

    For a harmonic group we plot the LOWEST partial -- that is the fundamental,
    the pitch a listener would name the track by. The harmonics move in
    parallel with it, so plotting them all would be visual noise.

    Args:
        track: The track to summarise.
        max_points: ADJUSTABLE ceiling on returned points.

    Returns:
        (time_s, frequency_hz, amplitude) triples, ascending in time.
    """
    if not track.partials:
        return []

    lowest = min(track.partials, key=lambda p: p.mean_frequency_hz)
    frames = lowest.frames
    if not frames:
        return []

    # Stride rather than average: a trajectory is a path, and averaging
    # neighbouring points would round off exactly the vibrato and glide detail
    # that makes the view worth looking at.
    stride = max(1, len(frames) // max_points)
    return [
        (round(f.time_s, 4), round(f.frequency_hz, 2), round(f.amplitude, 5))
        for f in frames[::stride]
    ]
