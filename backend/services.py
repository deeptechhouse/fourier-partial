"""Composition root — where the abstract ports meet concrete implementations.

Every other module either defines a contract or implements one. THIS module is
the only place that knows which implementation is currently plugged into which
port. That is the whole payoff of the ports-and-adapters design: to swap the
tracker, the renderer, or the audio backend, you change a line here and
nothing else in the system notices.

The adapter classes below are deliberately thin. The real work lives in plain
functions in `backend/analysis/` and `backend/synthesis/` -- functions are
easier to test and to reason about than methods with hidden state. The classes
exist only to give those functions the shape the protocols ask for.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from backend.analysis.grouping import group_partials
from backend.analysis.peaks import find_frame_peaks
from backend.analysis.pipeline import PartialAnalysisPipeline
from backend.analysis.stft import iter_stft_blocks
from backend.analysis.tracking import track_partials
from backend.audio_io.loader import SoundFileLoader
from backend.core.types import AnalysisParams, Partial, Peak, Track
from backend.synthesis.oscillator_bank import build_residual, render_track

logger = logging.getLogger(__name__)


class StftPeakAnalyzer:
    """Implements `SpectralAnalyzer`: waveform in, per-frame spectral peaks out.

    Bundles the two halves of the Fourier step -- transform, then find where
    the energy sits -- because no caller has ever wanted one without the other.

    STREAMS BY DEFAULT. The spectrogram is consumed a block of frames at a
    time and released as it goes, so peak memory depends on the block size
    rather than on how long the audio is. That is the difference between a
    30-second ceiling and a ten-minute one: the full spectrogram costs about
    11 MB per second of audio, so a ten-minute file would need ~6.8 GB before
    anything else happened. See `iter_stft_blocks`.
    """

    def find_peaks(
        self, samples: NDArray[np.float32], params: AnalysisParams
    ) -> list[list[Peak]]:
        """Transform the signal and locate the peaks in every frame."""
        frames: list[list[Peak]] = []
        for first_index, magnitude, phase in iter_stft_blocks(samples, params):
            for offset in range(magnitude.shape[0]):
                frames.append(
                    find_frame_peaks(
                        magnitude[offset], phase[offset], first_index + offset, params
                    )
                )
            # The block's arrays go out of scope here and are collected before
            # the next one is built, which is the whole point of streaming.
        return frames


class GreedyPartialTracker:
    """Implements `PartialTracker` using McAulay-Quatieri greedy matching."""

    def track(self, frames: list[list[Peak]], params: AnalysisParams) -> list[Partial]:
        """Link peaks across frames into partials."""
        return track_partials(frames, params)


class HarmonicGrouper:
    """Implements `TrackGrouper` by clustering partials onto fundamentals."""

    def group(self, partials: list[Partial], params: AnalysisParams) -> list[Track]:
        """Cluster partials into playable tracks."""
        return group_partials(partials, params)


class OscillatorBankRenderer:
    """Implements `Renderer`: synthesises tracks and derives the residual.

    Owns the system's central invariant. See `oscillator_bank.build_residual`.
    """

    def render_tracks(
        self, tracks: list[Track], params: AnalysisParams, num_samples: int
    ) -> None:
        """Render and attach audio to each track, in place."""
        for track in tracks:
            # A track that somehow already has audio is left alone --
            # attach_audio raises on a second call, and silently swallowing
            # that would hide a real double-render bug.
            if track.audio is None:
                track.attach_audio(render_track(track, params, num_samples))

    def build_residual(
        self,
        original: NDArray[np.float32],
        rendered_tracks: list[Track],
        track_id: int,
    ) -> Track:
        """Construct the residual as `original - sum(rendered)`."""
        return build_residual(original, rendered_tracks, track_id)


def build_pipeline() -> PartialAnalysisPipeline:
    """Assemble the default analysis pipeline.

    THIS is the line to edit when swapping an implementation.
    """
    return PartialAnalysisPipeline(
        analyzer=StftPeakAnalyzer(),
        tracker=GreedyPartialTracker(),
        grouper=HarmonicGrouper(),
        renderer=OscillatorBankRenderer(),
    )


def build_loader(max_duration_seconds: float) -> SoundFileLoader:
    """Assemble the default audio loader."""
    return SoundFileLoader(max_duration_seconds=max_duration_seconds)
