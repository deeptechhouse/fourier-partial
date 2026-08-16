"""Abstract ports for fourier_partial.

Every piece of the system that could plausibly be swapped out lives behind one
of the protocols below. Concrete implementations depend on these; nothing
depends on a concrete implementation. That is what lets you replace the
tracking algorithm, the audio backend, or the storage layer by writing one new
class and changing one line of wiring.

Why `Protocol` rather than abstract base classes? Structural typing: an
implementation does not need to inherit from anything to satisfy the contract,
which keeps third-party adapters trivial to write and keeps tests free to pass
in plain fakes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from backend.core.types import (
    AnalysisParams,
    AnalysisResult,
    Partial,
    Peak,
    ReconstructionReport,
    Track,
)


@runtime_checkable
class AudioLoader(Protocol):
    """Reads an audio file into a normalised numpy array.

    Implementations MUST validate the file before trusting it -- magic bytes,
    length, and duration -- because this is the system's boundary with the
    outside world. See TDD section 13.
    """

    def load(self, path: Path, target_sample_rate: int) -> tuple[NDArray[np.float32], int]:
        """Load and resample audio to mono float32 in [-1.0, 1.0].

        Args:
            path: File to read.
            target_sample_rate: Rate to resample to.

        Returns:
            A tuple of (samples, sample_rate).

        Raises:
            UnsupportedAudioError: Format not recognised or file malformed.
            AudioTooLongError: Duration exceeds the configured ceiling.
        """
        ...


@runtime_checkable
class SpectralAnalyzer(Protocol):
    """Turns a waveform into per-frame spectral peaks.

    This is the Fourier step proper: window the signal, transform each window,
    and find the frequencies where energy is concentrated.
    """

    def find_peaks(
        self, samples: NDArray[np.float32], params: AnalysisParams
    ) -> list[list[Peak]]:
        """Find spectral peaks in every analysis frame.

        Returns:
            One list of peaks per frame, in ascending time order.
        """
        ...


@runtime_checkable
class PartialTracker(Protocol):
    """Connects per-frame peaks into continuous trajectories through time.

    This is what turns a stack of independent snapshots into something you can
    call a *voice*: matching each frame's peaks to the previous frame's, so a
    tone that slides in pitch stays one object rather than becoming hundreds of
    unrelated dots.
    """

    def track(self, frames: list[list[Peak]], params: AnalysisParams) -> list[Partial]:
        """Link peaks across frames into partials.

        Args:
            frames: Per-frame peak lists, ascending in time.
            params: Drift tolerance, gap tolerance, minimum duration.

        Returns:
            Completed partials, unordered.
        """
        ...


@runtime_checkable
class TrackGrouper(Protocol):
    """Organises raw partials into a manageable set of tracks.

    A 20-second phrase can yield many hundreds of partials. Presenting that as
    a flat list is unusable, so a grouper clusters harmonically-related
    partials into voices and folds the least significant ones away.
    """

    def group(self, partials: list[Partial], params: AnalysisParams) -> list[Track]:
        """Cluster partials into playable tracks."""
        ...


@runtime_checkable
class Renderer(Protocol):
    """Synthesises audio for tracks, and derives the residual.

    The renderer owns the system's central invariant: after it runs, the sum of
    every track's audio must equal the original sample for sample. It achieves
    this not by being accurate but by *defining* the residual as the leftover.
    """

    def render_tracks(
        self, tracks: list[Track], params: AnalysisParams, num_samples: int
    ) -> None:
        """Render and attach audio to each track, in place."""
        ...

    def build_residual(
        self,
        original: NDArray[np.float32],
        rendered_tracks: list[Track],
        track_id: int,
    ) -> Track:
        """Construct the residual track as `original - sum(rendered)`.

        This subtraction is the reason reconstruction is exact rather than
        approximate. Do not replace it with a filtered or modelled estimate
        without also updating the invariant test.
        """
        ...


@runtime_checkable
class TrackExporter(Protocol):
    """Writes a single track to a file in some format."""

    @property
    def file_extension(self) -> str:
        """Extension this exporter produces, including the dot (e.g. '.wav')."""
        ...

    def supports(self, track: Track) -> bool:
        """Whether this exporter can meaningfully handle the given track."""
        ...

    def export(self, track: Track, destination: Path, sample_rate: int) -> Path:
        """Write the track and return the path actually written.

        Raises:
            UnsupportedExportError: If `supports()` is False for this track.
        """
        ...


@runtime_checkable
class AnalysisPipeline(Protocol):
    """The full front-to-back analysis, composed from the pieces above."""

    def analyze(
        self, samples: NDArray[np.float32], params: AnalysisParams
    ) -> AnalysisResult:
        """Run load-free analysis on raw samples and return a complete result."""
        ...

    def verify(self, result: AnalysisResult) -> ReconstructionReport:
        """Check the reconstruction invariant and summarise the decomposition."""
        ...


@runtime_checkable
class JobStore(Protocol):
    """Tracks long-running analyses so the API can report progress.

    Analysis of a 30-second phrase takes seconds, not milliseconds, so it runs
    as a background job with a WebSocket progress feed rather than blocking an
    HTTP request.
    """

    def create(self, job_id: str) -> None:
        """Register a new job in the pending state."""
        ...

    def set_progress(self, job_id: str, fraction: float, stage: str) -> None:
        """Update progress (0.0 - 1.0) and the human-readable current stage."""
        ...

    def complete(self, job_id: str, result: AnalysisResult) -> None:
        """Mark a job finished and store its result."""
        ...

    def fail(self, job_id: str, message: str) -> None:
        """Mark a job failed with a message safe to show a user.

        Implementations must not put exception text or stack traces here --
        those leak internals. Log the detail server-side, return something
        generic. See TDD section 13.
        """
        ...

    def get(self, job_id: str) -> AnalysisResult | None:
        """Retrieve a completed result, or None if not ready."""
        ...
