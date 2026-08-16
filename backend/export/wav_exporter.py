"""WAV export — the lossless path.

Nothing subtle here, which is the point: a track's rendered audio is already
exactly what the user heard when they soloed it, so exporting is a write.

The only decision is bit depth. 24-bit PCM is the default: it is what every
DAW expects, it has plenty of headroom for quiet partials, and it is half the
size of 32-bit float with no audible difference. Float is offered for users
who intend to re-sum the tracks and want the original's samples back bit for
bit.

STATUS: implemented (Phase 3).

WHY THE SUBTYPE IS CHECKED IN THE CONSTRUCTOR
--------------------------------------------
libsndfile reports an unknown subtype as a generic error at *write* time --
i.e. after the user has already waited through a full analysis and clicked
"export". Checking the string against `soundfile.available_subtypes("WAV")`
when the exporter is built moves that failure to configuration time, where it
is cheap and where the message can name the legal values. This is the usual
"fail fast at the boundary" rule from TDD section 13.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.core.errors import ExportError, UnsupportedExportError
from backend.core.types import Track

logger = logging.getLogger(__name__)


class WavExporter:
    """Writes a track's rendered audio to a WAV file."""

    # ADJUSTABLE: 'PCM_24' for DAW-friendly output, 'FLOAT' for bit-exact
    # round-tripping (needed if the user wants to re-sum tracks and land
    # exactly on the original).
    def __init__(self, subtype: str = "PCM_24") -> None:
        """Create an exporter writing the given libsndfile subtype.

        Args:
            subtype: A libsndfile WAV subtype name, e.g. 'PCM_24' or 'FLOAT'.

        Raises:
            ExportError: If libsndfile does not know this subtype for WAV.
        """
        # `available_subtypes` is queried from the linked libsndfile itself
        # rather than hardcoded, so this stays correct if the wheel is built
        # against a different libsndfile with more (or fewer) codecs.
        legal = sf.available_subtypes("WAV")
        if subtype not in legal:
            raise ExportError(f"unknown WAV subtype {subtype!r}; libsndfile offers {sorted(legal)}")
        self._subtype = subtype

    @property
    def subtype(self) -> str:
        """The libsndfile subtype this exporter writes (read-only)."""
        return self._subtype

    @property
    def file_extension(self) -> str:
        """Extension produced by this exporter."""
        return ".wav"

    def supports(self, track: Track) -> bool:
        """Any track with rendered audio can be written as WAV."""
        return track.audio is not None

    def export(self, track: Track, destination: Path, sample_rate: int) -> Path:
        """Write `track` to `destination` and return the path written.

        Raises:
            UnsupportedExportError: If the track has no rendered audio.
            ExportError: If libsndfile refuses to write the file.
        """
        if not self.supports(track):
            raise UnsupportedExportError(f"track {track.track_id} has no rendered audio")

        # `Track.audio` hands out a read-only *view* so callers cannot scribble
        # on the renderer's buffer. `np.asarray` is a no-op wrapper here (it
        # does not copy a numpy array that is already the right type), and
        # libsndfile only ever reads from the buffer, so the read-only flag is
        # not in the way.
        samples = np.asarray(track.audio)

        # The caller may hand us a path inside a directory that does not exist
        # yet (the API builds one temp directory per export). Creating it here
        # keeps every call site from having to remember to.
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            # `format="WAV"` is passed explicitly rather than left to be
            # guessed from the filename: a caller writing to a temp file with
            # no extension would otherwise get an opaque libsndfile error, and
            # this exporter has already promised WAV via `file_extension`.
            sf.write(destination, samples, sample_rate, subtype=self._subtype, format="WAV")
        except (OSError, RuntimeError, ValueError) as exc:
            # Wrap third-party failures in our own hierarchy so the API layer
            # can catch `FourierPartialError` and be sure nothing leaks past
            # it (see backend/core/errors.py). The detail goes to the log; the
            # user sees `ExportError.user_message`.
            logger.exception("WAV export failed for track %s -> %s", track.track_id, destination)
            raise ExportError(f"could not write WAV to {destination}: {exc}") from exc

        logger.debug(
            "wrote track %s to %s (%d samples, %s @ %d Hz)",
            track.track_id,
            destination,
            samples.shape[0],
            self._subtype,
            sample_rate,
        )
        return destination
