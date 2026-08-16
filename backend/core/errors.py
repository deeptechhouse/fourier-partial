"""Exception hierarchy for fourier_partial.

Two rules, both about the boundary between inside and outside:

  1. Every error raised by this system inherits from `FourierPartialError`, so
     the API layer can catch one type and be sure nothing leaks past it.
  2. Each error carries a `user_message` that is safe to send to a client.
     The exception's own text may contain paths, parameters, and internals --
     that goes to the log, never to the browser. See TDD section 13.
"""

from __future__ import annotations


class FourierPartialError(Exception):
    """Base class for every error this system raises deliberately."""

    #: Generic fallback shown to users when a subclass sets nothing better.
    user_message: str = "The audio could not be processed."


# --- Input / loading ---------------------------------------------------------


class AudioInputError(FourierPartialError):
    """Base for problems with the audio a user supplied."""

    user_message = "That audio file could not be read."


class UnsupportedAudioError(AudioInputError):
    """File is not a recognised audio format, or its contents contradict it."""

    user_message = "Unsupported audio format. Try WAV, FLAC, AIFF, or MP3."


class AudioTooLongError(AudioInputError):
    """Audio exceeds the configured duration ceiling."""

    user_message = "That audio is too long. Please use a clip of 30 seconds or less."


class AudioTooLargeError(AudioInputError):
    """Uploaded file exceeds the configured byte ceiling."""

    user_message = "That file is too large."


class EmptyAudioError(AudioInputError):
    """Audio decoded successfully but contains no samples, or is pure silence."""

    user_message = "That file contains no audible audio."


# --- Analysis ----------------------------------------------------------------


class AnalysisError(FourierPartialError):
    """Base for failures during decomposition."""

    user_message = "Analysis failed for this audio."


class InvalidParameterError(AnalysisError):
    """A supplied analysis parameter is out of range or self-contradictory."""

    user_message = "One of the analysis settings is invalid."


class ReconstructionInvariantError(AnalysisError):
    """The tracks do not sum back to the original.

    This is an internal consistency failure, not a user error. If it is ever
    raised in production, the renderer is broken -- the residual is defined as
    an exact subtraction, so the sum cannot drift unless something else has.
    """

    user_message = "An internal error occurred during analysis."


# --- Export ------------------------------------------------------------------


class ExportError(FourierPartialError):
    """Base for failures while writing output files."""

    user_message = "The export could not be completed."


class UnsupportedExportError(ExportError):
    """The requested format does not apply to this track.

    Most commonly: asking for MIDI from the residual track, which is noise and
    therefore has no notes to write.
    """

    user_message = "That track cannot be exported in the requested format."


# --- Jobs --------------------------------------------------------------------


class JobError(FourierPartialError):
    """Base for background job problems."""

    user_message = "The analysis job could not be completed."


class JobNotFoundError(JobError):
    """No job exists with the given identifier, or it has expired."""

    user_message = "That analysis session has expired. Please upload again."
