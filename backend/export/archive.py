"""Export everything at once — every track, in one download.

WHY THIS IS NOT JUST A LOOP OVER THE SINGLE-TRACK EXPORTER
-----------------------------------------------------------
Two things make a bulk export a different problem from forty individual ones.

SIZE. A ten-minute analysis yields around forty tracks, each rendered over the
whole timeline. As 24-bit WAV that is ~79 MB per track and **3.2 GB in total**.
Nothing here may hold that in memory, so every track is written to a scratch
file, streamed into the archive, and deleted before the next one starts. Peak
memory is one track's worth of buffering, not the whole export.

HONESTY. A bulk export is exactly where things get dropped quietly. The
residual cannot become MIDI; a single multi-channel MIDI file has fifteen
usable channels and an analysis routinely has more tracks than that. An archive
that silently contained fewer files than the user has rows on screen would be
the worst kind of wrong -- it looks complete. So every archive carries a
MANIFEST.txt naming what is inside, what was left out, and why.

The manifest also states the property a user unpacking forty WAVs has no way to
guess: they sum back to the original recording, exactly.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from backend.core.errors import ExportError
from backend.core.types import AnalysisResult, Track, TrackKind
from backend.export.midi_exporter import MidiExporter
from backend.export.wav_exporter import WavExporter

logger = logging.getLogger(__name__)

# ADJUSTABLE: WAV subtype for bulk export. 24-bit PCM is what a DAW expects and
# is 25% smaller than float. Use "FLOAT" if the point of the export is to sum
# the tracks back together and land on the original bit for bit -- 24-bit is
# inaudibly different but not *identical*, and the manifest says so.
BULK_WAV_SUBTYPE: str = "PCM_24"

# Compression is chosen per format, and the choice matters at this scale.
#
#   WAV  -> STORED. PCM audio is already dense; deflate typically saves a few
#           percent while costing minutes of CPU on a multi-gigabyte archive.
#           Storing is effectively free and the result is the same size.
#   MIDI -> DEFLATED. MIDI files are small and highly repetitive (long runs of
#           near-identical control messages), so they compress enormously and
#           the CPU cost is trivial.
_WAV_COMPRESSION = zipfile.ZIP_STORED
_MIDI_COMPRESSION = zipfile.ZIP_DEFLATED


class ArchiveFormat(Enum):
    """What an "export everything" request should produce."""

    WAV = "wav"
    """A .zip of one .wav per track, including the residual."""

    MIDI = "mid"
    """A .zip of one .mid per track. The residual is excluded -- it is noise."""

    MIDI_SINGLE = "mid-single"
    """One multi-channel .mid holding the most prominent tracks.

    Convenient for dragging into a DAW as a single object, but limited: pitch
    bend applies to a whole MIDI channel, so each track needs its own, and
    there are only fifteen usable (channel 10 is percussion by convention).
    """

    @property
    def is_archive(self) -> bool:
        """Whether this produces a .zip rather than a single file."""
        return self is not ArchiveFormat.MIDI_SINGLE

    @property
    def extension(self) -> str:
        """The extension of the thing the user ends up downloading."""
        return ".zip" if self.is_archive else ".mid"


@dataclass(frozen=True, slots=True)
class SkippedTrack:
    """One track that did not make it into the archive, and why.

    Carried separately from a log line because the user needs to see it: the
    reason belongs in the manifest they unzip, not in a server log they will
    never read.
    """

    track_id: int
    label: str
    reason: str


@dataclass(slots=True)
class ArchiveReport:
    """What an export actually contained."""

    path: Path
    format: ArchiveFormat
    included: list[str] = field(default_factory=list)
    skipped: list[SkippedTrack] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def is_complete(self) -> bool:
        """Whether every track made it in."""
        return not self.skipped

    def summary(self) -> str:
        """One line, safe to show a user."""
        n = len(self.included)
        if self.is_complete:
            return f"{n} file{'s' if n != 1 else ''}."
        return f"{n} file{'s' if n != 1 else ''}, {len(self.skipped)} skipped."


def estimate_bytes(result: AnalysisResult, fmt: ArchiveFormat) -> int:
    """Roughly how large the download will be, before building it.

    Exists so the interface can warn before someone commits to a multi-gigabyte
    download on a laptop. Deliberately an over-estimate for WAV (it ignores the
    handful of tracks that may be skipped) because a warning that turns out to
    be pessimistic is much better received than one that turns out to be
    optimistic.
    """
    if fmt is ArchiveFormat.WAV:
        # 3 bytes per sample at 24-bit, plus a negligible header per file.
        samples = result.original.shape[0]
        return int(samples * 3 * len(result.tracks))

    # MIDI is orders of magnitude smaller and depends on note density rather
    # than duration. A few tens of KB per track is a generous allowance.
    pitched = sum(1 for t in result.tracks if t.is_exportable_as_midi)
    return pitched * 40_000


def _safe_stem(track: Track) -> str:
    """A filename stem that is unique and survives every filesystem.

    The track id leads, which guarantees uniqueness: two occurrences of the
    same note produce identical labels, and on a case-insensitive filesystem
    even differing labels can collide. Without the id, an archive would quietly
    contain fewer files than tracks -- the exact failure this module exists to
    prevent.
    """
    label = "".join(c if c.isalnum() or c in "-_+" else "_" for c in track.label)
    return f"track{track.track_id:03d}_{label[:48]}"


def _manifest(result: AnalysisResult, report: ArchiveReport) -> str:
    """The README that travels inside the archive.

    Someone who unpacks forty WAV files has no way to know they sum back to the
    original recording, or why there are thirty-nine MIDI files where there
    were forty WAVs. This tells them, in the archive, without needing this
    documentation.
    """
    lines: list[str] = [
        "fourier_partial — export",
        "=" * 60,
        "",
        f"Source duration : {result.duration_s:.2f} seconds",
        f"Sample rate     : {result.sample_rate} Hz",
        f"Tracks analysed : {len(result.tracks)}",
        f"Files included  : {len(report.included)}",
        "",
    ]

    if report.format is ArchiveFormat.WAV:
        lines += [
            "THESE FILES SUM BACK TO THE ORIGINAL RECORDING.",
            "",
            "Every track, including the residual, added together at unity gain",
            "reproduces the audio that was analysed. That is a property of how",
            "the residual is computed -- it is the original minus the tones, so",
            "the parts cannot fail to add up.",
            "",
            f"One caveat: these are {BULK_WAV_SUBTYPE}. The reconstruction is exact in",
            "the renderer's own 32-bit float; writing 24-bit rounds each sample",
            "by an inaudible amount. Export tracks individually as float if you",
            "need the sum to be bit-identical.",
            "",
        ]
    else:
        lines += [
            "MIDI IS A LOSSY DESCRIPTION OF THIS AUDIO.",
            "",
            "Each file holds one track's tones as notes with pitch bend, so",
            "slides and vibrato survive rather than being snapped to the nearest",
            "semitone. What does not survive is timbre and phase: MIDI describes",
            "notes, not sound. The pitch-bend range is set to +/-48 semitones via",
            "RPN 0 at the start of every channel -- if your instrument ignores",
            "that, wide glides will play flattened.",
            "",
        ]

    lines += ["INCLUDED", "-" * 60]
    lines += [f"  {name}" for name in report.included]

    if report.skipped:
        lines += ["", "NOT INCLUDED", "-" * 60]
        for s in report.skipped:
            lines.append(f"  [{s.track_id:03d}] {s.label}")
            lines.append(f"        {s.reason}")
        lines += [
            "",
            "Nothing here is lost from the analysis -- these tracks are still",
            "in the app and can be exported individually or as WAV.",
        ]

    lines += ["", "=" * 60, "Generated by fourier_partial.", ""]
    return "\n".join(lines)


def build_archive(
    result: AnalysisResult,
    fmt: ArchiveFormat,
    destination: Path,
    scratch_dir: Path,
) -> ArchiveReport:
    """Write every track into one downloadable file.

    Args:
        result: The completed analysis.
        fmt: What to produce.
        destination: Where to write the archive. Its parent is created.
        scratch_dir: A directory for per-track temporary files. Each is deleted
            as soon as it has been streamed into the archive, so this never
            holds more than one track at a time.

    Returns:
        An ArchiveReport naming everything included and everything skipped.

    Raises:
        ExportError: If nothing could be exported at all. A partial archive is
            a fine outcome and is reported rather than raised; an empty one is
            not, because handing back a zip containing only a manifest saying
            "nothing here" is worse than an error message.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if fmt is ArchiveFormat.MIDI_SINGLE:
        return _build_single_midi(result, destination)

    report = ArchiveReport(path=destination, format=fmt)
    exporter = WavExporter(BULK_WAV_SUBTYPE) if fmt is ArchiveFormat.WAV else MidiExporter()
    compression = _WAV_COMPRESSION if fmt is ArchiveFormat.WAV else _MIDI_COMPRESSION

    with zipfile.ZipFile(destination, "w", compression=compression, allowZip64=True) as archive:
        for track in result.tracks:
            if not exporter.supports(track):
                report.skipped.append(
                    SkippedTrack(
                        track_id=track.track_id,
                        label=track.label,
                        reason=(
                            "The residual is broadband noise; it has no notes to write as "
                            "MIDI. Export it as WAV instead."
                            if track.kind is TrackKind.RESIDUAL
                            else "This track has nothing exportable in this format."
                        ),
                    )
                )
                continue

            name = _safe_stem(track) + exporter.file_extension
            scratch = scratch_dir / name
            try:
                exporter.export(track, scratch, result.sample_rate)
                # `write` streams from disk rather than reading the file into
                # memory, which is the whole reason each track goes via a
                # scratch file instead of a BytesIO.
                archive.write(scratch, arcname=name)
                report.included.append(name)
            except Exception:
                logger.exception("bulk export failed for track %s", track.track_id)
                report.skipped.append(
                    SkippedTrack(
                        track_id=track.track_id,
                        label=track.label,
                        reason="This track could not be written. The rest of the export is unaffected.",
                    )
                )
            finally:
                # Deleted immediately, so a forty-track WAV export never has
                # more than one loose track on disk beside the growing archive.
                scratch.unlink(missing_ok=True)

        if not report.included:
            raise ExportError(
                f"nothing could be exported as {fmt.value} "
                f"({len(report.skipped)} track(s) skipped)"
            )

        archive.writestr("MANIFEST.txt", _manifest(result, report))

    report.total_bytes = destination.stat().st_size
    logger.info(
        "bulk export %s: %d file(s), %d skipped, %.1f MB",
        fmt.value,
        len(report.included),
        len(report.skipped),
        report.total_bytes / 1e6,
    )
    return report


def _build_single_midi(result: AnalysisResult, destination: Path) -> ArchiveReport:
    """One multi-channel MIDI file, and an honest account of what fits.

    `MidiExporter.export_bundle` already handles the channel arithmetic and the
    selection. This wraps it so the caller gets the same ArchiveReport shape as
    the zip paths, including a per-track reason for anything left out.
    """
    report = ArchiveReport(path=destination, format=ArchiveFormat.MIDI_SINGLE)
    exporter = MidiExporter()

    eligible = [t for t in result.tracks if exporter.supports(t)]
    for track in result.tracks:
        if track not in eligible:
            report.skipped.append(
                SkippedTrack(
                    track_id=track.track_id,
                    label=track.label,
                    reason="Has no pitch content to write as MIDI.",
                )
            )

    if not eligible:
        raise ExportError("no track in this analysis has pitch content to export as MIDI")

    # Mirror export_bundle's own selection so the report matches the file.
    # Ranking by energy means what survives the channel limit is what a
    # listener would most notice missing.
    ranked = sorted(
        eligible,
        key=lambda t: sum(p.energy for p in t.partials),
        reverse=True,
    )
    kept = ranked[: _usable_channel_count()]
    for track in ranked[len(kept) :]:
        report.skipped.append(
            SkippedTrack(
                track_id=track.track_id,
                label=track.label,
                reason=(
                    "A single MIDI file has only 15 usable channels, and pitch bend "
                    "applies per channel. Export as separate MIDI files to get this one."
                ),
            )
        )

    exporter.export_bundle(ranked, destination, result.sample_rate)
    report.included = [f"{_safe_stem(t)} (channel {i})" for i, t in enumerate(kept)]
    report.total_bytes = destination.stat().st_size
    return report


def _usable_channel_count() -> int:
    """How many MIDI channels a bundle may use.

    Sixteen exist; channel 10 (0-indexed 9) is General MIDI percussion, and a
    pitched partial routed there plays as a drum hit on any GM instrument.
    Read from the exporter's own constant so the two cannot disagree.
    """
    from backend.export.midi_exporter import _USABLE_CHANNELS

    return len(_USABLE_CHANNELS)
