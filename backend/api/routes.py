"""HTTP + WebSocket routes.

API SURFACE
-----------
    POST   /api/analyze                          upload audio, returns {job_id}
    GET    /api/jobs/{job_id}                    job status and progress
    WS     /api/jobs/{job_id}/events             live progress stream
    GET    /api/jobs/{job_id}/tracks             track metadata (no audio)
    GET    /api/jobs/{job_id}/tracks/{tid}/audio rendered PCM for playback
    GET    /api/jobs/{job_id}/tracks/{tid}/export ?format=wav|mid
    GET    /api/jobs/{job_id}/mixdown            ?exclude=ids -> all tracks, minus those
    GET    /api/health                           liveness

WHY THERE IS A MIXDOWN ENDPOINT
-------------------------------
Because the tracks sum to the original exactly, "play everything" is one
buffer rather than one buffer per track. At ten minutes a track is ~106 MB
decoded, so for a long file that is the difference between ~106 MB in the
browser and several gigabytes. `exclude` extends the same trick to muting.

WHY A JOB RATHER THAN A SYNCHRONOUS RESPONSE
--------------------------------------------
Analysing 20 seconds of audio takes seconds, not milliseconds. Holding an HTTP
request open for that is fragile (proxies time out) and gives the user no
feedback. A job id plus a progress socket costs a little more plumbing and
makes the wait legible.

THIS LAYER OWNS THE SECURITY BOUNDARY. Everything arriving here came from
outside and is hostile until proven otherwise. Validation happens before
anything touches the disk, and no exception text ever reaches a client.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from backend.analysis.pipeline import downsample_trajectory
from backend.api.jobs import InMemoryJobStore, JobState
from backend.config import settings
from backend.core.errors import FourierPartialError, UnsupportedExportError
from backend.core.types import AnalysisParams, AnalysisResult, Track, TrackKind, pitch_label
from backend.export.midi_exporter import MidiExporter
from backend.export.wav_exporter import WavExporter
from backend.services import build_loader, build_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

# =============================================================================
# Application state
# =============================================================================
# Module-level singletons, built once. The job store is thread-safe; the
# pipeline and loader are stateless, so sharing them across requests is safe.

_jobs = InMemoryJobStore(ttl_seconds=settings.session_ttl_seconds)
_pipeline = build_pipeline()
_loader = build_loader(max_duration_seconds=settings.max_duration_seconds)
_params = AnalysisParams()

# ADJUSTABLE: how many analyses may run at once. Analysis is CPU-bound, so
# more threads than cores just adds contention and memory pressure.
_MAX_CONCURRENT_ANALYSES: int = 2
_analysis_slots = threading.Semaphore(_MAX_CONCURRENT_ANALYSES)

# ADJUSTABLE: rate limit on the one expensive endpoint. A simple sliding
# window per client address -- enough to stop a runaway script or an
# accidental refresh loop, which is the realistic threat for a local tool.
#
# 60 per minute, not 10. The threat here is a loop, not a determined attacker,
# and 10 was low enough to inconvenience the legitimate case: someone chopping
# a session into clips and feeding them in one after another hits it in under
# two minutes of ordinary work. It also made the test suite flaky, because the
# limiter is process-global and two test modules that each upload a handful of
# files together crossed the line -- surfacing as unrelated-looking failures
# in whichever test happened to run eleventh.
_RATE_LIMIT_REQUESTS: int = 60
_RATE_LIMIT_WINDOW_S: float = 60.0
_rate_lock = threading.Lock()
_rate_history: dict[str, deque[float]] = {}


def reset_rate_limit() -> None:
    """Forget all recorded request times.

    Exists for tests. The limiter is module-global by design (it must outlive
    any single request), which means it also outlives any single test, so a
    suite that uploads more than the allowance would otherwise fail in a way
    that points at the wrong test entirely.
    """
    with _rate_lock:
        _rate_history.clear()

# ADJUSTABLE: how often the progress socket samples the job store, in seconds.
# The store is polled rather than pushed-to because analysis runs in a worker
# thread: pushing across the thread/event-loop boundary needs
# call_soon_threadsafe plumbing that buys nothing at this update rate.
_SOCKET_POLL_S: float = 0.1

# ADJUSTABLE: bytes read per chunk while streaming an upload to disk.
_UPLOAD_CHUNK_BYTES: int = 1024 * 1024


def _session_dir() -> Path:
    """The scratch directory for uploads and exports, created on demand."""
    path = Path(settings.session_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Response schemas
# =============================================================================


class JobCreatedResponse(BaseModel):
    """Returned immediately after an upload is accepted."""

    job_id: str
    estimated_seconds: float = Field(description="Rough estimate of analysis time.")


class JobStatusResponse(BaseModel):
    """Current state of a background analysis."""

    job_id: str
    state: str = Field(description="pending | running | complete | failed")
    progress: float = Field(ge=0.0, le=1.0)
    stage: str
    message: str | None = Field(
        default=None,
        description="User-safe error text when state is 'failed'. Never a stack trace.",
    )


class TrackSummary(BaseModel):
    """Metadata for one track, enough to draw and label it without audio."""

    track_id: int
    kind: str
    label: str
    start_time_s: float
    end_time_s: float
    mean_frequency_hz: float
    peak_amplitude: float
    energy_ratio: float = Field(description="Share of total energy, 0.0 - 1.0.")
    partial_count: int
    exportable_as_midi: bool

    # The bare note name ('G4'), with no cent deviation. The interface groups
    # rows by this so a long recording reads as "G4, seven times" rather than
    # as seven near-identical rows. Empty for the residual, which has no pitch.
    pitch: str = ""

    # Downsampled trajectory for the visualiser. Full-resolution trajectories
    # would be megabytes per track; the display is a few hundred pixels wide,
    # so sending more than it can draw is pure waste.
    trajectory: list[tuple[float, float, float]] = Field(
        default_factory=list,
        description="(time_s, frequency_hz, amplitude) triples, downsampled for display.",
    )


class AnalysisSummary(BaseModel):
    """Everything the UI needs to render a completed analysis."""

    job_id: str
    sample_rate: int
    duration_s: float
    tracks: list[TrackSummary]

    # Diagnostics -- shown to the user so the decomposition is legible rather
    # than magic. See design document section 12.2.
    reconstruction_exact: bool
    partial_energy_ratio: float
    residual_energy_ratio: float
    warnings: list[str] = Field(default_factory=list)


# =============================================================================
# Helpers
# =============================================================================


def _client_key(request: Request) -> str:
    """Identify a caller for rate limiting."""
    return request.client.host if request.client else "unknown"


def _check_rate_limit(key: str) -> None:
    """Raise 429 if `key` has exceeded the sliding-window allowance."""
    now = time.monotonic()
    with _rate_lock:
        history = _rate_history.setdefault(key, deque())
        while history and now - history[0] > _RATE_LIMIT_WINDOW_S:
            history.popleft()
        if len(history) >= _RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many analyses started. Please wait a moment and try again.",
            )
        history.append(now)


def _track_energy(track: Track) -> float:
    """Sum of squares of a track's rendered audio."""
    audio = track.audio
    if audio is None:
        return 0.0
    return float(np.sum(np.square(audio.astype(np.float64))))


def _summarise(job_id: str, result: AnalysisResult) -> AnalysisSummary:
    """Build the UI payload for a completed analysis."""
    energies = [_track_energy(t) for t in result.tracks]
    total = sum(energies) or 1.0  # avoid dividing by zero on a silent upload

    summaries: list[TrackSummary] = []
    for track, energy in zip(result.tracks, energies, strict=True):
        partials = track.partials
        if partials:
            start = min(p.start_time_s for p in partials)
            end = max(p.end_time_s for p in partials)
            lowest = min(partials, key=lambda p: p.mean_frequency_hz)
            mean_hz = lowest.mean_frequency_hz
            peak_amp = max(p.peak_amplitude for p in partials)
        else:
            # The residual spans the whole clip and has no single frequency.
            start, end, mean_hz = 0.0, result.duration_s, 0.0
            audio = track.audio
            peak_amp = float(np.max(np.abs(audio))) if audio is not None else 0.0

        summaries.append(
            TrackSummary(
                track_id=track.track_id,
                kind=track.kind.value,
                label=track.label,
                start_time_s=round(start, 4),
                end_time_s=round(end, 4),
                mean_frequency_hz=round(mean_hz, 2),
                peak_amplitude=round(peak_amp, 5),
                energy_ratio=round(energy / total, 5),
                partial_count=len(partials),
                exportable_as_midi=track.is_exportable_as_midi,
                pitch=pitch_label(mean_hz) if track.kind is not TrackKind.RESIDUAL else "",
                trajectory=downsample_trajectory(track),
            )
        )

    report = _jobs.get_report(job_id)
    return AnalysisSummary(
        job_id=job_id,
        sample_rate=result.sample_rate,
        duration_s=round(result.duration_s, 4),
        tracks=summaries,
        reconstruction_exact=report.is_exact if report else True,
        partial_energy_ratio=round(report.partial_energy_ratio, 4) if report else 0.0,
        residual_energy_ratio=round(report.residual_energy_ratio, 4) if report else 0.0,
        warnings=list(report.warnings) if report else [],
    )


def _find_track(result: AnalysisResult, track_id: int) -> Track:
    """Look up a track by id, or 404."""
    for track in result.tracks:
        if track.track_id == track_id:
            return track
    raise HTTPException(status_code=404, detail="No such track.")


def _require_result(job_id: str) -> AnalysisResult:
    """Fetch a completed result, or raise the right HTTP error."""
    record = _jobs.snapshot(job_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail="That analysis session has expired. Please upload again."
        )
    if record.state is JobState.FAILED:
        raise HTTPException(status_code=400, detail=record.user_message or "Analysis failed.")
    result = _jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=409, detail="Analysis is still running.")
    return result


def _run_analysis(job_id: str, upload_path: Path) -> None:
    """Worker-thread entry point: load, analyse, verify, store.

    Runs off the event loop so the server keeps answering requests. Every
    exception is converted into a user-safe message here; nothing escapes.
    """
    acquired = _analysis_slots.acquire(timeout=120)
    if not acquired:
        _jobs.fail(job_id, "The server is busy. Please try again in a moment.")
        upload_path.unlink(missing_ok=True)
        return
    try:
        _jobs.set_progress(job_id, 0.01, "Reading audio")
        samples, _ = _loader.load(upload_path, _params.sample_rate)

        # Settings are chosen from the ACTUAL duration, not fixed up front.
        # A ten-minute file analysed at phrase settings would need ~6.8 GB for
        # the spectrogram alone and tens of thousands of frames; the coarser
        # tiers trade some onset precision for the file being analysable at
        # all. See AnalysisParams.for_duration.
        duration_s = len(samples) / _params.sample_rate
        params = AnalysisParams.for_duration(duration_s)
        if params.hop_size != _params.hop_size:
            logger.info(
                "job %s: %.1fs of audio -> hop=%d, zero_pad=%d (long-form settings)",
                job_id,
                duration_s,
                params.hop_size,
                params.zero_pad_factor,
            )

        def on_progress(fraction: float, stage: str) -> None:
            _jobs.set_progress(job_id, fraction, stage)

        result = _pipeline.analyze(samples, params, on_progress=on_progress)
        report = _pipeline.verify(result)
        _jobs.complete(job_id, result, report)
        logger.info(
            "job %s complete: %d tracks, partial energy %.1f%%, error %.2e",
            job_id,
            len(result.tracks),
            100 * report.partial_energy_ratio,
            report.max_sample_error,
        )
    except FourierPartialError as exc:
        # Known failure. The exception's own text may name paths and
        # parameters, so it goes to the log; the client gets user_message.
        logger.warning("job %s failed: %s", job_id, exc, exc_info=True)
        _jobs.fail(job_id, exc.user_message)
    except Exception:
        logger.exception("job %s crashed unexpectedly", job_id)
        _jobs.fail(job_id, "An unexpected error occurred while analysing that audio.")
    finally:
        _analysis_slots.release()
        # The upload is not needed once decoded, and holding user audio longer
        # than necessary has no upside.
        upload_path.unlink(missing_ok=True)
        _jobs.purge_expired()


# =============================================================================
# Routes
# =============================================================================


@router.post("/analyze", response_model=JobCreatedResponse)
async def analyze(request: Request, file: UploadFile) -> JobCreatedResponse:
    """Accept an audio upload and start a background analysis.

    Validation happens here, at the boundary, before anything is written to
    disk: rate limit, size ceiling, then (in the loader) magic bytes and
    duration. See design document section 13.
    """
    _check_rate_limit(_client_key(request))

    job_id = uuid.uuid4().hex
    # A generated name, never the uploader's. A user-supplied filename is an
    # untrusted string and has no business determining a path.
    upload_path = _session_dir() / f"upload_{job_id}"

    # Stream to disk with a running size check, so an oversized file is
    # rejected partway rather than after it has all been buffered in memory.
    total = 0
    try:
        with upload_path.open("wb") as sink:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    sink.close()
                    upload_path.unlink(missing_ok=True)
                    limit_mb = settings.max_upload_bytes / (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=f"That file is too large (limit {limit_mb:.0f} MB).",
                    )
                sink.write(chunk)
    except HTTPException:
        raise
    except OSError:
        upload_path.unlink(missing_ok=True)
        logger.exception("failed to store upload for job %s", job_id)
        raise HTTPException(status_code=500, detail="Could not store the upload.") from None
    finally:
        await file.close()

    if total == 0:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="That file is empty.")

    _jobs.create(job_id, filename=file.filename or "")
    threading.Thread(
        target=_run_analysis, args=(job_id, upload_path), daemon=True, name=f"analyze-{job_id[:8]}"
    ).start()

    # Rough guide for the UI, not a promise: analysis is a little slower than
    # real time on this class of machine.
    estimated = max(2.0, (total / (176_400)) * 1.5)
    return JobCreatedResponse(job_id=job_id, estimated_seconds=round(estimated, 1))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    """Poll a job's progress. The WebSocket endpoint is preferred."""
    record = _jobs.snapshot(job_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail="That analysis session has expired. Please upload again."
        )
    return JobStatusResponse(
        job_id=record.job_id,
        state=record.state.value,
        progress=record.progress,
        stage=record.stage,
        message=record.user_message,
    )


@router.websocket("/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    """Stream progress updates until the job reaches a terminal state."""
    await websocket.accept()
    last: tuple[str, float, str] | None = None
    try:
        while True:
            record = _jobs.snapshot(job_id)
            if record is None:
                await websocket.send_json(
                    {
                        "job_id": job_id,
                        "state": "failed",
                        "progress": 0.0,
                        "stage": "Expired",
                        "message": "That analysis session has expired. Please upload again.",
                    }
                )
                break

            current = (record.state.value, record.progress, record.stage)
            # Only send on change. A socket emitting ten identical frames a
            # second is just noise for the client to diff.
            if current != last:
                last = current
                await websocket.send_json(
                    {
                        "job_id": record.job_id,
                        "state": record.state.value,
                        "progress": record.progress,
                        "stage": record.stage,
                        "message": record.user_message,
                    }
                )
            if record.is_terminal:
                break
            await asyncio.sleep(_SOCKET_POLL_S)
    except WebSocketDisconnect:
        # The user navigated away mid-analysis. Entirely normal; the worker
        # thread carries on and the result stays fetchable by job id.
        return
    except Exception:
        logger.exception("progress socket for job %s failed", job_id)
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            # Already closed by the client. Nothing to do.
            pass


@router.get("/jobs/{job_id}/tracks", response_model=AnalysisSummary)
async def job_tracks(job_id: str) -> AnalysisSummary:
    """Return track metadata for a completed analysis."""
    return _summarise(job_id, _require_result(job_id))


@router.get("/jobs/{job_id}/tracks/{track_id}/audio")
async def track_audio(job_id: str, track_id: int) -> Response:
    """Return a track's rendered audio as a 32-bit float WAV.

    Float rather than 16-bit PCM so the browser receives exactly the samples
    the renderer produced. That is what lets "unmute everything" reproduce the
    upload bit for bit rather than approximately -- quantising here would
    quietly break the guarantee the whole design is built around.
    """
    result = _require_result(job_id)
    track = _find_track(result, track_id)
    audio = track.audio
    if audio is None:
        raise HTTPException(status_code=409, detail="That track has not been rendered.")

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio), result.sample_rate, format="WAV", subtype="FLOAT")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/jobs/{job_id}/mixdown")
async def mixdown(
    job_id: str,
    exclude: str = Query(default="", description="Comma-separated track ids to leave out."),
) -> Response:
    """Return the sum of every track, optionally minus a few, as one WAV.

    WHY THIS EXISTS — it is the invariant paying rent.
    -------------------------------------------------
    The browser used to fetch every track and decode them all before the first
    Play. At ten minutes a single track is ~106 MB decoded, so forty tracks is
    several gigabytes and the tab dies.

    But the tracks sum to the original *exactly* (design document section 7),
    so "play everything" never needed forty buffers -- it needed one, and we
    already have it. With `exclude`, muting a handful of tracks is the same
    trick: serve everything-except-those as one buffer, and the client holds
    two or three buffers instead of forty, however long the audio is.

    Args:
        job_id: The analysis to mix.
        exclude: Track ids to omit, e.g. "0,4,9". Unknown ids are ignored
            rather than rejected -- a stale id from a client that has moved on
            is not worth failing a playback request over.

    Returns:
        32-bit float WAV. Float, not PCM, because with nothing excluded this
        must be byte-identical to what the user uploaded.
    """
    result = _require_result(job_id)

    excluded: set[int] = set()
    for piece in exclude.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            excluded.add(int(piece))
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Track ids must be whole numbers."
            ) from None

    if not excluded:
        # Nothing to leave out, so the answer is the original by definition.
        # Serving it directly avoids re-summing forty buffers to arrive back at
        # a number we already hold -- and guarantees byte-identity rather than
        # merely achieving it.
        mixed = np.asarray(result.original)
    else:
        mixed = np.zeros(int(result.duration_s * result.sample_rate), dtype=np.float64)
        for track in result.tracks:
            if track.track_id in excluded:
                continue
            audio = track.audio
            if audio is not None:
                mixed += audio
        mixed = mixed.astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, mixed, result.sample_rate, format="WAV", subtype="FLOAT")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/jobs/{job_id}/tracks/{track_id}/export")
async def export_track(
    job_id: str,
    track_id: int,
    format: str = Query(default="wav", pattern="^(wav|mid)$"),
) -> FileResponse:
    """Export one track as a downloadable .wav or .mid file."""
    result = _require_result(job_id)
    track = _find_track(result, track_id)

    exporter = WavExporter() if format == "wav" else MidiExporter()
    if not exporter.supports(track):
        raise HTTPException(
            status_code=400,
            detail=(
                "The residual is broadband noise, so it has no notes to export as MIDI. "
                "Export it as WAV instead."
                if format == "mid"
                else "That track cannot be exported in the requested format."
            ),
        )

    # Sanitise the label into a filename. The label is generated by us, not by
    # the user, but building paths from display strings is a habit worth not
    # forming.
    safe = "".join(c if c.isalnum() or c in "-_+" else "_" for c in track.label)[:48]
    filename = f"track{track.track_id:03d}_{safe}{exporter.file_extension}"
    destination = Path(tempfile.mkdtemp(prefix="fp_export_", dir=_session_dir())) / filename

    try:
        exporter.export(track, destination, result.sample_rate)
    except UnsupportedExportError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from None
    except Exception:
        logger.exception("export failed for job %s track %s", job_id, track_id)
        raise HTTPException(status_code=500, detail="The export could not be completed.") from None

    return FileResponse(destination, filename=filename, media_type="application/octet-stream")


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. Also used by the .app launcher to detect readiness."""
    return {"status": "ok"}


def cleanup_sessions() -> None:
    """Remove this process's scratch files. Called on shutdown.

    DELIBERATELY SURGICAL. An earlier version removed the whole session
    directory with `rmtree`, which was wrong for a reason that only shows up
    when something has already gone wrong: the .app launcher keeps its
    `server.log` and `server.pid` in this same directory, so a clean shutdown
    deleted the very log the launcher's crash dialog exists to display. It also
    raced a concurrently-starting second instance, wiping files out from under
    it.

    So: remove only the two things this module creates, by name, and leave
    everything else alone. Files that do not match are somebody else's.
    """
    session_dir = Path(settings.session_dir)
    if not session_dir.is_dir():
        return

    removed = 0
    for entry in session_dir.iterdir():
        try:
            if entry.is_file() and entry.name.startswith("upload_"):
                entry.unlink(missing_ok=True)
                removed += 1
            elif entry.is_dir() and entry.name.startswith("fp_export_"):
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            # One stubborn file must not stop the rest being cleaned up.
            logger.warning("could not remove session entry %s", entry, exc_info=True)

    if removed:
        logger.info("cleaned up %d session artifact(s)", removed)
