"""In-memory job store with TTL expiry.

Deliberately not a database. Analyses are ephemeral by design (design document
section 13): a session's audio and rendered tracks are user data we have no
reason to retain, so they live in memory and are purged after
`session_ttl_seconds`.

If multi-worker deployment ever becomes necessary, this class is the seam --
implement the same `JobStore` protocol against Redis and change one line in
`backend/services.py`. Nothing else in the system knows where jobs live.

THREADING: analysis runs in a worker thread while the event loop keeps serving
requests, so every read and write here is guarded by a re-entrant lock. Getting
this wrong produces the worst class of bug in the project: intermittent,
unreproducible, and only under load.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum

from backend.core.types import AnalysisResult, ReconstructionReport

logger = logging.getLogger(__name__)


class JobState(Enum):
    """Lifecycle states for a background analysis."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Whether no further updates will arrive for a job in this state."""
        return self in (JobState.COMPLETE, JobState.FAILED)


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Immutable snapshot of one analysis job.

    Frozen on purpose: `snapshot()` hands these out to the API layer and to the
    WebSocket feed, both of which may read them long after the worker thread
    has moved on. An immutable record cannot be observed half-updated, which
    removes a whole class of race condition without any extra locking at the
    read site.
    """

    job_id: str
    state: JobState
    progress: float
    stage: str
    created_at: float
    filename: str = ""
    user_message: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether this job has finished, one way or the other."""
        return self.state.is_terminal


class InMemoryJobStore:
    """Thread-safe job store. Implements the `JobStore` port."""

    __slots__ = ("_ttl", "_lock", "_records", "_results", "_reports")

    def __init__(self, ttl_seconds: int) -> None:
        """Create a store that expires jobs after `ttl_seconds`."""
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._records: dict[str, JobRecord] = {}
        # Results are held separately from records because they are large
        # (hundreds of megabytes of rendered audio) and are only ever fetched
        # by id -- keeping them out of the snapshot means the WebSocket feed
        # can copy a record cheaply, many times a second.
        self._results: dict[str, AnalysisResult] = {}
        self._reports: dict[str, ReconstructionReport] = {}

    # -- Writes --------------------------------------------------------------

    def create(self, job_id: str, filename: str = "") -> None:
        """Register a new job in the pending state."""
        with self._lock:
            self._records[job_id] = JobRecord(
                job_id=job_id,
                state=JobState.PENDING,
                progress=0.0,
                stage="Queued",
                created_at=time.monotonic(),
                filename=filename,
            )

    def set_progress(self, job_id: str, fraction: float, stage: str) -> None:
        """Update progress (0.0 - 1.0) and the human-readable current stage."""
        with self._lock:
            record = self._records.get(job_id)
            # A vanished job means it expired or was never created. Dropping
            # the update is correct -- the alternative is resurrecting a job
            # nobody is waiting for.
            if record is None or record.is_terminal:
                return
            self._records[job_id] = replace(
                record,
                state=JobState.RUNNING,
                progress=max(0.0, min(1.0, fraction)),
                stage=stage,
            )

    def complete(
        self,
        job_id: str,
        result: AnalysisResult,
        report: ReconstructionReport | None = None,
    ) -> None:
        """Mark a job finished and store its result."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            self._records[job_id] = replace(
                record, state=JobState.COMPLETE, progress=1.0, stage="Complete"
            )
            self._results[job_id] = result
            if report is not None:
                self._reports[job_id] = report

    def fail(self, job_id: str, message: str) -> None:
        """Mark a job failed with a message safe to show a user.

        `message` must come from an error's `user_message`, never from
        `str(exception)` -- exception text carries paths, parameters and stack
        frames. Log the detail; return something generic.
        """
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            self._records[job_id] = replace(
                record,
                state=JobState.FAILED,
                stage="Failed",
                user_message=message,
            )

    # -- Reads ---------------------------------------------------------------

    def snapshot(self, job_id: str) -> JobRecord | None:
        """Current state of a job, or None if unknown or expired."""
        with self._lock:
            return self._records.get(job_id)

    def get(self, job_id: str) -> AnalysisResult | None:
        """Retrieve a completed result, or None if not ready."""
        with self._lock:
            return self._results.get(job_id)

    def get_report(self, job_id: str) -> ReconstructionReport | None:
        """Retrieve a completed job's diagnostics, if any."""
        with self._lock:
            return self._reports.get(job_id)

    # -- Housekeeping --------------------------------------------------------

    def purge_expired(self) -> int:
        """Drop jobs older than the TTL. Returns how many were removed."""
        cutoff = time.monotonic() - self._ttl
        with self._lock:
            stale = [
                job_id
                for job_id, record in self._records.items()
                if record.created_at < cutoff
            ]
            for job_id in stale:
                self._records.pop(job_id, None)
                self._results.pop(job_id, None)
                self._reports.pop(job_id, None)
        if stale:
            logger.info("purged %d expired job(s)", len(stale))
        return len(stale)
