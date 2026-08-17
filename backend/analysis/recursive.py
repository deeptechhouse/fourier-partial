"""Analyse the residual, repeatedly.

WHY THIS EXISTS
---------------
One analysis pass is not exhaustive, and not because the analyser is weak.
Peak picking is *relative*: a spectral peak has to clear `peak_threshold_db`
measured against the loudest peak in its own frame. A quiet tone sitting under
a loud one therefore never gets picked, however clearly it is there.

Subtract the loud one and that quiet tone becomes the loudest thing left. The
same analyser, at the same settings, now finds it. This is matching pursuit in
its plainest form: explain what you can, then look again at what is left.

THE INVARIANT COMPOSES
----------------------
The residual is defined as a subtraction at every level, never as a model:

    original   = sum(partials_1) + residual_1
    residual_1 = sum(partials_2) + residual_2
    ...
    residual_n = sum(partials_n+1) + residual_n+1

Substituting downwards, every level's partials plus the FINAL residual sum to
the original, sample for sample. Recursion does not weaken the product's
central promise -- it is preserved by arithmetic, exactly as at depth 1. The
check at the bottom of `analyse` asserts precisely this.

THE ONE PLACE IT DOES CHANGE
----------------------------
`residual_skip` throws the early levels away. Those partials are then genuinely
gone, so what comes back sums to the stripped residual and not to the file the
user opened. That is the entire point of the setting -- "show me what was
underneath" -- but it must never be silent. It is handled by *substituting the
original*: the result's `original` becomes the stripped residual, so the
invariant is still exactly true, and every downstream consumer (the mixdown,
the export manifest, the reconstruction check) is automatically talking about
the right reference signal without needing to know recursion happened.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from backend.analysis.pipeline import ProgressCallback
from backend.core.types import AnalysisParams, AnalysisResult, Track, TrackKind

logger = logging.getLogger(__name__)

#: There is deliberately NO energy-based stopping rule.
#:
#: The first version of this module stopped once a pass explained less than a
#: millionth of the incoming energy, on the reasoning that such a pass was not
#: worth its cost. Measuring it showed that was exactly backwards. On a loud
#: tone hiding two quiet ones, pass 2 explained 0.0% of the energy -- and
#: passes 3 and 4 went on to find six more tracks. Quiet is the *point*: a user
#: who asked for depth 4 is asking to dig past the loud material, and an energy
#: floor throws away the only thing they wanted.
#:
#: The one stop that remains is provable rather than tuned; see `_exhausted`.


class LevelReport:
    """What one pass of the analyser achieved.

    Kept because "depth 5" is meaningless without knowing that levels 4 and 5
    found nothing: the user needs to see where the material actually ran out
    rather than guessing at a number.
    """

    __slots__ = ("_level", "_tracks_found", "_energy_share", "_kept")

    def __init__(self, level: int, tracks_found: int, energy_share: float, kept: bool) -> None:
        self._level = level
        self._tracks_found = tracks_found
        self._energy_share = energy_share
        self._kept = kept

    @property
    def level(self) -> int:
        """1-based pass number, counting skipped passes."""
        return self._level

    @property
    def tracks_found(self) -> int:
        """Pitched tracks this pass produced, excluding its residual."""
        return self._tracks_found

    @property
    def energy_share(self) -> float:
        """Share of the incoming energy this pass explained, in [0, 1]."""
        return self._energy_share

    @property
    def kept(self) -> bool:
        """False for passes discarded by `residual_skip`."""
        return self._kept

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form, for the API and the CLI's report."""
        return {
            "level": self._level,
            "tracks_found": self._tracks_found,
            "energy_share": round(self._energy_share, 6),
            "kept": self._kept,
        }

    def __repr__(self) -> str:
        state = "kept" if self._kept else "discarded"
        return (
            f"LevelReport(L{self._level}: {self._tracks_found} tracks, "
            f"{self._energy_share:.1%} of energy, {state})"
        )


class RecursiveAnalyser:
    """Runs a single-pass pipeline repeatedly over its own residual.

    Deliberately a wrapper rather than a change to the pipeline. The pipeline's
    job -- decompose this buffer, exactly once, and prove the result sums back
    -- is complete and correct, and stays that way. Recursion is a policy about
    how many times to invoke it and what to do with the pieces, which is a
    different concern and belongs in a different object.
    """

    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: object) -> None:
        """Wrap any object with a `.analyze(samples, params, on_progress)`."""
        self._pipeline = pipeline

    def analyse(
        self,
        samples: NDArray[np.float32],
        params: AnalysisParams,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[AnalysisResult, list[LevelReport]]:
        """Decompose `samples`, optionally re-analysing the residual.

        Args:
            samples: Mono float32 audio.
            params: Settings, including `residual_skip` and `recursion_depth`.
            on_progress: Called with overall progress across ALL levels, so a
                depth-4 run does not show the bar completing four times.

        Returns:
            The combined result, and a report per level.

        The returned result's `original` is the audio the tracks actually sum
        to: the input normally, or the stripped residual when levels were
        skipped. Nothing downstream then needs to know recursion happened.
        """
        skip = int(params.residual_skip)
        depth = int(params.recursion_depth)
        total_levels = skip + depth

        audio = np.asarray(samples, dtype=np.float32)
        # The signal each pass works on. Starts as the input and becomes the
        # previous pass's residual.
        current = audio
        # What the kept tracks are ultimately measured against. Moves to the
        # stripped residual if any levels are skipped -- see the module note.
        reference = audio

        kept: list[Track] = []
        reports: list[LevelReport] = []
        final_residual: Track | None = None

        for index in range(total_levels):
            level = index + 1
            is_skipped = index < skip

            result = self._pipeline.analyze(
                current,
                params,
                self._level_progress(on_progress, index, total_levels),
            )

            pitched = [t for t in result.tracks if t.kind is not TrackKind.RESIDUAL]
            residual = result.residual_track
            explained = _energy_share(current, residual)
            reports.append(LevelReport(level, len(pitched), explained, not is_skipped))

            if is_skipped:
                # A skipped pass can exhaust the signal too, and then there is
                # nothing left to hand the passes we were going to keep.
                # Everything this pass found is thrown away; only what it could
                # NOT explain survives. The reference moves with it, because
                # that is now the whole of what the user is being shown.
                logger.info("level %d discarded: %d tracks, %.1f%% of energy",
                            level, len(pitched), 100 * explained)
                current = _residual_audio(residual, current)
                reference = current
                continue

            for track in pitched:
                track.set_level(level)
                track.relabel(_level_label(track.label, level, total_levels, skip))
                kept.append(track)
            if residual is not None:
                # The residual belongs to the last pass that produced one, and
                # the manifest reports levels, so it must be stamped too.
                residual.set_level(level)
            final_residual = residual

            logger.info("level %d kept: %d tracks, %.1f%% of energy",
                        level, len(pitched), 100 * explained)

            # --- stop early, on a fact rather than a guess -------------------
            # See `_exhausted`: this triggers only when the pass provably
            # cannot lead anywhere, never because its yield looked small.
            following = _residual_audio(residual, current)
            if _exhausted(following, current):
                if level < total_levels:
                    logger.info(
                        "stopping at level %d of %d: the residual is unchanged, "
                        "so every further pass would repeat this one",
                        level, total_levels,
                    )
                break

            current = following

        return self._combine(kept, final_residual, reference, params, reports), reports

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _level_progress(
        on_progress: ProgressCallback | None, index: int, total: int
    ) -> ProgressCallback | None:
        """Rescale one level's 0..1 progress into its slice of the whole run."""
        if on_progress is None:
            return None

        def report(fraction: float, stage: str) -> None:
            overall = (index + max(0.0, min(1.0, fraction))) / total
            label = stage if total == 1 else f"{stage} (pass {index + 1} of {total})"
            on_progress(overall, label)

        return report

    @staticmethod
    def _combine(
        kept: list[Track],
        residual: Track | None,
        reference: NDArray[np.float32],
        params: AnalysisParams,
        reports: list[LevelReport],
    ) -> AnalysisResult:
        """Renumber the kept tracks, append the final residual, and check.

        Renumbering is not cosmetic. Each pass numbers its tracks from zero, so
        without this two tracks from different levels would share an id -- and
        an id collision silently breaks mute, solo, per-track export, and the
        `only=` selection used by record-out.
        """
        tracks: list[Track] = []
        for track in kept:
            track.renumber(len(tracks))
            tracks.append(track)
        if residual is not None:
            residual.renumber(len(tracks))
            tracks.append(residual)

        duration_s = reference.shape[0] / params.sample_rate
        result = AnalysisResult(params.sample_rate, duration_s, tracks, params, reference)

        # The claim this whole module has to earn. Checked here as well as
        # inside each pass, because the composition across levels is the new
        # thing and is exactly what could be got wrong.
        error = result.reconstruction_error()
        if not np.isfinite(error) or error > 1e-4:
            raise ValueError(
                f"recursive analysis broke the reconstruction invariant: "
                f"max sample error {error:.3e} across {len(reports)} level(s)"
            )
        return result


def _exhausted(
    following: NDArray[np.float32], current: NDArray[np.float32]
) -> bool:
    """Whether another pass could possibly find anything.

    True only when the residual came back bit-identical to what the pass was
    given. That is a proof, not an estimate: the next pass would receive the
    same samples, and the analyser is deterministic, so it would produce the
    same nothing. Running it could only spend another full analysis confirming
    that.

    Note what is NOT here. A pass that explains very little energy is not
    exhausted -- it has still changed the residual, so the next pass sees new
    input and routinely finds more. Stopping on a small yield was a real bug:
    it cut a depth-4 run short at level 2 and threw away six tracks.
    """
    return following.shape == current.shape and bool(np.array_equal(following, current))


def _residual_audio(
    residual: Track | None, fallback: NDArray[np.float32]
) -> NDArray[np.float32]:
    """The audio to feed the next pass.

    A pass with no residual explained nothing, so the next pass gets what this
    one was given. Copied because the next pass's result will hold it as its
    `original`, and results expose that array as a read-only view of memory
    they assume they own.
    """
    if residual is None or residual.audio is None:
        return np.array(fallback, dtype=np.float32)
    return np.array(residual.audio, dtype=np.float32)


def _energy_share(source: NDArray[np.float32], residual: Track | None) -> float:
    """How much of `source`'s energy this pass explained, in [0, 1]."""
    total = float(np.sum(np.square(source, dtype=np.float64)))
    if total <= 0.0:
        return 0.0
    if residual is None or residual.audio is None:
        return 1.0
    left = float(np.sum(np.square(np.asarray(residual.audio, dtype=np.float64))))
    return max(0.0, min(1.0, 1.0 - left / total))


def _level_label(label: str, level: int, total_levels: int, skip: int) -> str:
    """Mark which pass found a track, but only when that is information.

    An ordinary depth-1 analysis gains nothing from every row saying "L1", so
    the marker appears only when more than one pass contributed. Skipped passes
    still count towards the numbering, so "L3" means the third pass of the run
    rather than the third kept one -- which is what a user comparing settings
    needs it to mean.
    """
    if total_levels <= 1 or (skip == 0 and total_levels - skip <= 1):
        return label
    return f"{label} · L{level}"
