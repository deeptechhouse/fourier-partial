"""Recursive analysis: does it find more, and does the arithmetic survive?

TWO CLAIMS, TESTED SEPARATELY
-----------------------------
1. RECURSION FINDS THINGS ONE PASS CANNOT. Not a nice-to-have — it is the
   entire justification. Peak picking is relative to the loudest peak in each
   frame, so a quiet tone under a loud one is invisible until the loud one is
   subtracted. The fixture below is built to make that measurable: one loud
   tone and two much quieter ones at known frequencies.

2. THE INVARIANT COMPOSES. Every level's partials plus the final residual must
   still sum to the input, sample for sample, at any depth. This is the
   product's central promise and recursion is the most plausible way to break
   it, so it is asserted at every depth rather than once.

The one place the promise legitimately changes is `residual_skip`, which
discards whole layers. There the tracks sum to the stripped residual instead —
and that is tested too, including that the result *says so* by reporting the
stripped signal as its `original`.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.analysis.recursive import (
    LevelReport,
    RecursiveAnalyser,
    _exhausted,
    _level_label,
)
from backend.core.types import AnalysisParams, TrackKind
from backend.services import build_pipeline

SR = 44_100

#: The quiet tones hidden under the loud one, in Hz.
#:
#: Chosen to sit well away from every multiple of LOUD. 700 Hz was tried first
#: and is 20 Hz from 4x180 -- close enough that leakage around the loud tone
#: registered as "found", making a depth-1 pass look successful when it had
#: found nothing of the sort. A masking test needs the maskee to be
#: unmistakably its own tone.
HIDDEN = (631.0, 1013.0)
LOUD = 180.0


@pytest.fixture(scope="module")
def analyser():
    return RecursiveAnalyser(build_pipeline())


@pytest.fixture(scope="module")
def masked():
    """One loud tone with two much quieter ones underneath.

    The amplitudes are the point: 0.02 and 0.006 against 0.8 is roughly -32 dB
    and -42 dB, far enough down that a single pass will not pick them out from
    under the loud partial's skirts.
    """
    t = np.arange(int(SR * 1.2)) / SR
    signal = 0.8 * np.sin(2 * np.pi * LOUD * t)
    signal += 0.02 * np.sin(2 * np.pi * HIDDEN[0] * t)
    signal += 0.006 * np.sin(2 * np.pi * HIDDEN[1] * t)
    return signal.astype(np.float32)


def _pitched(result):
    return [t for t in result.tracks if t.kind is not TrackKind.RESIDUAL]


def _energy_at(signal, hz, bins=3):
    """Magnitude near `hz` in a windowed spectrum of `signal`."""
    windowed = np.asarray(signal, dtype=np.float64) * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    centre = int(round(hz * len(signal) / SR))
    lo = max(0, centre - bins)
    return float(spectrum[lo : centre + bins + 1].max())


def _finds(result, hz, tolerance_hz=25.0):
    """Whether any track's partials centre near `hz`."""
    for track in _pitched(result):
        for partial in track.partials:
            freqs = [f.frequency_hz for f in partial.frames]
            if freqs and abs(float(np.median(freqs)) - hz) <= tolerance_hz:
                return True
    return False


# =============================================================================
# Claim 1: deeper passes find more
# =============================================================================


class TestItFindsMore:
    def test_one_pass_misses_the_quiet_tones(self, analyser, masked):
        """The premise. If a single pass already found them, this whole
        feature would be solving a problem that does not exist."""
        result, _ = analyser.analyse(masked, AnalysisParams(recursion_depth=1))
        assert _finds(result, LOUD), "the loud tone should be found immediately"
        assert not _finds(result, HIDDEN[0]), "quiet tone should be masked at depth 1"

    def test_a_second_pass_uncovers_what_was_masked(self, analyser, masked):
        """Subtract the loud tone and the quiet one becomes findable."""
        result, _ = analyser.analyse(masked, AnalysisParams(recursion_depth=3))
        assert _finds(result, HIDDEN[0]), f"{HIDDEN[0]} Hz should surface by depth 3"

    def test_more_depth_never_yields_fewer_tracks(self, analyser, masked):
        """Monotonic by construction: each pass only adds to the kept set."""
        counts = [
            len(_pitched(analyser.analyse(masked, AnalysisParams(recursion_depth=d))[0]))
            for d in (1, 2, 3)
        ]
        assert counts == sorted(counts), counts
        assert counts[-1] > counts[0], "depth should actually buy something"

    def test_each_pass_is_reported(self, analyser, masked):
        _, levels = analyser.analyse(masked, AnalysisParams(recursion_depth=3))
        assert [lv.level for lv in levels] == [1, 2, 3]
        assert all(lv.kept for lv in levels)

    def test_tracks_carry_the_pass_that_found_them(self, analyser, masked):
        """The level must be on the track, not only in its name.

        Setting the label alone was a real bug: the API reported every track as
        level 1 while its own name said "L3".
        """
        result, _ = analyser.analyse(masked, AnalysisParams(recursion_depth=3))
        levels = {t.level for t in _pitched(result)}
        assert len(levels) > 1, "several passes contributed, so levels must differ"
        for track in _pitched(result):
            assert f"L{track.level}" in track.label


# =============================================================================
# Claim 2: the arithmetic survives
# =============================================================================


class TestTheInvariantComposes:
    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_tracks_still_sum_to_the_input(self, analyser, masked, depth):
        """The central promise, at every depth.

        Asserted against the input array rather than against a recomputed
        expectation, so it cannot pass by sharing a bug with the code.
        """
        result, _ = analyser.analyse(masked, AnalysisParams(recursion_depth=depth))
        summed = np.zeros(masked.shape[0], dtype=np.float64)
        for track in result.tracks:
            if track.audio is not None:
                summed += track.audio
        assert np.max(np.abs(summed - masked)) < 1e-4

    @pytest.mark.parametrize("depth", [2, 3])
    def test_exactly_one_residual_survives(self, analyser, masked, depth):
        """Intermediate residuals are superseded and must not be kept.

        Keeping them would double-count: residual_1 already contains everything
        residual_2 does, so shipping both makes the sum wrong by a whole layer.
        """
        result, _ = analyser.analyse(masked, AnalysisParams(recursion_depth=depth))
        residuals = [t for t in result.tracks if t.kind is TrackKind.RESIDUAL]
        assert len(residuals) == 1

    @pytest.mark.parametrize("depth", [2, 3, 4])
    def test_track_ids_are_unique_across_passes(self, analyser, masked, depth):
        """Each pass numbers from zero, so combining must renumber.

        A collision here is silent and severe: mute, solo, per-track export and
        the record-out selection all address tracks by id.
        """
        result, _ = analyser.analyse(masked, AnalysisParams(recursion_depth=depth))
        ids = [t.track_id for t in result.tracks]
        assert len(ids) == len(set(ids)), ids
        assert ids == sorted(ids), "ids should stay in display order"


# =============================================================================
# Stripping layers
# =============================================================================


class TestStripping:
    def test_stripping_removes_the_loud_layer(self, analyser, masked):
        """The stripped signal must have lost the loud tone's energy.

        Measured spectrally on the reference rather than by asking whether any
        track still mentions that frequency. The distinction is real: pass 2
        routinely re-finds a faint *residue* of what pass 1 removed, because
        subtraction is exact in the time domain but the tracker's model of a
        tone never is. Demanding the frequency vanish entirely would be
        demanding something stripping does not claim to do. What it does claim
        is that the material is no longer dominant, and that is what is checked.
        """
        result, levels = analyser.analyse(
            masked, AnalysisParams(residual_skip=1, recursion_depth=1)
        )
        assert not levels[0].kept and levels[1].kept

        before = _energy_at(masked, LOUD)
        after = _energy_at(np.asarray(result.original), LOUD)
        assert after < before / 10, (
            f"the loud tone should be largely gone: {before:.1f} -> {after:.1f}"
        )

    def test_the_reference_moves_to_the_stripped_signal(self, analyser, masked):
        """The result must report what its tracks actually sum to.

        This is the mechanism that keeps everything downstream honest: the
        mixdown, the reconstruction check and the export manifest all read
        `original`, so moving it means none of them need to know recursion
        happened.
        """
        result, _ = analyser.analyse(
            masked, AnalysisParams(residual_skip=1, recursion_depth=1)
        )
        reference = np.asarray(result.original)
        assert np.max(np.abs(reference - masked)) > 0.1, "reference should have changed"

        summed = np.zeros(reference.shape[0], dtype=np.float64)
        for track in result.tracks:
            if track.audio is not None:
                summed += track.audio
        assert np.max(np.abs(summed - reference)) < 1e-4

    def test_stripping_more_strips_more(self, analyser, masked):
        """Each stripped layer takes more of the signal away."""
        distances = []
        for skip in (1, 2):
            result, _ = analyser.analyse(
                masked, AnalysisParams(residual_skip=skip, recursion_depth=1)
            )
            distances.append(float(np.max(np.abs(np.asarray(result.original) - masked))))
        assert distances[1] >= distances[0]

    def test_stripped_passes_still_count_in_the_level_numbers(self, analyser, masked):
        """"L2" must mean the second pass of the run, not the second kept one.

        Someone comparing `--strip 1 --depth 1` against `--depth 2` needs the
        numbering to line up, or the two are impossible to reason about.
        """
        result, levels = analyser.analyse(
            masked, AnalysisParams(residual_skip=1, recursion_depth=1)
        )
        assert [lv.level for lv in levels] == [1, 2]
        assert all(t.level == 2 for t in _pitched(result))


# =============================================================================
# Stopping
# =============================================================================


class TestStopping:
    def test_silence_stops_immediately(self, analyser):
        """Nothing to find, so no pass after the first can do anything."""
        quiet = np.zeros(int(SR * 0.5), dtype=np.float32)
        result, levels = analyser.analyse(quiet, AnalysisParams(recursion_depth=5))
        assert len(levels) < 5
        assert result.reconstruction_error() < 1e-6

    def test_the_stop_is_provable_not_a_yield_heuristic(self, analyser, masked):
        """Regression: a low-yield pass must NOT end the run.

        The first version stopped once a pass explained less than a millionth
        of the incoming energy. On this fixture pass 2 explains ~0% — and
        passes 3 and 4 then found six more tracks. Stopping on a small yield
        threw away exactly the quiet material recursion exists to reach.
        """
        _, levels = analyser.analyse(masked, AnalysisParams(recursion_depth=4))
        assert len(levels) == 4, "a low-yield pass must not stop the run"
        assert any(lv.energy_share < 1e-6 for lv in levels), (
            "this fixture is supposed to contain a near-zero-yield pass; "
            "if it no longer does, the regression is no longer being tested"
        )

    def test_exhausted_is_bit_equality(self):
        a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert _exhausted(a.copy(), a)
        assert not _exhausted(a * 0.5, a)
        # Different lengths are not "unchanged", and must not raise either.
        assert not _exhausted(np.zeros(2, dtype=np.float32), a)


# =============================================================================
# Details
# =============================================================================


class TestLabelling:
    def test_a_single_pass_gets_no_level_suffix(self):
        """Every row saying "L1" on an ordinary analysis is noise, not data."""
        assert _level_label("A4", 1, 1, 0) == "A4"

    def test_several_passes_are_marked(self):
        assert _level_label("A4", 2, 3, 0) == "A4 · L2"

    def test_stripping_marks_even_a_single_kept_pass(self):
        """With layers stripped, "L2" is the only clue that pass 1 is missing."""
        assert _level_label("A4", 2, 2, 1) == "A4 · L2"


class TestReportShape:
    def test_report_serialises_for_the_api(self):
        report = LevelReport(2, 5, 0.1234567, kept=True)
        assert report.as_dict() == {
            "level": 2,
            "tracks_found": 5,
            "energy_share": 0.123457,
            "kept": True,
        }

    def test_repr_is_readable(self):
        assert "L2" in repr(LevelReport(2, 5, 0.5, kept=False))
        assert "discarded" in repr(LevelReport(2, 5, 0.5, kept=False))


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs", [{"recursion_depth": 0}, {"recursion_depth": -2}, {"residual_skip": -1}]
    )
    def test_impossible_settings_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            AnalysisParams(**kwargs)

    def test_defaults_are_an_ordinary_single_analysis(self):
        """Nobody who has not asked for recursion should pay for it."""
        params = AnalysisParams()
        assert params.recursion_depth == 1
        assert params.residual_skip == 0

    def test_track_rejects_a_level_below_one(self):
        from backend.core.types import Track

        track = Track(0, TrackKind.PARTIAL, "A4")
        with pytest.raises(ValueError):
            track.set_level(0)


class TestProgress:
    def test_progress_spans_the_whole_run_once(self, analyser, masked):
        """A depth-3 run must not fill the bar three times."""
        seen: list[float] = []
        analyser.analyse(
            masked,
            AnalysisParams(recursion_depth=3),
            on_progress=lambda f, _s: seen.append(f),
        )
        assert seen, "progress should be reported"
        # Tolerance, not equality: the fractions are computed per pass and
        # rescaled, so two consecutive values can differ by one ulp in the
        # wrong direction. A bar that jumps back by 1e-16 is not a bug; one
        # that jumps back by a whole pass is.
        for earlier, later in zip(seen, seen[1:]):
            assert later >= earlier - 1e-9, f"progress went backwards: {earlier} -> {later}"
        assert max(seen) <= 1.0
        assert min(seen) >= 0.0

    def test_progress_names_the_pass(self, analyser, masked):
        stages: list[str] = []
        analyser.analyse(
            masked,
            AnalysisParams(recursion_depth=2),
            on_progress=lambda _f, s: stages.append(s),
        )
        assert any("pass 2 of 2" in s for s in stages)
