"""The command line, driven in-process.

WHAT MATTERS HERE
-----------------
The CLI's job is to be *scriptable*. That makes two things load-bearing that a
browser interface never has to care about:

  - EXIT CODES. A shell loop reacts differently to "nothing worked" and "most
    of it worked", so total failure (1) and partial failure (2) must stay
    distinguishable, and neither may be confused with bad usage (3).
  - NOT STOPPING. One unreadable file in a directory of sixty must not end the
    run. A batch tool that aborts on the first bad input is a batch tool
    nobody can leave unattended.

Everything else the CLI does -- analysis, export, manifests -- is the server's
code reached from a different door, and is tested where it lives. What is
tested here is the door: argument handling, file discovery, and reporting.

`main()` returns its exit code rather than calling sys.exit, so these run
in-process. Subprocess tests would be slower and would hide tracebacks.
"""

from __future__ import annotations

import io
import json
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.cli import (
    AUDIO_SUFFIXES,
    find_audio,
    main,
    resolve_params,
)
from backend.core.types import AnalysisParams

SR = 44_100


def write_tone(path: Path, freqs=(220.0, 330.0), duration_s: float = 0.7) -> Path:
    """A short chord, long enough to produce several tracks."""
    t = np.arange(int(SR * duration_s)) / SR
    signal = sum(0.4 / (i + 1) * np.sin(2 * np.pi * f * t) for i, f in enumerate(freqs))
    envelope = np.minimum(1.0, np.minimum(t * 40, (t[-1] - t) * 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (signal * envelope).astype(np.float32), SR, subtype="FLOAT")
    return path


def run(*argv: str) -> tuple[int, str]:
    """Invoke the CLI, capturing stdout."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(list(argv))
    return code, buffer.getvalue()


@pytest.fixture
def library(tmp_path):
    """A small directory of audio, one nested file, and a decoy."""
    write_tone(tmp_path / "chord.wav")
    write_tone(tmp_path / "second.wav", freqs=(261.0, 392.0))
    write_tone(tmp_path / "nested" / "deep.wav", freqs=(147.0,))
    (tmp_path / "notes.txt").write_text("not audio")
    return tmp_path


# =============================================================================
# Settings assembly
# =============================================================================


class TestResolveParams:
    def test_nothing_given_is_the_default_analysis(self):
        assert resolve_params(None, [], None, None) == AnalysisParams()

    def test_a_setting_can_be_overridden(self):
        params = resolve_params(None, ["window_size=8192"], None, None)
        assert params.window_size == 8192

    def test_integral_settings_do_not_arrive_as_floats(self):
        """`window_size=8192` parses as a float and must not stay one.

        A float window size propagates into array shapes and FFT lengths, where
        it becomes a TypeError several layers away from the flag that caused it.
        """
        params = resolve_params(None, ["window_size=8192", "max_gap_frames=4"], None, None)
        assert isinstance(params.window_size, int)
        assert isinstance(params.max_gap_frames, int)

    def test_flags_beat_a_preset(self):
        """`--depth 3` typed now outranks a depth saved in a preset last week."""
        params = resolve_params(None, ["recursion_depth=2"], 3, None)
        assert params.recursion_depth == 3

    def test_out_of_range_values_are_clamped_not_rejected(self):
        """A parameter sweep that overshoots should keep running.

        Failing the whole batch because step 12 of a loop asked for depth 99 is
        less useful than doing the nearest legal thing.
        """
        params = resolve_params(None, [], 999, None)
        assert params.recursion_depth == 8

    def test_an_unknown_setting_is_refused_with_a_suggestion(self):
        with pytest.raises(ValueError, match="window_size"):
            resolve_params(None, ["windo_size=4096"], None, None)

    def test_a_malformed_pair_is_refused(self):
        with pytest.raises(ValueError, match="KEY=VALUE"):
            resolve_params(None, ["window_size"], None, None)

    def test_a_non_numeric_value_is_refused(self):
        with pytest.raises(ValueError, match="number"):
            resolve_params(None, ["window_size=big"], None, None)

    def test_an_unknown_preset_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="Sharp attacks"):
            resolve_params("no such preset", [], None, None)

    def test_a_builtin_preset_applies(self):
        """The same presets the browser offers, by name."""
        params = resolve_params("Bells and struck metal", [], None, None)
        assert params.window_size == 8192


# =============================================================================
# Finding files
# =============================================================================


class TestDiscovery:
    def test_only_audio_is_picked_up(self, library):
        found = find_audio(library, recursive=False)
        assert [p.name for p in found] == ["chord.wav", "second.wav"]

    def test_recursive_descends(self, library):
        found = find_audio(library, recursive=True)
        assert "deep.wav" in {p.name for p in found}

    def test_results_are_sorted(self, library):
        """Filesystem order differs between runs, which makes two logs
        impossible to diff. Sorting makes a batch reproducible."""
        found = find_audio(library, recursive=True)
        assert [str(p) for p in found] == sorted(str(p) for p in found)

    def test_common_formats_are_all_attempted(self):
        for suffix in (".wav", ".flac", ".aiff", ".mp3", ".ogg", ".m4a"):
            assert suffix in AUDIO_SUFFIXES


# =============================================================================
# One file
# =============================================================================


class TestOne:
    def test_it_writes_the_requested_format(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        code, _ = run("one", str(source), "-o", str(out), "-f", "wav")
        assert code == 0
        assert (out / "in_wav.zip").is_file()

    def test_several_formats_at_once(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        code, _ = run("one", str(source), "-o", str(out), "-f", "wav", "-f", "mid")
        assert code == 0
        assert {p.name for p in out.iterdir()} == {"in_wav.zip", "in_mid.zip"}

    def test_the_exported_parts_reconstruct_the_input(self, tmp_path):
        """The promise, end to end through the CLI rather than the library.

        Compared against the file on disk, so a bug anywhere between reading
        the input and writing the archive shows up here.
        """
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        assert run("one", str(source), "-o", str(out), "-f", "wav")[0] == 0

        with zipfile.ZipFile(out / "in_wav.zip") as archive:
            names = [n for n in archive.namelist() if n.endswith(".wav")]
            total = None
            for name in names:
                data, _ = sf.read(io.BytesIO(archive.read(name)), dtype="float64")
                total = data if total is None else total + data
        original, _ = sf.read(source, dtype="float64")
        assert np.max(np.abs(total - original)) < 1e-4

    def test_depth_reaches_the_analyser(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        code, text = run("one", str(source), "-o", str(out), "--depth", "2", "--json")
        assert code == 0
        levels = json.loads(text)[0]["levels"]
        assert len(levels) >= 1
        assert all(lv["kept"] for lv in levels)

    def test_stripping_discards_the_first_pass(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        code, text = run(
            "one", str(source), "-o", str(out), "--strip", "1", "--depth", "1", "--json"
        )
        assert code == 0
        levels = json.loads(text)[0]["levels"]
        assert levels[0]["kept"] is False
        assert levels[1]["kept"] is True

    def test_the_manifest_admits_when_it_cannot_reconstruct(self, tmp_path):
        """Stripping breaks the sum-to-original property, so the archive that
        travels to someone else has to say so."""
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        run("one", str(source), "-o", str(out), "-f", "wav", "--strip", "1")
        with zipfile.ZipFile(out / "in_wav.zip") as archive:
            manifest = archive.read("MANIFEST.txt").decode()
        assert "DO NOT SUM BACK" in manifest

    def test_an_ordinary_export_still_claims_reconstruction(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        run("one", str(source), "-o", str(out), "-f", "wav", "--depth", "2")
        with zipfile.ZipFile(out / "in_wav.zip") as archive:
            manifest = archive.read("MANIFEST.txt").decode()
        assert "SUM BACK TO THE ORIGINAL" in manifest
        assert "DO NOT SUM BACK" not in manifest

    def test_tones_only_leaves_the_residual_out(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        run("one", str(source), "-o", str(out), "-f", "wav", "--tones-only")
        with zipfile.ZipFile(out / "in_wav.zip") as archive:
            names = " ".join(archive.namelist()).lower()
        assert "residual" not in names

    def test_a_dry_run_writes_nothing(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        code, text = run("one", str(source), "-o", str(out), "-n")
        assert code == 0
        assert not out.exists()
        assert "Would analyse" in text

    def test_a_dry_run_warns_that_stripping_breaks_the_sum(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        _, text = run("one", str(source), "-n", "--strip", "2")
        assert "sum to what was left" in text

    def test_output_defaults_to_beside_the_input(self, tmp_path):
        source = write_tone(tmp_path / "here" / "in.wav")
        assert run("one", str(source), "-f", "wav")[0] == 0
        assert (tmp_path / "here" / "in_wav.zip").is_file()


# =============================================================================
# A directory
# =============================================================================


class TestMany:
    def test_every_file_is_processed(self, library, tmp_path):
        out = tmp_path / "out"
        code, _ = run("many", str(library), "-o", str(out), "-f", "wav")
        assert code == 0
        assert {p.name for p in out.glob("*.zip")} == {"chord_wav.zip", "second_wav.zip"}

    def test_recursive_mirrors_the_input_tree(self, library, tmp_path):
        """Two files called kick.wav in different folders must not collide."""
        out = tmp_path / "out"
        assert run("many", str(library), "-o", str(out), "-r", "-f", "wav")[0] == 0
        assert (out / "nested" / "deep_wav.zip").is_file()

    def test_a_bad_file_does_not_stop_the_others(self, library, tmp_path):
        """The reason a batch tool exists: one broken input among sixty."""
        (library / "broken.wav").write_bytes(b"RIFFxxxxWAVEnonsense")
        out = tmp_path / "out"
        code, _ = run("many", str(library), "-o", str(out), "-f", "wav")
        assert code == 2, "partial failure has its own exit code"
        assert (out / "chord_wav.zip").is_file(), "the good files must still be written"

    def test_all_bad_is_a_different_code_from_some_bad(self, tmp_path):
        """A shell loop reacts differently to these two, so they must differ."""
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "a.wav").write_bytes(b"RIFFxxxxWAVEnonsense")
        (bad / "b.wav").write_bytes(b"RIFFyyyyWAVEnonsense")
        code, _ = run("many", str(bad), "-o", str(tmp_path / "out"), "-q")
        assert code == 1

    def test_an_empty_directory_is_usage_not_failure(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert run("many", str(empty))[0] == 3

    def test_a_file_where_a_directory_belongs_is_usage(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        assert run("many", str(source))[0] == 3

    def test_json_output_is_machine_readable(self, library, tmp_path):
        code, text = run("many", str(library), "-o", str(tmp_path / "o"), "--json")
        assert code == 0
        payload = json.loads(text)
        assert len(payload) == 2
        assert all(entry["ok"] for entry in payload)
        assert all(entry["written"] for entry in payload)

    def test_parallel_produces_the_same_files(self, library, tmp_path):
        """`--jobs` must change only the timing.

        Worth pinning because the worker path pickles arguments across a
        process boundary, and something unpicklable there fails only under -j.
        """
        serial = tmp_path / "serial"
        parallel = tmp_path / "parallel"
        assert run("many", str(library), "-o", str(serial), "-f", "wav")[0] == 0
        assert run("many", str(library), "-o", str(parallel), "-f", "wav", "-j", "2")[0] == 0
        assert {p.name for p in serial.glob("*.zip")} == {
            p.name for p in parallel.glob("*.zip")
        }


# =============================================================================
# Informational commands
# =============================================================================


class TestInformational:
    def test_settings_lists_the_recursion_controls(self):
        code, text = run("settings")
        assert code == 0
        assert "recursion_depth" in text
        assert "residual_skip" in text

    def test_settings_as_json_matches_the_served_schema(self):
        """The CLI and the browser panel must describe the same settings.

        Same function, so this is really checking that neither grew its own
        copy -- which is exactly the drift the schema exists to prevent.
        """
        from backend.api.settings_schema import schema

        code, text = run("settings", "--json")
        assert code == 0
        assert json.loads(text) == schema()

    def test_presets_lists_the_builtins(self):
        code, text = run("presets")
        assert code == 0
        assert "Sharp attacks" in text

    def test_presets_as_json(self):
        code, text = run("presets", "--json")
        assert code == 0
        assert any(p["name"] == "Dig deep" for p in json.loads(text))


# =============================================================================
# Usage errors
# =============================================================================


class TestUsage:
    def test_a_missing_file_is_reported_not_crashed(self, tmp_path):
        assert run("one", str(tmp_path / "nope.wav"))[0] == 3

    def test_a_bad_setting_never_reaches_the_analyser(self, tmp_path):
        source = write_tone(tmp_path / "in.wav")
        assert run("one", str(source), "-s", "nonsense=1")[0] == 3

    def test_repeating_the_format_flag_does_not_keep_the_default(self, tmp_path):
        """argparse's `append` extends the default list rather than replacing
        it, so a default of ["wav"] would silently add a wav zip to every
        export the user asked to be MIDI only."""
        source = write_tone(tmp_path / "in.wav")
        out = tmp_path / "out"
        assert run("one", str(source), "-o", str(out), "-f", "mid")[0] == 0
        assert {p.name for p in out.iterdir()} == {"in_mid.zip"}


# =============================================================================
# The launcher script
# =============================================================================


class TestLauncher:
    """`./fourier-partial`, run as a real subprocess.

    Worth the cost of spawning a shell because the bug this catches is invisible
    to every in-process test: the launcher's job is to make `backend`
    importable, and the obvious way to do that -- `cd` to the project root --
    silently reinterprets every relative path the user typed. `fourier-partial
    one take.wav` then looks for take.wav inside the install directory. Nothing
    errors; it simply cannot find a file that is plainly there.
    """

    LAUNCHER = Path(__file__).resolve().parent.parent / "fourier-partial"

    def _run(self, cwd: Path, *args: str):
        import subprocess

        return subprocess.run(
            [str(self.LAUNCHER), *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_the_launcher_is_executable(self):
        import os

        assert self.LAUNCHER.is_file()
        assert os.access(self.LAUNCHER, os.X_OK), "chmod +x fourier-partial"

    def test_it_runs_from_an_unrelated_directory(self, tmp_path):
        result = self._run(tmp_path, "settings")
        assert result.returncode == 0, result.stderr
        assert "window_size" in result.stdout

    def test_a_relative_path_resolves_against_the_users_directory(self, tmp_path):
        """The regression. `one take.wav` must mean the caller's take.wav."""
        write_tone(tmp_path / "take.wav")
        result = self._run(tmp_path, "one", "take.wav", "-f", "wav")
        assert result.returncode == 0, result.stderr + result.stdout
        assert (tmp_path / "take_wav.zip").is_file()

    def test_a_relative_output_directory_lands_where_the_user_is(self, tmp_path):
        write_tone(tmp_path / "take.wav")
        result = self._run(tmp_path, "one", "./take.wav", "-o", "./out", "-f", "wav")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "out" / "take_wav.zip").is_file()

    def test_a_path_with_spaces_survives(self, tmp_path):
        """Field recordings live in folders with spaces in the name."""
        folder = tmp_path / "field recordings"
        write_tone(folder / "a take.wav")
        result = self._run(tmp_path, "many", "./field recordings", "-o", "./out")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "out" / "a take_wav.zip").is_file()
