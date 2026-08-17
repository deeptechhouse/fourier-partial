"""Command-line access to the analyser, one file or a whole directory.

WHY A CLI AT ALL
----------------
The browser interface exists to let you *listen* — mute a track, solo another,
decide by ear what the sound is made of. That is the right shape for one file
you are curious about, and the wrong shape entirely for sixty field recordings
you already know you want decomposed. Nobody drags sixty files into a browser
one at a time.

So this is deliberately not a second interface. It is the same analysis, the
same exporters, and the same settings, reached without a person in the loop.
Every module it uses is the one the server uses:

    backend.services.build_pipeline   the analyser
    backend.analysis.recursive        the passes
    backend.export.archive            the exporters, manifests and all
    backend.api.settings_schema       the settable names and their bounds
    backend.api.presets               the same saved presets as the GUI

Nothing here re-implements analysis or export. A bug fixed in one place is
fixed in both, and a preset saved in the browser works on the command line.

USAGE
-----
    python3 -m backend.cli one  INPUT.wav        [options]
    python3 -m backend.cli many INPUT_DIR/       [options]
    python3 -m backend.cli settings              # what can be tuned
    python3 -m backend.cli presets               # what is saved

    -o, --output DIR       where to write (default: alongside the input)
    -f, --format FMT       repeatable: wav, wav-tones, wav-mix, mid, mid-single
        --depth N          analysis passes; each finds what the last could not
        --strip N          passes to run and discard first, keeping the residual
        --preset NAME      start from a saved preset
    -s, --set KEY=VALUE    repeatable: override any single setting
    -j, --jobs N           files to process at once (many)
    -r, --recursive        descend into subdirectories (many)
    -n, --dry-run          say what would happen, write nothing
    -q, --quiet            errors only

EXIT CODES
----------
    0  everything succeeded
    1  every file failed
    2  some files succeeded and some failed (batch only)
    3  bad usage — nothing was attempted

The distinction between 1 and 2 matters in a shell loop: "nothing worked" and
"most of it worked" call for different reactions, and collapsing them into a
single non-zero code makes a batch run over a mixed directory unscriptable.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from backend.analysis.recursive import RecursiveAnalyser
from backend.api.presets import PresetStore
from backend.api.settings_schema import SETTABLE, clamp, schema
from backend.config import settings as app_settings
from backend.core.errors import FourierPartialError
from backend.core.types import AnalysisParams, TrackKind
from backend.export.archive import ArchiveFormat, build_archive
from backend.services import build_loader, build_pipeline

logger = logging.getLogger("fourier_partial.cli")

#: Extensions we will attempt in a batch. The loader sniffs the actual bytes
#: and will reject a mislabelled file, so this only decides what to *try* —
#: it stops a directory sweep from feeding it every .txt and .DS_Store.
AUDIO_SUFFIXES: frozenset[str] = frozenset(
    {".wav", ".wave", ".aif", ".aiff", ".aifc", ".flac", ".ogg", ".oga", ".mp3", ".m4a"}
)

#: ADJUSTABLE: refuse audio longer than this, in seconds. Matches the server.
MAX_DURATION_S: float = 600.0


# =============================================================================
# Result of processing one file
# =============================================================================


@dataclass(slots=True)
class FileOutcome:
    """What happened to one input file."""

    source: Path
    ok: bool
    written: list[Path]
    tracks: int
    levels: list[dict[str, object]]
    seconds: float
    error: str = ""

    def line(self) -> str:
        """One line for the console."""
        if not self.ok:
            return f"FAILED  {self.source.name}: {self.error}"
        passes = len(self.levels)
        detail = f"{self.tracks} tracks"
        if passes > 1:
            kept = sum(1 for lv in self.levels if lv["kept"])
            detail += f" over {kept} of {passes} passes"
        return f"ok      {self.source.name}  ->  {detail}, {self.seconds:.1f}s"


# =============================================================================
# Settings assembly
# =============================================================================


def resolve_params(
    preset: str | None, overrides: Sequence[str], depth: int | None, strip: int | None
) -> AnalysisParams:
    """Build the analysis settings from a preset, --set pairs, and the flags.

    Precedence, loosest to tightest: defaults, then the preset, then --set,
    then --depth/--strip. The explicit flag wins over a preset that also
    carries a depth, because someone who typed `--depth 3` on this invocation
    means it more than a preset they saved last week.

    Raises:
        ValueError: on an unknown setting name or unparseable value. Raised
            rather than exiting so tests can drive this without a subprocess.
    """
    values: dict[str, float] = {}

    if preset:
        store = PresetStore(app_settings.presets_file)
        saved = store.get(preset)
        if saved is None:
            known = ", ".join(p.name for p in store.all())
            raise ValueError(f"no preset named {preset!r}. Available: {known}")
        values.update(saved.settings)

    for pair in overrides:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in SETTABLE:
            near = ", ".join(sorted(n for n in SETTABLE if n.startswith(key[:4])))
            hint = f" Did you mean: {near}?" if near else ""
            raise ValueError(f"{key!r} is not a setting.{hint} Try `settings`.")
        try:
            values[key] = float(raw)
        except ValueError:
            raise ValueError(f"{key} needs a number, got {raw!r}") from None

    if depth is not None:
        values["recursion_depth"] = depth
    if strip is not None:
        values["residual_skip"] = strip

    # Clamped, not rejected. A CLI is often driven by a script sweeping a
    # range, and failing the whole run because one step overshot a bound is
    # less useful than doing the nearest legal thing and saying so.
    cleaned: dict[str, float] = {}
    for key, value in values.items():
        limited = clamp(key, value)
        if limited != value:
            logger.warning("%s clamped from %g to %g", key, value, limited)
        cleaned[key] = limited

    integral = {
        "window_size", "hop_size", "zero_pad_factor", "max_peaks_per_frame",
        "max_gap_frames", "max_tracks", "sample_rate", "recursion_depth",
        "residual_skip",
    }
    typed: dict[str, object] = {
        k: int(v) if k in integral else float(v) for k, v in cleaned.items()
    }
    return AnalysisParams(**typed)  # type: ignore[arg-type]


# =============================================================================
# Processing
# =============================================================================


def process_file(
    source: Path,
    out_dir: Path,
    formats: Sequence[ArchiveFormat],
    params: AnalysisParams,
    tones_only: bool = False,
) -> FileOutcome:
    """Analyse one file and write every requested format.

    Self-contained on purpose: it opens the file, analyses, exports, and
    returns a plain summary. That makes it safe to hand to a worker process,
    which is what `--jobs` does — nothing crosses the process boundary except
    this small result.
    """
    started = time.monotonic()
    try:
        loader = build_loader(MAX_DURATION_S)
        samples, sample_rate = loader.load(source, params.sample_rate)

        analyser = RecursiveAnalyser(build_pipeline())
        result, levels = analyser.analyse(samples, params)

        only: set[int] | None = None
        if tones_only:
            only = {
                t.track_id for t in result.tracks if t.kind is not TrackKind.RESIDUAL
            }

        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        # One scratch directory for the whole file, removed however we leave.
        scratch = Path(tempfile.mkdtemp(prefix="fp-cli-"))
        try:
            for fmt in formats:
                destination = out_dir / f"{source.stem}_{fmt.value}{fmt.extension}"
                build_archive(result, fmt, destination, scratch / fmt.value, only)
                written.append(destination)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        return FileOutcome(
            source=source,
            ok=True,
            written=written,
            tracks=len(result.tracks),
            levels=[lv.as_dict() for lv in levels],
            seconds=time.monotonic() - started,
        )
    except FourierPartialError as exc:
        return FileOutcome(source, False, [], 0, [], time.monotonic() - started,
                           exc.user_message)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
        logger.debug("unexpected failure on %s", source, exc_info=True)
        return FileOutcome(source, False, [], 0, [], time.monotonic() - started, str(exc))


def find_audio(directory: Path, recursive: bool) -> list[Path]:
    """Every plausible audio file in `directory`, sorted for reproducibility.

    Sorted because a batch that processes files in filesystem order produces a
    different log every run, which makes two runs impossible to diff.
    """
    walker: Iterable[Path] = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        p for p in walker if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )


# =============================================================================
# Commands
# =============================================================================


def cmd_settings(args: argparse.Namespace) -> int:
    """Print what can be tuned, from the same schema the browser panel uses."""
    data = schema()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    defaults = AnalysisParams()
    for group in data["groups"]:  # type: ignore[index]
        print(f"\n{group['label']}")
        print(f"  {group['blurb']}")
        for setting in group["settings"]:
            name = setting["name"]
            current = getattr(defaults, name)
            flag = " (advanced)" if setting.get("advanced") else ""
            print(f"\n  {name}{flag}")
            print(f"    {setting['label']}: {setting['help']}")
            print(
                f"    range {setting['min']}..{setting['max']}"
                f"  step {setting['step']}  default {current}"
            )
    print()
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    """List the saved presets — the same ones the browser offers."""
    store = PresetStore(app_settings.presets_file)
    entries = store.all()
    if args.json:
        print(json.dumps([p.as_dict() for p in entries], indent=2))
        return 0
    if not entries:
        print("No presets saved.")
        return 0
    for preset in entries:
        origin = "built-in" if preset.builtin else "yours"
        summary = (
            ", ".join(f"{k}={v}" for k, v in sorted(preset.settings.items())) or "defaults"
        )
        print(f"  {preset.name}  ({origin})")
        print(f"      {summary}")
    print(f"\nYour own presets live in {store.path}")
    return 0


def cmd_one(args: argparse.Namespace) -> int:
    """Analyse a single file."""
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        print(f"Not a file: {source}", file=sys.stderr)
        return 3

    params = resolve_params(args.preset, args.set, args.depth, args.strip)
    formats = [ArchiveFormat(f) for f in args.format]
    out_dir = Path(args.output).expanduser() if args.output else source.parent

    if args.dry_run:
        _describe(source, out_dir, formats, params)
        return 0

    outcome = process_file(source, out_dir, formats, params, args.tones_only)
    _report([outcome], args)
    return 0 if outcome.ok else 1


def cmd_many(args: argparse.Namespace) -> int:
    """Analyse every audio file in a directory."""
    directory = Path(args.input).expanduser().resolve()
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 3

    sources = find_audio(directory, args.recursive)
    if not sources:
        where = "or below " if args.recursive else ""
        print(f"No audio files in {where}{directory}", file=sys.stderr)
        return 3

    params = resolve_params(args.preset, args.set, args.depth, args.strip)
    formats = [ArchiveFormat(f) for f in args.format]
    out_root = Path(args.output).expanduser() if args.output else directory / "decomposed"

    if args.dry_run:
        print(f"Would process {len(sources)} file(s) into {out_root}:")
        for source in sources:
            print(f"  {source.relative_to(directory)}")
        _describe(None, out_root, formats, params)
        return 0

    # --json means JSON on stdout and nothing else. Progress lines interleaved
    # with the payload make the output unparseable, which defeats the flag.
    chatty = not args.quiet and not args.json
    if chatty:
        print(f"Processing {len(sources)} file(s) -> {out_root}")
        if params.recursion_depth > 1 or params.residual_skip:
            total = params.recursion_depth + params.residual_skip
            print(
                f"  {total} passes per file "
                f"({params.residual_skip} stripped, {params.recursion_depth} kept) "
                f"— roughly {total}x the usual time"
            )

    def destination_for(source: Path) -> Path:
        # Mirror the input tree so a recursive sweep does not collapse
        # a/kick.wav and b/kick.wav onto the same output name.
        relative = source.relative_to(directory).parent
        return out_root / relative

    outcomes: list[FileOutcome] = []
    if args.jobs > 1:
        # Processes rather than threads: analysis is numpy-heavy Python, and
        # threads would serialise on the GIL for most of it. Memory is the real
        # constraint — each worker holds a whole analysis — so this is opt-in
        # and never the default.
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    process_file, s, destination_for(s), formats, params, args.tones_only
                ): s
                for s in sources
            }
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                if chatty:
                    print(f"  {outcome.line()}", flush=True)
    else:
        for source in sources:
            outcome = process_file(
                source, destination_for(source), formats, params, args.tones_only
            )
            outcomes.append(outcome)
            if chatty:
                print(f"  {outcome.line()}", flush=True)

    outcomes.sort(key=lambda o: str(o.source))
    _report(outcomes, args, batch=True)

    failed = [o for o in outcomes if not o.ok]
    if not failed:
        return 0
    return 1 if len(failed) == len(outcomes) else 2


# =============================================================================
# Output
# =============================================================================


def _describe(
    source: Path | None,
    out_dir: Path,
    formats: Sequence[ArchiveFormat],
    params: AnalysisParams,
) -> None:
    """Explain a dry run."""
    if source is not None:
        print(f"Would analyse {source}")
    print(f"  output    {out_dir}")
    print(f"  formats   {', '.join(f.value for f in formats)}")
    print(f"  window    {params.window_size} samples, hop {params.hop_size}")
    total = params.recursion_depth + params.residual_skip
    if total > 1:
        print(
            f"  passes    {total} "
            f"({params.residual_skip} stripped, {params.recursion_depth} kept)"
        )
        if params.residual_skip:
            print(
                "            note: stripped passes are discarded, so the exported "
                "tracks sum to what was left, not to the input file"
            )
    else:
        print("  passes    1")


def _report(outcomes: Sequence[FileOutcome], args: argparse.Namespace, batch: bool = False) -> None:
    """Print the outcome, as JSON or for a person."""
    if args.json:
        print(json.dumps(
            [
                {
                    "source": str(o.source),
                    "ok": o.ok,
                    "written": [str(p) for p in o.written],
                    "tracks": o.tracks,
                    "levels": o.levels,
                    "seconds": round(o.seconds, 3),
                    "error": o.error,
                }
                for o in outcomes
            ],
            indent=2,
        ))
        return

    if args.quiet:
        for outcome in outcomes:
            if not outcome.ok:
                print(outcome.line(), file=sys.stderr)
        return

    if not batch:
        outcome = outcomes[0]
        print(outcome.line())
        for path in outcome.written:
            print(f"  wrote {path}")
        for level in outcome.levels:
            state = "kept" if level["kept"] else "discarded"
            print(
                f"  pass {level['level']}: {level['tracks_found']} tracks, "
                f"{float(level['energy_share']):.1%} of energy, {state}"
            )
        if outcome.error:
            print(f"  {outcome.error}", file=sys.stderr)
        return

    ok = [o for o in outcomes if o.ok]
    failed = [o for o in outcomes if not o.ok]
    total_files = sum(len(o.written) for o in ok)
    seconds = sum(o.seconds for o in outcomes)
    print(
        f"\n{len(ok)} of {len(outcomes)} file(s) succeeded, "
        f"{total_files} archive(s) written, {seconds:.1f}s total"
    )
    for outcome in failed:
        print(f"  {outcome.line()}", file=sys.stderr)


# =============================================================================
# Argument parsing
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface."""
    parser = argparse.ArgumentParser(
        prog="fourier_partial",
        description="Break audio into individually-audible pitched tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s one take.wav -f wav -f mid\n"
            "  %(prog)s many ./recordings -o ./out --depth 3 -j 4\n"
            "  %(prog)s one cymbal.wav --strip 1 --depth 2   # what is under the loud part\n"
            "  %(prog)s one bell.wav --preset 'Bells and struck metal'\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("input", help="audio file, or directory for `many`")
        p.add_argument("-o", "--output", help="where to write (default: beside the input)")
        p.add_argument(
            "-f", "--format", action="append", default=None,
            choices=[f.value for f in ArchiveFormat],
            help="repeatable; default wav",
        )
        p.add_argument(
            "--depth", type=int, default=None,
            help="analysis passes; each works on what the last could not explain",
        )
        p.add_argument(
            "--strip", type=int, default=None,
            help="passes to run and DISCARD first, keeping only their residual",
        )
        p.add_argument("--preset", default=None, help="start from a saved preset")
        p.add_argument(
            "-s", "--set", action="append", default=[], metavar="KEY=VALUE",
            help="override one setting; repeatable. See `settings`.",
        )
        p.add_argument(
            "--tones-only", action="store_true",
            help="leave the leftover/residual track out of the export",
        )
        p.add_argument("-n", "--dry-run", action="store_true", help="write nothing")
        p.add_argument("-q", "--quiet", action="store_true", help="errors only")
        p.add_argument("--json", action="store_true", help="machine-readable output")

    one = sub.add_parser("one", help="analyse a single file")
    shared(one)
    one.set_defaults(func=cmd_one)

    many = sub.add_parser("many", help="analyse every audio file in a directory")
    shared(many)
    many.add_argument("-r", "--recursive", action="store_true", help="descend into subdirectories")
    many.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="files at once. Each worker holds a whole analysis in memory, so "
             "raise this only if you have the RAM for it.",
    )
    many.set_defaults(func=cmd_many)

    for name, handler, help_text in (
        ("settings", cmd_settings, "list every tunable setting"),
        ("presets", cmd_presets, "list saved presets"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--json", action="store_true", help="machine-readable output")
        p.set_defaults(func=handler)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the exit code rather than calling sys.exit."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.ERROR if getattr(args, "quiet", False) else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Defaulted here rather than in add_argument: argparse's `append` action
    # extends the default list instead of replacing it, so a default of ["wav"]
    # would make every -f produce "wav" plus the choice.
    if getattr(args, "format", None) is None:
        args.format = ["wav"]

    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
