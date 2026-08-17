"""Saved settings combinations.

Finding a good set of analysis settings for a kind of material takes several
passes, and the result is a combination — window plus threshold plus minimum
duration — that is genuinely hard to remember and worthless written on paper.
This stores them by name.

THE ONE RULE THAT MATTERS: A PRESET HOLDS ONLY WHAT DIFFERS FROM THE DEFAULTS
----------------------------------------------------------------------------
Not the full set of twelve values. This is the same constraint the settings
panel works under, for the same reason, and it is worth restating because a
preset is exactly the shape of thing someone would naturally implement as a
complete snapshot.

The server chooses cheaper settings for long audio -- a bigger hop, no
zero-padding, fewer tracks -- and it can only choose once the duration is
known. A preset that pinned all twelve values would count as twelve explicit
overrides, so applying a preset saved on a five-second phrase and then feeding
in a ten-minute recording would drag it back to settings that need gigabytes.
The user would have chosen a preset and broken the file.

Holding only the differences means a preset composes with the duration tier
instead of replacing it: "smaller window and a lower threshold" stays exactly
that at any length.

STORAGE
-------
One JSON file, keyed by name -- deliberately not one file per preset. A name
comes from a user and would otherwise have to be sanitised into a path, and
the safest way to handle a filename supplied by someone else is to not have
one. It is plain, readable JSON in the project folder, so it can be edited by
hand, copied to another machine, or committed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from backend.api.settings_schema import SETTABLE, clamp
from backend.core.types import AnalysisParams

logger = logging.getLogger(__name__)

# ADJUSTABLE: how long a preset name may be. Long enough to be descriptive,
# short enough to fit a dropdown without truncation.
MAX_NAME_LENGTH: int = 48

# ADJUSTABLE: how many presets a user may keep. A guard against a runaway
# client filling the disk, not a judgement about how many anyone needs.
MAX_USER_PRESETS: int = 200

#: Names are compared case-insensitively after collapsing whitespace, so
#: "Bells" and "bells " are the same preset rather than two confusing ones.
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Preset:
    """One named settings combination."""

    name: str
    settings: dict[str, float]
    description: str = ""
    builtin: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "settings": dict(self.settings),
            "description": self.description,
            "builtin": self.builtin,
        }


# =============================================================================
# Built-in presets
# =============================================================================
# Shipped, read-only, and chosen to be useful rather than exhaustive. They also
# serve as documentation: someone who has never opened the settings panel can
# pick "Bells and struck metal", see which sliders move, and learn what those
# settings are for by watching them.
#
# Every value here is a deliberate departure from a default, and the
# description says what it buys.

BUILTIN: tuple[Preset, ...] = (
    Preset(
        name="Sharp attacks",
        description=(
            "Plucks, drums, anything where the moment of the hit matters more than "
            "the pitch. A short window keeps transients crisp; brief tones are kept "
            "rather than discarded as noise."
        ),
        settings={"window_size": 1024, "min_partial_duration_ms": 20.0},
        builtin=True,
    ),
    Preset(
        name="Sustained and tonal",
        description=(
            "Drones, bowed strings, feedback, held voice. A long window resolves "
            "pitches that sit close together, and a longer minimum keeps flutter "
            "out of the track list."
        ),
        settings={
            "window_size": 8192,
            "min_partial_duration_ms": 150.0,
            "max_freq_drift_cents": 40.0,
        },
        builtin=True,
    ),
    Preset(
        name="Bells and struck metal",
        description=(
            "Inharmonic material, where the partials are NOT whole-number multiples "
            "and are interesting individually. Tight harmonic tolerance stops them "
            "being gathered into one 'note', so each is its own soloable track."
        ),
        settings={
            "window_size": 8192,
            "harmonic_tolerance_cents": 10.0,
            "min_partial_duration_ms": 100.0,
        },
        builtin=True,
    ),
    Preset(
        name="Voice and glides",
        description=(
            "Singing, portamento, tape wobble, anything that slides. A wide drift "
            "tolerance lets a tone move a long way and still be tracked as one tone "
            "rather than shattering into fragments."
        ),
        settings={"max_freq_drift_cents": 200.0, "max_gap_frames": 5},
        builtin=True,
    ),
    Preset(
        name="Dig deep",
        description=(
            "Show everything the analysis can find, including the junk. Useful for "
            "understanding a sound or hunting for one odd partial; unwieldy for "
            "actually working."
        ),
        settings={
            "peak_threshold_db": -85.0,
            "min_track_energy_share": 0.0,
            "max_tracks": 300,
            "min_partial_duration_ms": 20.0,
        },
        builtin=True,
    ),
    Preset(
        name="Just the notes",
        description=(
            "The opposite. Only prominent, confident tones, for when you want a "
            "handful of usable stems rather than a survey."
        ),
        settings={
            "peak_threshold_db": -40.0,
            "min_track_energy_share": 0.03,
            "min_partial_duration_ms": 120.0,
        },
        builtin=True,
    ),
)


class PresetError(ValueError):
    """A preset could not be saved. The message is safe to show a user."""


def normalise_name(raw: str) -> str:
    """Tidy a name, or explain why it cannot be used.

    Raises:
        PresetError: If the name is empty or too long, with a message written
            for the person who typed it.
    """
    name = _WHITESPACE.sub(" ", str(raw)).strip()
    if not name:
        raise PresetError("A preset needs a name.")
    if len(name) > MAX_NAME_LENGTH:
        raise PresetError(f"That name is too long (limit {MAX_NAME_LENGTH} characters).")
    return name


def clean_settings(raw: dict[str, float]) -> dict[str, float]:
    """Keep only real, in-range settings that differ from the defaults.

    The "differ from the defaults" part is the important one -- see the note at
    the top of this module. A preset carrying every value would override the
    duration tier and break long files.
    """
    defaults = AnalysisParams()
    clean: dict[str, float] = {}
    for key, value in (raw or {}).items():
        if key not in SETTABLE:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PresetError(f"{key} must be a number.")
        default = getattr(defaults, key)
        bounded = clamp(key, float(value))
        bounded = int(round(bounded)) if isinstance(default, int) else bounded
        if bounded != default:
            clean[key] = bounded

    # A preset with nothing in it would silently do nothing when applied, which
    # reads as a broken preset rather than an empty one.
    if not clean:
        raise PresetError(
            "Nothing to save — change a setting first, then save it as a preset."
        )
    return clean


class PresetStore:
    """User presets, kept in one JSON file.

    Thread-safe: the API can be handling two requests at once, and a
    read-modify-write of a shared file is exactly where that goes wrong.
    """

    __slots__ = ("_path", "_lock")

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        """Where the presets live. Shown to the user so the file is findable."""
        return self._path

    # -- Reading --------------------------------------------------------------

    def _read(self) -> dict[str, Preset]:
        """Load user presets. A damaged file yields none rather than raising.

        Losing presets is bad; refusing to start the app because a
        hand-editable file has a stray comma is worse. The corrupt file is left
        on disk untouched so it can be repaired.
        """
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("preset file at %s could not be read; ignoring it", self._path)
            return {}
        if not isinstance(raw, dict):
            return {}

        presets: dict[str, Preset] = {}
        for name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            settings = entry.get("settings")
            if not isinstance(settings, dict):
                continue
            # Re-clean on read: the file is hand-editable, so what comes back
            # may not satisfy the rules that applied when it was written.
            try:
                presets[name] = Preset(
                    name=normalise_name(name),
                    settings=clean_settings(settings),
                    description=str(entry.get("description", ""))[:200],
                )
            except PresetError:
                logger.warning("preset %r in %s is not usable; skipping", name, self._path)
        return presets

    def all(self) -> list[Preset]:
        """Built-ins first, then the user's own, each alphabetical."""
        with self._lock:
            user = sorted(self._read().values(), key=lambda p: p.name.lower())
        return [*BUILTIN, *user]

    def get(self, name: str) -> Preset | None:
        key = normalise_name(name).lower()
        for preset in self.all():
            if preset.name.lower() == key:
                return preset
        return None

    # -- Writing --------------------------------------------------------------

    def save(self, name: str, settings: dict[str, float], description: str = "") -> Preset:
        """Create or overwrite a user preset.

        Raises:
            PresetError: If the name clashes with a built-in, the settings are
                unusable, or there are already too many presets.
        """
        clean_name = normalise_name(name)
        if any(p.name.lower() == clean_name.lower() for p in BUILTIN):
            raise PresetError(
                f"'{clean_name}' is a built-in preset. Please choose another name."
            )
        clean = clean_settings(settings)

        with self._lock:
            presets = self._read()
            existing = next(
                (k for k in presets if k.lower() == clean_name.lower()), None
            )
            if existing is None and len(presets) >= MAX_USER_PRESETS:
                raise PresetError(
                    f"You already have {MAX_USER_PRESETS} presets. Delete one first."
                )
            if existing is not None:
                del presets[existing]

            preset = Preset(name=clean_name, settings=clean, description=str(description)[:200])
            presets[clean_name] = preset
            self._write(presets)
        return preset

    def delete(self, name: str) -> bool:
        """Remove a user preset. Returns False if there was nothing to remove."""
        clean_name = normalise_name(name)
        if any(p.name.lower() == clean_name.lower() for p in BUILTIN):
            raise PresetError("Built-in presets cannot be deleted.")

        with self._lock:
            presets = self._read()
            key = next((k for k in presets if k.lower() == clean_name.lower()), None)
            if key is None:
                return False
            del presets[key]
            self._write(presets)
        return True

    def _write(self, presets: dict[str, Preset]) -> None:
        """Persist, atomically.

        Written to a neighbouring temp file and renamed, because a process that
        dies mid-write would otherwise leave a truncated file -- and losing
        every preset to a crash while saving one is a poor trade.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            p.name: {"settings": p.settings, "description": p.description}
            for p in presets.values()
        }
        temp = self._path.with_suffix(self._path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temp.replace(self._path)
