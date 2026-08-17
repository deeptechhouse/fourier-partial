"""The tunable analysis settings, described once, for both sides.

WHY A SCHEMA RATHER THAN A FORM
-------------------------------
The interface needs a label, a range, a step, a default and a sentence of help
for every setting it offers. Written in JavaScript, that duplicates knowledge
the Python already holds -- and the two drift the moment a default changes,
producing a slider whose "reset" restores a value the analyser no longer uses.

So the settings describe themselves here, the API serves the description, and
the panel builds itself from what it is given. Adding a knob means adding one
entry to this file.

ON THE RANGES
-------------
They are deliberately wide. This is a tool for taking sounds apart creatively,
and "wrong" settings are a legitimate thing to want -- someone hunting for the
strange partials in a cymbal has every reason to drop the threshold to -90 dB
and wade through the junk. The bounds exist to prevent values that would hang
or crash, not to enforce good taste.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.core.types import AnalysisParams


@dataclass(frozen=True, slots=True)
class Setting:
    """One adjustable parameter, described for a person."""

    name: str
    label: str
    help: str
    minimum: float
    maximum: float
    step: float
    #: Powers of two only (window and hop sizes). The UI snaps to them.
    power_of_two: bool = False
    #: Shown under the group heading rather than inline; advanced knobs.
    advanced: bool = False
    #: What raising it does, in the user's terms. Paired with `lower`.
    higher: str = ""
    lower: str = ""

    def default(self) -> float:
        return getattr(AnalysisParams(), self.name)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "help": self.help,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "default": self.default(),
            "power_of_two": self.power_of_two,
            "advanced": self.advanced,
            "higher": self.higher,
            "lower": self.lower,
        }


@dataclass(frozen=True, slots=True)
class SettingGroup:
    """A handful of settings that belong to the same stage of the pipeline."""

    key: str
    label: str
    blurb: str
    settings: list[Setting] = field(default_factory=list)


# The order here is the order of the pipeline itself, which is also the order
# in which a change stops mattering: a different window changes everything
# downstream, whereas the track ceiling changes only what you are shown.
GROUPS: tuple[SettingGroup, ...] = (
    SettingGroup(
        key="listening",
        label="How closely it listens",
        blurb=(
            "The window is the one true trade-off in the whole program: you can have "
            "precise pitch or precise timing, never both. Everything else here decides "
            "how quiet a tone has to be before it stops counting."
        ),
        settings=[
            Setting(
                name="window_size",
                label="Window size",
                help=(
                    "How much audio it looks at in one go. This is not a quality dial — "
                    "it picks a side. Bigger separates closer pitches and smears attacks; "
                    "smaller sharpens attacks and blurs pitch."
                ),
                minimum=512,
                maximum=16384,
                step=1,
                power_of_two=True,
                higher="separates closer pitches, smears attacks",
                lower="sharper attacks, coarser pitch",
            ),
            Setting(
                name="peak_threshold_db",
                label="Quiet-tone threshold",
                help=(
                    "How far below the loudest tone in each moment something can be and "
                    "still count. The most direct 'more tracks / fewer tracks' control."
                ),
                minimum=-90,
                maximum=-20,
                step=1,
                higher="fewer, cleaner tones",
                lower="finds quiet detail, and more junk",
            ),
            Setting(
                name="hop_size",
                label="Step between windows",
                help=(
                    "How far it moves between looks. Smaller means smoother trajectories "
                    "and proportionally more work; it is the main cost dial."
                ),
                minimum=64,
                maximum=4096,
                step=1,
                power_of_two=True,
                advanced=True,
                higher="faster analysis, coarser timing",
                lower="smoother tracking, slower",
            ),
            Setting(
                name="zero_pad_factor",
                label="Spectrum interpolation",
                help=(
                    "Adds no real detail, but samples the spectrum more finely so pitch "
                    "estimates land closer to the truth. Costs memory."
                ),
                minimum=1,
                maximum=8,
                step=1,
                power_of_two=True,
                advanced=True,
                higher="slightly more accurate pitch, more memory",
                lower="cheaper",
            ),
            Setting(
                name="max_peaks_per_frame",
                label="Tones per moment",
                help="A ceiling on how many tones it will consider at any instant.",
                minimum=10,
                maximum=400,
                step=10,
                advanced=True,
                higher="handles dense, crowded material",
                lower="bounds the worst case",
            ),
        ],
    ),
    SettingGroup(
        key="tracking",
        label="What counts as one tone",
        blurb=(
            "Having found tones moment by moment, it has to decide which ones are the "
            "same tone continuing. These settings are where vibrato either stays one "
            "note or shatters into fragments."
        ),
        settings=[
            Setting(
                name="max_freq_drift_cents",
                label="Allowed pitch drift",
                help=(
                    "How far a tone may move between moments and still be considered the "
                    "same tone. In cents — 100 is a semitone."
                ),
                minimum=10,
                maximum=300,
                step=5,
                higher="follows wide vibrato and glides",
                lower="stops tracks jumping between pitches",
            ),
            Setting(
                name="min_partial_duration_ms",
                label="Shortest tone kept",
                help=(
                    "Anything briefer than this is treated as noise and folded into the "
                    "leftover track. The main defence against hiss arriving as notes."
                ),
                minimum=10,
                maximum=500,
                step=10,
                higher="rejects noise-as-tone",
                lower="keeps short blips and clicks",
            ),
            Setting(
                name="max_gap_frames",
                label="Tolerated silences",
                help=(
                    "How long a tone may be masked by something louder and still resume "
                    "as the same track rather than starting a new one."
                ),
                minimum=0,
                maximum=10,
                step=1,
                advanced=True,
                higher="survives heavy masking",
                lower="strict, unforgiving tracking",
            ),
        ],
    ),
    SettingGroup(
        key="grouping",
        label="How tones become notes",
        blurb=(
            "A note is a fundamental plus its harmonics sounding together. These decide "
            "how generously that family is drawn, and how much makes it onto the screen."
        ),
        settings=[
            Setting(
                name="harmonic_tolerance_cents",
                label="Harmonic tolerance",
                help=(
                    "How far from an exact multiple a tone can sit and still be counted "
                    "as part of the same note. Piano strings and struck metal are "
                    "genuinely stretched, so strictness loses real harmonics."
                ),
                minimum=5,
                maximum=200,
                step=5,
                higher="groups stretched harmonics (piano, bells)",
                lower="stricter, more separate tracks",
            ),
            Setting(
                name="min_track_energy_share",
                label="Quiet-track cull",
                help=(
                    "A track must carry at least this share of the loudest track's energy "
                    "to get a row. Set it to zero to see absolutely everything found — "
                    "illuminating, and unwieldy."
                ),
                minimum=0.0,
                maximum=0.2,
                step=0.001,
                higher="fewer, more significant tracks",
                lower="shows everything, including junk",
            ),
            Setting(
                name="min_temporal_overlap",
                label="Simultaneity required",
                help=(
                    "How much two tones must overlap in time to belong to the same note. "
                    "Low values start merging separate occurrences of the same pitch."
                ),
                minimum=0.0,
                maximum=0.9,
                step=0.05,
                advanced=True,
                higher="splits notes whose harmonics die early",
                lower="merges repeats of the same pitch",
            ),
            Setting(
                name="max_tracks",
                label="Most tracks shown",
                help=(
                    "A ceiling on rows. Anything beyond it is folded into the leftover "
                    "track — still audible and exportable, just not on its own row."
                ),
                minimum=1,
                maximum=300,
                step=1,
                higher="more detail, heavier interface",
                lower="only the most prominent notes",
            ),
        ],
    ),
)

#: Every settable field name, for validating an incoming payload.
SETTABLE: frozenset[str] = frozenset(
    s.name for group in GROUPS for s in group.settings
)


def schema() -> dict[str, object]:
    """The whole description, for the interface to build itself from."""
    return {
        "groups": [
            {
                "key": g.key,
                "label": g.label,
                "blurb": g.blurb,
                "settings": [s.as_dict() for s in g.settings],
            }
            for g in GROUPS
        ]
    }


def clamp(name: str, value: float) -> float:
    """Hold a value inside its declared range.

    Applied server-side as well as in the panel, because the panel is not the
    only possible client and a slider is not a validation boundary.
    """
    for group in GROUPS:
        for setting in group.settings:
            if setting.name == name:
                return max(setting.minimum, min(setting.maximum, value))
    return value
