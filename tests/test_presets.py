"""Saved settings combinations.

The property most worth pinning is the one a preset would naturally get wrong:
**a preset stores only what differs from the defaults.** A full twelve-value
snapshot would count as twelve explicit overrides, so applying a preset saved
on a short phrase and then analysing a ten-minute file would drag it off its
duration-appropriate settings and onto ones that need gigabytes. Several tests
below exist purely for that.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api.presets import (
    BUILTIN,
    MAX_NAME_LENGTH,
    MAX_USER_PRESETS,
    PresetError,
    PresetStore,
    clean_settings,
    normalise_name,
)
from backend.core.types import AnalysisParams
from backend.main import create_app


@pytest.fixture
def store(tmp_path):
    return PresetStore(tmp_path / "presets.json")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client whose presets go to a temp file, not the real one."""
    from backend.api import routes

    monkeypatch.setattr(routes, "_presets", PresetStore(tmp_path / "presets.json"))
    with TestClient(create_app()) as test_client:
        yield test_client


# =============================================================================
# The rule that makes presets safe
# =============================================================================


def test_a_preset_stores_only_what_differs_from_the_defaults():
    """The whole reason this is not a twelve-value snapshot."""
    defaults = AnalysisParams()
    cleaned = clean_settings(
        {
            "window_size": 1024,                                  # changed
            "peak_threshold_db": defaults.peak_threshold_db,      # unchanged
            "max_tracks": defaults.max_tracks,                    # unchanged
        }
    )
    assert cleaned == {"window_size": 1024}


def test_a_preset_of_only_defaults_is_refused():
    """Saving one would produce a preset that silently does nothing."""
    with pytest.raises(PresetError, match="change a setting"):
        clean_settings({"max_tracks": AnalysisParams().max_tracks})


def test_no_builtin_pins_a_value_that_equals_its_default():
    """A built-in carrying a default would be dead weight with a real cost.

    It would count as an override and displace the duration tier for that
    field, so a preset about window size could quietly change track limits on a
    long file.
    """
    defaults = AnalysisParams()
    for preset in BUILTIN:
        for key, value in preset.settings.items():
            assert value != getattr(defaults, key), (
                f"built-in {preset.name!r} pins {key} to its own default"
            )


def test_applying_a_preset_leaves_the_duration_tier_intact():
    """A preset composes with the tier rather than replacing it.

    This is the failure the whole design guards against, checked end to end:
    the 'Sharp attacks' preset changes the window, and everything it does not
    mention must still come from the long-form tier.
    """
    from backend.api.routes import _clean_settings, _params_for

    sharp = next(p for p in BUILTIN if p.name == "Sharp attacks")
    tier = AnalysisParams.for_duration(600.0)
    applied = _params_for(600.0, _clean_settings(dict(sharp.settings)))

    assert applied.window_size == sharp.settings["window_size"]
    # Untouched fields still come from the tier, not from phrase defaults.
    assert applied.zero_pad_factor == tier.zero_pad_factor
    assert applied.max_tracks == tier.max_tracks


# =============================================================================
# Built-ins
# =============================================================================


def test_builtins_are_usable_settings():
    """Every built-in must survive validation and name a real field."""
    fields = set(AnalysisParams().__dataclass_fields__)
    for preset in BUILTIN:
        assert preset.settings, f"{preset.name} is empty"
        assert preset.description.strip(), f"{preset.name} does not explain itself"
        for key in preset.settings:
            assert key in fields, f"{preset.name} names {key}, which is not a setting"
        # It has to actually build.
        AnalysisParams(**preset.settings)


def test_builtin_names_are_unique():
    names = [p.name.lower() for p in BUILTIN]
    assert len(names) == len(set(names))


def test_builtins_are_offered_before_user_presets(store):
    store.save("Aaa first alphabetically", {"window_size": 2048})
    names = [p.name for p in store.all()]
    assert names[: len(BUILTIN)] == [p.name for p in BUILTIN]


def test_builtins_cannot_be_overwritten(store):
    with pytest.raises(PresetError, match="built-in"):
        store.save(BUILTIN[0].name, {"window_size": 2048})


def test_builtins_cannot_be_deleted(store):
    with pytest.raises(PresetError, match="(?i)built-in"):
        store.delete(BUILTIN[0].name)


# =============================================================================
# Saving, loading, deleting
# =============================================================================


def test_save_then_read_back(store):
    store.save("Cymbals", {"window_size": 8192, "harmonic_tolerance_cents": 10.0}, "for metal")
    found = store.get("Cymbals")
    assert found is not None
    assert found.settings == {"window_size": 8192, "harmonic_tolerance_cents": 10.0}
    assert found.description == "for metal"
    assert found.builtin is False


def test_presets_survive_a_new_store(tmp_path):
    """They are on disk, not in memory — that is the point."""
    path = tmp_path / "presets.json"
    PresetStore(path).save("Kept", {"window_size": 1024})
    assert PresetStore(path).get("Kept") is not None


def test_the_file_is_readable_json(store):
    """Hand-editable on purpose, so it has to be legible."""
    store.save("Mine", {"window_size": 1024}, "notes")
    data = json.loads(store.path.read_text())
    assert data["Mine"]["settings"] == {"window_size": 1024}
    assert data["Mine"]["description"] == "notes"


def test_saving_the_same_name_overwrites(store):
    store.save("X", {"window_size": 1024})
    store.save("X", {"window_size": 8192})
    assert store.get("X").settings == {"window_size": 8192}
    assert sum(1 for p in store.all() if not p.builtin) == 1


def test_names_differing_only_in_case_or_spacing_are_the_same_preset(store):
    store.save("My Preset", {"window_size": 1024})
    store.save("  my   preset ", {"window_size": 8192})
    assert sum(1 for p in store.all() if not p.builtin) == 1


def test_delete_removes_it(store):
    store.save("Temp", {"window_size": 1024})
    assert store.delete("Temp") is True
    assert store.get("Temp") is None


def test_deleting_something_absent_reports_it(store):
    assert store.delete("never existed") is False


def test_empty_and_overlong_names_are_refused(store):
    for bad in ("", "   ", "\t"):
        with pytest.raises(PresetError, match="needs a name"):
            store.save(bad, {"window_size": 1024})
    with pytest.raises(PresetError, match="too long"):
        store.save("x" * (MAX_NAME_LENGTH + 1), {"window_size": 1024})


def test_a_name_cannot_escape_the_store(store, tmp_path):
    """Names come from users. Storing by key rather than by filename means a
    path-shaped name is just an odd string, not a traversal."""
    store.save("../../etc/passwd", {"window_size": 1024})
    assert not (tmp_path.parent / "etc").exists()
    assert store.get("../../etc/passwd") is not None
    assert json.loads(store.path.read_text())


def test_too_many_presets_is_refused(store):
    for i in range(MAX_USER_PRESETS):
        # Values chosen to never equal the default, which would be refused as
        # "nothing to save" and fail this test for the wrong reason.
        store.save(f"p{i}", {"max_tracks": 1 + (i % 150)})
    with pytest.raises(PresetError, match="Delete one first"):
        store.save("one too many", {"window_size": 1024})


def test_a_corrupt_file_does_not_break_the_app(store):
    store.path.write_text("{ this is not json")
    assert [p.name for p in store.all()] == [p.name for p in BUILTIN]
    # And it can still be written to afterwards.
    store.save("Recovered", {"window_size": 1024})
    assert store.get("Recovered") is not None


def test_unusable_entries_in_the_file_are_skipped(store):
    """Hand-edited files will contain mistakes; one bad entry must not lose the rest."""
    store.path.write_text(
        json.dumps(
            {
                "Good": {"settings": {"window_size": 1024}},
                "Bad": {"settings": "not a dict"},
                "Empty": {"settings": {}},
            }
        )
    )
    user = [p.name for p in store.all() if not p.builtin]
    assert user == ["Good"]


def test_out_of_range_values_are_clamped_on_save(store):
    store.save("Extreme", {"peak_threshold_db": -9999})
    assert store.get("Extreme").settings["peak_threshold_db"] == -90


def test_unknown_settings_are_dropped(store):
    store.save("Odd", {"window_size": 1024, "not_a_setting": 5})
    assert store.get("Odd").settings == {"window_size": 1024}


def test_booleans_are_not_numbers(store):
    with pytest.raises(PresetError):
        store.save("Bool", {"max_tracks": True})


def test_normalise_collapses_whitespace():
    assert normalise_name("  a   b  ") == "a b"


# =============================================================================
# Over HTTP
# =============================================================================


def test_list_includes_builtins_and_the_file_location(client):
    data = client.get("/api/presets").json()
    names = [p["name"] for p in data["presets"]]
    for preset in BUILTIN:
        assert preset.name in names
    assert data["file"].endswith("presets.json")


def test_save_and_list_round_trip(client):
    response = client.put(
        "/api/presets/My Cymbals",
        json={"settings": {"window_size": 8192}, "description": "metal"},
    )
    assert response.status_code == 200
    assert response.json()["settings"] == {"window_size": 8192}

    listed = client.get("/api/presets").json()["presets"]
    mine = next(p for p in listed if p["name"] == "My Cymbals")
    assert mine["builtin"] is False
    assert mine["description"] == "metal"


def test_delete_over_http(client):
    client.put("/api/presets/Temp", json={"settings": {"window_size": 1024}})
    assert client.delete("/api/presets/Temp").status_code == 200
    assert client.delete("/api/presets/Temp").status_code == 404


def test_saving_over_a_builtin_is_refused_with_a_reason(client):
    response = client.put(
        f"/api/presets/{BUILTIN[0].name}", json={"settings": {"window_size": 1024}}
    )
    assert response.status_code == 400
    assert "built-in" in response.json()["detail"].lower()


def test_saving_nothing_is_refused_with_a_reason(client):
    response = client.put("/api/presets/Empty", json={"settings": {}})
    assert response.status_code == 400
    assert "change a setting" in response.json()["detail"].lower()


def test_a_saved_preset_can_be_used_for_an_analysis(client):
    """The round trip that matters: save, read back, analyse with it."""
    import io
    import time

    import numpy as np
    import soundfile as sf

    client.put("/api/presets/Deep", json={"settings": {"min_track_energy_share": 0.0}})
    preset = next(
        p for p in client.get("/api/presets").json()["presets"] if p["name"] == "Deep"
    )

    sr = 44_100
    t = np.arange(int(0.8 * sr)) / sr
    audio = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 7))
    audio += 0.03 * np.random.default_rng(1).standard_normal(len(t))
    audio = (0.7 * audio / np.max(np.abs(audio))).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="FLOAT")

    response = client.post(
        "/api/analyze",
        files={"file": ("a.wav", buffer.getvalue(), "audio/wav")},
        data={"settings": json.dumps(preset["settings"])},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    for _ in range(900):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert status["state"] == "complete", status
    assert client.get(f"/api/jobs/{job_id}/tracks").json()["reconstruction_exact"] is True
