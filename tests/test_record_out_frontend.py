"""The browser half of record out, executed rather than inspected.

WHY THIS IS A SEPARATE FILE FROM test_record_out.py
---------------------------------------------------
The Python tests prove the server sums the right tracks. They cannot prove the
browser *asks* for the right tracks, and that is the half where every frontend
bug in this project has actually lived — each one passing `node --check`, and
one of them needing to be reported twice before it was found.

Two things decide whether record-out is correct in the browser:

  1. `_audibleIds()` — which tracks are considered audible. If this disagrees
     with playback by even one track, the bounce is not what the user heard,
     and nothing about the file looks wrong.
  2. `exportAllUrl()` — whether that selection survives into the request. An
     empty `only=` means "no tracks" while an absent one means "all tracks",
     so the difference between those two is the whole feature.

Both are pure functions of their inputs, so both can be run for real in node
with nothing stubbed but the browser globals the module touches at load time.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "js"

node = shutil.which("node")
requires_node = pytest.mark.skipif(node is None, reason="node is not installed")

# Enough browser for the module to evaluate; see test_frontend_modules.py.
PRELUDE = """
const el = () => new Proxy(function () {}, {
  get: (t, k) => (k === 'children' ? [] : k === 'length' ? 0 : el()),
  set: () => true,
  apply: () => el(),
});
global.window = {
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  addEventListener() {}, devicePixelRatio: 1,
  AudioContext: function () { return el(); },
};
global.document = {
  getElementById: () => el(), querySelector: () => el(),
  querySelectorAll: () => [], createElement: () => el(),
  addEventListener() {}, documentElement: el(), body: el(),
  readyState: 'complete',
};
global.fetch = async () => ({ ok: false, json: async () => ({}) });
global.AudioContext = function () { return el(); };
global.ResizeObserver = function () { return { observe() {}, disconnect() {} }; };
global.URL = { createObjectURL: () => 'blob:x', revokeObjectURL() {} };
const { App, ApiClient, audibleIds } = await import(process.argv[1]);

/** The shared rule, called directly. */
function audible(all, muted, soloed) {
  return audibleIds(all, new Set(muted), new Set(soloed));
}

/** The App's own accessor, on a hand-made state -- proves it is wired up.
 *
 * Object.create(App.prototype), not a bare object with one borrowed method:
 * _audibleIds calls _allTrackIds, so both must genuinely live on App. Handing
 * the stub its own copies would defeat the entire point of this check. */
function viaApp(rows, muted, soloed) {
  const app = Object.create(App.prototype);
  app._rows = new Map(rows.map((id) => [id, {}]));
  app._muted = new Set(muted);
  app._soloed = new Set(soloed);
  return app._audibleIds();
}
"""


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Copy the modules with .js imports rewritten to .mjs so node loads them."""
    out = tmp_path_factory.mktemp("frontend-record")
    for path in FRONTEND.glob("*.js"):
        rewritten = re.sub(r"(from\s+'\./[a-z]+)\.js'", r"\1.mjs'", path.read_text())
        (out / f"{path.stem}.mjs").write_text(rewritten)
    return out


def run(prepared, body: str):
    """Evaluate `body` after the prelude and return its JSON result."""
    script = PRELUDE + f"\nconsole.log(JSON.stringify((() => {{ {body} }})()));"
    result = subprocess.run(
        [node, "--input-type=module", "-e", script, str(prepared / "app.mjs")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.strip()
    return json.loads(result.stdout.strip().splitlines()[-1])


# =============================================================================
# Which tracks count as audible
# =============================================================================


@requires_node
def test_nothing_muted_or_soloed_means_everything(prepared):
    assert run(prepared, "return audible([1,2,3], [], []);") == [1, 2, 3]


@requires_node
def test_muting_removes_only_the_muted(prepared):
    assert run(prepared, "return audible([1,2,3,4], [2], []);") == [1, 3, 4]


@requires_node
def test_soloing_overrides_everything_not_soloed(prepared):
    """A solo is exclusive: unsoloed tracks are silent whether muted or not."""
    assert run(prepared, "return audible([1,2,3,4], [], [3]);") == [3]
    assert run(prepared, "return audible([1,2,3,4], [1], [3]);") == [3]


@requires_node
def test_a_track_both_soloed_and_muted_is_silent(prepared):
    """The exact rule EagerPlayback.apply uses: `soloed minus muted`.

    This is the case that separates a correct implementation from a plausible
    one. Solo normally wins over mute, but a track carrying both makes no sound
    either way, so it must not appear in the bounce.
    """
    assert run(prepared, "return audible([1,2,3], [2], [2,3]);") == [3]


@requires_node
def test_soloing_and_muting_the_same_single_track_leaves_nothing(prepared):
    """The empty case is reachable, so it must be represented, not crash."""
    assert run(prepared, "return audible([1,2,3], [2], [2]);") == []


@requires_node
def test_muting_every_track_leaves_nothing(prepared):
    assert run(prepared, "return audible([1,2], [1,2], []);") == []


@requires_node
def test_the_result_is_sorted_and_deduplicated(prepared):
    """Stable ordering keeps the URL stable, which keeps it cacheable."""
    assert run(prepared, "return audible([5,1,3], [], [3,5,3]);") == [3, 5]


@requires_node
def test_the_app_reaches_the_rule_through_its_own_state(prepared):
    """Not the rule this time — the wiring.

    `_audibleIds` was first written as a method on PlaybackStrategy and called
    from App, which has no `_allTrackIds` at all. Every record-out click would
    have thrown, and nothing static caught it. This calls App's accessor
    against App's actual data shape (a Map of rows keyed by track id), so a
    method landing on the wrong class fails here rather than in the browser.
    """
    assert run(prepared, "return viaApp([1,2,3], [2], []);") == [1, 3]
    assert run(prepared, "return viaApp([1,2,3], [], [3]);") == [3]


# =============================================================================
# What the menu tells the user it would capture
# =============================================================================


@requires_node
def test_the_menu_describes_the_current_mix(prepared):
    """`_refreshRecordOut` rewrites the detail lines on every mute and solo.

    Run against a tiny fake DOM rather than the load-time Proxy, because the
    Proxy answers every query and would make any assertion pass. The point is
    the text: "3 of 4 tracks" is the answer to the question the user has at
    that moment, and a stale "the current mix" is not.
    """
    result = run(
        prepared,
        """
        const made = [];
        const mk = (fmt) => {
          const detail = { textContent: '' };
          return { dataset: { format: fmt, record: '1' }, disabled: false,
                   querySelector: () => detail, _detail: detail };
        };
        const buttons = [mk('wav-mix'), mk('wav')];
        global.document.querySelectorAll = (sel) =>
          sel.includes('data-record') ? buttons : [];

        const app = Object.create(App.prototype);
        app._rows = new Map([[0,{}],[1,{}],[2,{}],[3,{}]]);
        const snap = () => ({
          mix: buttons[0]._detail.textContent,
          sep: buttons[1]._detail.textContent,
          disabled: buttons[0].disabled,
        });

        app._muted = new Set(); app._soloed = new Set();
        app._refreshRecordOut(); const all = snap();

        app._muted = new Set([1]); app._soloed = new Set();
        app._refreshRecordOut(); const some = snap();

        app._muted = new Set([0,1,2,3]); app._soloed = new Set();
        app._refreshRecordOut(); const none = snap();

        return { all, some, none };
        """,
    )

    # Nothing touched: it says so plainly, rather than "4 of 4".
    assert "all 4 tracks" in result["all"]["mix"]
    assert "one file" in result["all"]["mix"]
    assert result["all"]["disabled"] is False

    # One muted: the count is live.
    assert "3 of 4 tracks" in result["some"]["mix"]
    assert "3 of 4 tracks" in result["some"]["sep"]
    assert "one file each" in result["some"]["sep"]

    # Everything muted: the option refuses rather than bouncing silence.
    assert result["none"]["disabled"] is True
    assert "nothing is audible" in result["none"]["mix"].lower()


# =============================================================================
# Whether the selection survives into the request
# =============================================================================


@requires_node
def test_a_selection_becomes_an_only_parameter(prepared):
    url = run(
        prepared,
        "return new ApiClient('/api').exportAllUrl('j1', 'wav-mix', '2,3');",
    )
    assert "format=wav-mix" in url
    assert "only=2%2C3" in url or "only=2,3" in url


@requires_node
def test_no_selection_sends_no_only_parameter(prepared):
    """Absent, not empty.

    `only=` would be read as "no tracks selected" and export nothing. This is
    the single most consequential line in the frontend change.
    """
    url = run(prepared, "return new ApiClient('/api').exportAllUrl('j1', 'wav');")
    assert "only" not in url
    assert "format=wav" in url


@requires_node
def test_an_empty_string_selection_is_also_omitted(prepared):
    """Defence in depth: an empty join must not become `only=` either."""
    url = run(prepared, "return new ApiClient('/api').exportAllUrl('j1', 'wav', '');")
    assert "only" not in url


# =============================================================================
# The markup the handlers depend on
# =============================================================================


@requires_node
def test_the_page_carries_both_record_out_options():
    """The buttons must exist, be marked as record-out, and have detail slots.

    `_refreshRecordOut` finds them by `[data-record]` and writes into
    `[data-record-detail]`. If either attribute were dropped the labels would
    silently stop updating and the menu would describe a mix that is no longer
    current.
    """
    index = (FRONTEND.parent / "index.html").read_text()
    options = re.findall(r'<button[^>]*data-record="1"[^>]*>', index)
    assert len(options) == 2, "expected a mixed bounce and a separate-files bounce"

    formats = {re.search(r'data-format="([^"]+)"', o).group(1) for o in options}
    assert formats == {"wav-mix", "wav"}

    for fmt in formats:
        assert f'data-record-detail="{fmt}"' in index


@requires_node
def test_record_out_options_are_not_mistaken_for_plain_exports():
    """`[data-detail=...]` is the estimate loop's selector; record-out uses its
    own. Sharing it would let the estimate overwrite the live "what you're
    hearing" text with a static file count."""
    index = (FRONTEND.parent / "index.html").read_text()
    for option in re.findall(r'<button[^>]*data-record="1"[^>]*>.*?</button>', index, re.S):
        assert "data-detail=" not in option
