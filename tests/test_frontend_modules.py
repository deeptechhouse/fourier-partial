"""The frontend modules must actually load, not merely parse.

WHY THIS TEST EXISTS
--------------------
A call to `createSettingsPanel(...)` shipped with no import for it. The entire
settings panel was dead, and so was everything wired after it in the same
method, because the ReferenceError aborted the rest.

`node --check` passed the whole time. It is a *syntax* check: an identifier
that is spelled correctly and defined nowhere is perfectly valid syntax and
fails only when the line runs. So the guard that would have caught it is not a
stricter parse — it is evaluating the module.

That is what this does: rewrite the imports for node, stub the browser APIs the
modules touch at load time, and import them. If an identifier is missing, or a
module imports a name its dependency does not export, this fails.

It cannot exercise behaviour — there is no DOM, no audio clock, no canvas — and
it is not trying to. It answers one question: would this file explode the
moment the browser loaded it?
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "js"
# Discovered, not listed. A hardcoded list goes stale the moment a module is
# added — which happened immediately: presets.js was written, settings.js
# imported it, and the copy step silently skipped it, so the test failed with a
# missing-module error that had nothing to do with the code under test.
MODULES = tuple(sorted(p.stem for p in (FRONTEND).glob("*.js")))

# Enough of the browser for the modules to *evaluate*. Deliberately a Proxy
# rather than hand-written stubs: the modules touch a wide and shifting surface
# at load time, and a stub list would need updating every time one more
# property was read, turning a useful test into a maintenance tax.
HARNESS = """
const el = () => new Proxy(function () {}, {
  get: (t, k) => (k === 'children' ? [] : k === 'length' ? 0 : el()),
  set: () => true,
  apply: () => el(),
});
global.window = {
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  addEventListener() {},
  devicePixelRatio: 1,
  AudioContext: function () { return el(); },
};
global.document = {
  getElementById: () => el(),
  querySelector: () => el(),
  querySelectorAll: () => [],
  createElement: () => el(),
  addEventListener() {},
  documentElement: el(),
  body: el(),
  readyState: 'complete',
};
global.fetch = async () => ({ ok: false, json: async () => ({}) });
global.AudioContext = function () { return el(); };
global.ResizeObserver = function () { return { observe() {}, disconnect() {} }; };
global.URL = { createObjectURL: () => 'blob:x', revokeObjectURL() {} };

const mod = await import(process.argv[1]);
console.log('EXPORTS:' + Object.keys(mod).join(','));
"""

node = shutil.which("node")
requires_node = pytest.mark.skipif(node is None, reason="node is not installed")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Copy the modules with .js imports rewritten to .mjs so node loads them."""
    out = tmp_path_factory.mktemp("frontend")
    for name in MODULES:
        source = (FRONTEND / f"{name}.js").read_text()
        # Only rewrite our own relative imports; leave anything else alone.
        rewritten = re.sub(r"(from\s+'\./[a-z]+)\.js'", r"\1.mjs'", source)
        (out / f"{name}.mjs").write_text(rewritten)
    return out


@requires_node
@pytest.mark.parametrize("name", MODULES)
def test_module_evaluates_without_a_reference_error(prepared, name):
    """Load the module for real. This is what `node --check` cannot do."""
    result = subprocess.run(
        [node, "--input-type=module", "-e", HARNESS, str(prepared / f"{name}.mjs")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{name}.js failed to evaluate:\n{result.stderr.strip()}\n\n"
        "A ReferenceError here almost always means a function is called but "
        "never imported."
    )


@requires_node
def test_app_imports_everything_it_calls(prepared):
    """The specific failure that prompted this file.

    app.js is where the cross-module calls live, so it is the one that can lose
    an import. Checking the names it uses against the names it imports catches
    the mistake at the point it is made rather than at the point it is noticed.
    """
    source = (FRONTEND / "app.js").read_text()

    imported: set[str] = set()
    for names in re.findall(r"import\s*\{([^}]+)\}\s*from", source):
        for part in names.split(","):
            imported.add(part.split(" as ")[-1].strip())

    # Names app.js is known to obtain from its siblings. Checked explicitly
    # rather than inferred, because inferring "is this call local or imported?"
    # from a regex over 3,000 lines produces more false alarms than findings.
    from_siblings = {
        "Transport",
        "MIX_TRACK_ID",
        "drawTrajectory",
        "drawWaveform",
        "clear",
        "createSettingsPanel",
    }
    for name in sorted(from_siblings):
        if re.search(rf"\b{name}\b", source):
            assert name in imported, (
                f"app.js uses {name} but does not import it — this parses "
                f"cleanly and throws a ReferenceError the moment it runs."
            )


@requires_node
def test_every_module_the_page_loads_is_present():
    """index.html must not reference a module that does not exist."""
    index = (FRONTEND.parent / "index.html").read_text()
    for src in re.findall(r'<script[^>]+src="([^"]+)"', index):
        if src.startswith("http"):
            continue
        target = FRONTEND.parent / src.lstrip("/")
        assert target.is_file(), f"index.html loads {src}, which does not exist"
