#!/bin/bash
#
# fourier_partial — one-command setup.
#
#   ./setup.sh
#
# Creates the Python environment and installs pinned dependencies. Run this
# once after cloning, or after moving the project to a different machine.
#
# WHY YOU MUST RUN THIS AFTER MOVING MACHINES
# -------------------------------------------
# .venv contains COMPILED, ARCHITECTURE-SPECIFIC binaries. numpy, scipy and
# soundfile all ship native code, so an Intel (x86_64) virtual environment
# simply will not run on Apple Silicon (arm64), and vice versa. Copying the
# folder across is fine; copying .venv across is not. This script deletes and
# rebuilds it, which is why .venv is in .gitignore and never committed.
#
# Everything else in the project is architecture-independent: the Python source,
# the frontend, and FourierPartial.app (which is a shell script, a plist and an
# icon, not a compiled binary).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$PROJECT_ROOT"

REQUIRED_PYTHON="$(cat .python-version 2>/dev/null || echo '3.13')"

printf '\n=== fourier_partial setup ===\n'
printf 'project : %s\n' "$PROJECT_ROOT"
printf 'arch    : %s\n' "$(uname -m)"

# --- 1. Find a usable Python -------------------------------------------------
# Prefer the exact pinned minor version, then fall back to whatever python3 is,
# checking it is new enough rather than assuming.
PYTHON=""
for candidate in "python${REQUIRED_PYTHON%.*}" "python3.13" python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)' 2>/dev/null; then
            PYTHON="$(command -v "$candidate")"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    cat >&2 <<EOF

ERROR: Python 3.13 or newer was not found.

Install it, then run this script again:

    brew install python@3.13

(or download from python.org — either works)
EOF
    exit 1
fi

printf 'python  : %s (%s)\n\n' "$PYTHON" "$("$PYTHON" --version)"

# --- 2. Rebuild the virtual environment --------------------------------------
if [ -d .venv ]; then
    printf 'Removing the existing .venv (architecture-specific — see the note above)...\n'
    rm -rf .venv
fi

printf 'Creating .venv...\n'
"$PYTHON" -m venv .venv

printf 'Upgrading pip...\n'
./.venv/bin/python -m pip install --quiet --upgrade pip

# --- 3. Install dependencies -------------------------------------------------
# --only-binary=:all: refuses to compile from source. That is deliberate: every
# dependency here publishes wheels for both macOS architectures, and if pip ever
# starts building numpy or scipy locally, something is wrong with the
# environment and you want to know immediately rather than after 40 minutes of
# compilation.
printf 'Installing dependencies (wheels only)...\n'
if ! ./.venv/bin/pip install --quiet --only-binary=:all: -r requirements.txt; then
    cat >&2 <<'EOF'

ERROR: dependency installation failed.

The most likely cause is that no prebuilt wheel exists for this Python version
on this architecture, and installation refused to compile from source.

Check which package failed above, then either use a Python version with wheels
available, or drop --only-binary=:all: from this script if you genuinely want
to compile (expect it to take a long time).
EOF
    exit 1
fi

# --- 4. Verify it actually works ---------------------------------------------
printf '\nVerifying...\n'
./.venv/bin/python - <<'PYEOF'
import numpy, scipy, soundfile, mido, fastapi
print(f"  numpy      {numpy.__version__}")
print(f"  scipy      {scipy.__version__}")
print(f"  soundfile  {soundfile.__version__} (libsndfile {soundfile.__libsndfile_version__})")
print(f"  fastapi    {fastapi.__version__}")
import backend.main  # the real smoke test: does the app import?
print("  backend    imports cleanly")
PYEOF

cat <<EOF

=== Ready ===

Start it any of these ways:

  Double-click     FourierPartial.app
  Terminal         ./run.sh
  Dev (reload)     .venv/bin/uvicorn backend.main:app --reload

If macOS refuses to open FourierPartial.app because it came from another
machine, clear the quarantine flag once:

  xattr -dr com.apple.quarantine FourierPartial.app

To run the tests:

  .venv/bin/pip install -r requirements-dev.txt
  .venv/bin/python -m pytest

EOF
