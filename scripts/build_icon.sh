#!/bin/bash
#
# Build FourierPartial.app/Contents/Resources/AppIcon.icns from scratch.
#
# Uses only stock macOS tooling (sips, iconutil) plus the project virtualenv's
# numpy, so it runs on a bare macOS 12 Intel box with nothing installed.
#
#   ./scripts/build_icon.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
BUILD_DIR="$PROJECT_ROOT/build"
MASTER_PNG="$BUILD_DIR/icon_master_1024.png"
ICONSET_DIR="$BUILD_DIR/AppIcon.iconset"
ICNS_OUT="$PROJECT_ROOT/FourierPartial.app/Contents/Resources/AppIcon.icns"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "error: virtualenv python not found at $PYTHON_BIN" >&2
    echo "       create it with:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

echo "==> rendering 1024x1024 master"
"$PYTHON_BIN" "$SCRIPT_DIR/make_icon.py" "$MASTER_PNG"

# --- the ten renditions macOS expects inside an .iconset ----------------------
# Format is "<pixels>:<filename>". iconutil rejects the bundle outright if a
# name does not match Apple's scheme, so these strings are not free-form.
RENDITIONS=(
    "16:icon_16x16.png"
    "32:icon_16x16@2x.png"
    "32:icon_32x32.png"
    "64:icon_32x32@2x.png"
    "128:icon_128x128.png"
    "256:icon_128x128@2x.png"
    "256:icon_256x256.png"
    "512:icon_256x256@2x.png"
    "512:icon_512x512.png"
    "1024:icon_512x512@2x.png"
)

echo "==> building iconset"
rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"
for rendition in "${RENDITIONS[@]}"; do
    px="${rendition%%:*}"
    name="${rendition##*:}"
    sips -z "$px" "$px" "$MASTER_PNG" --out "$ICONSET_DIR/$name" >/dev/null
done

echo "==> compiling AppIcon.icns"
mkdir -p "$(dirname "$ICNS_OUT")"
iconutil -c icns "$ICONSET_DIR" -o "$ICNS_OUT"

# A valid multi-resolution .icns for this artwork is comfortably over 100 KB.
# Anything tiny means iconutil silently produced a stub, which would show up as
# a blank Dock icon long after anyone remembers building it.
bytes=$(stat -f%z "$ICNS_OUT")
if [ "$bytes" -lt 20000 ]; then
    echo "error: $ICNS_OUT is only $bytes bytes -- expected a multi-resolution icon" >&2
    exit 1
fi

echo "==> ok: $ICNS_OUT ($bytes bytes)"
