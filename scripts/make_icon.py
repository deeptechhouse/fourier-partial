#!/usr/bin/env python3
"""Generate the fourier_partial app icon from scratch — no image libraries.

Why hand-rolled? The only hard dependency in this project's virtualenv is
numpy. Pillow is deliberately NOT installed, and macOS 12 on Intel has no
Homebrew bottles, so "just pip install a drawing library" is a multi-hour
source build. Everything here is numpy maths plus a ~40-line PNG writer built
on the standard library's ``zlib`` and ``struct``.

The picture: a single composite waveform on the left splits into six coloured
partials on the right — literally what the application does to your audio.

Run it with the project virtualenv::

    .venv/bin/python scripts/make_icon.py [output.png]

Default output is ``build/icon_master_1024.png`` (1024x1024 RGBA).
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# ADJUSTABLE PARAMETERS — everything you would want to tweak lives here.
# ---------------------------------------------------------------------------

#: Master canvas size in pixels. macOS wants a 1024x1024 master for .icns.
SIZE = 1024

#: Background of the rounded "app tile". Near-black with a hint of blue.
GROUND_TOP = (0x12, 0x12, 0x1A)  # slightly lifted at the top edge
GROUND_BOTTOM = (0x08, 0x08, 0x0B)  # the canonical #08080b at the bottom

#: How far the rounded tile is inset from the canvas edge, as a fraction.
TILE_INSET = 0.035
#: Corner radius of the tile, as a fraction of the canvas. Apple's squircle is
#: close to 0.225 of the icon width; this reads the same at every size.
TILE_RADIUS = 0.205

#: The spectral ramp, ordered from the highest partial (top) to the
#: fundamental (bottom). Violet -> blue -> cyan -> green -> yellow-green -> amber.
PARTIAL_COLORS = (
    (0x7D, 0x5F, 0xE8),  # violet   — 6th partial
    (0x4A, 0x7F, 0xE0),  # blue     — 5th partial
    (0x35, 0xB5, 0xD8),  # cyan     — 4th partial
    (0x3F, 0xC9, 0xA8),  # green    — 3rd partial
    (0xA8, 0xCF, 0x5C),  # yellow-green — 2nd partial
    (0xE8, 0xA3, 0x3D),  # amber    — fundamental
)

#: Colour of the composite (undecomposed) waveform on the left.
COMPOSITE_COLOR = (0xEC, 0xEA, 0xF6)

#: Stroke half-widths in master pixels. Bold on purpose: at 32px a 26px
#: half-width becomes ~1.6px, which is the thinnest a line may be and still
#: hold its colour after downsampling.
PARTIAL_HALF_WIDTH = 25.0
COMPOSITE_HALF_WIDTH = 31.0

#: Soft glow radius around every stroke, and its peak opacity. Pure decoration
#: at large sizes; kept faint so it does not muddy the 16px rendering.
GLOW_RADIUS = 38.0
GLOW_STRENGTH = 0.20

#: Ceiling on the slope used for the stroke-width correction. Without it, a
#: near-vertical segment divides the vertical distance by a huge number, every
#: pixel in that column counts as "on the curve", and the glow paints a
#: full-height vertical band. Clamping trades a slightly thin stroke on the
#: steepest segments for an artefact-free image.
SLOPE_CLAMP = 3.0

#: Horizontal drawing area, as a fraction of the canvas.
X_START, X_END = 0.115, 0.885
#: Vertical band occupied by the fanned-out partials.
Y_TOP, Y_BOTTOM = 0.150, 0.850

#: Where the bundle splits, in normalised x. The ranges overlap on purpose so
#: the composite hands off to the partials through a single bright neck: the
#: signal oscillates, collapses to one line, then fans into its components.
COMPOSITE_COLLAPSE = (0.12, 0.36)  # composite amplitude 1 -> 0 (flattens to centre)
COMPOSITE_FADE = (0.40, 0.52)  # composite alpha 1 -> 0
PARTIAL_FADE = (0.36, 0.50)  # partial alpha 0 -> 1
FAN_SPREAD = (0.36, 0.74)  # partials move from centre to their own lanes
AMP_OPEN = (0.40, 0.80)  # partials grow from flat to full amplitude

#: Cycles of the fundamental across the full drawing width. Keep this low: the
#: 6th partial runs six times faster, and anything above ~6 cycles across the
#: canvas turns into grey mush once the icon is downsampled to 32px.
BASE_CYCLES = 0.95
#: Amplitude of partial *n* (1 = fundamental) relative to its lane half-height.
#: Higher partials are quieter, as in a real harmonic series — but not so much
#: quieter that the top lanes flatten into straight rules.
AMP_FALLOFF = 0.78

#: The composite waveform is the sum of only the first few harmonics. Summing
#: all six produced a near-vertical spike that read as a glitch rather than as
#: a waveform, so the left-hand signal is deliberately the smooth low end.
COMPOSITE_HARMONICS = ((1, 1.0), (2, 0.40), (3, 0.18))
#: Composite peak deviation from the centre line, as a fraction of the canvas.
COMPOSITE_AMPLITUDE = 0.105
#: Cycles of the composite fundamental across the full drawing width. Only the
#: left ~40% is visible, so this needs to be high enough to show two or three
#: cycles there — fewer than that and it reads as a diagonal slash, not a wave.
COMPOSITE_CYCLES = 4.0


# ---------------------------------------------------------------------------
# PNG output — 8-bit RGBA, no external dependencies.
# ---------------------------------------------------------------------------


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk: length, type, payload, CRC32."""
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, rgba: np.ndarray) -> None:
    """Write an (H, W, 4) uint8 array to `path` as a PNG.

    Uses filter type 0 (None) on every scanline — the image is smooth gradients,
    so zlib already compresses it well and skipping filter selection keeps this
    function short enough to audit at a glance.
    """
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected an (H, W, 4) uint8 array")
    height, width = rgba.shape[:2]

    # Prefix each row with a zero byte (the per-scanline filter type).
    prefixed = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), rgba.reshape(height, width * 4)], axis=1
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(prefixed.tobytes(), 9))
        + _chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite ease between two edges — 0 below `edge0`, 1 above `edge1`."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rounded_rect_coverage(size: int, inset: float, radius: float) -> np.ndarray:
    """Antialiased coverage mask (H, W) in [0, 1] for a rounded square.

    Evaluated as a signed distance field rather than drawn: an SDF gives exact
    sub-pixel antialiasing on the corners with no supersampling, which matters
    because those corners survive all the way down to the 16px rendering.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    centre = (size - 1) / 2.0
    half = size * (0.5 - inset)
    rad = size * radius

    # Distance to a rounded box centred on the canvas.
    dx = np.abs(xx - centre) - (half - rad)
    dy = np.abs(yy - centre) - (half - rad)
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    distance = outside + inside - rad
    return np.clip(0.5 - distance, 0.0, 1.0)


def stroke_coverage(
    grid_y: np.ndarray,
    curve_y: np.ndarray,
    curve_slope: np.ndarray,
    half_width: float,
    column_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage for a stroked curve y = f(x), plus its glow.

    The perpendicular distance from a pixel to the curve is approximated as the
    vertical gap divided by sqrt(1 + f'(x)^2). That correction is what stops the
    steep parts of a sine wave from rendering thinner than the flat parts.

    Returns:
        (core, glow) coverage arrays, both (H, W) in [0, 1].
    """
    clamped = np.clip(np.abs(curve_slope), 0.0, SLOPE_CLAMP)
    normaliser = np.sqrt(1.0 + clamped**2)
    distance = np.abs(grid_y - curve_y[None, :]) / normaliser[None, :]
    core = np.clip(half_width + 0.5 - distance, 0.0, 1.0) * column_alpha[None, :]
    glow = np.exp(-((distance / GLOW_RADIUS) ** 2)) * GLOW_STRENGTH * column_alpha[None, :]
    return core, glow


def _composite_over(
    dst: np.ndarray, colour: tuple[int, int, int], coverage: np.ndarray
) -> None:
    """Alpha-blend a flat colour onto an (H, W, 3) float image, in place."""
    rgb = np.array(colour, dtype=np.float64) / 255.0
    dst *= 1.0 - coverage[..., None]
    dst += rgb[None, None, :] * coverage[..., None]


def _add_light(dst: np.ndarray, colour: tuple[int, int, int], amount: np.ndarray) -> None:
    """Additively mix a colour in — used for glows, which emit rather than cover."""
    rgb = np.array(colour, dtype=np.float64) / 255.0
    dst += rgb[None, None, :] * amount[..., None]


# ---------------------------------------------------------------------------
# The icon itself
# ---------------------------------------------------------------------------


def render_icon(size: int = SIZE) -> np.ndarray:
    """Render the icon and return an (H, W, 4) uint8 RGBA array."""
    n_partials = len(PARTIAL_COLORS)

    # --- ground: vertical gradient, clipped to the rounded tile --------------
    tile = rounded_rect_coverage(size, TILE_INSET, TILE_RADIUS)
    ramp = (np.arange(size, dtype=np.float64) / (size - 1))[:, None, None]
    top = np.array(GROUND_TOP, dtype=np.float64).reshape(1, 1, 3) / 255.0
    bottom = np.array(GROUND_BOTTOM, dtype=np.float64).reshape(1, 1, 3) / 255.0
    column = top * (1.0 - ramp) + bottom * ramp  # (size, 1, 3)
    image = np.repeat(column, size, axis=1)  # (size, size, 3)

    # --- geometry of the waves ----------------------------------------------
    px = np.arange(size, dtype=np.float64)  # pixel x
    grid_y = np.arange(size, dtype=np.float64)[:, None]  # pixel y, as a column
    xn = (px / (size - 1) - X_START) / (X_END - X_START)  # normalised 0..1 across the art
    inside_art = (xn >= 0.0) & (xn <= 1.0)

    lane_top, lane_bottom = Y_TOP * size, Y_BOTTOM * size
    lane_height = (lane_bottom - lane_top) / n_partials
    centre_y = (lane_top + lane_bottom) / 2.0

    spread = smoothstep(*FAN_SPREAD, xn)
    opening = smoothstep(*AMP_OPEN, xn)
    partial_alpha = smoothstep(*PARTIAL_FADE, xn) * inside_art
    composite_alpha = (1.0 - smoothstep(*COMPOSITE_FADE, xn)) * inside_art

    # Fade both ends of every stroke so nothing terminates in a hard chop.
    edge_fade = smoothstep(0.0, 0.035, xn) * (1.0 - smoothstep(0.965, 1.0, xn))
    partial_alpha = partial_alpha * edge_fade
    composite_alpha = composite_alpha * edge_fade

    # Two-pixel-wide finite difference stands in for the analytic derivative;
    # it is exact enough for the stroke-width correction and keeps the maths in
    # one place no matter how the curves are defined.
    def slope_of(curve: np.ndarray) -> np.ndarray:
        return np.gradient(curve)

    partial_curves: list[np.ndarray] = []
    for index in range(n_partials):
        harmonic = n_partials - index  # top lane is the highest harmonic
        lane_centre = lane_top + lane_height * (index + 0.5)
        # Amplitude: quieter as the harmonic climbs, capped so lanes never touch.
        amp_ceiling = lane_height * 0.5 - PARTIAL_HALF_WIDTH - 3.0
        amplitude = amp_ceiling * (AMP_FALLOFF ** (harmonic - 1))
        wave = np.sin(2.0 * np.pi * BASE_CYCLES * harmonic * xn)
        baseline = centre_y + spread * (lane_centre - centre_y)
        partial_curves.append(baseline + amplitude * opening * wave)

    # The composite: a weighted sum of the low harmonics about the centre line,
    # normalised so its peak lands exactly on COMPOSITE_AMPLITUDE.
    raw_sum = np.zeros_like(xn)
    for harmonic, weight in COMPOSITE_HARMONICS:
        raw_sum += weight * np.sin(2.0 * np.pi * COMPOSITE_CYCLES * harmonic * xn)
    peak = float(np.max(np.abs(raw_sum))) or 1.0
    collapse = 1.0 - smoothstep(*COMPOSITE_COLLAPSE, xn)
    composite_curve = centre_y + raw_sum * collapse * (size * COMPOSITE_AMPLITUDE / peak)

    # --- paint: all glows first, so no glow washes out a neighbouring core ---
    glow_layers: list[tuple[tuple[int, int, int], np.ndarray]] = []
    core_layers: list[tuple[tuple[int, int, int], np.ndarray]] = []

    comp_core, comp_glow = stroke_coverage(
        grid_y, composite_curve, slope_of(composite_curve), COMPOSITE_HALF_WIDTH, composite_alpha
    )
    glow_layers.append((COMPOSITE_COLOR, comp_glow))
    core_layers.append((COMPOSITE_COLOR, comp_core))

    for colour, curve in zip(PARTIAL_COLORS, partial_curves):
        core, glow = stroke_coverage(
            grid_y, curve, slope_of(curve), PARTIAL_HALF_WIDTH, partial_alpha
        )
        glow_layers.append((colour, glow))
        core_layers.append((colour, core))

    for colour, glow in glow_layers:
        _add_light(image, colour, glow)
    for colour, core in core_layers:
        _composite_over(image, colour, core)

    np.clip(image, 0.0, 1.0, out=image)

    # --- assemble RGBA, with the tile mask as the alpha channel -------------
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    # Premultiply by the tile mask so the colour channels are black wherever the
    # icon is transparent. Without this, a stroke that runs past the rounded
    # corner leaves coloured-but-invisible pixels that some scalers bleed back
    # in as a fringe when they downsample.
    rgba[..., :3] = np.rint(image * tile[..., None] * 255.0).astype(np.uint8)
    rgba[..., 3] = np.rint(tile * 255.0).astype(np.uint8)
    return rgba


def main(argv: list[str]) -> int:
    """Render the master PNG to the path given on the command line."""
    default = Path(__file__).resolve().parent.parent / "build" / "icon_master_1024.png"
    out = Path(argv[1]).resolve() if len(argv) > 1 else default
    write_png(out, render_icon(SIZE))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
