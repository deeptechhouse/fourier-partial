/**
 * waveform.js — canvas drawing for track rows.
 *
 * TWO VIEWS, ONE TIMELINE
 * -----------------------
 * Each track row can show either:
 *
 *   AMPLITUDE  the familiar waveform envelope — good for spotting when a
 *              track is active, useless for seeing what pitch it is.
 *
 *   TRAJECTORY frequency plotted against time, line brightness following
 *              amplitude — this is the view that makes a partial legible as
 *              a *tone*: you see the vibrato, the glide, the decay.
 *
 * Trajectory is the default, because "which of these is the note that slides"
 * is the question a user actually has, and the amplitude view cannot answer it.
 *
 * DRAWING CHEAPLY
 * ---------------
 * Never draw more points than the canvas has pixels. A 20-second track at the
 * default hop has ~1700 trajectory points and the row is ~800px wide, so
 * downsample by min/max per column. The server already sends a downsampled
 * trajectory for exactly this reason.
 *
 * WHY SEGMENTS, NOT A PATH
 * ------------------------
 * The trajectory is stroked as a run of short segments rather than one long
 * path. A path has a single globalAlpha for its whole length, so a decaying
 * tone would fade all at once or not at all; per-segment alpha is what makes
 * the decay legible as a decay. It costs one stroke() per point, which at
 * display resolution is a few hundred calls per row — cheap, and paid only
 * when a row is (re)drawn, never per animation frame.
 *
 * WHO OWNS THE TRANSFORM
 * ----------------------
 * The caller sets up device-pixel scaling (canvas.width = cssWidth * dpr, then
 * ctx.scale(dpr, dpr)) because only the caller knows the row's CSS size. These
 * functions therefore never call setTransform: they read the backing-store size
 * off g.canvas and divide out whatever scale is already on the context, so they
 * draw in the same CSS-pixel space the caller set up. save()/restore() is used
 * for style state, which leaves that transform untouched.
 *
 * STATUS: implemented (Phase 5). Contract unchanged from the scaffold.
 */

/** Frequency axis bounds for the trajectory view, in Hz. Log-scaled. */
const FREQ_MIN_HZ = 20;
const FREQ_MAX_HZ = 20000;

/**
 * The spectral ramp, low register to high. Read from CSS at draw time so
 * restyling the app in app.css restyles the canvases too — the tokens are the
 * single source of truth, and a hardcoded palette here would silently diverge
 * from the one the rest of the UI uses.
 */
const SPEC_RAMP_VARS = ['--spec-0', '--spec-1', '--spec-2', '--spec-3', '--spec-4', '--spec-5'];

/**
 * Fallbacks matching app.css, used when there is no document to read from
 * (tests, any headless rendering). Drawing something in the right family beats
 * throwing or drawing invisible black.
 */
const SPEC_RAMP_FALLBACK = ['#6b4fd8', '#4a7fe0', '#35b5d8', '#3fc9a8', '#a8cf5c', '#e8a33d'];

/**
 * The amplitude view is deliberately neutral. It shows *when*, not *what
 * pitch* — colouring it from the spectral ramp would imply a register it is
 * not actually reporting.
 */
const ENVELOPE_VAR = '--paper-dim';
const ENVELOPE_FALLBACK = '#a8a49a';

/**
 * Floor for trajectory line alpha. A partial's tail should fade but never
 * vanish: disappearing ink reads as "the track stopped", which is a different
 * claim than "the track got quiet".
 */
const MIN_ALPHA = 0.12;

/** Trajectory stroke width, in CSS pixels. */
const TRAJECTORY_LINE_WIDTH = 1.5;

/** Radius of the dot drawn for a trajectory point with no neighbour to join. */
const DOT_RADIUS = 1.5;

/**
 * Map a frequency to a vertical position, log-scaled.
 *
 * Log rather than linear because pitch perception is logarithmic: an octave
 * should occupy the same height whether it runs 100-200 Hz or 5000-10000 Hz.
 * A linear axis crushes every musically interesting partial into the bottom
 * few pixels.
 *
 * @param {number} hz
 * @param {number} height
 * @returns {number} y coordinate, 0 at top
 */
export function freqToY(hz, height) {
  const clamped = Math.min(FREQ_MAX_HZ, Math.max(FREQ_MIN_HZ, hz));
  const t = Math.log2(clamped / FREQ_MIN_HZ) / Math.log2(FREQ_MAX_HZ / FREQ_MIN_HZ);
  return height * (1 - t);
}

/**
 * Read a CSS custom property off the document root.
 *
 * Guarded rather than assumed: this module is imported by tests that have no
 * DOM, and a ReferenceError there would fail a test for a reason unrelated to
 * the drawing being tested.
 *
 * @param {string} name e.g. '--spec-2'
 * @param {string} fallback
 * @returns {string}
 */
function readCssVar(name, fallback) {
  if (typeof document === 'undefined' || typeof getComputedStyle !== 'function') return fallback;
  const root = document.documentElement;
  if (!root) return fallback;
  const value = getComputedStyle(root).getPropertyValue(name);
  return value && value.trim() ? value.trim() : fallback;
}

/**
 * The canvas size in the caller's drawing units.
 *
 * g.canvas.width is the backing store — device pixels. If the caller has
 * scaled the context for a retina display, drawing to those numbers would
 * paint at twice the intended size and spill off the row. Divide the scale
 * back out to land in the caller's space. getTransform is feature-detected
 * because not every context (or test double) provides it.
 *
 * @param {CanvasRenderingContext2D} g
 * @returns {{width: number, height: number}}
 */
function canvasSize(g) {
  const canvas = g.canvas;
  if (!canvas) return { width: 0, height: 0 };

  let width = Number(canvas.width) || 0;
  let height = Number(canvas.height) || 0;

  if (typeof g.getTransform === 'function') {
    const m = g.getTransform();
    if (m && Number.isFinite(m.a) && m.a > 0) width /= m.a;
    if (m && Number.isFinite(m.d) && m.d > 0) height /= m.d;
  }
  return { width, height };
}

/**
 * The time the x axis spans.
 *
 * durationS is the track's timeline and is what keeps every row aligned to the
 * same clock — a per-track axis would make two rows uncomparable. It is only
 * ignored when it is missing or nonsensical, in which case falling back to the
 * trajectory's own extent at least draws something honest.
 *
 * @param {Array<[number, number, number]>} trajectory
 * @param {number} durationS
 * @returns {number} seconds, always > 0
 */
function timeSpan(trajectory, durationS) {
  if (Number.isFinite(durationS) && durationS > 0) return durationS;

  let latest = 0;
  for (const frame of trajectory) {
    const t = frame && Number(frame[0]);
    if (Number.isFinite(t) && t > latest) latest = t;
  }
  return latest > 0 ? latest : 1;
}

/**
 * Loudest point on the trajectory. Alpha is relative to this rather than to
 * an absolute scale, so a quiet partial is still fully drawn at its own peak —
 * the view is about the shape of one track's envelope, not its rank in the mix.
 *
 * @param {Array<[number, number, number]>} trajectory
 * @returns {number} 0 when the trajectory carries no usable amplitude
 */
function peakAmplitude(trajectory) {
  let peak = 0;
  for (const frame of trajectory) {
    const a = frame && Number(frame[2]);
    if (Number.isFinite(a) && a > peak) peak = a;
  }
  return peak;
}

/**
 * Line alpha for one amplitude.
 *
 * Square-rooted because perceived brightness is not linear in amplitude: a
 * linear ramp makes a decay drop to near-invisible while it is still clearly
 * audible, which misrepresents the tail. With no amplitude information at all
 * (peak of zero) everything is drawn solid rather than uniformly faint.
 *
 * @param {number} amp
 * @param {number} peak
 * @returns {number} 0..1
 */
function alphaFor(amp, peak) {
  if (!(peak > 0)) return 1;
  const a = Number.isFinite(amp) && amp > 0 ? amp : 0;
  const ratio = Math.min(1, a / peak);
  return MIN_ALPHA + (1 - MIN_ALPHA) * Math.sqrt(ratio);
}

/**
 * The colour for a whole track, from its amplitude-weighted mean frequency.
 *
 * One colour per track, not per point: the ramp's job is to let you scan a
 * 40-row stack and see roughly where each track sits before reading a single
 * label. A per-point gradient would say the same thing the y axis already says
 * and cost the rows their identity. Weighted by amplitude so the register a
 * track is actually heard in wins over a brief inaudible excursion.
 *
 * @param {Array<[number, number, number]>} trajectory
 * @returns {string} a CSS colour
 */
function trackColour(trajectory) {
  let weighted = 0;
  let weight = 0;
  let plain = 0;
  let count = 0;

  for (const frame of trajectory) {
    if (!frame) continue;
    const hz = Number(frame[1]);
    if (!Number.isFinite(hz) || hz <= 0) continue;
    const amp = Number(frame[2]);
    const w = Number.isFinite(amp) && amp > 0 ? amp : 0;
    weighted += hz * w;
    weight += w;
    plain += hz;
    count += 1;
  }

  if (count === 0) return SPEC_RAMP_FALLBACK[0];
  const meanHz = weight > 0 ? weighted / weight : plain / count;

  // Bucket on the same log axis as freqToY, so a track's colour and its
  // vertical position agree about what "high" means.
  const clamped = Math.min(FREQ_MAX_HZ, Math.max(FREQ_MIN_HZ, meanHz));
  const t = Math.log2(clamped / FREQ_MIN_HZ) / Math.log2(FREQ_MAX_HZ / FREQ_MIN_HZ);
  const index = Math.min(SPEC_RAMP_VARS.length - 1, Math.max(0, Math.round(t * (SPEC_RAMP_VARS.length - 1))));

  return readCssVar(SPEC_RAMP_VARS[index], SPEC_RAMP_FALLBACK[index]);
}

/**
 * Stroke one unbroken run of trajectory points.
 *
 * A run of one gets a dot. That happens for a genuinely single-point
 * trajectory, and also for a point isolated between two unusable ones — in
 * both cases the alternative is drawing nothing, which loses evidence that the
 * partial existed there at all.
 *
 * @param {CanvasRenderingContext2D} g
 * @param {Array<{x: number, y: number, amp: number}>} run
 * @param {number} peak
 */
function strokeRun(g, run, peak) {
  if (run.length === 0) return;

  if (run.length === 1) {
    const p = run[0];
    g.globalAlpha = alphaFor(p.amp, peak);
    g.beginPath();
    g.arc(p.x, p.y, DOT_RADIUS, 0, Math.PI * 2);
    g.fill();
    return;
  }

  for (let i = 1; i < run.length; i += 1) {
    const from = run[i - 1];
    const to = run[i];
    // Mean of the two endpoints: the segment spans both, and taking either end
    // alone makes an attack or a release look like a staircase.
    g.globalAlpha = alphaFor((from.amp + to.amp) / 2, peak);
    g.beginPath();
    g.moveTo(from.x, from.y);
    g.lineTo(to.x, to.y);
    g.stroke();
  }
}

/**
 * Draw a partial's frequency trajectory.
 * @param {CanvasRenderingContext2D} g
 * @param {Array<[number, number, number]>} trajectory (time_s, hz, amp)
 * @param {number} durationS
 */
export function drawTrajectory(g, trajectory, durationS) {
  if (!Array.isArray(trajectory) || trajectory.length === 0) return;

  const { width, height } = canvasSize(g);
  if (!(width > 0) || !(height > 0)) return;

  const span = timeSpan(trajectory, durationS);
  const peak = peakAmplitude(trajectory);
  const colour = trackColour(trajectory);

  g.save();
  g.strokeStyle = colour;
  g.fillStyle = colour;
  g.lineWidth = TRAJECTORY_LINE_WIDTH;
  // Round caps let independently stroked segments meet without visible notches
  // at the joints, which at 1.5px would otherwise read as a dotted line.
  g.lineCap = 'round';

  // A point with a non-positive frequency has no position on a log axis — that
  // is a silent frame, not a frequency of zero. Skipping it *and breaking the
  // run* matters: joining across the gap would draw a line through pitches the
  // partial never had.
  let run = [];
  for (const frame of trajectory) {
    const t = frame && Number(frame[0]);
    const hz = frame && Number(frame[1]);
    const amp = frame && Number(frame[2]);

    if (!Number.isFinite(t) || !Number.isFinite(hz) || hz <= 0) {
      strokeRun(g, run, peak);
      run = [];
      continue;
    }

    const x = Math.min(width, Math.max(0, (t / span) * width));
    run.push({ x, y: freqToY(hz, height), amp });
  }
  strokeRun(g, run, peak);

  g.restore();
}

/**
 * Draw an amplitude envelope from a decoded buffer.
 * @param {CanvasRenderingContext2D} g
 * @param {AudioBuffer} buffer
 */
export function drawWaveform(g, buffer) {
  if (!buffer || !(buffer.length > 0) || typeof buffer.getChannelData !== 'function') return;

  const { width, height } = canvasSize(g);
  if (!(width > 0) || !(height > 0)) return;

  // Channel 0 only. Tracks are rendered mono by the synthesiser, and for the
  // rare stereo upload the left channel is a fair envelope — averaging the
  // channels would cancel out-of-phase content and under-report the level.
  const data = buffer.getChannelData(0);
  if (!data || data.length === 0) return;

  // One column per pixel at most. More is invisible work; fewer only when the
  // buffer is shorter than the row is wide, where each sample gets its own
  // column and the columns widen to fill the canvas.
  const columns = Math.max(1, Math.min(Math.floor(width), data.length));
  const columnWidth = width / columns;
  const samplesPerColumn = data.length / columns;
  const mid = height / 2;

  g.save();
  g.strokeStyle = readCssVar(ENVELOPE_VAR, ENVELOPE_FALLBACK);
  g.lineWidth = Math.max(1, columnWidth);

  // One path, one stroke: unlike the trajectory, every column here is the same
  // colour and opacity, so there is nothing to gain from stroking separately.
  g.beginPath();
  for (let col = 0; col < columns; col += 1) {
    const start = Math.floor(col * samplesPerColumn);
    const end = Math.min(data.length, Math.max(start + 1, Math.floor((col + 1) * samplesPerColumn)));

    let min = Infinity;
    let max = -Infinity;
    for (let i = start; i < end; i += 1) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!(min <= max)) continue; // every sample in the column was NaN

    const x = (col + 0.5) * columnWidth;
    const yTop = mid - max * mid;
    const yBottom = mid - min * mid;
    g.moveTo(x, yTop);
    // Silence has min === max and would stroke a zero-length line, which some
    // engines drop entirely. Force a hairline so a silent stretch reads as a
    // centre line rather than as a hole in the canvas.
    g.lineTo(x, Math.max(yBottom, yTop + 0.5));
  }
  g.stroke();

  g.restore();
}

/**
 * Wipe a canvas.
 *
 * Cleared in the caller's units and deliberately over-reaching: under a retina
 * transform the backing-store dimensions describe an area larger than the
 * canvas, and clearRect clips, so over-clearing is free while under-clearing
 * would leave a stale band down the right edge.
 *
 * Both directions have to be covered, which is why this takes the larger of the
 * two candidate sizes. clearRect is interpreted in the caller's units, so the
 * area it wipes is (rect * scale) device pixels:
 *
 *   scale >= 1 (retina)  backing store is the bigger number; clearing to it
 *                        over-reaches, and the excess is clipped away.
 *   scale <  1 (browser zoomed below 100%, where devicePixelRatio is e.g. 0.8)
 *                        the backing store is the *smaller* number, so clearing
 *                        to it wipes only backing*scale device pixels and leaves
 *                        the last (1-scale) of the row untouched. At 80% zoom
 *                        that is a fifth of every track row still showing the
 *                        previous view after a mode toggle.
 *
 * canvasSize is backing/scale, so max(backing, backing/scale) covers the canvas
 * for any scale without ever under-clearing.
 *
 * @param {CanvasRenderingContext2D} g
 */
export function clear(g) {
  const canvas = g.canvas;
  if (!canvas) return;

  const backingWidth = Number(canvas.width) || 0;
  const backingHeight = Number(canvas.height) || 0;
  const { width, height } = canvasSize(g);

  g.clearRect(0, 0, Math.max(backingWidth, width), Math.max(backingHeight, height));
}
