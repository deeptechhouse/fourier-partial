/**
 * app.js — wiring. Upload -> job feed -> workbench.
 *
 * Deliberately thin: this module owns no signal processing and no audio
 * scheduling. It owns the *order* in which the other modules are called, and
 * the DOM those modules are shown through. Anything that deserves a unit test
 * lives in transport.js, waveform.js, or on the server.
 *
 * THE FLOW, END TO END
 * --------------------
 *   1. A file arrives (drop or file picker).
 *   2. POST /api/analyze                -> { job_id, estimated_seconds }
 *   3. WS  /api/jobs/{id}/events        -> progress frames until terminal
 *      (falls back to polling GET /api/jobs/{id} if the socket never opens)
 *   4. GET /api/jobs/{id}/tracks        -> the whole workbench payload
 *   5. audio, by one of two strategies (below) -> register with the Transport
 *   6. Play.
 *
 * STEP 5 HAS TWO STRATEGIES, AND A MEMORY ESTIMATE PICKS ONE
 * ----------------------------------------------------------
 * transport.js schedules every source at one shared AudioContext timestamp, and
 * a track registered *after* play() has begun has no source and stays silent
 * until the next play(). So "press Play and hear the whole mix" requires every
 * buffer that should be audible to be present first. The workbench is rendered
 * immediately either way (the trajectories need no audio) with Play disabled
 * and an honest status line until the audio is ready.
 *
 * EAGER (short files). Fetch every track, decode it, register it. Solo and mute
 * are then pure gain changes with no network in the loop: instant, offline, and
 * exactly what comparing two partials against each other wants.
 *
 * BOUNDED (long files). Fetching everything stops being possible. One track of
 * ten-minute audio is ~106 MB decoded and a long file can yield forty of them,
 * so eager loading would ask the tab for ~4.3 GB and it would die. Instead:
 *
 *   - GET /api/jobs/{id}/mixdown is the sum of every track as ONE buffer, and
 *     with nothing excluded it is byte-identical to the uploaded file. So
 *     "play everything" costs one buffer, not forty, at any duration.
 *   - `?exclude=3,7` is the same trick for muting: the server answers with
 *     everything-except-those, still as one buffer.
 *   - Soloing needs the soloed tracks themselves, so those are fetched on
 *     demand and kept in a small least-recently-used cache.
 *
 * The switch is a memory estimate rather than a duration — see
 * PLAYBACK_MEMORY_BUDGET_BYTES. A phrase never leaves eager mode, so nothing
 * about the good experience regresses for the case this tool is mostly used on.
 *
 * WHAT MUST STAY TRUE IN BOTH MODES
 * ---------------------------------
 * With nothing muted and nothing soloed, what comes out of the speakers is the
 * uploaded file. In eager mode that is transport.js's arithmetic: a unity-gain
 * sum of buffers the server guarantees sum to the original. In bounded mode it
 * is the mixdown endpoint's own guarantee, which is asserted in
 * tests/test_mixdown.py. The one way to break the bounded case would be to play
 * an individually-fetched track *on top of* a mix that already contains it, so
 * tracks fetched for soloing are registered `soloOnly` and are silent unless
 * soloed. That rule lives in transport.js, in one place, deliberately.
 *
 * ERRORS ARE PAGE CONTENT, NOT CONSOLE CONTENT
 * --------------------------------------------
 * Every failure path in here ends at _showError(), which puts the server's own
 * message in a role="alert" banner and returns the UI to the dropzone. A
 * spinner that never resolves is the one outcome this module must never
 * produce, so every await is inside a try/catch that reaches that function.
 *
 * The one deliberate exception is a failed export, which stops at _setBanner()
 * instead: the analysis behind it is still on screen and still playable, so it
 * gets the message without losing the session. Nothing reports a failure only
 * to the console.
 *
 * STATUS: implemented (Phase 5; bounded playback and pitch grouping added in
 * Phase 6). The exported contract is unchanged from the scaffold.
 */

import { Transport, MIX_TRACK_ID } from './transport.js';
import { drawTrajectory, drawWaveform, clear } from './waveform.js';
import { createSettingsPanel } from './settings.js';

// =============================================================================
// Tunables
// =============================================================================

/**
 * ADJUSTABLE. How often to poll GET /api/jobs/{id} when the WebSocket is
 * unavailable. 400 ms is fast enough that the bar still looks live and slow
 * enough that a 20-second analysis costs ~50 requests rather than ~600.
 */
const POLL_INTERVAL_MS = 400;

/**
 * ADJUSTABLE. How long to wait for the progress socket to open before giving
 * up on it and polling instead. A same-origin socket opens in single-digit
 * milliseconds; anything past ~2 s means a proxy is eating the upgrade, and
 * continuing to wait just leaves the user staring at a bar at 0%.
 */
const SOCKET_OPEN_TIMEOUT_MS = 2000;

/**
 * ADJUSTABLE. How many track-audio requests are in flight at once.
 *
 * A trade with no dramatic edges: 1 is needlessly serial on a connection that
 * can do better, and 30 makes the browser queue them anyway while the server
 * renders WAV encodes back to back. 4 keeps the pipe full without turning a
 * 200-track analysis into a self-inflicted denial of service.
 */
const AUDIO_FETCH_CONCURRENCY = 4;

/** Bytes per sample of a decoded AudioBuffer. Web Audio is 32-bit float, always. */
const BYTES_PER_SAMPLE = 4;

/**
 * ADJUSTABLE. How much decoded audio this page will hold before it stops
 * loading every track and switches to the bounded strategy. 600 MB.
 *
 * WHY 600 MB, AND WHY A BUDGET RATHER THAN A DURATION.
 *
 * The quantity that kills the tab is decoded bytes, and that is a product of
 * three things — duration, sample rate and track count — so any single one of
 * them is a poor proxy. A ten-minute file with five tracks is comfortable; a
 * thirty-second file with two hundred is not, and it is the same ~1 GB the
 * design document's section 9 measured. Estimating the product catches both.
 *
 * The number itself is chosen from what a browser tab actually survives. A
 * decoded AudioBuffer lives outside the JavaScript heap, so the ~4 GB heap cap
 * is not the ceiling — the renderer process is, and it is killed well before
 * the machine runs out. 600 MB leaves room for the DOM, forty canvases, the
 * decoder's own scratch space (decodeAudioData holds the encoded bytes AND the
 * decoded output at once, so a decode transiently costs ~1.25x the buffer) and
 * whatever else the user has open. It is deliberately conservative: crossing
 * into bounded mode costs a little latency on solo, whereas guessing too high
 * costs the user the whole session.
 *
 * At the ten-minute ceiling one track is ~106 MB, so this admits the mix plus
 * four cached tracks — see BOUNDED_CACHE_MAX_TRACKS.
 *
 * Raise it if you are targeting machines you control and have measured them.
 */
const PLAYBACK_MEMORY_BUDGET_BYTES = 600 * 1024 * 1024;

/**
 * ADJUSTABLE. Ceiling on how many individually-fetched tracks the bounded
 * strategy keeps decoded at once.
 *
 * The budget above already limits this by size; this is a second limit by
 * count, for the other shape of long file — hundreds of short tracks, where the
 * budget alone would happily cache a hundred of them and spend the time
 * decoding audio nobody asked to hear. Twelve is enough for the comparison this
 * tool is for (A/B a handful of partials) without becoming a second eager mode.
 */
const BOUNDED_CACHE_MAX_TRACKS = 12;

/**
 * ADJUSTABLE. Hard ceiling on how many times one user action may re-reconcile.
 *
 * A reconcile can revert a control it could not honour and run again against
 * the corrected intent, which is a loop, and loops that make network requests
 * need a bound that does not depend on the logic inside them being right. Four
 * is far more than the two the design actually uses (act, then settle).
 */
const APPLY_MAX_PASSES = 4;

/** Fallback text when a response carries no usable message of its own. */
const GENERIC_ERROR = 'Something went wrong. Please try another file.';

/**
 * The one line the interface owes a user whose file went into bounded mode.
 *
 * Kept here rather than buried in the strategy because it is a promise about
 * behaviour, and a user who notices that solo takes a moment deserves to find
 * the reason on screen rather than infer it. Never hidden, never folded away.
 */
const BOUNDED_MODE_NOTE =
  'Long file: playing one combined stream, and loading individual tracks only ' +
  'when you solo them — so solo and mute take a moment.';

// =============================================================================
// Small DOM and formatting helpers
// =============================================================================

/**
 * Build an element.
 *
 * Text always goes in via textContent, never innerHTML. Labels and warnings
 * are generated by our own server rather than by a user, but building markup
 * out of strings that came off the wire is a habit worth not forming — the day
 * one of those strings starts echoing a filename, the escaping is already
 * there.
 *
 * @param {string} tag
 * @param {object} [props] className, textContent, and attributes via attrs:{}
 * @param {Array<Node>} [children]
 * @returns {HTMLElement}
 */
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  if (props.className) node.className = props.className;
  if (props.text !== undefined) node.textContent = String(props.text);
  if (props.attrs) {
    for (const [name, value] of Object.entries(props.attrs)) {
      if (value === null || value === undefined || value === false) continue;
      node.setAttribute(name, value === true ? '' : String(value));
    }
  }
  for (const child of children) {
    if (child) node.appendChild(child);
  }
  return node;
}

/** Seconds, fixed to hundredths — enough resolution to read a transient. */
function formatSeconds(value) {
  const s = Number.isFinite(value) ? Math.max(0, value) : 0;
  return `${s.toFixed(2)} s`;
}

/** A 0..1 ratio as a percentage string. */
function formatPercent(ratio) {
  const r = Number.isFinite(ratio) ? ratio : 0;
  return `${(r * 100).toFixed(1)}%`;
}

/** Frequency, switching to kHz where Hz would need five digits. */
function formatHz(hz) {
  if (!Number.isFinite(hz) || hz <= 0) return 'broadband';
  return hz >= 1000 ? `${(hz / 1000).toFixed(2)} kHz` : `${hz.toFixed(1)} Hz`;
}

/** Bytes as a whole number of megabytes — for status text, not for arithmetic. */
function formatMegabytes(bytes) {
  const mb = Number.isFinite(bytes) && bytes > 0 ? bytes / (1024 * 1024) : 0;
  return `${Math.round(mb)} MB`;
}

/**
 * Decoded bytes one track of this analysis will occupy.
 *
 * duration x sample rate x 4 bytes, mono. The tracks are all rendered over the
 * whole timeline (a partial that dies early is padded with silence, not
 * truncated), so every track costs the same and the mixdown costs one more.
 *
 * A caveat worth knowing: decodeAudioData resamples to the AudioContext's rate,
 * so a 44.1 kHz file decoded into a 48 kHz context is ~9% larger than this says.
 * The estimate is a threshold test, not an allocator, and the budget it is
 * compared against is conservative enough to absorb that.
 *
 * @param {object} summary AnalysisSummary
 * @returns {number} bytes
 */
function trackDecodedBytes(summary) {
  const duration = Number(summary && summary.duration_s);
  const rate = Number(summary && summary.sample_rate);
  if (!Number.isFinite(duration) || !Number.isFinite(rate)) return 0;
  return Math.max(0, duration) * Math.max(0, rate) * BYTES_PER_SAMPLE;
}

/**
 * Decoded bytes the eager strategy would need for this whole analysis.
 * @param {object} summary AnalysisSummary
 * @returns {number} bytes
 */
function estimateDecodedBytes(summary) {
  const count = ((summary && summary.tracks) || []).length;
  return trackDecodedBytes(summary) * count;
}

/**
 * Whether two sets hold the same members.
 *
 * Used to decide whether a fallback actually moved the user's intent. "Did this
 * change anything" is the question that makes the reconcile loop terminate, so
 * it is asked explicitly rather than assumed.
 *
 * @param {Set<number>} a
 * @param {Set<number>} b
 * @returns {boolean}
 */
function sameSet(a, b) {
  if (a.size !== b.size) return false;
  for (const value of a) {
    if (!b.has(value)) return false;
  }
  return true;
}

/**
 * A stable key for a set of excluded track ids.
 *
 * Sorted numerically and de-duplicated, so {7,3} and {3,7,3} produce the same
 * string. That matters because the key is what decides whether the mix we
 * already hold still answers the question being asked — an unsorted key would
 * refetch ~106 MB just because the user muted two tracks in the other order.
 *
 * @param {Iterable<number>} ids
 * @returns {string} e.g. "3,7", or "" for nothing excluded
 */
function excludeKey(ids) {
  return [...new Set(ids)].sort((a, b) => a - b).join(',');
}

/**
 * Decode compressed or uncompressed audio bytes into an AudioBuffer.
 *
 * Both call signatures are supported because decodeAudioData predates
 * promises: older Safari returns undefined and only ever calls the callbacks.
 * Wrapping once here means the rest of the module can simply await it.
 *
 * @param {AudioContext} ctx
 * @param {ArrayBuffer} bytes
 * @returns {Promise<AudioBuffer>}
 */
function decodeAudio(ctx, bytes) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const ok = (buffer) => {
      if (settled) return;
      settled = true;
      resolve(buffer);
    };
    const fail = (err) => {
      if (settled) return;
      settled = true;
      reject(err instanceof Error ? err : new Error('Could not decode that track.'));
    };
    const maybePromise = ctx.decodeAudioData(bytes, ok, fail);
    if (maybePromise && typeof maybePromise.then === 'function') {
      maybePromise.then(ok, fail);
    }
  });
}

// =============================================================================
// API client
// =============================================================================

/**
 * An HTTP failure carrying the status code and a message safe to show a user.
 */
class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/**
 * Every call to the server, in one place.
 *
 * The URLs are relative because the frontend is served as static files by the
 * same FastAPI app that answers /api — so there is no origin to configure, and
 * hardcoding one would break the .app launcher, which picks whatever port
 * happens to be free.
 *
 * This is also the seam the design document's service-abstraction rule asks
 * for: swapping the transport (a different base path, a mock in a test)
 * touches this class and nothing else.
 */
class ApiClient {
  /** @param {string} [base] path prefix for the API, relative to the origin */
  constructor(base = '/api') {
    this._base = base;
  }

  /**
   * Upload one file and start an analysis.
   *
   * The field name is "file" because that is the parameter name the endpoint
   * declares; FastAPI matches multipart parts by name, so a mismatch here
   * produces a 422 that reads like a validation bug rather than a typo.
   *
   * @param {File} file
   * @returns {Promise<{job_id: string, estimated_seconds: number}>}
   */
  async analyze(file, settings) {
    const body = new FormData();
    body.append('file', file, file.name);
    // Only sent when something was actually changed. An empty object would be
    // harmless, but omitting the field entirely keeps the common request
    // identical to what it was before settings existed.
    if (settings && Object.keys(settings).length) {
      body.append('settings', JSON.stringify(settings));
    }
    return this._json(await this._fetch(`${this._base}/analyze`, { method: 'POST', body }));
  }

  /** Run the analysis again on the same audio, with different settings. */
  async reanalyze(jobId, settings) {
    return this._json(
      await this._fetch(`${this._jobPath(jobId)}/reanalyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: settings || {} }),
      })
    );
  }

  /**
   * Poll one job's progress.
   * @param {string} jobId
   * @returns {Promise<{job_id: string, state: string, progress: number, stage: string, message: ?string}>}
   */
  async jobStatus(jobId) {
    return this._json(await this._fetch(this._jobPath(jobId)));
  }

  /**
   * Fetch the completed analysis: every track's metadata plus the diagnostics.
   * @param {string} jobId
   * @returns {Promise<object>} AnalysisSummary
   */
  async tracks(jobId) {
    return this._json(await this._fetch(`${this._jobPath(jobId)}/tracks`));
  }

  /**
   * Fetch one track's rendered PCM as raw bytes.
   *
   * Returned as an ArrayBuffer rather than decoded here because decoding needs
   * an AudioContext, and this class deliberately knows nothing about audio.
   *
   * @param {string} jobId
   * @param {number} trackId
   * @returns {Promise<ArrayBuffer>}
   */
  async trackAudio(jobId, trackId) {
    const response = await this._fetch(`${this._jobPath(jobId)}/tracks/${Number(trackId)}/audio`);
    if (!response.ok) throw await this._error(response);
    return response.arrayBuffer();
  }

  /**
   * Fetch the sum of every track — optionally minus a few — as one WAV.
   *
   * This is the endpoint that makes long files playable at all. The server
   * guarantees that with nothing excluded the bytes are the uploaded file
   * itself, so the browser can hold one buffer where it would otherwise hold
   * forty. `exclude` extends the same trick to muting.
   *
   * The ids are sent as one comma-separated `exclude` parameter because that is
   * the shape the endpoint declares; URLSearchParams handles the encoding, so
   * nothing here builds a query string by concatenation.
   *
   * @param {string} jobId
   * @param {Iterable<number>} [excludeIds]
   * @returns {Promise<ArrayBuffer>}
   */
  async mixdown(jobId, excludeIds = []) {
    const query = new URLSearchParams({ exclude: excludeKey(excludeIds) });
    const response = await this._fetch(`${this._jobPath(jobId)}/mixdown?${query}`);
    if (!response.ok) throw await this._error(response);
    return response.arrayBuffer();
  }

  /**
   * The download URL for one track. Not fetched here — see exportFile.
   * @param {string} jobId
   * @param {number} trackId
   * @param {'wav'|'mid'} format
   * @returns {string}
   */
  /**
   * The download URL for EVERY track at once.
   *
   * Separate from exportUrl because it is a different resource, not the same
   * one with a wildcard: it returns a zip (or, for mid-single, one multi-track
   * file) built server-side, and it can be gigabytes.
   */
  exportAllUrl(jobId, format) {
    const query = new URLSearchParams({ format });
    return `${this._jobPath(jobId)}/export?${query}`;
  }

  /**
   * Fetch the whole-analysis export.
   *
   * Returns the counts the server reports in headers alongside the blob, so
   * the UI can say "39 files, 1 skipped" without unzipping anything. A bulk
   * export that quietly contains fewer files than the user has rows is the
   * failure this feature exists to avoid, so the count travels with the file.
   */
  async exportAll(jobId, format) {
    const response = await this._fetch(this.exportAllUrl(jobId, format));
    if (!response.ok) throw await this._error(response);
    const blob = await response.blob();
    const ext = format === 'mid-single' ? 'mid' : 'zip';
    return {
      blob,
      filename: filenameFrom(response.headers, `fourier_partial_export.${ext}`),
      included: Number(response.headers.get('X-Export-Included') || 0),
      skipped: Number(response.headers.get('X-Export-Skipped') || 0),
    };
  }

  /** How big each export format would be, without building any of them. */
  async exportEstimate(jobId) {
    const response = await this._fetch(`${this._jobPath(jobId)}/export/estimate`);
    if (!response.ok) throw await this._error(response);
    return response.json();
  }

  exportUrl(jobId, trackId, format) {
    const query = new URLSearchParams({ format });
    return `${this._jobPath(jobId)}/tracks/${Number(trackId)}/export?${query}`;
  }

  /**
   * Fetch one track's export as bytes, with the filename the server chose.
   *
   * Fetched rather than navigated to. The export endpoint answers a *success*
   * with Content-Disposition: attachment, which a browser downloads without
   * leaving the page — but a 404 (expired session) or a 400 answers JSON with
   * no disposition at all, and pointing window.location at that navigates the
   * whole workbench away. Going through fetch means a failed export raises the
   * same ApiError as every other call and the page survives it.
   *
   * @param {string} jobId
   * @param {number} trackId
   * @param {'wav'|'mid'} format
   * @returns {Promise<{blob: Blob, filename: string}>}
   */
  async exportFile(jobId, trackId, format) {
    const response = await this._fetch(this.exportUrl(jobId, trackId, format));
    if (!response.ok) throw await this._error(response);
    const blob = await response.blob();
    const fallback = `track${Number(trackId)}.${format === 'mid' ? 'mid' : 'wav'}`;
    return { blob, filename: filenameFrom(response.headers, fallback) };
  }

  /**
   * The progress socket's URL.
   *
   * Built from window.location rather than hardcoded so it follows the page:
   * same host, same port, and wss:// when the page itself is https, which is
   * what a browser requires of a secure page.
   *
   * @param {string} jobId
   * @returns {string}
   */
  eventsUrl(jobId) {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${scheme}//${window.location.host}${this._jobPath(jobId)}/events`;
  }

  /** @param {string} jobId */
  _jobPath(jobId) {
    return `${this._base}/jobs/${encodeURIComponent(jobId)}`;
  }

  /**
   * fetch(), with a network failure turned into an ApiError.
   *
   * A dropped connection rejects with a bare TypeError whose message differs
   * per browser and means nothing to a user. Converting it here means every
   * caller has exactly one error type to handle.
   */
  async _fetch(url, init) {
    try {
      return await fetch(url, init);
    } catch {
      throw new ApiError('Could not reach the server. Is it still running?', 0);
    }
  }

  /** Parse a JSON response, or raise the server's own error message. */
  async _json(response) {
    if (!response.ok) throw await this._error(response);
    try {
      return await response.json();
    } catch {
      throw new ApiError('The server sent a response this page could not read.', response.status);
    }
  }

  /**
   * Turn a non-2xx response into an ApiError carrying user-facing text.
   *
   * FastAPI's HTTPException serialises as {"detail": "..."} and its request
   * validation errors as {"detail": [{msg, loc}, ...]}. Both shapes are handled
   * so the user sees the server's actual explanation ("That file is too large
   * (limit 50 MB)") rather than a status code we re-worded from a lookup table.
   * The table is only the fallback for responses that carry no detail at all.
   */
  async _error(response) {
    let detail = null;
    try {
      const body = await response.json();
      if (typeof body?.detail === 'string') {
        detail = body.detail;
      } else if (Array.isArray(body?.detail)) {
        detail = body.detail
          .map((item) => (item && typeof item.msg === 'string' ? item.msg : null))
          .filter(Boolean)
          .join('; ');
      }
    } catch {
      detail = null;
    }
    return new ApiError(detail || fallbackMessage(response.status), response.status);
  }
}

/**
 * The filename a Content-Disposition header asks for.
 *
 * The server already picks a good name (track007_A4.wav) and it is the one the
 * user would have got from a plain navigation, so honouring it keeps the
 * fetch-based download indistinguishable from the browser's own. RFC 5987's
 * `filename*` form is checked first because it is the one that survives
 * non-ASCII; the quoted form is the fallback, and a missing header is not an
 * error — the caller always supplies a usable default.
 *
 * @param {Headers} headers
 * @param {string} fallback
 * @returns {string}
 */
function filenameFrom(headers, fallback) {
  const raw = headers && typeof headers.get === 'function' ? headers.get('content-disposition') : null;
  if (!raw) return fallback;

  const encoded = /filename\*=\s*UTF-8''([^;]+)/i.exec(raw);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1].trim());
    } catch {
      // Malformed percent-encoding. Fall through to the plain form.
    }
  }

  const plain = /filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;]+)/i.exec(raw);
  const name = plain && (plain[1] || plain[2]);
  return name ? name.trim() : fallback;
}

/**
 * A sentence for a status code that arrived without a message body.
 * @param {number} status
 * @returns {string}
 */
function fallbackMessage(status) {
  switch (status) {
    case 400:
      return 'The server could not use that file.';
    case 404:
      return 'That analysis session has expired. Please upload again.';
    case 409:
      return 'That analysis is still running.';
    case 413:
      return 'That file is too large for the server to accept.';
    case 415:
      return 'That file type is not supported. Try WAV, FLAC, AIFF, or MP3.';
    case 422:
      return 'The server rejected that upload.';
    case 429:
      return 'Too many analyses started. Please wait a moment and try again.';
    case 500:
      return 'The server hit an unexpected error. Please try again.';
    default:
      return GENERIC_ERROR;
  }
}

// =============================================================================
// Job feed: socket first, polling second
// =============================================================================

/**
 * Progress updates for one job, from whichever channel works.
 *
 * The socket is preferred because the server pushes only on change, so an
 * analysis that sits in one stage for two seconds costs one frame instead of
 * twenty polls. But a socket is the part of this stack most likely to be
 * broken by something outside the app — a corporate proxy that drops the
 * Upgrade header, a browser extension, an origin policy — and the failure is
 * silent: the socket simply never opens and the bar never moves.
 *
 * So the fallback is not an error path, it is a second supported channel. Both
 * deliver the identical JSON shape, so everything downstream is unaware of
 * which one is running.
 */
class JobFeed {
  /**
   * @param {ApiClient} api
   * @param {string} jobId
   * @param {{onUpdate: Function, onDone: Function, onError: Function}} handlers
   */
  constructor(api, jobId, handlers) {
    this._api = api;
    this._jobId = jobId;
    this._handlers = handlers;
    this._socket = null;
    this._openTimer = null;
    this._pollTimer = null;
    this._polling = false;
    /** Guards against a slow poll overlapping the next interval tick. */
    this._inFlight = false;
    /** Latched once the job reaches a terminal state, or once stop() is called.
     *  Every callback checks it, so a late socket close or an in-flight poll
     *  cannot deliver a second "complete" after the workbench has been built. */
    this._settled = false;
  }

  /** Begin listening. Safe to call once. */
  start() {
    this._openSocket();
  }

  /** Stop listening and release both channels. */
  stop() {
    this._settled = true;
    this._clearTimers();
    if (this._socket) {
      // Drop the handlers before closing: close() fires onclose, and the
      // onclose handler's job is to decide whether to fall back to polling.
      // Leaving it attached would start a poll loop for a job we are done with.
      this._socket.onopen = null;
      this._socket.onmessage = null;
      this._socket.onerror = null;
      this._socket.onclose = null;
      try {
        this._socket.close();
      } catch {
        // Already closing or closed. Nothing to do.
      }
      this._socket = null;
    }
  }

  _clearTimers() {
    if (this._openTimer !== null) {
      clearTimeout(this._openTimer);
      this._openTimer = null;
    }
    if (this._pollTimer !== null) {
      clearInterval(this._pollTimer);
      this._pollTimer = null;
    }
  }

  _openSocket() {
    if (typeof WebSocket !== 'function') {
      this._startPolling();
      return;
    }

    let socket;
    try {
      socket = new WebSocket(this._api.eventsUrl(this._jobId));
    } catch {
      // Constructing a WebSocket throws synchronously on a malformed URL or a
      // blocking Content-Security-Policy. Either way, polling still works.
      this._startPolling();
      return;
    }
    this._socket = socket;

    // The one failure a socket does not report: it never opens and never
    // errors, it just sits in CONNECTING. This timer is what makes that case
    // recoverable instead of a permanently stalled progress bar.
    this._openTimer = setTimeout(() => {
      if (this._settled || socket.readyState === WebSocket.OPEN) return;
      this._startPolling();
    }, SOCKET_OPEN_TIMEOUT_MS);

    socket.onopen = () => {
      if (this._openTimer !== null) {
        clearTimeout(this._openTimer);
        this._openTimer = null;
      }
    };

    socket.onmessage = (event) => {
      let status;
      try {
        status = JSON.parse(event.data);
      } catch {
        return; // A frame we cannot parse is not worth failing the job over.
      }
      this._handle(status);
    };

    socket.onerror = () => {
      // Fires before onclose on most engines and carries no useful detail.
      // The close handler below owns the decision.
    };

    socket.onclose = () => {
      // The server closes the socket as soon as the job is terminal, which is
      // the normal ending — _settled is already true and there is nothing to
      // do. A close *before* that means we lost the channel mid-analysis, and
      // the job itself is still running server-side, so pick it up by polling.
      if (this._settled) return;
      this._startPolling();
    };
  }

  _startPolling() {
    if (this._settled || this._polling) return;
    this._polling = true;
    this._clearTimers();

    // Poll immediately as well as on the interval, so switching channels does
    // not cost the user a visible pause at whatever percentage we stalled on.
    const poll = async () => {
      if (this._settled || this._inFlight) return;
      this._inFlight = true;
      try {
        this._handle(await this._api.jobStatus(this._jobId));
      } catch (err) {
        // A failed poll is fatal in a way a failed socket is not: there is no
        // third channel to fall back to, so the user has to be told.
        this._settled = true;
        this._clearTimers();
        this._handlers.onError(err instanceof ApiError ? err.message : GENERIC_ERROR);
      } finally {
        this._inFlight = false;
      }
    };

    this._pollTimer = setInterval(poll, POLL_INTERVAL_MS);
    poll();
  }

  /**
   * Route one status frame. Identical shape from either channel.
   * @param {{state: string, progress: number, stage: string, message: ?string}} status
   */
  _handle(status) {
    if (this._settled || !status) return;
    this._handlers.onUpdate(status);

    if (status.state === 'complete') {
      this.stop();
      this._handlers.onDone(status);
    } else if (status.state === 'failed') {
      this.stop();
      this._handlers.onError(status.message || 'The analysis failed.');
    }
  }
}

// =============================================================================
// One track row
// =============================================================================

/**
 * A row in the workbench: mute, solo, label, canvas, exports.
 *
 * The row owns its own DOM and its own drawing, but not its own audio state —
 * mute and solo are reported upward and applied by the Transport, then
 * reflected back onto the buttons. Keeping the aria-pressed attribute a
 * *reflection* of transport state rather than a second copy of it is what stops
 * the two from disagreeing after "Clear solos".
 */
class TrackRow {
  /**
   * @param {object} track TrackSummary from GET /api/jobs/{id}/tracks
   * @param {{durationS: number, onMute: Function, onSolo: Function,
   *          exportUrlFor: Function, inGroup?: boolean}} deps
   */
  constructor(track, deps) {
    this._track = track;
    this._durationS = deps.durationS;
    this._buffer = null;
    this._mode = 'trajectory';
    /** Latched when this track's PCM never arrived. See setAudioFailed. */
    this._audioFailed = false;
    this._build(deps);
  }

  /** @returns {number} */
  get trackId() {
    return this._track.track_id;
  }

  /** @returns {HTMLElement} */
  get element() {
    return this._root;
  }

  /** The bare note name this row groups under, e.g. "G4". Empty for the residual. */
  get pitch() {
    return typeof this._track.pitch === 'string' ? this._track.pitch : '';
  }

  /** Share of total energy, 0..1. */
  get energyRatio() {
    const ratio = Number(this._track.energy_ratio);
    return Number.isFinite(ratio) ? ratio : 0;
  }

  /** The name this row shows, falling back to the track number. */
  get label() {
    return this._track.label || `Track ${this._track.track_id}`;
  }

  /** This track's (time_s, hz, amplitude) triples. A group row overlays these. */
  get trajectory() {
    return Array.isArray(this._track.trajectory) ? this._track.trajectory : [];
  }

  /** Amplitude-weighted mean frequency, in Hz. Zero for the residual. */
  get meanFrequencyHz() {
    const hz = Number(this._track.mean_frequency_hz);
    return Number.isFinite(hz) ? hz : 0;
  }

  /** When this track starts and stops sounding, in seconds. */
  get startTimeS() {
    const value = Number(this._track.start_time_s);
    return Number.isFinite(value) ? value : 0;
  }

  /** @see startTimeS */
  get endTimeS() {
    const value = Number(this._track.end_time_s);
    return Number.isFinite(value) ? value : 0;
  }

  /**
   * Whether this row's audio was asked for and could not be had.
   *
   * Read by the playback strategies to decide whether the row's controls can do
   * anything: in eager mode a row needs a decoded buffer, in bounded mode it
   * needs only to not have failed, because its audio is fetched on demand.
   *
   * @returns {boolean}
   */
  get audioFailed() {
    return this._audioFailed;
  }

  /**
   * Whether this row's PCM actually arrived.
   *
   * The Transport only knows track ids it has been handed a buffer for, so this
   * is what decides whether the row's mute and solo buttons can do anything.
   *
   * @returns {boolean}
   */
  get hasAudio() {
    // Boolean(), not `!== null`: a decoder that resolves with nothing would
    // otherwise read as "has audio" and re-enable a row the Transport cannot
    // act on, which is the exact state this getter exists to keep off screen.
    return Boolean(this._buffer);
  }

  /** Reflect the transport's mute state onto the button. */
  setMuted(muted) {
    this._muteButton.setAttribute('aria-pressed', muted ? 'true' : 'false');
  }

  /** Reflect the transport's solo state onto the button. */
  setSoloed(soloed) {
    this._soloButton.setAttribute('aria-pressed', soloed ? 'true' : 'false');
  }

  /**
   * Enable or disable this row's mute and solo buttons.
   *
   * Disabled until the audio is registered with the Transport, because
   * setMuted on an id the Transport has never seen is a documented no-op — a
   * button that looks live and does nothing is worse than one that admits it
   * is not ready yet.
   *
   * A row whose audio *failed* is never playable, whatever the rest of the
   * workbench is doing — the Transport has no track under this id at all, so
   * re-enabling it when the other tracks finish loading would hand back the
   * very no-op button this method exists to prevent.
   *
   * @param {boolean} playable
   */
  setPlayable(playable) {
    const usable = Boolean(playable) && !this._audioFailed;
    this._muteButton.disabled = !usable;
    this._soloButton.disabled = !usable;
  }

  /**
   * Hand the row its decoded audio.
   *
   * Only needed for the amplitude view — the trajectory comes from the JSON
   * summary and is drawable the moment the row exists, which is why the
   * workbench appears before a single byte of PCM has arrived.
   *
   * @param {AudioBuffer} buffer
   */
  setBuffer(buffer) {
    this._buffer = buffer;
    this._root.dataset.audio = 'ready';
    // Arriving audio can change which of the two views is on screen (see
    // _showsTrajectory), and role="img" means the accessible name IS the
    // picture — so the name has to be re-derived, not just the pixels.
    this._refreshCanvasLabel();
    if (this._mode === 'amplitude' || !this._hasTrajectory()) this.draw();
  }

  /**
   * Give up this row's decoded audio.
   *
   * The bounded strategy caches only a few tracks at a time, and evicting one
   * from that cache frees nothing while this row still references the same
   * AudioBuffer — the buffer is the hundred megabytes. So eviction reaches in
   * here, exactly as it reaches into the Transport.
   *
   * Not a failure: the row stays fully usable, and soloing it again simply
   * fetches it again.
   */
  clearBuffer() {
    if (!this._buffer) return;
    this._buffer = null;
    if (!this._audioFailed) delete this._root.dataset.audio;
    this._refreshCanvasLabel();
    this.draw();
  }

  /** Mark this row's audio as unavailable, and say so where it is visible. */
  setAudioFailed() {
    this._audioFailed = true;
    this._root.dataset.audio = 'failed';
    this._meta.textContent = `${this._metaText()} · audio unavailable`;
    // The Transport never received this track, so setMuted and setSoloed on
    // its id are documented no-ops. Leaving the buttons live would let them
    // show "soloed" while the mix is unchanged — a control that reports a
    // state the audio does not have is worse than one that is plainly off.
    this.setPlayable(false);
  }

  /**
   * @param {'trajectory'|'amplitude'} mode
   */
  setViewMode(mode) {
    this._mode = mode;
    this._refreshCanvasLabel();
    this.draw();
  }

  /**
   * Re-derive the canvas's accessible name from what it is about to draw.
   *
   * role="img" means the accessible name IS the picture, so it has to follow
   * what is actually drawn rather than what was drawn when the row was built.
   * Three things can change that: the view toggle, audio arriving, and audio
   * being evicted — hence one function all three call.
   */
  _refreshCanvasLabel() {
    this._canvas.setAttribute('aria-label', this._canvasLabel());
  }

  /**
   * Size the canvas for the display and redraw it.
   *
   * WHO OWNS THE TRANSFORM: waveform.js documents that the *caller* sets up
   * device-pixel scaling, because only the caller knows the row's CSS size. So
   * that happens here: match the backing store to the physical pixels, then
   * scale the context so the drawing code can keep thinking in CSS pixels.
   * Skipping this is what makes canvas text and 1.5px lines look furry on a
   * retina display.
   */
  draw() {
    const canvas = this._canvas;
    const rect = canvas.getBoundingClientRect();
    // Zero-sized means the row is not laid out yet (or is inside a hidden
    // section). Drawing into a 0x0 backing store is silently discarded, so
    // bail and let the caller redraw once it is visible.
    if (!(rect.width > 0) || !(rect.height > 0)) return;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    // Assigning width/height wipes the canvas *and* resets its transform, so
    // only do it on an actual size change — otherwise every redraw would flash.
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const g = canvas.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    clear(g);

    // The residual has no single frequency and normally no trajectory at all,
    // so it falls through to the amplitude view by itself — which is the
    // honest view for it. Section 8 of the design document: colouring a
    // broadband track from the frequency ramp would imply a register it does
    // not have.
    if (this._showsTrajectory()) {
      drawTrajectory(g, this._track.trajectory, this._durationS);
    } else if (this._buffer) {
      drawWaveform(g, this._buffer);
    }
  }

  _hasTrajectory() {
    return Array.isArray(this._track.trajectory) && this._track.trajectory.length > 0;
  }

  /**
   * Which of the two views this row is actually drawing right now.
   *
   * Asked for the drawing and for the accessible name from the same place, so
   * the label cannot claim a view the canvas is not showing — the residual has
   * no trajectory and therefore never draws one, whatever the global toggle
   * says.
   *
   * The `!this._buffer` clause is the other half of the same honesty. The
   * amplitude view is drawn from decoded audio, and in bounded mode most rows
   * have none — their audio is only fetched if you solo them. Without this the
   * amplitude toggle would blank forty canvases and say nothing about why;
   * with it, a row that cannot draw an envelope keeps drawing the trajectory it
   * does have, and the accessible name (which comes through here too) says
   * "frequency over time" rather than claiming an envelope that is not there.
   */
  _showsTrajectory() {
    if (!this._hasTrajectory()) return false;
    return this._mode === 'trajectory' || !this._buffer;
  }

  /** The canvas's accessible name, describing the view currently drawn. */
  _canvasLabel() {
    const track = this._track;
    const label = track.label || `Track ${track.track_id}`;
    const what = this._showsTrajectory() ? 'frequency over time' : 'amplitude over time';
    return (
      `${label}: ${what}, ` +
      `${formatSeconds(track.start_time_s)} to ${formatSeconds(track.end_time_s)}`
    );
  }

  /** The one-line summary under the label. */
  _metaText() {
    const track = this._track;
    const parts = [formatPercent(track.energy_ratio), formatHz(track.mean_frequency_hz)];
    if (track.partial_count > 0) {
      parts.push(`${track.partial_count} partial${track.partial_count === 1 ? '' : 's'}`);
    }
    return parts.join(' · ');
  }

  /**
   * Build the row's DOM.
   * @param {{onMute: Function, onSolo: Function, exportUrlFor: Function}} deps
   */
  _build(deps) {
    const track = this._track;
    const label = track.label || `Track ${track.track_id}`;

    // Real <button>s with aria-pressed, not styled divs: a toggle button is a
    // solved accessibility problem and re-solving it with role/tabindex/keydown
    // is how the Space key stops working.
    this._muteButton = el('button', {
      className: 'track-toggle mute',
      text: 'M',
      attrs: { type: 'button', 'aria-pressed': 'false', 'aria-label': `Mute ${label}` },
    });
    this._muteButton.addEventListener('click', () => {
      deps.onMute(this.trackId, this._muteButton.getAttribute('aria-pressed') !== 'true');
    });

    this._soloButton = el('button', {
      className: 'track-toggle solo',
      text: 'S',
      attrs: { type: 'button', 'aria-pressed': 'false', 'aria-label': `Solo ${label}` },
    });
    this._soloButton.addEventListener('click', () => {
      deps.onSolo(this.trackId, this._soloButton.getAttribute('aria-pressed') !== 'true');
    });

    this._meta = el('span', { className: 'track-meta', text: this._metaText() });

    this._canvas = el('canvas', {
      attrs: {
        // The canvas is a picture of data that is already stated in text one
        // column to the left, so it is labelled rather than described twice.
        // The name comes from _canvasLabel so it names the view that will
        // actually be drawn — the residual has no trajectory and starts on the
        // amplitude view even though the global default is the trajectory.
        role: 'img',
        'aria-label': this._canvasLabel(),
      },
    });

    const wav = el('button', {
      className: 'track-export',
      text: 'wav',
      attrs: { type: 'button', 'aria-label': `Export ${label} as WAV` },
    });
    wav.addEventListener('click', () => deps.exportUrlFor(this.trackId, 'wav'));

    // The residual is broadband noise: writing it as notes would fabricate
    // information rather than convert it, so the server refuses and the button
    // is disabled here rather than failing after the click. The reason is in
    // the accessible name, because a disabled control cannot explain itself.
    const midEnabled = Boolean(track.exportable_as_midi);
    const mid = el('button', {
      className: 'track-export',
      text: 'mid',
      attrs: {
        type: 'button',
        disabled: !midEnabled,
        title: midEnabled ? null : 'Broadband noise has no notes to export as MIDI.',
        'aria-label': midEnabled
          ? `Export ${label} as MIDI`
          : `MIDI export unavailable for ${label}: broadband noise has no notes`,
      },
    });
    if (midEnabled) mid.addEventListener('click', () => deps.exportUrlFor(this.trackId, 'mid'));

    // `is-child` only changes how the row is indented and dimmed. It is not a
    // second kind of row: a track inside a pitch group is the same track with
    // the same controls and the same exports, which is the point of grouping.
    this._root = el(
      'div',
      {
        className: deps.inGroup ? 'track-row is-child' : 'track-row',
        attrs: { role: 'listitem', 'data-kind': track.kind, 'data-track-id': track.track_id },
      },
      [
        el('div', { className: 'track-controls' }, [this._muteButton, this._soloButton]),
        el('div', { className: 'track-id-cell' }, [
          el('span', { className: 'track-label', text: label, attrs: { title: label } }),
          this._meta,
        ]),
        el('div', { className: 'track-canvas' }, [
          this._canvas,
          // The playhead. One element per row, but all of them read the same
          // --playhead custom property off the list container, so the animation
          // loop below does a single style write per frame no matter how many
          // rows there are.
          el('span', { className: 'track-cursor', attrs: { 'aria-hidden': 'true' } }),
        ]),
        el('div', { className: 'track-exports' }, [wav, mid]),
      ]
    );
  }
}

// =============================================================================
// One pitch group
// =============================================================================

/**
 * A collapsible parent for every occurrence of one pitch.
 *
 * WHY THIS EXISTS. A ten-minute recording plays the same note many times, and
 * each occurrence is legitimately its own track — they are separate notes, not
 * a mistake to be merged. But as a flat list that reads as forty near-identical
 * rows ("G4 -1c + 25 harmonics", "G4 + 58 harmonics", "G4 -2c + 33 harmonics")
 * and the list stops being scannable. Collapsing them under one "G4" row that
 * says how many there are, and opening on request, keeps both facts: the note
 * is one thing, and it happened seven times.
 *
 * WHAT IT IS NOT. It is not a new kind of audio. The group has no track of its
 * own on the server and no export of its own; its mute and solo simply fan out
 * to its members, and every member row underneath is still independently
 * mutable, soloable and exportable. Nothing is reachable only through the
 * group, which is what makes collapsing it safe.
 *
 * A GROUP OF ONE IS NOT A GROUP. Pitches that occur once render as an ordinary
 * row — see App.renderWorkbench. A disclosure triangle you must open to find a
 * single row is ceremony, and on a short file it would be every row.
 */
class TrackGroupRow {
  /**
   * @param {string} pitch e.g. "G4"
   * @param {Array<TrackRow>} rows the occurrence rows, in display order
   * @param {{durationS: number, onMute: Function, onSolo: Function}} deps
   */
  constructor(pitch, rows, deps) {
    this._pitch = pitch;
    this._rows = rows;
    this._durationS = deps.durationS;
    this._expanded = false;
    this._build(deps);
  }

  /** @returns {HTMLElement} */
  get element() {
    return this._root;
  }

  /** Every track id this group speaks for. @returns {Array<number>} */
  get trackIds() {
    return this._rows.map((row) => row.trackId);
  }

  /**
   * Combined share of total energy.
   *
   * Summed rather than averaged: the occurrences are disjoint in time, so their
   * energies add, and the sum is what tells you how much of the recording this
   * note actually is. It is also what the list is ordered by.
   *
   * @returns {number} 0..1
   */
  get energyRatio() {
    return this._rows.reduce((total, row) => total + row.energyRatio, 0);
  }

  /**
   * Reflect the current mute and solo picture onto the group's two toggles.
   *
   * Three states, not two, because a group can be half muted — and saying so is
   * the whole reason the parent is trustworthy. aria-pressed="mixed" is the ARIA
   * value for exactly this, so a screen reader says "partially pressed" without
   * any invented markup. The dashed border in app.css is the non-colour cue
   * that carries the same fact visually (WCAG 1.4.1).
   *
   * @param {Set<number>} muted
   * @param {Set<number>} soloed
   */
  refresh(muted, soloed) {
    this._applyAggregate(this._muteButton, muted, 'Mute', 'Unmute');
    this._applyAggregate(this._soloButton, soloed, 'Solo', 'Unsolo');
  }

  /**
   * Enable or disable the group's toggles.
   * @param {boolean} playable
   */
  setPlayable(playable) {
    const usable = Boolean(playable);
    this._muteButton.disabled = !usable;
    this._soloButton.disabled = !usable;
  }

  /**
   * Show or hide the occurrence rows.
   *
   * The children are hidden with the `hidden` attribute rather than moved out
   * of the DOM, so their canvases keep their pixels and their buttons keep
   * their state across a collapse. The cost is that a hidden canvas measures
   * 0x0 and cannot be drawn, which is why expanding asks the caller to redraw.
   *
   * @param {boolean} expanded
   */
  setExpanded(expanded) {
    this._expanded = Boolean(expanded);
    this._childList.hidden = !this._expanded;
    this._root.dataset.expanded = this._expanded ? 'true' : 'false';
    this._toggleButton.setAttribute('aria-expanded', this._expanded ? 'true' : 'false');
    this._toggleButton.setAttribute('aria-label', this._toggleLabel());
    this._toggleGlyph.textContent = this._expanded ? '▾' : '▸';
  }

  /**
   * Draw every occurrence's trajectory onto one canvas.
   *
   * The honest picture of a group is all of its members at once: the note sits
   * at the same height each time, so the canvas shows you where in the piece
   * that note occurs, which is precisely what the parent row is claiming to
   * summarise. Deliberately drawTrajectory once per member rather than a new
   * drawing primitive — the members are drawn by the same code that draws them
   * individually, so the parent cannot disagree with its own children.
   *
   * There is no amplitude view here: a group has no single decoded buffer to
   * take an envelope from, and inventing one by summing would claim a track
   * that does not exist. The accessible name says which view this is.
   */
  draw() {
    const canvas = this._canvas;
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0) || !(rect.height > 0)) return;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const g = canvas.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    clear(g);

    // One clear, then every member on top of it. drawTrajectory save()s and
    // restore()s its own style state, so the calls do not contaminate each
    // other and the order does not matter.
    for (const row of this._rows) {
      drawTrajectory(g, row.trajectory, this._durationS);
    }
  }

  /**
   * Set one toggle from how many of the group's members are in `active`.
   *
   * @param {HTMLButtonElement} button
   * @param {Set<number>} active
   * @param {string} onVerb what pressing it will do when it is not fully on
   * @param {string} offVerb what pressing it will do when it is fully on
   */
  _applyAggregate(button, active, onVerb, offVerb) {
    const ids = this.trackIds;
    const count = ids.filter((id) => active.has(id)).length;
    const state = count === 0 ? 'false' : count === ids.length ? 'true' : 'mixed';
    button.setAttribute('aria-pressed', state);
    // The accessible name says what the NEXT press does, because that is what a
    // user is deciding about. Pressing a partly-muted group mutes the rest,
    // which is why "mixed" reads the same as "off" here.
    const verb = state === 'true' ? offVerb : onVerb;
    button.setAttribute('aria-label', `${verb} all ${ids.length} ${this._pitch} tracks`);
  }

  /** The disclosure button's accessible name, describing what pressing it does. */
  _toggleLabel() {
    const n = this._rows.length;
    return this._expanded
      ? `Hide the ${n} ${this._pitch} tracks`
      : `Show the ${n} ${this._pitch} tracks`;
  }

  /** The one-line summary under the pitch name. */
  _metaText() {
    const n = this._rows.length;
    // The mean frequency of the first occurrence stands for the group: they are
    // the same note by construction, so they agree to within a few cents, and
    // averaging would only add a digit of false precision.
    return [
      `${n} occurrence${n === 1 ? '' : 's'}`,
      formatPercent(this.energyRatio),
      formatHz(this._rows[0] ? this._rows[0].meanFrequencyHz : 0),
    ].join(' · ');
  }

  /** The canvas's accessible name. Describes the overlay, not one track. */
  _canvasLabel() {
    const n = this._rows.length;
    return (
      `${this._pitch}: frequency over time for all ${n} occurrence${n === 1 ? '' : 's'}, ` +
      `overlaid on one timeline`
    );
  }

  /**
   * Build the group's DOM.
   * @param {{onMute: Function, onSolo: Function}} deps
   */
  _build(deps) {
    const pitch = this._pitch;
    const ids = this.trackIds;

    // A real <button> with aria-expanded and aria-controls, not a styled span:
    // disclosure is a solved accessibility problem, and Enter, Space and the
    // screen reader's own "expand" affordance all come free with the element.
    this._toggleGlyph = el('span', { text: '▸', attrs: { 'aria-hidden': 'true' } });
    // A DOM id has to be unique and must not start with a digit; the pitch can
    // be "A#4" and the residual's neighbours can be anything, so the id is
    // built from the member track ids, which are unique integers by definition.
    this._childListId = `group-${ids.join('-')}-tracks`;
    this._toggleButton = el(
      'button',
      {
        className: 'track-disclosure',
        attrs: {
          type: 'button',
          'aria-expanded': 'false',
          'aria-controls': this._childListId,
          'aria-label': this._toggleLabel(),
        },
      },
      [this._toggleGlyph]
    );
    this._toggleButton.addEventListener('click', () => {
      this.setExpanded(!this._expanded);
      // Children that were hidden had zero-sized canvases and could not draw.
      // The caller owns redrawing because it also owns the resize observer.
      if (typeof deps.onToggle === 'function') deps.onToggle(this);
    });

    this._muteButton = el('button', {
      className: 'track-toggle mute',
      text: 'M',
      attrs: { type: 'button', 'aria-pressed': 'false', 'aria-label': `Mute all ${ids.length} ${pitch} tracks` },
    });
    this._muteButton.addEventListener('click', () => {
      // Derived from the buttons' own state, which refresh() keeps as a
      // reflection of the App's intent: a fully muted group unmutes, anything
      // else mutes the rest. That makes a half-muted group one press from tidy.
      const fullyOn = this._muteButton.getAttribute('aria-pressed') === 'true';
      deps.onMute(ids, !fullyOn);
    });

    this._soloButton = el('button', {
      className: 'track-toggle solo',
      text: 'S',
      attrs: { type: 'button', 'aria-pressed': 'false', 'aria-label': `Solo all ${ids.length} ${pitch} tracks` },
    });
    this._soloButton.addEventListener('click', () => {
      const fullyOn = this._soloButton.getAttribute('aria-pressed') === 'true';
      deps.onSolo(ids, !fullyOn);
    });

    this._canvas = el('canvas', {
      attrs: { role: 'img', 'aria-label': this._canvasLabel() },
    });

    const head = el('div', { className: 'track-row track-group-head' }, [
      el('div', { className: 'track-controls' }, [this._muteButton, this._soloButton]),
      el('div', { className: 'track-id-cell is-group' }, [
        this._toggleButton,
        el('div', { className: 'track-id-text' }, [
          el('span', { className: 'track-label', text: pitch, attrs: { title: pitch } }),
          el('span', { className: 'track-meta', text: this._metaText() }),
        ]),
      ]),
      el('div', { className: 'track-canvas' }, [
        this._canvas,
        el('span', { className: 'track-cursor', attrs: { 'aria-hidden': 'true' } }),
      ]),
      // The export column is deliberately empty. There is no server endpoint
      // that exports a group, and a button that cannot work is worse than no
      // button; the occurrences underneath each carry their own.
      el('div', { className: 'track-exports' }),
    ]);

    // A nested list inside the group's own list item. That is what ARIA's list
    // semantics are for, and it means a screen reader announces "3 items"
    // inside "G4" rather than flattening the occurrences into the main list.
    this._childList = el(
      'div',
      {
        className: 'track-group-children',
        attrs: { role: 'list', id: this._childListId, hidden: true, 'aria-label': `${pitch} tracks` },
      },
      this._rows.map((row) => row.element)
    );

    this._root = el(
      'div',
      {
        className: 'track-group',
        attrs: { role: 'listitem', 'data-pitch': pitch, 'data-expanded': 'false' },
      },
      [head, this._childList]
    );
  }
}

// =============================================================================
// Playback strategies
// =============================================================================

/**
 * How the workbench gets sound out of the server.
 *
 * Two implementations, one interface, chosen by App from a memory estimate.
 * They are separate classes rather than branches inside App because they differ
 * in almost everything that matters — what is fetched, when, how much is held —
 * and because "the eager path is exactly what it always was" is much easier to
 * believe when it is a class nobody edited than when it is every other line of
 * an if-statement.
 *
 * THE CONTRACT
 *   prepare()          load whatever the first Play needs. Rejects on failure.
 *   apply(muted, solo) make the audible mix match that intent. Never rejects;
 *                      it reports what it could not do so the caller can put
 *                      the controls back.
 *   isRowPlayable(row) whether a row's mute/solo can do anything yet.
 *   note               one line for the UI, or '' when there is nothing to say.
 *
 * `apply` deliberately reports rather than throws. A failed fetch mid-session
 * is not fatal to the workbench — the analysis is still on screen and still
 * exportable — so the strategy hands back the ids it could not honour and the
 * App reverts those controls and banners the reason. A control that stays lit
 * while the audio does not match it is the one outcome neither mode allows.
 */
class PlaybackStrategy {
  /**
   * @param {{api: ApiClient, jobId: string, ctx: AudioContext, transport: Transport,
   *          summary: object, rowFor: Function, setStatus: Function,
   *          isCurrent: Function}} deps
   */
  constructor(deps) {
    this._api = deps.api;
    this._jobId = deps.jobId;
    this._ctx = deps.ctx;
    this._transport = deps.transport;
    this._summary = deps.summary;
    this._rowFor = deps.rowFor;
    this._setStatus = deps.setStatus;
    /** False once a newer analysis has superseded this one. Guards every await. */
    this._isCurrent = deps.isCurrent;
  }

  /** @returns {string} the honesty line for the UI, or '' */
  get note() {
    return '';
  }

  /**
   * Something the user needs told about how loading went, or ''.
   *
   * Separate from `note`, which describes the strategy itself: this one is
   * about this particular run — "some tracks are missing and the mix will not
   * reproduce the original". It is announced as well as shown, because it
   * qualifies the one claim this product makes.
   *
   * @returns {string}
   */
  get loadWarning() {
    return '';
  }

  /** @returns {Promise<void>} */
  async prepare() {
    throw new Error('not implemented');
  }

  /**
   * @param {Set<number>} muted
   * @param {Set<number>} soloed
   * @returns {Promise<{revertSolos: Array<number>, revertMutedTo: ?Set<number>, message: ?string}>}
   */
  async apply() {
    return { revertSolos: [], revertMutedTo: null, message: null };
  }

  /** @param {TrackRow} row @returns {boolean} */
  isRowPlayable(row) {
    return row.hasAudio;
  }

  /** Release anything held beyond what the Transport owns. */
  dispose() {}

  /** All track ids in this analysis, including the residual. @returns {Array<number>} */
  _allTrackIds() {
    return (this._summary.tracks || []).map((track) => track.track_id);
  }

  /**
   * Fetch and decode one WAV.
   * @param {Promise<ArrayBuffer>} request
   * @returns {Promise<?AudioBuffer>} null if the session was superseded
   */
  async _decode(request) {
    const bytes = await request;
    const buffer = await decodeAudio(this._ctx, bytes);
    // Guard against a session that was reset (new upload, or an error) while
    // this fetch was in flight — registering into a dead transport would
    // resurrect the old analysis's audio under the new one's rows.
    return this._isCurrent() ? buffer : null;
  }
}

/**
 * Load every track up front. The original behaviour, unchanged.
 *
 * Once prepare() has finished, nothing here touches the network again: mute and
 * solo are a single batched gain update. That is the experience worth keeping
 * for anything short enough to afford it.
 */
class EagerPlayback extends PlaybackStrategy {
  constructor(deps) {
    super(deps);
    /** Set when some tracks never arrived, so the mix is no longer exact. */
    this._loadWarning = '';
  }

  get loadWarning() {
    return this._loadWarning;
  }

  /**
   * Fetch, decode, and register every track's PCM.
   *
   * Progress is reported as a count rather than a spinner because a spinner
   * cannot tell you whether 3 of 200 tracks have arrived or 199.
   */
  async prepare() {
    const tracks = (this._summary.tracks || []).slice();
    const total = tracks.length;

    // Counted, not inferred. `loaded` is incremented where a buffer actually
    // reaches the Transport, because that is the only event that makes a track
    // audible — deriving it as `total - failures.length` would count every
    // track we never got around to asking for as a success.
    let loaded = 0;
    const failures = [];
    this._setStatus(`Loading audio — 0 of ${total}`);

    /**
     * Record a track the user will not hear, and say so on its row.
     *
     * Every path that gives up on a track goes through here, so the count in
     * the summary below and the marks on the rows can never disagree.
     */
    const recordFailure = (track) => {
      failures.push(track.label || `Track ${track.track_id}`);
      const failedRow = this._rowFor(track.track_id);
      if (failedRow) failedRow.setAudioFailed();
    };

    // A fixed pool of workers pulling from one queue: simpler than batching,
    // and it keeps exactly AUDIO_FETCH_CONCURRENCY requests in flight even when
    // one track is much larger than the others.
    const worker = async () => {
      while (tracks.length > 0) {
        const track = tracks.shift();
        const row = this._rowFor(track.track_id);
        try {
          const buffer = await this._decode(this._api.trackAudio(this._jobId, track.track_id));
          if (buffer === null) return; // superseded by a newer analysis
          this._transport.addTrack(track.track_id, buffer);
          if (row) row.setBuffer(buffer);
          loaded += 1;
        } catch (err) {
          recordFailure(track);
          if (err instanceof ApiError && err.status === 404) {
            // The session expired mid-load. Nothing else will succeed either,
            // so stop asking — but a track abandoned here is still a track the
            // user will not hear, and it has to be counted and marked as one.
            // Draining the queue silently would leave the summary below
            // reporting only the handful that happened to be in flight, which
            // is how "1 of 200 tracks loaded" gets announced as "3 failed".
            // splice() is atomic against the other workers: there is no await
            // between it and the loop, so no worker can pick up a track that
            // is about to be recorded as abandoned.
            for (const abandoned of tracks.splice(0)) recordFailure(abandoned);
          }
        } finally {
          if (this._isCurrent()) {
            this._setStatus(`Loading audio — ${loaded + failures.length} of ${total}`);
          }
        }
      }
    };

    const pool = Math.max(1, Math.min(AUDIO_FETCH_CONCURRENCY, total));
    await Promise.all(Array.from({ length: pool }, worker));

    if (!this._isCurrent()) return;

    if (loaded === 0) {
      throw new ApiError(
        'The analysis finished but none of its audio could be loaded. ' +
          'The session may have expired — please try the file again.',
        0
      );
    }

    if (failures.length > 0) {
      // Say it plainly, on screen AND out loud. Silently playing 198 of 200
      // tracks would break the one claim this product makes about its own mix.
      this._loadWarning =
        `${failures.length} of ${total} tracks could not be loaded and will be silent. ` +
        'The mix will not reproduce the original exactly.';
      this._setStatus(this._loadWarning);
    } else {
      this._setStatus(`${total} track${total === 1 ? '' : 's'} ready.`);
    }
  }

  /**
   * Every buffer is already here, so this is one batched gain update and no
   * network at all. Nothing can fail, hence nothing to revert.
   */
  async apply(muted, soloed) {
    this._transport.setMixState(muted, soloed);
    return { revertSolos: [], revertMutedTo: null, message: null };
  }
}

/**
 * Play one server-side mix, and fetch individual tracks only when soloed.
 *
 * THE ARRANGEMENT, IN FULL. There are exactly three configurations, and the
 * point of listing them is that in every one of them a given track's audio is
 * audible through exactly ONE path — never through both the mix and a
 * separately-fetched copy, which would double it.
 *
 *   (1) NOTHING SOLOED.  The mix holds `mixdown?exclude=<muted>` and is the
 *       only audible source. Cached tracks are registered `soloOnly`, so they
 *       are silent. With nothing muted the exclude list is empty and the mix
 *       is the uploaded file, byte for byte.
 *
 *   (2) A FEW TRACKS SOLOED (at most `_capacity` of them). Each soloed,
 *       unmuted track is fetched, cached and registered `soloOnly`; the mix is
 *       muted outright. The audible sum is exactly the soloed tracks, which is
 *       what solo means, and re-soloing something already cached costs nothing.
 *
 *   (3) MORE SOLOED THAN FIT (soloing a whole pitch group, usually). Holding
 *       them individually would be the memory problem again, so the same
 *       endpoint answers the same question in one buffer: the mix becomes
 *       `mixdown?exclude=<everything except the soloed tracks>`, and every
 *       cached individual track is evicted. Audibly identical to (2); one
 *       buffer instead of forty.
 *
 * The mix is only fetched when it is about to be heard. While solos are up in
 * case (2) the mix is silent, so a mute pressed there costs no download at all —
 * it is settled later, when the last solo is cleared and the mix is needed.
 */
class BoundedPlayback extends PlaybackStrategy {
  constructor(deps) {
    super(deps);

    /**
     * Decoded tracks, most-recently-used last.
     *
     * A Map iterates in insertion order, so "delete then set" on every use is
     * all an LRU needs — no separate recency list to keep in step with it.
     * @type {Map<number, AudioBuffer>}
     */
    this._cache = new Map();

    /**
     * The exclude list the currently registered mix buffer actually represents,
     * or null when no mix has ever loaded. Compared against what the user's
     * intent asks for; equal means the ~106 MB download can be skipped.
     *
     * Only ever advanced on a *successful* load, so after a failed one it still
     * describes the buffer we are still holding. That is what makes the failure
     * path terminate: the app falls back to the state this key names, and the
     * next reconcile finds the key already correct and asks for nothing.
     * @type {?string}
     */
    this._mixKey = null;

    /**
     * What the held mix's exclude list *meant*: 'mute' if it was
     * everything-minus-the-muted-tracks, 'solo' if it was
     * everything-minus-the-unsoloed-tracks.
     *
     * The two look identical (a list of ids) and mean opposite things, so
     * reading a solo mix's list back as a mute picture would light up the mute
     * button on thirty tracks the user never touched. Recording which one it is
     * costs a field and removes the whole class of mistake.
     * @type {?('mute'|'solo')}
     */
    this._mixKind = null;

    // How many individual tracks fit alongside the mix inside the budget. The
    // "- 1" is the mix itself, which is the same size as a track because every
    // track is rendered over the whole timeline.
    const perTrack = trackDecodedBytes(this._summary) || 1;
    const affordable = Math.floor(PLAYBACK_MEMORY_BUDGET_BYTES / perTrack) - 1;
    this._capacity = Math.max(1, Math.min(BOUNDED_CACHE_MAX_TRACKS, affordable));
  }

  get note() {
    return BOUNDED_MODE_NOTE;
  }

  /** The first Play needs one thing: the whole mix. */
  async prepare() {
    this._setStatus('Loading the combined mix…');
    const buffer = await this._decode(this._api.mixdown(this._jobId, []));
    if (buffer === null) return; // superseded by a newer analysis
    this._transport.setMix(buffer);
    this._mixKey = '';
    this._mixKind = 'mute';
    this._setStatus(this._readyStatus());
  }

  /**
   * In bounded mode a row's controls work whether or not its audio is here yet
   * — fetching it is what pressing them does. Only a row whose fetch already
   * failed is genuinely dead, and that is what `audioFailed` records.
   */
  isRowPlayable(row) {
    return !row.audioFailed;
  }

  dispose() {
    this._cache.clear();
    this._mixKey = null;
    this._mixKind = null;
  }

  /**
   * Reconcile the audible mix with the user's intent.
   *
   * Reads only the two sets it is given, so a call that was queued behind a
   * slow fetch still lands on the latest intent rather than a stale delta.
   *
   * @param {Set<number>} muted
   * @param {Set<number>} soloed
   */
  async apply(muted, soloed) {
    const failed = [];
    let message = null;
    let revertMutedTo = null;

    // The only tracks that can make a sound: soloed, and not muted on top.
    // A track that is both is silent either way, so it is never worth 106 MB.
    const wanted = [...soloed].filter((id) => !muted.has(id));
    const useIndividualTracks = wanted.length > 0 && wanted.length <= this._capacity;

    /**
     * Something is soloed, and every soloed track is also muted. Nothing may be
     * audible at all — which is a case the mix cannot express, and cannot be
     * left to the Transport to notice.
     *
     * WHY IT NEEDS SAYING HERE. transport.js silences a non-soloed track only
     * when some *registered* track is soloed, and case (3) below evicts every
     * individual buffer, so after soloing a whole pitch group the Transport
     * holds nothing but the mix and sees no solo at all. Muting that group would
     * then fall into case (1) and load `mixdown?exclude=<the group>` — the sum
     * of every track the user did NOT solo — and play it. The user asked to hear
     * seven tracks, muted them, and would hear the other twenty-three.
     *
     * Deciding it from the intent, where the solo is known whether or not its
     * audio happens to be resident, closes that. It also saves a ~106 MB
     * download of a buffer that could not have been heard either way.
     */
    const nothingAudible = wanted.length === 0 && soloed.size > 0;

    /**
     * Set when a *track* buffer joined mid-session. Only tracks need the
     * restart below: Transport.setMix restores playback itself, because only it
     * knows whether replacing the mix's source stopped the transport.
     */
    let addedTracks = false;
    /** Set when the mix we are holding does not answer the question being asked. */
    let mixUnusable = false;

    if (useIndividualTracks) {
      // --- case (2): hold the soloed tracks themselves ----------------------
      for (const id of wanted) {
        if (this._cache.has(id)) {
          this._touch(id);
          continue;
        }
        // Make room BEFORE fetching, not after the loop. Evicting afterwards
        // means the outgoing set and the incoming set are decoded at the same
        // time, so soloing four tracks and then four different ones peaks at
        // 2 x capacity + 1 buffers — nine of them, ~1 GB at ten minutes, which
        // is well past the budget that put this file in bounded mode in the
        // first place, and the peak is what kills a tab rather than the
        // average. Trimming to capacity - 1 first bounds the peak at
        // capacity + 1, which is the number this mode promises.
        //
        // `wanted` is the keep set, so a track already fetched for this same
        // solo is never thrown away to make room for its neighbour. That is
        // safe: case (2) only runs when wanted.length <= capacity.
        this._evict(new Set(wanted), Math.max(0, this._capacity - 1));
        try {
          if (await this._loadTrack(id)) addedTracks = true;
        } catch (err) {
          failed.push(id);
          message = err instanceof ApiError ? err.message : GENERIC_ERROR;
        }
      }
      this._evict(new Set(wanted));
    } else {
      // --- cases (1) and (3): one buffer answers the whole question ---------
      // (1) nothing soloed  -> everything except what is muted
      // (3) too many soloed -> everything except what is NOT soloed
      const wantMuteMix = wanted.length === 0;
      const exclude = wantMuteMix
        ? [...muted]
        : this._allTrackIds().filter((id) => !wanted.includes(id));
      const key = excludeKey(exclude);

      // `nothingAudible` asks for no mix at all: whatever we are holding is
      // about to be silenced below, so fetching a different one would spend a
      // hundred megabytes to change which buffer is inaudible.
      if (!nothingAudible && key !== this._mixKey) {
        try {
          await this._loadMix(exclude, key, wantMuteMix ? 'mute' : 'solo');
        } catch (err) {
          message = err instanceof ApiError ? err.message : GENERIC_ERROR;
          // The load failed, so we are still holding the previous mix. What to
          // do with it depends entirely on what it is — and this is where the
          // fallback has to terminate rather than retry, because a server that
          // is down will fail the retry too.
          if (!wantMuteMix) {
            // A solo we cannot deliver. The mix we hold contains every track,
            // so playing it would be the opposite of soloing. Silence it and
            // put the solo buttons back.
            failed.push(...wanted);
            mixUnusable = true;
          } else if (this._mixKind === 'mute') {
            // A mute we cannot deliver, but the mix we hold IS a mute picture.
            // Fall back to it: the buttons move to describe what is audible,
            // which is both honest and stable — the next reconcile finds the
            // key already correct and asks the server for nothing.
            revertMutedTo = this._parseMixKey();
          } else {
            // We hold a solo mix and were asked for a mute mix. It cannot stand
            // in for one, and its exclude list cannot be read as a mute picture
            // either, so the only honest option is silence plus the banner.
            mixUnusable = true;
          }
        }
      }

      // In case (3) the individual buffers are both unnecessary and the thing
      // we are short of, so let all of them go — capacity 0. In case (1) they
      // are harmless (silent, and capacity-limited) and worth keeping, because
      // re-soloing something already decoded should cost nothing.
      if (wanted.length > 0) this._evict(new Set(), 0);
    }

    // --- gains -------------------------------------------------------------
    // The mix is muted whenever it must not be heard: whenever individual
    // tracks are carrying the solo, whenever every soloed track is muted so
    // nothing may sound, and whenever the mix we hold no longer answers the
    // question being asked. Muting it is how a failed fetch stays honest — the
    // user hears less than they asked for, never something they muted.
    const silenceMix =
      useIndividualTracks || nothingAudible || this._mixKey === null || mixUnusable;
    const mutedForTransport = silenceMix ? [...muted, MIX_TRACK_ID] : [...muted];
    this._transport.setMixState(mutedForTransport, soloed);

    // --- make newly fetched tracks audible ---------------------------------
    // A buffer registered mid-playback has no source and stays silent until the
    // next play() (transport.js says so, and says why). Restarting every source
    // from the current position is the sanctioned way to add a voice: one fresh
    // shared timestamp for all of them, so nothing lands a sample out.
    //
    // Only for tracks. A mix swap has already restored itself inside setMix,
    // and doing it twice would cost the user a second gap for nothing.
    if (addedTracks && this._transport.isPlaying) {
      this._transport.play(this._transport.currentTime);
    }

    if (this._isCurrent()) this._setStatus(this._readyStatus());
    return { revertSolos: failed, revertMutedTo, message };
  }

  /** Move an id to the most-recently-used end of the cache. */
  _touch(id) {
    const buffer = this._cache.get(id);
    this._cache.delete(id);
    this._cache.set(id, buffer);
  }

  /**
   * Fetch, decode and register one track for soloing.
   * @param {number} id
   * @returns {Promise<boolean>} whether a buffer actually reached the Transport
   */
  async _loadTrack(id) {
    const row = this._rowFor(id);
    this._setStatus(`Loading ${row ? row.label : `track ${id}`}…`);

    const buffer = await this._decode(this._api.trackAudio(this._jobId, id));
    if (buffer === null) return false; // superseded by a newer analysis

    this._cache.set(id, buffer);
    // soloOnly: this track's audio is already inside the mix, so it must be
    // silent unless it is the thing being soloed. See transport.js.
    this._transport.addTrack(id, buffer, { soloOnly: true });
    if (row) row.setBuffer(buffer);
    return true;
  }

  /**
   * Fetch, decode and install a new mix.
   *
   * Nothing is mutated until the bytes are here and decoded. If this throws,
   * `_mixKey` and `_mixKind` still describe the buffer the Transport is still
   * holding — which is what lets the caller fall back to a state it can
   * actually produce, instead of a state it will fail to fetch again.
   *
   * @param {Array<number>} exclude
   * @param {string} key the stable form of `exclude`, already computed
   * @param {'mute'|'solo'} kind what that exclude list means
   */
  async _loadMix(exclude, key, kind) {
    this._setStatus(exclude.length > 0 ? 'Updating the mix…' : 'Loading the combined mix…');

    const buffer = await this._decode(this._api.mixdown(this._jobId, exclude));
    if (buffer === null) return; // superseded by a newer analysis

    this._transport.setMix(buffer);
    this._mixKey = key;
    this._mixKind = kind;
  }

  /**
   * The mute picture the currently held mix represents.
   *
   * Only meaningful when `_mixKind === 'mute'`; the one caller checks that
   * first, because the same list of ids read off a solo mix would mean the
   * exact opposite.
   *
   * @returns {Set<number>}
   */
  _parseMixKey() {
    if (!this._mixKey) return new Set();
    return new Set(this._mixKey.split(',').map(Number).filter(Number.isFinite));
  }

  /**
   * Drop cached tracks until the cache is within capacity.
   *
   * `keep` is never evicted however old it is: those are the tracks currently
   * making a sound, and freeing memory by silencing something the user is
   * listening to is not a trade worth making. If a user solos more tracks than
   * capacity we never get here — that is case (3), which holds none at all.
   *
   * @param {Set<number>} keep
   * @param {number} [capacity] how many to keep. Defaults to the budgeted
   *   number; case (3) passes 0, because there the individual buffers are the
   *   very thing being traded away for one mix.
   */
  _evict(keep, capacity = this._capacity) {
    for (const id of [...this._cache.keys()]) {
      if (this._cache.size <= capacity) break;
      if (keep.has(id)) continue;
      this._cache.delete(id);
      // Both holders have to let go. The Transport keeps the same AudioBuffer,
      // so dropping only the cache entry would free nothing at all.
      this._transport.removeTrack(id);
      const row = this._rowFor(id);
      if (row) row.clearBuffer();
    }
  }

  /** The steady-state status line: what is loaded, and what it costs. */
  _readyStatus() {
    if (this._mixKey === null) return 'No mix loaded.';
    const held = this._cache.size + 1;
    const bytes = held * trackDecodedBytes(this._summary);
    return this._cache.size === 0
      ? `Full mix ready · ${formatMegabytes(bytes)} in memory`
      : `Full mix plus ${this._cache.size} track${this._cache.size === 1 ? '' : 's'} · ` +
          `${formatMegabytes(bytes)} in memory`;
  }
}

// =============================================================================
// The application
// =============================================================================

/**
 * Owns the three stages, the transport, and the rows.
 *
 * A class rather than a pile of module-level variables so that the state a
 * failed analysis has to unwind — job id, feed, transport, rows, animation
 * frame — is enumerable in one place. _reset() is the proof: every field it
 * clears is a field that would otherwise leak into the next upload.
 */
class App {
  constructor(api = new ApiClient()) {
    this._api = api;

    // --- Stage elements -----------------------------------------------------
    this._dropzone = document.getElementById('dropzone');
    this._fileInput = document.getElementById('file-input');
    this._progress = document.getElementById('progress');
    this._progressBar = document.getElementById('progress-bar');
    this._progressFill = document.getElementById('progress-fill');
    this._progressStage = document.getElementById('progress-stage');
    this._workbench = document.getElementById('workbench');
    this._tracksEl = document.getElementById('tracks');
    this._diagnostics = document.getElementById('diagnostics');
    this._errorBanner = document.getElementById('error');
    this._noticeBanner = document.getElementById('notice');
    this._announcer = document.getElementById('announcer');

    // --- Transport controls -------------------------------------------------
    this._playButton = document.getElementById('play');
    this._stopButton = document.getElementById('stop');
    this._clearSolosButton = document.getElementById('solo-none');
    this._viewToggle = document.getElementById('view-toggle');
    this._newFileButton = document.getElementById('new-file');
    this._settingsHost = document.getElementById('settings');
    this._reanalyzeButton = document.getElementById('reanalyze');
    /** The settings panel, or null if its schema could not be fetched. */
    this._settings = null;
    /** Guards against a second bulk export starting on top of one in flight. */
    this._exportingAll = false;
    this._playheadEl = document.getElementById('playhead');
    this._audioStatus = document.getElementById('audio-status');
    this._playbackNote = document.getElementById('playback-mode');

    // --- Session state ------------------------------------------------------
    /** AudioContext is created on first user gesture — browsers require it. */
    this._ctx = null;
    this._transport = null;
    this._feed = null;
    this._jobId = null;
    this._summary = null;
    /** @type {Map<number, TrackRow>} */
    this._rows = new Map();
    /** @type {Array<TrackGroupRow>} */
    this._groups = [];
    /** @type {?PlaybackStrategy} */
    this._playback = null;

    /**
     * WHAT THE USER ASKED FOR, as opposed to what is currently audible.
     *
     * These two sets are the single source of truth for mute and solo. The
     * Transport's gains and the rows' aria-pressed attributes are both
     * *derived* from them, never a second copy — which is what lets a group
     * toggle, a row toggle and "Clear solos" all be the same operation, and
     * what lets a bounded-mode fetch that arrives late reconcile against the
     * latest intent instead of replaying a stale delta.
     *
     * @type {Set<number>}
     */
    this._muted = new Set();
    /** @type {Set<number>} */
    this._soloed = new Set();

    /** Serialises _applyMix so two fast clicks cannot interleave two fetches. */
    this._applyRunning = false;
    this._applyPending = false;

    this._audioReady = false;
    this._raf = null;
    this._resizeObserver = null;
    this._busy = false;
  }

  /** Wire every listener. Call once, after the DOM exists. */
  start() {
    this._wireDropzone();
    this._wireTransport();
    this._wireResize();
    this._setStage('drop');
  }

  // ---------------------------------------------------------------------------
  // Stage 1: getting a file
  // ---------------------------------------------------------------------------

  _wireDropzone() {
    const zone = this._dropzone;
    const input = this._fileInput;
    if (!zone || !input) return;

    // Clicking anywhere in the zone opens the picker. The <button id="browse">
    // inside it needs no handler of its own — its click bubbles here, which is
    // one code path instead of two and keeps the keyboard route identical to
    // the mouse route.
    zone.addEventListener('click', (event) => {
      // The hidden <input> lives inside the zone, and input.click() dispatches
      // a click that bubbles right back up to this handler. Without this guard
      // that is an infinite loop, and the file picker never appears.
      if (event.target === input) return;
      if (this._busy) return;
      input.click();
    });

    input.addEventListener('change', () => {
      const file = input.files && input.files[0];
      // Clear the value so choosing the *same* file twice still fires change.
      // Without this, "analyse it again" silently does nothing.
      input.value = '';
      if (file) this._handleFile(file);
    });

    zone.addEventListener('dragenter', (event) => {
      event.preventDefault();
      zone.classList.add('is-over');
    });

    zone.addEventListener('dragover', (event) => {
      // Without preventDefault the browser refuses the drop entirely — this is
      // the single most common reason a drop target "does nothing".
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
      zone.classList.add('is-over');
    });

    zone.addEventListener('dragleave', (event) => {
      // dragleave also fires when the pointer crosses from the zone onto one of
      // its own children, which would make the highlight flicker. relatedTarget
      // is where the pointer went; if it is still inside the zone, stay lit.
      // It is null when the pointer leaves the window, which is a real leave.
      if (event.relatedTarget && zone.contains(event.relatedTarget)) return;
      zone.classList.remove('is-over');
    });

    zone.addEventListener('drop', (event) => {
      event.preventDefault();
      zone.classList.remove('is-over');
      const files = event.dataTransfer ? event.dataTransfer.files : null;
      if (!files || files.length === 0) return;
      if (files.length > 1) {
        this._announce('Several files were dropped. Analysing the first one.');
      }
      this._handleFile(files[0]);
    });

    // A file dropped anywhere *else* on the page would otherwise navigate the
    // window to it, throwing away a finished analysis. Swallow those.
    for (const type of ['dragover', 'drop']) {
      window.addEventListener(type, (event) => {
        if (zone.contains(event.target)) return;
        event.preventDefault();
      });
    }
  }

  /**
   * Take one file all the way to a rendered workbench.
   * @param {File} file
   */
  async _handleFile(file) {
    if (this._busy) return;

    if (!file || file.size === 0) {
      this._showError('That file is empty. Please choose an audio file.');
      return;
    }

    this._reset();
    this._busy = true;
    this._clearError();
    this._setStage('progress');
    this._setProgress(0, 'Uploading');

    // The click or drop that got us here IS the user gesture browsers require,
    // so this is the right moment to build the AudioContext. Creating it later,
    // from the Play button, works too — but creating it now means a context
    // that is already running by the time the first buffer decodes.
    this._ensureAudioContext();

    try {
      const created = await this._api.analyze(
        file,
        this._settings ? this._settings.values() : null
      );
      if (!created || !created.job_id) throw new ApiError(GENERIC_ERROR, 0);
      this._jobId = created.job_id;

      const estimate = Number(created.estimated_seconds);
      this._setProgress(
        0.01,
        Number.isFinite(estimate) && estimate > 0
          ? `Analysing — about ${Math.round(estimate)} s`
          : 'Analysing'
      );

      this._startFeed(created.job_id);
    } catch (err) {
      this._showError(err instanceof ApiError ? err.message : GENERIC_ERROR);
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 2: watching the job
  // ---------------------------------------------------------------------------

  /** @param {string} jobId */
  _startFeed(jobId) {
    this._feed = new JobFeed(this._api, jobId, {
      onUpdate: (status) => {
        // The stage text lives in the aria-live region the markup already
        // declares, so a screen-reader user hears "Following tones through
        // time" rather than a percentage they cannot see.
        this._setProgress(status.progress, status.stage);
      },
      onDone: () => this._loadSummary(jobId),
      onError: (message) => this._showError(message || 'The analysis failed.'),
    });
    this._feed.start();
  }

  /** @param {string} jobId */
  async _loadSummary(jobId) {
    try {
      this._setProgress(1, 'Preparing the workbench');
      const summary = await this._api.tracks(jobId);
      this._summary = summary;
      this.renderWorkbench(summary);
      this._setStage('workbench');
      // Fire and forget: the buttons work without their size labels, so this
      // must never block the workbench appearing or fail it.
      this._loadExportEstimates();
      this._refreshReanalyze();
      // Canvases cannot be sized while their section is hidden, so the first
      // real draw has to happen after the stage switch, not during render.
      this._drawAllRows();

      // The upload lock is released here rather than after the audio finishes:
      // the workbench is on screen and readable, so starting a different
      // analysis from this point is a legitimate thing to want to do.
      this._busy = false;

      const count = (summary.tracks || []).length;
      this._announce(
        `Analysis complete. ${count} track${count === 1 ? '' : 's'}. Loading audio.`
      );
      await this._loadAllAudio(summary);
    } catch (err) {
      this._showError(err instanceof ApiError ? err.message : GENERIC_ERROR);
    }
  }

  // ---------------------------------------------------------------------------
  // Stage 3: the workbench
  // ---------------------------------------------------------------------------

  /**
   * Build the track list — grouped by pitch — plus the diagnostics panel.
   *
   * THE GROUPING RULE, in one place so it can be checked against the intent:
   *
   *   - Non-residual tracks that share a `pitch` and occur more than once
   *     become one collapsible parent row.
   *   - A pitch that occurs exactly once stays a plain row. A disclosure
   *     triangle hiding a single row is ceremony, and on a short file every
   *     pitch occurs once, so grouping there would add friction and no signal.
   *   - Groups and plain rows are ordered together by energy, descending —
   *     a group by the sum of its occurrences, which is how much of the
   *     recording that note actually is.
   *   - The residual keeps its own ungrouped row and is always last, whatever
   *     its energy. It is the leftovers, not a competitor for the top of the
   *     list.
   *
   * @param {object} summary AnalysisSummary
   */
  renderWorkbench(summary) {
    this._rows.clear();
    this._groups = [];
    this._muted.clear();
    this._soloed.clear();
    this._tracksEl.replaceChildren();
    this._tracksEl.style.setProperty('--playhead', '0%');

    const durationS = Number(summary.duration_s) || 0;
    const tracks = summary.tracks || [];

    // Build every row first. Grouping only decides where a row is *placed* —
    // the rows themselves are identical either way, which is what keeps a
    // grouped occurrence exactly as usable as an ungrouped one.
    const makeRow = (track, inGroup) => {
      const row = new TrackRow(track, {
        durationS,
        inGroup,
        onMute: (id, muted) => this._setMuted([id], muted),
        onSolo: (id, soloed) => this._setSoloed([id], soloed),
        exportUrlFor: (id, format) => this._exportTrack(id, format),
      });
      this._rows.set(row.trackId, row);
      return row;
    };

    // Bucket by pitch, preserving the server's order inside each bucket (it is
    // already energy-descending, and time order within one note is not
    // something the summary reports).
    const buckets = new Map();
    const residuals = [];
    for (const track of tracks) {
      if (track.kind === 'residual') {
        residuals.push(track);
        continue;
      }
      const pitch = typeof track.pitch === 'string' ? track.pitch : '';
      // A non-residual track with no pitch should not happen — the server
      // always labels one — but if it ever does, it gets a plain row rather
      // than being silently swept into an "" group with unrelated tracks.
      const key = pitch || `#ungrouped-${track.track_id}`;
      if (!buckets.has(key)) buckets.set(key, { pitch, tracks: [] });
      buckets.get(key).tracks.push(track);
    }

    /** @type {Array<{energy: number, build: Function}>} */
    const entries = [];
    for (const { pitch, tracks: members } of buckets.values()) {
      if (members.length < 2) {
        const track = members[0];
        entries.push({
          energy: Number(track.energy_ratio) || 0,
          build: () => makeRow(track, false).element,
        });
        continue;
      }
      entries.push({
        energy: members.reduce((total, t) => total + (Number(t.energy_ratio) || 0), 0),
        build: () => {
          const childRows = members.map((track) => makeRow(track, true));
          const group = new TrackGroupRow(pitch, childRows, {
            durationS,
            onMute: (ids, muted) => this._setMuted(ids, muted),
            onSolo: (ids, soloed) => this._setSoloed(ids, soloed),
            // Children were hidden, so their canvases measured 0x0 and drew
            // nothing. This is the first moment they have a size.
            onToggle: () => this._drawAllRows(),
          });
          this._groups.push(group);
          return group.element;
        },
      });
    }

    entries.sort((a, b) => b.energy - a.energy);

    const fragment = document.createDocumentFragment();
    for (const entry of entries) fragment.appendChild(entry.build());
    for (const track of residuals) fragment.appendChild(makeRow(track, false).element);
    this._tracksEl.appendChild(fragment);

    this._renderDiagnostics(summary);
    this._observeRows();

    // Nothing is playable until the PCM lands. Saying so on the buttons is the
    // difference between "loading" and "broken".
    this._setAudioReady(false);
    this._refreshRowStates();
    this._updatePlayhead();
  }

  /**
   * The honesty panel: where the energy went, whether the sum is exact, and
   * anything the analysis had to drop.
   * @param {object} summary
   */
  _renderDiagnostics(summary) {
    const partial = Number(summary.partial_energy_ratio) || 0;
    const residual = Number(summary.residual_energy_ratio) || 0;
    const exact = Boolean(summary.reconstruction_exact);
    const warnings = Array.isArray(summary.warnings) ? summary.warnings.filter(Boolean) : [];

    // The two ratios are measured independently and need not sum to exactly
    // 1.0. The *numbers* shown are the raw ones; only the bar's widths are
    // normalised, because a bar that overflows its track would misreport the
    // split more badly than a rounding difference does.
    const total = partial + residual || 1;

    /**
     * One segment of the bar, sized through the CSSOM.
     *
     * NOT `attrs: { style: '…' }`. The server sends `style-src 'self'` with no
     * `'unsafe-inline'` (backend/main.py), and that blocks inline style
     * *attributes* as well as <style> blocks — so a width written as an
     * attribute is dropped by the browser and both segments render at zero
     * width, silently turning the honesty panel into an empty bar. Assigning
     * through element.style is a scripted change rather than parsed markup, so
     * CSP does not apply to it, which is why every other dynamic size in this
     * module (the playhead) is written the same way.
     */
    const segment = (className, ratio) => {
      const node = el('span', { className });
      node.style.setProperty('width', `${(100 * ratio) / total}%`);
      return node;
    };

    const bar = el(
      'div',
      {
        className: 'energy-bar',
        attrs: {
          role: 'img',
          'aria-label':
            `Energy split: partials ${formatPercent(partial)}, ` +
            `residual ${formatPercent(residual)}`,
        },
      },
      [segment('seg seg-partial', partial), segment('seg seg-residual', residual)]
    );

    const stat = (label, value, className) =>
      el('div', { className: 'diag-item' }, [
        el('span', { className: 'diag-label', text: label }),
        el('span', { className: `diag-value ${className || ''}`.trim(), text: value }),
      ]);

    const children = [
      el('h2', { className: 'diag-title', text: 'Analysis' }),
      bar,
      el('div', { className: 'diag-grid' }, [
        stat('Partial energy', formatPercent(partial), 'is-partial'),
        stat('Residual energy', formatPercent(residual), 'is-residual'),
        stat('Tracks', String((summary.tracks || []).length)),
        stat('Duration', formatSeconds(summary.duration_s)),
        stat('Sample rate', `${summary.sample_rate || 0} Hz`),
        stat(
          'Reconstruction',
          exact ? 'exact' : 'NOT exact',
          exact ? 'is-exact' : 'is-inexact'
        ),
      ]),
      el('p', {
        className: 'diag-note',
        text: exact
          ? 'Unmuting every track sums to the original file, sample for sample.'
          : 'The tracks do not sum back to the original. Treat these tracks as approximate.',
      }),
    ];

    // Warnings are how the backend reports that it dropped tracks over the cap,
    // or that a stage degraded. Folding them away — or leaving them in the
    // console — would let the interface read as "here is everything" when it is
    // not. So they go at the top of the panel, styled to be noticed.
    if (warnings.length > 0) {
      children.unshift(
        el(
          'div',
          { className: 'diag-warnings', attrs: { role: 'status' } },
          [
            el('h3', {
              className: 'diag-warnings-title',
              text: warnings.length === 1 ? 'One thing to know' : `${warnings.length} things to know`,
            }),
            el(
              'ul',
              { className: 'diag-warning-list' },
              warnings.map((warning) => el('li', { text: warning }))
            ),
          ]
        )
      );
    }

    this._diagnostics.replaceChildren(...children);
  }

  // ---------------------------------------------------------------------------
  // Audio loading
  // ---------------------------------------------------------------------------

  /**
   * Get this analysis's audio into the Transport, by whichever strategy fits.
   *
   * The choice is made here, once, from an estimate of what eager loading would
   * cost: duration x sample rate x 4 bytes x track count. Over the budget and
   * the bounded strategy takes over; under it, nothing about the original path
   * changes. See the module header for what each strategy does.
   *
   * @param {object} summary
   */
  async _loadAllAudio(summary) {
    const jobId = summary.job_id || this._jobId;
    const total = (summary.tracks || []).length;

    if (total === 0) {
      this._setAudioStatus('No tracks to play.');
      this._setAudioReady(false);
      return;
    }

    const ctx = this._ensureAudioContext();
    if (!ctx) {
      // No Web Audio in this browser. The analysis is still fully readable and
      // fully exportable, so degrade to that rather than failing the session.
      this._setAudioStatus('This browser cannot play audio. Export still works.');
      this._setAudioReady(false);
      return;
    }

    // A fresh Transport per analysis, not a reused one. Transport keys tracks
    // by id and derives `duration` from the longest registered buffer, so
    // carrying one over would let a previous upload's tracks lengthen this
    // one's timeline. (The AudioContext itself IS reused — browsers cap how
    // many a page may create, and the earlier context is still perfectly good.)
    this._transport = new Transport(ctx);

    const estimate = estimateDecodedBytes(summary);
    const bounded = estimate > PLAYBACK_MEMORY_BUDGET_BYTES;
    const deps = {
      api: this._api,
      jobId,
      ctx,
      transport: this._transport,
      summary,
      rowFor: (id) => this._rows.get(id) || null,
      // Guarded once, here, so no strategy can write a status line belonging to
      // an analysis the user has already replaced.
      setStatus: (text) => {
        if (this._summary === summary) this._setAudioStatus(text);
      },
      isCurrent: () => this._summary === summary,
    };
    this._playback = bounded ? new BoundedPlayback(deps) : new EagerPlayback(deps);

    // Say which mode is running before the loading starts, not after. A user
    // watching a long file load deserves to know why it is behaving this way
    // while it is happening.
    this._setPlaybackNote(this._playback.note);
    if (bounded) {
      this._announce(
        `${total} tracks. This file is long — about ${formatMegabytes(estimate)} of audio if it ` +
          'were all loaded — so tracks load on demand. ' + BOUNDED_MODE_NOTE
      );
    }

    try {
      await this._playback.prepare();
      if (this._summary !== summary) return; // superseded by a newer analysis

      this._setAudioReady(true);
      // The Transport starts with no mute or solo picture at all, so push the
      // (empty) intent through the same path every later change uses. One code
      // path from the first frame means there is no separate "initial state"
      // to get wrong.
      await this._applyMix();
      if (this._summary !== summary) return;

      // A partial failure outranks "ready": the user is about to hear something
      // that is not the file they uploaded, and that is the sentence that has
      // to reach them.
      const warning = this._playback.loadWarning;
      this._announce(
        warning ||
          (bounded ? 'The combined mix is ready to play.' : 'All track audio loaded. Ready to play.')
      );
    } catch (err) {
      if (this._summary !== summary) return;
      const message = err instanceof ApiError ? err.message : GENERIC_ERROR;
      this._setAudioReady(false);
      if (bounded) {
        // Keep the workbench. Unlike the eager path, one failed request has not
        // cost the user their analysis — every trajectory is still drawn and
        // every track is still exportable — so this reports without tearing the
        // session down.
        this._setAudioStatus('Playback unavailable. Exports still work.');
        this._setBanner(`The mix could not be loaded: ${message}`);
        this._announce(`Playback unavailable. ${message}`);
      } else {
        this._setAudioStatus('No track audio could be loaded.');
        this._showError(message);
      }
    }
  }

  /**
   * Create the AudioContext, once, lazily.
   * @returns {?AudioContext}
   */
  _ensureAudioContext() {
    if (this._ctx) return this._ctx;
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (typeof Ctor !== 'function') return null;
    try {
      this._ctx = new Ctor();
    } catch {
      return null;
    }
    return this._ctx;
  }

  // ---------------------------------------------------------------------------
  // Transport
  // ---------------------------------------------------------------------------

  _wireTransport() {
    if (this._playButton) {
      this._playButton.addEventListener('click', () => {
        if (!this._transport || !this._audioReady) return;
        if (this._transport.isPlaying) {
          // Pause: transport.stop() remembers the position, so the next play()
          // resumes rather than restarting.
          this._transport.stop();
          this._onPlaybackStopped();
        } else {
          this._transport.play(this._transport.currentTime);
          this._setPlayingUi(true);
          this._startPlayheadLoop();
        }
      });
    }

    if (this._stopButton) {
      this._stopButton.addEventListener('click', () => {
        if (!this._transport) return;
        this._transport.stop();
        this._transport.seek(0);
        this._onPlaybackStopped();
      });
    }

    if (this._clearSolosButton) {
      this._clearSolosButton.addEventListener('click', () => this._clearSolos());
    }

    if (this._viewToggle) {
      this._viewToggle.addEventListener('click', () => {
        const toAmplitude = this._viewToggle.getAttribute('aria-pressed') !== 'true';
        this._viewToggle.setAttribute('aria-pressed', toAmplitude ? 'true' : 'false');
        const mode = toAmplitude ? 'amplitude' : 'trajectory';
        for (const row of this._rows.values()) row.setViewMode(mode);
        this._announce(toAmplitude ? 'Amplitude view.' : 'Frequency trajectory view.');
      });
    }

    // Export-all: one handler per format option. Wired here with the other
    // transport controls rather than per-row, because this acts on the whole
    // analysis rather than on any one track.
    // The settings panel builds itself from the server's schema. It is an
    // enhancement: if the fetch fails the app works exactly as before, on
    // defaults, so nothing here is awaited before the page becomes usable.
    createSettingsPanel(document.getElementById('settings-body'), '/api', {
      onChange: () => this._refreshReanalyze(),
    }).then((panel) => {
      this._settings = panel;
      if (panel && this._settingsHost) this._settingsHost.hidden = false;
      this._refreshReanalyze();
    });

    if (this._reanalyzeButton) {
      this._reanalyzeButton.addEventListener('click', () => this._reanalyze());
    }

    for (const button of document.querySelectorAll('.export-option')) {
      button.addEventListener('click', () => this._exportAll(button.dataset.format));
    }

    if (this._newFileButton) {
      this._newFileButton.addEventListener('click', () => {
        this._reset();
        this._clearError();
        this._setStage('drop');
        if (this._fileInput) this._fileInput.click();
      });
    }
  }

  /**
   * Mute or unmute one track or a whole pitch group.
   *
   * Takes a list because a group toggle is the same operation on several ids,
   * and doing it as one intent change means one reconcile and — in bounded
   * mode — one mixdown request instead of one per member.
   *
   * @param {Iterable<number>} trackIds
   * @param {boolean} muted
   */
  _setMuted(trackIds, muted) {
    if (!this._transport) return;
    this._updateIntent(this._muted, trackIds, muted);
  }

  /**
   * Solo or unsolo one track or a whole pitch group.
   * @param {Iterable<number>} trackIds
   * @param {boolean} soloed
   */
  _setSoloed(trackIds, soloed) {
    if (!this._transport) return;
    this._updateIntent(this._soloed, trackIds, soloed);
  }

  /**
   * Add or remove ids from one intent set, then reflect and reconcile.
   *
   * Every control ends up here, which is why the buttons cannot drift from the
   * audio: the set changes, the rows are redrawn *from* the set, and the
   * strategy is asked to make the sound match the same set.
   *
   * @param {Set<number>} set
   * @param {Iterable<number>} trackIds
   * @param {boolean} on
   */
  _updateIntent(set, trackIds, on) {
    for (const id of trackIds) {
      if (on) {
        set.add(id);
      } else {
        set.delete(id);
      }
    }
    this._refreshRowStates();
    this._applyMix();
  }

  _clearSolos() {
    if (!this._transport) return;
    this._soloed.clear();
    this._refreshRowStates();
    this._applyMix();
    this._announce('Solos cleared.');
  }

  /**
   * Push the current intent onto every row and group.
   *
   * The buttons are a *reflection* of `_muted` and `_soloed`, never a second
   * copy of them. That is what makes "Clear solos" and a group toggle correct
   * without either knowing about the other.
   */
  _refreshRowStates() {
    for (const [id, row] of this._rows) {
      row.setMuted(this._muted.has(id));
      row.setSoloed(this._soloed.has(id));
    }
    for (const group of this._groups) group.refresh(this._muted, this._soloed);
  }

  /**
   * Make the audible mix match the intent, one reconcile at a time.
   *
   * SERIALISED, AND COALESCED. In bounded mode a reconcile can involve a
   * hundred-megabyte download, and a user clicking three rows quickly would
   * otherwise start three overlapping fetches whose completion order decides
   * what they hear. So one runs at a time, and anything that arrives during it
   * sets a flag rather than queueing: the next pass reads the intent sets
   * afresh, so it lands on the latest state and the intermediate ones are
   * simply skipped. Fewer requests AND a more correct answer.
   */
  async _applyMix() {
    if (!this._playback || !this._transport) return;
    if (this._applyRunning) {
      this._applyPending = true;
      return;
    }

    this._applyRunning = true;
    const summary = this._summary;
    try {
      // The cap bounds only the passes this loop asks for ITSELF (a revert
      // wanting to settle). Passes asked for by the user — a second click
      // arriving while the first is still fetching — are never dropped, because
      // they are bounded by the person clicking and dropping one would leave the
      // controls describing a mix nobody is hearing. Belt and braces on the
      // self-induced kind: a reconcile that can re-enter itself can spin the
      // network forever if a later edit gets the progress argument subtly
      // wrong, and a tab hammering the server is much worse than a stale button.
      let passes = 0;
      do {
        this._applyPending = false;
        let wantsAnotherPass = false;
        const result = await this._playback.apply(this._muted, this._soloed);
        if (this._summary !== summary) return; // superseded by a newer analysis

        // The strategy could not deliver part of what was asked for. Put those
        // controls back so the buttons describe what is actually audible, then
        // reconcile once more against the corrected intent.
        //
        // Only re-run if the revert CHANGED something. That is what guarantees
        // termination: a revert either moves the intent (so the next pass is
        // asking a different question) or it does not (so asking again would
        // fail identically). A server that is down therefore costs one request
        // per click, not an infinite retry loop.
        const revertSolos = (result && result.revertSolos) || [];
        const revertMutedTo = result && result.revertMutedTo;
        const soloChanged = revertSolos.some((id) => this._soloed.has(id));
        const muteChanged = Boolean(revertMutedTo) && !sameSet(revertMutedTo, this._muted);
        if (soloChanged || muteChanged) {
          for (const id of revertSolos) this._soloed.delete(id);
          if (revertMutedTo) this._muted = new Set(revertMutedTo);
          this._refreshRowStates();
          wantsAnotherPass = true;
        }
        if (result && result.message) {
          // _setBanner, not _showError: the workbench is intact and still
          // playable, and tearing it down would cost far more than the failed
          // request did. But it does reach the banner — nothing here fails only
          // into the console.
          this._setBanner(`Could not update playback: ${result.message}`);
          this._announce(`Could not update playback. ${result.message}`);
        }
        passes += 1;
        if (wantsAnotherPass && passes < APPLY_MAX_PASSES) this._applyPending = true;
      } while (this._applyPending);
    } catch (err) {
      // A strategy is not supposed to reject — it reports instead — so this is
      // the belt to that braces. Reaching the banner matters more than which
      // path got here.
      const message = err instanceof ApiError ? err.message : GENERIC_ERROR;
      this._setBanner(`Could not update playback: ${message}`);
      this._announce(`Could not update playback. ${message}`);
    } finally {
      this._applyRunning = false;
    }
  }

  /**
   * Enable or disable everything that needs decoded audio.
   * @param {boolean} ready
   */
  _setAudioReady(ready) {
    this._audioReady = Boolean(ready);
    const controls = [
      this._playButton,
      this._stopButton,
      this._clearSolosButton,
      this._viewToggle,
    ];
    for (const button of controls) {
      if (button) button.disabled = !this._audioReady;
    }
    // Per row, not globally, and the *strategy* decides what "playable" means.
    // In eager mode a row needs a decoded buffer, because without one its mute
    // and solo are no-ops on a Transport that has never seen that id — the
    // "looks live and does nothing" case setPlayable exists to prevent. In
    // bounded mode the row has no buffer by design and pressing the button is
    // what fetches it, so the same test would disable the entire workbench.
    for (const row of this._rows.values()) {
      row.setPlayable(this._audioReady && this._rowIsPlayable(row));
    }
    // A group is usable while any one of its occurrences is.
    for (const group of this._groups) {
      const usable = group.trackIds.some((id) => {
        const row = this._rows.get(id);
        return Boolean(row) && this._rowIsPlayable(row);
      });
      group.setPlayable(this._audioReady && usable);
    }
    if (this._workbench) this._workbench.dataset.audio = this._audioReady ? 'ready' : 'loading';
  }

  /** @param {TrackRow} row @returns {boolean} */
  _rowIsPlayable(row) {
    return this._playback ? this._playback.isRowPlayable(row) : row.hasAudio;
  }

  /**
   * Show or hide the one line that admits bounded mode is running.
   * @param {string} note empty to hide it
   */
  _setPlaybackNote(note) {
    if (!this._playbackNote) return;
    this._playbackNote.textContent = note || '';
    this._playbackNote.hidden = !note;
  }

  /** @param {boolean} playing */
  _setPlayingUi(playing) {
    if (!this._playButton) return;
    this._playButton.textContent = playing ? 'Pause' : 'Play';
    this._playButton.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  }

  _onPlaybackStopped() {
    this._setPlayingUi(false);
    this._stopPlayheadLoop();
    this._updatePlayhead();
  }

  /**
   * Follow the playhead while sound is running.
   *
   * requestAnimationFrame rather than setInterval, so the readout is painted
   * once per frame the browser is actually drawing — and so it stops entirely
   * when the tab is hidden, instead of burning battery updating text nobody is
   * looking at.
   */
  _startPlayheadLoop() {
    this._stopPlayheadLoop();
    const tick = () => {
      this._updatePlayhead();
      if (this._transport && this._transport.isPlaying) {
        this._raf = requestAnimationFrame(tick);
      } else {
        // Playback ended on its own (the transport's own onended handler fired
        // and rewound to zero). This loop is the only thing watching for that,
        // so it is where the button gets put back.
        this._raf = null;
        this._setPlayingUi(false);
        this._updatePlayhead();
      }
    };
    this._raf = requestAnimationFrame(tick);
  }

  _stopPlayheadLoop() {
    if (this._raf !== null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  _updatePlayhead() {
    const duration = this._transport ? this._transport.duration : 0;
    // currentTime runs slightly negative during the transport's scheduling
    // lookahead, which is honest but reads as a bug. Clamp for display only.
    const raw = this._transport ? this._transport.currentTime : 0;
    const position = Math.min(duration, Math.max(0, raw));

    if (this._playheadEl) {
      this._playheadEl.textContent = `${formatSeconds(position)} / ${formatSeconds(duration)}`;
    }
    if (this._tracksEl) {
      const percent = duration > 0 ? (position / duration) * 100 : 0;
      // One custom property on the container; every row's cursor inherits it.
      this._tracksEl.style.setProperty('--playhead', `${percent}%`);
    }
  }

  // ---------------------------------------------------------------------------
  // Export
  // ---------------------------------------------------------------------------

  /**
   * Start a download for one track.
   *
   * The bytes are fetched, then handed to the user through an object URL,
   * rather than reached by assigning window.location. Navigation works while
   * the server answers with Content-Disposition: attachment — but a 404 from
   * an expired session (the job store's TTL is an hour) answers JSON with no
   * disposition, and the browser then navigates the page away, taking every
   * decoded buffer, the mute/solo picture and the diagnostics with it. That
   * would also be the one failure in this module that never reaches the
   * banner. Fetching first makes an export failure read like every other
   * failure and leaves the workbench standing.
   *
   * @param {number} trackId
   * @param {'wav'|'mid'} format
   */
  /**
   * Human-readable byte size. Deliberately coarse — this is a warning, not a
   * measurement, and "3.2 GB" communicates better than "3,247 MB".
   */
  _formatSize(bytes) {
    const n = Number(bytes) || 0;
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
    if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
    if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
    return `${n} bytes`;
  }

  /**
   * Label each export option with what it will actually cost.
   *
   * The point of putting the size on the button is that the decision and the
   * warning become one thing. A ten-minute analysis is 3.2 GB of WAV, and a
   * user should learn that while choosing rather than while waiting. Failing
   * to fetch estimates is not worth an error banner — the buttons simply keep
   * their generic descriptions and still work.
   */
  async _loadExportEstimates() {
    if (!this._jobId) return;
    let data;
    try {
      data = await this._api.exportEstimate(this._jobId);
    } catch {
      return;
    }
    this._exportEstimate = data.formats || {};
    for (const [format, info] of Object.entries(this._exportEstimate)) {
      const el = document.querySelector(`[data-detail="${format}"]`);
      if (!el || !info) continue;
      const files = Number(info.files) || 0;
      if (files === 0) {
        el.textContent = 'nothing to export in this format';
        const button = el.closest('.export-option');
        if (button) button.disabled = true;
        continue;
      }
      const noun = format === 'mid-single' ? 'track' : 'file';
      el.textContent =
        `${files} ${noun}${files === 1 ? '' : 's'} · ${this._formatSize(info.bytes)}`;
    }
  }

  /**
   * Download every track at once.
   *
   * The server builds the archive, which for a long analysis takes real time,
   * so the control reports that it is working and refuses to start a second
   * one on top. Whatever comes back, the user is told how many files it holds
   * and how many were skipped — the counts arrive in headers precisely so this
   * can be said without opening the zip.
   */
  /**
   * Show the re-analyse button only when it would do something.
   *
   * Offering it with no changed settings would promise a different result for
   * a minute of work and then produce the identical analysis.
   */
  _refreshReanalyze() {
    if (!this._reanalyzeButton) return;
    const changed = this._settings ? this._settings.changedCount() : 0;
    const usable = Boolean(this._jobId) && changed > 0;
    this._reanalyzeButton.hidden = !usable;
    this._reanalyzeButton.textContent =
      changed === 1 ? 'Re-analyse (1 change)' : `Re-analyse (${changed} changes)`;
  }

  /**
   * Analyse the same audio again with the current settings.
   *
   * The server still holds the decoded audio from the first pass, so this
   * costs one analysis rather than an upload and a decode as well. It returns
   * a NEW job; the previous one is left alone until it expires, so backing out
   * of a change means reloading a result that still exists rather than
   * rebuilding it.
   */
  async _reanalyze() {
    if (!this._jobId || !this._settings) return;
    const settings = this._settings.values();
    if (!Object.keys(settings).length) return;

    const previousJob = this._jobId;
    try {
      this._clearError();
      const created = await this._api.reanalyze(previousJob, settings);
      if (!created || !created.job_id) throw new ApiError(GENERIC_ERROR, 0);

      // Tear the workbench down before following the new job. The tracks,
      // transport and buffers on screen belong to the previous analysis, and a
      // re-analysis produces a different set — leaving them up would show the
      // old rows filling with the new one's audio.
      this._reset();
      this._busy = true;
      this._setStage('progress');
      this._setProgress(0.01, 'Re-analysing with your settings');

      this._jobId = created.job_id;
      // The same feed the upload path uses. A re-analysis differs only in
      // where the job came from, so it gets the same progress handling rather
      // than a second copy of it.
      this._startFeed(created.job_id);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : GENERIC_ERROR;
      this._setBanner(message);
      this._announce(message);
    }
  }

  async _exportAll(format) {
    if (!this._jobId || this._exportingAll) return;

    const info = (this._exportEstimate && this._exportEstimate[format]) || null;
    const files = info ? Number(info.files) || 0 : 0;
    if (info && files === 0) {
      const nothing = 'There is nothing to export in that format.';
      this._setNotice(nothing);
      this._announce(nothing);
      return;
    }

    const menu = document.getElementById('export-all');
    if (menu) menu.open = false;

    // *** THE BROWSER DOWNLOADS THIS ITSELF. WE DO NOT FETCH IT. ***
    //
    // The obvious implementation — fetch, response.blob(), object URL, click a
    // hidden <a> — is what every "download a file" snippet shows, and it is
    // wrong here for one decisive reason: blob() buffers the ENTIRE archive in
    // memory before a single byte reaches disk. A ten-minute analysis is about
    // 3.2 GB of WAV. The tab cannot hold that, so the download simply never
    // happens, and because the failure is an allocation rather than an
    // exception there is often nothing useful to report either.
    //
    // Navigating to the URL instead hands the transfer to the browser, which
    // streams it straight to disk with no memory ceiling, shows its own
    // progress, and lets the user cancel. The server already sends
    // Content-Disposition: attachment, so this downloads rather than navigates
    // away — the page stays exactly as it is.
    //
    // The cost is that we no longer see the X-Export-Included / X-Export-Skipped
    // response headers, so the counts below come from the estimate fetched when
    // the workbench loaded. That is the same data the server derives them from.
    const url = this._api.exportAllUrl(this._jobId, format);

    this._exportingAll = true;
    const summary = document.getElementById('export-all-summary');
    const previous = summary ? summary.textContent : 'Export all';
    if (summary) summary.textContent = 'Preparing…';

    try {
      // A real anchor with a real href, clicked inside the user's own click
      // handler. Not an iframe, and not a blob.
      //
      //   blob  -> buffers the whole archive in memory first; a 3.2 GB export
      //            never completes.
      //   iframe-> the browser cannot always tell the navigation apart from an
      //            automatic one, and blocks it as an unsolicited download.
      //   anchor-> a download the user visibly asked for, streamed to disk by
      //            the browser, with its own progress and cancel.
      //
      // The `download` attribute names the file; the server's
      // Content-Disposition supplies the real one and wins, so the two agree.
      const link = document.createElement('a');
      link.href = url;
      link.download = '';
      link.rel = 'noopener';
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      link.remove();

      const skipped = Math.max(0, (this._summary ? this._summary.tracks.length : files) - files);
      const message = skipped
        ? `Preparing ${files} file${files === 1 ? '' : 's'}. ` +
          `${skipped} track${skipped === 1 ? ' is' : 's are'} not included — see MANIFEST.txt inside.`
        : `Preparing ${files} file${files === 1 ? '' : 's'}. Your browser will save it when it is ready.`;
      this._announce(message);
      this._setNotice(skipped ? message : null);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : GENERIC_ERROR;
      this._setNotice(null);
      this._setBanner(message);
      this._announce(message);
    } finally {
      // A large archive takes real time to build server-side, and the button
      // must not stay stuck on "Preparing…" for a transfer we can no longer
      // observe. Released on a timer rather than on completion, deliberately.
      setTimeout(() => {
        this._exportingAll = false;
        if (summary) summary.textContent = previous || 'Export all';
      }, 2500);
    }
  }


  async _exportTrack(trackId, format) {
    if (!this._jobId) return;
    try {
      const { blob, filename } = await this._api.exportFile(this._jobId, trackId, format);
      this._saveBlob(blob, filename);
    } catch (err) {
      // Deliberately _setBanner and not _showError: the analysis is still on
      // screen and still playable, and tearing it down would cost the user far
      // more than the failed download did.
      const message = err instanceof ApiError ? err.message : GENERIC_ERROR;
      this._setBanner(message);
      this._announce(message);
    }
  }

  /**
   * Hand a fetched export to the user as a file.
   *
   * A synthetic <a download> click rather than window.open: it is the only
   * route that both names the file and cannot navigate the page. The anchor is
   * attached to the document because Firefox ignores a click on a detached
   * one, and removed immediately after.
   *
   * @param {Blob} blob
   * @param {string} filename
   */
  _saveBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = el('a', { attrs: { href: url, download: filename, rel: 'noopener' } });
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    link.remove();
    // Revoking synchronously can beat the browser's own read of the URL in
    // some engines, which produces a silently empty file. One second is far
    // longer than that read and still bounds the leak.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ---------------------------------------------------------------------------
  // Stages, errors, and teardown
  // ---------------------------------------------------------------------------

  /** @param {'drop'|'progress'|'workbench'} stage */
  _setStage(stage) {
    if (this._dropzone) this._dropzone.hidden = stage !== 'drop';
    if (this._progress) this._progress.hidden = stage !== 'progress';
    if (this._workbench) this._workbench.hidden = stage !== 'workbench';
  }

  /**
   * @param {number} fraction 0..1
   * @param {string} stage user-facing stage name
   */
  _setProgress(fraction, stage) {
    const value = Number.isFinite(fraction) ? Math.min(1, Math.max(0, fraction)) : 0;
    if (this._progressFill) this._progressFill.style.width = `${value * 100}%`;

    // The stage text is the thing worth hearing, and it sits in the section's
    // aria-live region, so changing it announces "Following tones through time"
    // rather than a number. The percentage goes on the bar's own progressbar
    // role, where an assistive technology can read it on demand instead of
    // interrupting with it four times a second.
    if (this._progressStage && stage) this._progressStage.textContent = stage;
    if (this._progressBar) {
      this._progressBar.setAttribute('aria-valuenow', String(Math.round(value * 100)));
    }
  }

  /** @param {string} text */
  _setAudioStatus(text) {
    if (this._audioStatus) this._audioStatus.textContent = text;
  }

  /**
   * Say something to a screen reader without putting it on screen twice.
   *
   * A separate polite live region, rather than making the visible counter live:
   * a counter that announces "3 of 200", "4 of 200", "5 of 200" is worse than
   * useless, so only milestones come through here.
   *
   * @param {string} text
   */
  _announce(text) {
    if (!this._announcer) return;
    this._announcer.textContent = text;
  }

  /**
   * Show a failure in the page and hand the user back the dropzone.
   *
   * The banner is role="alert", so it is announced immediately rather than
   * waiting for the user to happen to move focus near it. And the UI is *never*
   * left on the progress stage: the point of this function is that there is no
   * path to a spinner that spins forever.
   *
   * @param {string} message
   */
  _showError(message) {
    this._reset();
    this._setBanner(message);
    this._setStage('drop');
  }

  /**
   * Put a message in the alert banner and touch nothing else.
   *
   * Split out of _showError because not every failure is fatal to the session.
   * A failed export leaves a perfectly good workbench behind — decoded audio,
   * mutes, solos, diagnostics — and unwinding all of that would cost the user
   * more than the error did.
   *
   * @param {string} message
   */
  /**
   * Show something the user should know that is NOT a failure.
   *
   * Deliberately a different element and a different colour from _setBanner.
   * MIDI export always leaves the residual behind, so "1 track is not
   * included" fires on every MIDI export — and shown in the red error banner,
   * every successful export looked like a broken one. Passing null clears it.
   */
  _setNotice(message) {
    if (!this._noticeBanner) return;
    if (!message) {
      this._noticeBanner.hidden = true;
      this._noticeBanner.textContent = '';
      return;
    }
    this._noticeBanner.textContent = message;
    this._noticeBanner.hidden = false;
  }

  _setBanner(message) {
    if (!this._errorBanner) return;
    this._errorBanner.textContent = message || GENERIC_ERROR;
    this._errorBanner.hidden = false;
  }

  _clearError() {
    if (this._errorBanner) {
      this._errorBanner.hidden = true;
      this._errorBanner.textContent = '';
    }
    // The notice goes with it. Every caller of this is starting something new,
    // and a note about the previous export's skipped tracks hanging over a
    // fresh analysis would describe a file that is no longer on screen.
    this._setNotice(null);
  }

  /**
   * Tear the session down.
   *
   * Order matters: stop the feed before dropping the job id, and stop the
   * transport before dropping the rows, so nothing can call back into state
   * that has already been cleared.
   */
  _reset() {
    if (this._feed) {
      this._feed.stop();
      this._feed = null;
    }
    this._stopPlayheadLoop();
    if (this._transport) {
      // dispose(), not just stop(): the AudioContext is reused for the whole
      // session, so a Transport that is merely dropped leaves its master and
      // one GainNode per track wired to the destination forever. Two hundred
      // nodes per analysis, never collected. dispose() stops playback and then
      // unwires the graph it built.
      this._transport.dispose();
      this._transport = null;
    }
    if (this._playback) {
      // The strategy's own cache holds decoded AudioBuffers — the megabytes
      // this whole mode exists to bound — so dropping the reference is not
      // enough while an in-flight fetch could still be holding the object.
      this._playback.dispose();
      this._playback = null;
    }
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }

    this._jobId = null;
    // Cleared BEFORE the rows: every in-flight fetch checks `_summary` against
    // the one it started with, so nulling it here is what makes them all
    // stand down rather than register audio into the next analysis.
    this._summary = null;
    this._rows.clear();
    this._groups = [];
    this._muted.clear();
    this._soloed.clear();
    this._applyPending = false;
    this._audioReady = false;
    this._busy = false;

    if (this._tracksEl) this._tracksEl.replaceChildren();
    if (this._diagnostics) this._diagnostics.replaceChildren();
    this._setAudioStatus('');
    this._setPlaybackNote('');
    this._setPlayingUi(false);
    this._setProgress(0, 'Analysing…');
    if (this._playheadEl) this._playheadEl.textContent = '0.00 s / 0.00 s';
    if (this._viewToggle) this._viewToggle.setAttribute('aria-pressed', 'false');
  }

  // ---------------------------------------------------------------------------
  // Redrawing
  // ---------------------------------------------------------------------------

  /**
   * Redraw every canvas on screen.
   *
   * Groups as well as rows: a group's canvas is its own overlay of its
   * children's trajectories, so it needs the same resize and expand treatment.
   * A collapsed group's children measure 0x0 and bail out by themselves, which
   * is why this can safely be called for everything every time.
   */
  _drawAllRows() {
    for (const row of this._rows.values()) row.draw();
    for (const group of this._groups) group.draw();
  }

  /**
   * Redraw rows when the window resizes.
   *
   * A canvas does not reflow: its backing store is fixed until something
   * reassigns it, so a widened window stretches yesterday's pixels. Debounced
   * with rAF because a drag-resize fires dozens of events per second and each
   * redraw is a few hundred strokes per row.
   */
  _wireResize() {
    let pending = null;
    window.addEventListener('resize', () => {
      if (pending !== null) return;
      pending = requestAnimationFrame(() => {
        pending = null;
        this._drawAllRows();
      });
    });
  }

  /**
   * Draw a row the moment it first has a size.
   *
   * The section is hidden while the rows are built, so the first draw attempt
   * measures 0x0 and bails. ResizeObserver fires once the section is shown,
   * which is the earliest point the canvas can be sized correctly.
   */
  _observeRows() {
    if (typeof ResizeObserver !== 'function' || !this._tracksEl) return;
    if (this._resizeObserver) this._resizeObserver.disconnect();

    let pending = null;
    this._resizeObserver = new ResizeObserver(() => {
      if (pending !== null) return;
      pending = requestAnimationFrame(() => {
        pending = null;
        this._drawAllRows();
      });
    });
    this._resizeObserver.observe(this._tracksEl);
  }
}

// =============================================================================
// Boot, and the module's declared contract
// =============================================================================

/** The single App instance. Created on DOM ready. */
let app = null;

function boot() {
  app = new App();
  app.start();
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
}

/**
 * Upload a file and open the progress socket.
 * @param {File} file
 */
async function analyze(file) {
  if (!app) throw new Error('The page is not ready yet.');
  return app._handleFile(file);
}

/**
 * Fetch each track's rendered PCM and hand it to the transport.
 * @param {object} summary
 */
async function loadTracks(summary) {
  if (!app) throw new Error('The page is not ready yet.');
  return app._loadAllAudio(summary);
}

/**
 * Build one row per track: mute, solo, label, waveform, export buttons.
 * @param {object} summary
 */
function renderWorkbench(summary) {
  if (!app) throw new Error('The page is not ready yet.');
  return app.renderWorkbench(summary);
}

export {
  analyze,
  loadTracks,
  renderWorkbench,
  Transport,
  App,
  ApiClient,
  ApiError,
  JobFeed,
  TrackRow,
  // Added in Phase 6. The scaffold's contract above is unchanged; these are
  // exported so the grouping rule and the two playback strategies can be
  // exercised on their own rather than only through a whole session.
  TrackGroupRow,
  PlaybackStrategy,
  EagerPlayback,
  BoundedPlayback,
  estimateDecodedBytes,
  excludeKey,
  PLAYBACK_MEMORY_BUDGET_BYTES,
};
