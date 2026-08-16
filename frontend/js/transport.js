/**
 * transport.js — synchronised multi-track playback.
 *
 * THE WHOLE TRICK
 * ---------------
 * Every track is a decoded AudioBuffer. To play them in sync we do NOT start
 * them "at the same time" in the JavaScript sense — setTimeout and friends are
 * milliseconds-jittery, and a few ms of skew between two partials is an
 * audible phasing artifact.
 *
 * Instead we schedule every source at one shared AudioContext timestamp:
 *
 *     const startAt = ctx.currentTime + LOOKAHEAD;
 *     sources.forEach(s => s.start(startAt, offset));
 *
 * The audio thread honours that timestamp to the sample. Two hundred tracks
 * start on exactly the same sample, every time, regardless of what the main
 * thread is doing.
 *
 * WHY SOLO/MUTE IS JUST GAIN
 * --------------------------
 * We never stop and restart sources to mute a track — that would resynchronise
 * everything and click. Each track has a GainNode; muting sets its gain to 0
 * with a short ramp. The track keeps playing silently, stays in sync, and
 * unmutes instantly.
 *
 * WHY THIS GUARANTEES THE PRODUCT'S CLAIM
 * ---------------------------------------
 * The backend already guarantees sum(tracks) == original sample-for-sample.
 * Because playback is a sample-aligned sum of those same buffers at unity
 * gain, "play everything" reproduces the original exactly. The promise is
 * kept by arithmetic rather than by careful mixing.
 *
 * STATUS: implemented (Phase 5). The contract is unchanged from the scaffold.
 */

/**
 * Seconds of scheduling lookahead. Enough to absorb main-thread hiccups.
 *
 * ADJUSTABLE. This is the only latency the user feels between clicking Play
 * and hearing sound, so it is a trade: too small and a garbage-collection
 * pause on the main thread can push `startAt` into the past, at which point
 * the audio thread starts the source *immediately* and that one track lands
 * late — the exact skew this whole module exists to prevent. Too large and
 * the button feels sluggish. 80 ms is comfortably above a typical GC pause
 * and comfortably below the ~100 ms where a human notices the delay.
 */
const LOOKAHEAD_S = 0.08;

/**
 * Gain ramp time for mute/solo, in seconds. Long enough to avoid clicks.
 *
 * ADJUSTABLE. A gain change is a step discontinuity in the waveform, and a
 * step is broadband noise — that is the "click". Spreading it over 15 ms
 * turns the step into a slope no one can hear as a transient, while staying
 * short enough that solo still feels instant when A/B-ing two tracks.
 */
const RAMP_S = 0.015;

export class Transport {
  /** @param {AudioContext} ctx */
  constructor(ctx) {
    this._ctx = ctx;
    /** @type {Map<number, {buffer: AudioBuffer, gain: GainNode, source: AudioBufferSourceNode|null}>} */
    this._tracks = new Map();
    this._master = ctx.createGain();
    this._master.connect(ctx.destination);
    this._startedAt = 0;
    this._offset = 0;
    this._playing = false;
    this._soloed = new Set();
  }

  /**
   * Register a track's decoded audio.
   * @param {number} trackId
   * @param {AudioBuffer} buffer
   */
  addTrack(trackId, buffer) {
    const existing = this._tracks.get(trackId);

    if (existing) {
      // Re-registering an id replaces its audio (the fetch-on-demand path in
      // TDD section 9 can hand us a track twice). Keep the GainNode: it holds
      // this track's mute/solo position and is already wired to the master
      // bus, so reusing it means a re-fetched track cannot silently un-mute
      // itself. Only the source is stale, because it is still playing the old
      // buffer.
      //
      // Check for the end watcher BEFORE releasing, because releasing clears
      // the handler. If this happened to be the longest track, dropping its
      // source would leave nothing to notice that playback finished, and the
      // transport would believe it was playing forever.
      const wasEndWatcher = existing.source !== null && existing.source.onended !== null;
      this._releaseSource(existing);
      existing.buffer = buffer;
      if (this._playing && wasEndWatcher) this._armEndWatcher();
    } else {
      // One GainNode per track, created once and kept for the track's whole
      // life. This is the node mute and solo write to; sources come and go
      // underneath it. Starting at unity matters — the "unmute everything
      // reproduces the original" claim is only true if every gain is exactly
      // 1.0, so the default has to be exact, not approximately exact.
      const gain = this._ctx.createGain();
      gain.gain.value = 1;
      gain.connect(this._master);

      // `muted` is kept per track and deliberately separate from the soloed
      // Set. Two independent facts, two independent stores — collapsing them
      // into one "is this audible" boolean is what makes solo/mute interaction
      // go wrong, because you can no longer answer "what should this track do
      // when the last solo is cleared?"
      this._tracks.set(trackId, { buffer, gain, source: null, muted: false });
    }

    // A brand-new track must obey a solo that is already active, otherwise
    // adding a track mid-session would leak audio into a soloed mix.
    this._updateGains();

    // NOTE: a track registered while the transport is running does not join
    // mid-flight — it has no source yet, so it stays silent until the next
    // play(). Deliberate: joining late means picking a second start timestamp,
    // and every timestamp that is not the shared one is a chance to be a
    // sample out. Callers that want it audible now should call
    // `seek(transport.currentTime)`, which restarts every track together.
  }

  /**
   * Start all tracks from `offset` seconds, sample-aligned.
   * @param {number} offset
   */
  play(offset = 0) {
    // Play while already playing means "restart from here". Without this we
    // would strand the running sources: still connected, still audible, no
    // longer referenced by the map, and impossible to stop.
    if (this._playing) this.stop();

    // Clamp before scheduling. A negative offset makes source.start() throw a
    // RangeError, which would abort the loop half-way and leave some tracks
    // started and some not — the worst possible failure for a sync engine.
    // (_clampToTimeline also rejects NaN, which Math.min/max would let straight
    // through into start() as a TypeError with the same half-started result.)
    const startOffset = this._clampToTimeline(offset);

    // Nothing decoded yet. Remember where the playhead was asked to sit so a
    // later play() resumes from the right place, but do not claim to be
    // playing — there is no source to ever end, so `_playing` would stick true
    // forever and the Play button would never reset.
    if (this._tracks.size === 0) {
      this._offset = startOffset;
      return;
    }

    // Browsers hand back a suspended AudioContext until a user gesture, and a
    // suspended context's currentTime does not advance. Resuming is async but
    // we deliberately do not await it: awaiting would move the scheduling below
    // into a later microtask, and the whole point is that every source gets a
    // timestamp computed in one uninterrupted synchronous block.
    if (this._ctx.state === 'suspended') this._ctx.resume();

    // *** THE LINE THE MODULE IS ABOUT ***
    // Read the clock ONCE. Every source below gets this identical value, so
    // they are sample-aligned by construction. Calling ctx.currentTime inside
    // the loop would give each track a slightly later time — a few hundred
    // microseconds of drift per track, which is precisely the phasing artifact
    // this design exists to eliminate.
    const startAt = this._ctx.currentTime + LOOKAHEAD_S;

    for (const track of this._tracks.values()) {
      // A fresh AudioBufferSourceNode per track, per play(). Sources are
      // single-use: once started they can never be started again, and calling
      // start() twice throws InvalidStateError. Reusing a source is the number
      // one bug in code shaped like this — the second Play silently does
      // nothing, or throws, depending on the browser.
      const source = this._ctx.createBufferSource();
      source.buffer = track.buffer;

      // source -> this track's gain -> master -> destination. Routing through
      // the per-track gain is what lets mute and solo work without touching
      // the source at all.
      source.connect(track.gain);

      // Start BEFORE handing the source to the track. If start() ever throws,
      // the map must not be left holding a source that was never started —
      // _releaseSource() would then call stop() on it, and stop()-without-start()
      // is an InvalidStateError, which would make every later stop() throw and
      // wedge the transport permanently. This ordering means a failed start
      // leaves the track exactly as it was: source-less, and recoverable.
      source.start(startAt, startOffset);
      track.source = source;
    }

    // `_startedAt` is the SCHEDULED time, not "now". The currentTime getter
    // subtracts it from the clock, so anchoring to startAt makes the playhead
    // read exactly `startOffset` at the instant sound actually begins. (During
    // the ~80 ms lookahead it reads slightly *before* startOffset, which is
    // honest: playback has not started yet.)
    this._startedAt = startAt;
    this._offset = startOffset;
    this._playing = true;

    // Somebody has to notice when the audio runs out. Must come after
    // `_playing` is true, because the watcher's job is to turn it back off.
    this._armEndWatcher();

    // Re-assert mute/solo on the new sources. The gains already hold the right
    // values, so this normally ramps to where it already is — a no-op that
    // costs nothing and guarantees a track muted before Play starts silent.
    this._updateGains();
  }

  /** Stop all tracks and release their sources. */
  stop() {
    // Read the playhead BEFORE clearing `_playing`, because the currentTime
    // getter branches on it: once `_playing` is false it just returns the
    // stored offset, and we would be storing the value we were trying to
    // measure. Clamped because during the lookahead window the elapsed time is
    // still negative, and a negative resume position would throw in play().
    const position = Math.min(this.duration, Math.max(0, this.currentTime));

    this._releaseAllSources();

    // Remember where we were so the next play() resumes rather than restarts.
    this._offset = position;
    this._playing = false;
  }

  /**
   * Mute or unmute a track without disturbing sync.
   * @param {number} trackId
   * @param {boolean} muted
   */
  setMuted(trackId, muted) {
    const track = this._tracks.get(trackId);
    // Forgiving on unknown ids: the UI can race a click against a track list
    // that is still loading, and throwing there would break the whole row.
    if (!track) return;

    // Record the fact, then let one function decide what it sounds like. Note
    // what is NOT here: no source is stopped, started, or touched. The track
    // keeps playing silently and stays sample-aligned with everything else,
    // which is why unmuting is instant instead of a resync.
    track.muted = Boolean(muted);
    this._updateGains();
  }

  /**
   * Solo a track. With any track soloed, all non-soloed tracks are silenced.
   * @param {number} trackId
   * @param {boolean} soloed
   */
  setSoloed(trackId, soloed) {
    // Guard the add path especially: soloing an id we do not have would
    // silence every real track in favour of a track that does not exist.
    if (!this._tracks.has(trackId)) return;

    // A Set, not a single "the soloed track" field, because multiple solos are
    // the useful case — hearing the 1st and 3rd harmonic together is exactly
    // the comparison this product is for.
    if (soloed) {
      this._soloed.add(trackId);
    } else {
      this._soloed.delete(trackId);
    }

    // Every track's gain is recomputed, not just this one: soloing track 4
    // changes what tracks 1, 2 and 3 sound like.
    this._updateGains();
  }

  /**
   * Jump to `offset` seconds, staying in sync.
   *
   * There is no way to move a running AudioBufferSourceNode's playback
   * position — the offset is fixed at start(). So seeking while playing is
   * literally stop-and-play-again, which is fine: play() re-derives one shared
   * timestamp, so the tracks land back together at the new position.
   *
   * @param {number} offset
   */
  seek(offset) {
    const target = this._clampToTimeline(offset);

    if (this._playing) {
      this.stop();
      this.play(target);
    } else {
      // Stopped: just move the playhead. The next play() will start here.
      this._offset = target;
    }
  }

  /**
   * Release every node this Transport built, permanently.
   *
   * WHY THIS EXISTS. A Transport is per-analysis, but the AudioContext is not —
   * the page deliberately reuses one context for the whole session, because
   * browsers cap how many a page may create. Nodes are only collectable once
   * they are unreachable from the destination, so a Transport that is simply
   * dropped leaves its master plus one GainNode per track wired into a live
   * graph forever. That is the same leak the source path is careful about, only
   * worse: it is one per track per analysis, and nothing ever clears it.
   *
   * After this the instance is spent. Call it when you drop the reference.
   */
  dispose() {
    this.stop();

    for (const track of this._tracks.values()) {
      track.gain.disconnect();
    }

    this._master.disconnect();

    // Drop the buffers too. A decoded AudioBuffer is the largest object here by
    // far — a few megabytes per track — so holding the map alive after the
    // graph is gone would leak the audio itself, not just the nodes.
    this._tracks.clear();
    this._soloed.clear();
    this._playing = false;
    this._offset = 0;
  }

  /** Current playhead position in seconds. */
  get currentTime() {
    if (!this._playing) return this._offset;
    return this._offset + (this._ctx.currentTime - this._startedAt);
  }

  /**
   * Length of the piece in seconds — the longest registered track.
   *
   * The longest, not the sum and not the first: tracks are layers that all
   * start at zero, so the piece lasts as long as its longest layer. Partials
   * genuinely differ in length (a partial that dies early has a shorter
   * buffer), so this cannot be read off any single track.
   */
  get duration() {
    let longest = 0;
    for (const track of this._tracks.values()) {
      if (track.buffer.duration > longest) longest = track.buffer.duration;
    }
    return longest;
  }

  /** Whether sources are currently scheduled or sounding. */
  get isPlaying() {
    return this._playing;
  }

  /**
   * Clamp a caller-supplied position into [0, duration].
   *
   * Math.min/Math.max alone are not enough: they pass NaN straight through
   * (Math.min(NaN, 5) is NaN), and a NaN offset reaches source.start(), whose
   * IDL takes a restricted double and throws TypeError — mid-loop, which is the
   * half-started graph this clamp exists to prevent. Worse, a NaN stored in
   * `_offset` is sticky: currentTime returns NaN forever after, so the display
   * and every later play() inherit it. One place to reject it, used by both
   * entry points that accept a position from outside.
   *
   * @param {number} offset
   * @returns {number}
   */
  _clampToTimeline(offset) {
    const seconds = Number(offset);
    // NaN only. The infinities are left to Math, which already does the right
    // thing with them: -Infinity clamps to 0, +Infinity clamps to the duration.
    if (Number.isNaN(seconds)) return 0;
    return Math.max(0, Math.min(seconds, this.duration));
  }

  /**
   * Recompute every track's gain from (muted, soloedSet).
   *
   * ONE PLACE, ON PURPOSE. Mute and solo are not independent controls — solo
   * changes what mute means for every other track — so deriving the audible
   * state at each call site is how the two drift apart. Instead both are
   * stored as plain facts and this function is the only thing that turns them
   * into sound. Every mutation ends with a call to it.
   *
   * The truth table, with `anySolo = soloed.size > 0`:
   *
   *   anySolo  isSoloed  muted   gain   why
   *   -------  --------  -----   ----   ----------------------------------
   *   false    -         false   1.0    normal playback
   *   false    -         true    0.0    its own mute is the only rule
   *   true     true      false   1.0    the point of soloing it
   *   true     true      true    0.0    mute wins (see below)
   *   true     false     false   0.0    silenced by someone else's solo
   *   true     false     true    0.0    silenced twice over
   *
   * Mute wins over solo because it keeps each control's meaning single: mute
   * means "never make sound", solo means "of the unmuted tracks, only these".
   * Clearing the last solo therefore restores exactly the mute picture the
   * user built before soloing — nothing is lost and nothing is remembered
   * implicitly. (Pro Tools takes the other option, where solo overrides mute.
   * That is defensible too, but it makes solo silently rewrite mute state,
   * which is harder to undo than to explain.)
   */
  _updateGains() {
    const anySolo = this._soloed.size > 0;
    const now = this._ctx.currentTime;

    for (const [id, track] of this._tracks) {
      // Derived from both facts, never from a cached "audible" flag.
      const audible = !track.muted && (!anySolo || this._soloed.has(id));
      const target = audible ? 1 : 0;

      const param = track.gain.gain;

      // Cancel first, then pin the current value at `now`. Without the pin, a
      // ramp interpolates from the last *scheduled* event — which may be
      // seconds old — and the gain jumps to that stale value before ramping.
      // The pin makes the ramp always start from where the signal actually is,
      // so rapid solo-clicking cross-fades cleanly instead of clicking.
      param.cancelScheduledValues(now);
      param.setValueAtTime(param.value, now);

      // linearRampToValueAtTime, not setTargetAtTime. setTargetAtTime is
      // exponential: it approaches the target asymptotically and never
      // actually arrives, so an "unmuted" track would sit at 0.999… forever
      // and the sum of the tracks would no longer equal the original exactly.
      // A linear ramp lands on precisely 1.0, which is what the product's
      // whole claim rests on.
      param.linearRampToValueAtTime(target, now + RAMP_S);
    }
  }

  /**
   * Put exactly one `onended` handler on the source that will finish last.
   *
   * The longest buffer is the one still sounding when every other track has
   * run out, so its source is the only one whose end means "playback is over".
   * Watching a shorter track would reset the transport mid-phrase; watching
   * all of them would reset it on the FIRST track to finish, which is worse.
   *
   * Kept as its own function because the watcher has to be re-armed if the
   * source carrying it is ever released early (see addTrack).
   */
  _armEndWatcher() {
    let watched = null;
    let longestSoFar = -1;

    for (const track of this._tracks.values()) {
      // Only a live source can report an end. A track with no source (added
      // mid-playback, or just re-registered) cannot be the watcher.
      if (track.source === null) continue;
      if (track.buffer.duration > longestSoFar) {
        longestSoFar = track.buffer.duration;
        watched = track.source;
      }
    }

    if (watched === null) {
      // Nothing is sounding, so playback has in effect already ended. Say so
      // now rather than waiting for a callback that can never arrive.
      this._handlePlaybackEnd();
      return;
    }

    watched.onended = () => this._handlePlaybackEnd();
  }

  /**
   * Natural end of playback: return to a clean stopped state.
   *
   * Without this the UI would still believe it was playing, and the next click
   * on Play would be read as a restart-in-place of sources that have already
   * ended. Note this is NOT stop(): stop() saves the playhead so you can
   * resume, whereas a finished piece rewinds to the top — a player that
   * resumes from the very end plays nothing, which reads as a broken button.
   */
  _handlePlaybackEnd() {
    this._releaseAllSources();
    this._playing = false;
    this._offset = 0;
  }

  /**
   * Stop and discard one track's source, if it has one.
   * @param {{source: AudioBufferSourceNode|null}} track
   */
  _releaseSource(track) {
    const source = track.source;
    if (!source) return;

    // Drop the handler BEFORE stopping. stop() fires `onended` just like a
    // natural end does, and the natural-end handler rewinds the playhead to
    // zero — so leaving it attached would make every manual stop silently
    // forget the position we are about to save.
    source.onended = null;
    source.stop();

    // Disconnect so the node is unreachable and can be collected. A source
    // left connected to a live graph is a leak the GC cannot clear, and with
    // 200 tracks per play that adds up fast.
    source.disconnect();

    // Null it out. A source is single-use — keeping the reference would invite
    // exactly the "start it again" bug this module is careful to avoid.
    track.source = null;
  }

  /** Stop and discard every track's source. */
  _releaseAllSources() {
    for (const track of this._tracks.values()) {
      this._releaseSource(track);
    }
  }
}
