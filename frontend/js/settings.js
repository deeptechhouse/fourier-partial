/**
 * settings.js — the analysis settings panel.
 *
 * Builds itself from GET /api/settings/schema. Nothing about a setting's range,
 * default, or help text is written here: duplicating that in JavaScript would
 * drift from the Python the moment a default changed, leaving a "reset" button
 * that restores a value the analyser no longer uses.
 *
 * ONLY CHANGED VALUES ARE SENT. This is the single most important thing in the
 * file, and it is not an optimisation.
 *
 * The server picks settings by duration: a ten-minute file starts from cheaper
 * long-form settings (bigger hop, no zero-padding, fewer tracks) because at
 * phrase settings it needs ~6.8 GB and thousands of frames. Those defaults are
 * chosen AFTER the audio is decoded, when its length is known.
 *
 * If the panel sent every value it holds, it would send the phrase defaults it
 * was built with — and every one of them would count as an explicit override,
 * silently dragging a long recording back to settings that cannot analyse it.
 * The user would have changed one slider and broken the file.
 *
 * So `values()` returns only what differs from the default. Touch nothing and
 * it sends nothing, and the duration tier applies exactly as it would have.
 */

/** Powers of two between min and max, for the window and hop controls. */
function powersOfTwo(min, max) {
  const out = [];
  for (let v = 1; v <= max; v *= 2) if (v >= min) out.push(v);
  return out;
}

/** Round to a sensible number of decimals for display. */
function tidy(value, step) {
  if (step >= 1) return String(Math.round(value));
  const decimals = Math.min(4, String(step).split('.')[1]?.length ?? 2);
  return value.toFixed(decimals);
}

export class SettingsPanel {
  /**
   * @param {HTMLElement} host where to render
   * @param {object} schema the payload from /api/settings/schema
   * @param {{onChange?: Function}} [deps]
   */
  constructor(host, schema, deps = {}) {
    this._host = host;
    this._schema = schema;
    this._onChange = typeof deps.onChange === 'function' ? deps.onChange : () => {};

    /** @type {Map<string, {setting: object, input: HTMLElement, readout: HTMLElement}>} */
    this._controls = new Map();
    /** Current value per setting name. Starts at the defaults. */
    this._values = new Map();

    this._showAdvanced = false;
    this._render();
  }

  // --- Public API ----------------------------------------------------------

  /**
   * The settings the user actually changed.
   *
   * Deliberately NOT every value — see the note at the top of this file. An
   * untouched panel returns {}, which lets the server apply its
   * duration-appropriate defaults rather than being overridden with phrase
   * defaults that would make a long file unanalysable.
   */
  values() {
    const changed = {};
    for (const [name, { setting }] of this._controls) {
      const value = this._values.get(name);
      if (value !== setting.default) changed[name] = value;
    }
    return changed;
  }

  /** How many settings differ from the defaults. */
  changedCount() {
    return Object.keys(this.values()).length;
  }

  /** Put everything back, and tell the caller. */
  reset() {
    for (const [name, { setting }] of this._controls) this._set(name, setting.default);
    this._refreshSummary();
    this._onChange(this.values());
  }

  /** Apply a saved set of values (used when re-opening after an analysis). */
  apply(values) {
    for (const [name, value] of Object.entries(values || {})) {
      if (this._controls.has(name)) this._set(name, Number(value));
    }
    this._refreshSummary();
  }

  // --- Rendering -----------------------------------------------------------

  _render() {
    this._host.textContent = '';

    for (const group of this._schema.groups || []) {
      const section = document.createElement('section');
      section.className = 'settings-group';

      const heading = document.createElement('h3');
      heading.textContent = group.label;
      section.appendChild(heading);

      const blurb = document.createElement('p');
      blurb.className = 'settings-blurb';
      blurb.textContent = group.blurb;
      section.appendChild(blurb);

      for (const setting of group.settings) {
        section.appendChild(this._control(setting));
      }
      this._host.appendChild(section);
    }

    // Derived consequences of the window and hop. Shown because the raw
    // numbers mean nothing on their own: "4096" is not information, "93 ms
    // window, 10.8 Hz resolution" is the actual trade being made.
    this._derived = document.createElement('p');
    this._derived.className = 'settings-derived';
    this._host.appendChild(this._derived);

    const footer = document.createElement('div');
    footer.className = 'settings-footer';

    this._advancedToggle = document.createElement('button');
    this._advancedToggle.type = 'button';
    this._advancedToggle.className = 'ghost';
    this._advancedToggle.setAttribute('aria-pressed', 'false');
    this._advancedToggle.textContent = 'Show advanced';
    this._advancedToggle.addEventListener('click', () => {
      this._showAdvanced = !this._showAdvanced;
      this._advancedToggle.setAttribute('aria-pressed', String(this._showAdvanced));
      this._advancedToggle.textContent = this._showAdvanced ? 'Hide advanced' : 'Show advanced';
      this._host.classList.toggle('shows-advanced', this._showAdvanced);
    });
    footer.appendChild(this._advancedToggle);

    const resetButton = document.createElement('button');
    resetButton.type = 'button';
    resetButton.className = 'ghost';
    resetButton.textContent = 'Reset to defaults';
    resetButton.addEventListener('click', () => this.reset());
    footer.appendChild(resetButton);

    this._summary = document.createElement('span');
    this._summary.className = 'settings-summary';
    footer.appendChild(this._summary);

    this._host.appendChild(footer);
    this._refreshDerived();
    this._refreshSummary();
  }

  _control(setting) {
    const row = document.createElement('div');
    row.className = 'setting' + (setting.advanced ? ' is-advanced' : '');

    const id = `set-${setting.name}`;
    const label = document.createElement('label');
    label.setAttribute('for', id);
    label.className = 'setting-label';
    label.textContent = setting.label;

    const readout = document.createElement('output');
    readout.className = 'setting-value';
    readout.setAttribute('for', id);

    const input = document.createElement('input');
    input.type = 'range';
    input.id = id;
    input.step = String(setting.step);

    // Powers of two are indexed rather than valued: a linear slider from 512
    // to 16384 spends most of its travel in a range nobody wants, and can land
    // on values the analyser rejects. Indexing makes every stop legal and
    // gives each one equal room.
    const scale = setting.power_of_two ? powersOfTwo(setting.min, setting.max) : null;
    if (scale) {
      input.min = '0';
      input.max = String(scale.length - 1);
      input.step = '1';
      input.value = String(Math.max(0, scale.indexOf(setting.default)));
    } else {
      input.min = String(setting.min);
      input.max = String(setting.max);
      input.value = String(setting.default);
    }

    const help = document.createElement('p');
    help.className = 'setting-help';
    help.textContent = setting.help;

    // What moving it does, in the user's terms rather than the parameter's.
    const arrows = document.createElement('p');
    arrows.className = 'setting-arrows';
    if (setting.lower || setting.higher) {
      arrows.innerHTML = '';
      const lower = document.createElement('span');
      lower.textContent = `↓ ${setting.lower}`;
      const higher = document.createElement('span');
      higher.textContent = `↑ ${setting.higher}`;
      arrows.appendChild(lower);
      arrows.appendChild(higher);
    }

    row.appendChild(label);
    row.appendChild(readout);
    row.appendChild(input);
    row.appendChild(help);
    row.appendChild(arrows);

    this._controls.set(setting.name, { setting, input, readout, scale, row });
    this._values.set(setting.name, setting.default);

    input.addEventListener('input', () => {
      const raw = Number(input.value);
      this._values.set(setting.name, scale ? scale[raw] : raw);
      this._paint(setting.name);
      this._refreshDerived();
      this._refreshSummary();
      this._onChange(this.values());
    });

    this._paint(setting.name);
    return row;
  }

  _set(name, value) {
    const entry = this._controls.get(name);
    if (!entry) return;
    this._values.set(name, value);
    entry.input.value = entry.scale ? String(entry.scale.indexOf(value)) : String(value);
    this._paint(name);
    this._refreshDerived();
  }

  _paint(name) {
    const { setting, readout, row } = this._controls.get(name);
    const value = this._values.get(name);

    // A share reads better as a percentage than as 0.005.
    readout.textContent =
      setting.max <= 1 && setting.step < 0.1
        ? `${(value * 100).toFixed(1)}%`
        : tidy(value, setting.step);

    // Mark anything the user has moved, so "what did I change?" is answerable
    // at a glance rather than by comparing eleven numbers to remembered ones.
    row.classList.toggle('is-changed', value !== setting.default);
  }

  /**
   * Translate the window and hop into what they actually mean.
   *
   * The uncertainty principle is the reason this exists: window size is not a
   * quality setting, it is a choice between knowing *when* and knowing *what
   * pitch*, and the numbers below are the two halves of that trade.
   */
  _refreshDerived() {
    if (!this._derived) return;
    const sampleRate = 44100;
    const window = this._values.get('window_size') ?? 4096;
    const hop = this._values.get('hop_size') ?? 512;

    const windowMs = (1000 * window) / sampleRate;
    const resolutionHz = sampleRate / window;
    const framesPerSecond = sampleRate / hop;

    this._derived.textContent =
      `At these settings: each look spans ${windowMs.toFixed(1)} ms, ` +
      `tones closer than ${resolutionHz.toFixed(1)} Hz merge, ` +
      `and it looks ${Math.round(framesPerSecond)} times a second.`;
  }

  _refreshSummary() {
    if (!this._summary) return;
    const n = this.changedCount();
    this._summary.textContent = n === 0 ? 'Using defaults' : `${n} changed`;
    this._summary.classList.toggle('is-changed', n > 0);
  }
}

/** Fetch the schema and build a panel into `host`. Resolves to null on failure. */
export async function createSettingsPanel(host, apiBase, deps) {
  try {
    const response = await fetch(`${apiBase}/settings/schema`);
    if (!response.ok) return null;
    return new SettingsPanel(host, await response.json(), deps);
  } catch {
    // The panel is an enhancement. If it cannot be built, analysis still works
    // perfectly well with the defaults, so this must never be fatal.
    return null;
  }
}
