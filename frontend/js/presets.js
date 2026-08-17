/**
 * presets.js — saved settings combinations, in the settings panel.
 *
 * Sits above the sliders: choose a preset to load a combination, or save the
 * current one under a name. Built-ins are read-only; your own can be
 * overwritten and deleted.
 *
 * WHAT IT SENDS IS WHAT THE PANEL CHANGED, NOT EVERY VALUE
 * --------------------------------------------------------
 * `panel.values()` already returns only the settings that differ from their
 * defaults, and that is exactly what a preset stores. The reason matters: the
 * server chooses cheaper settings for long audio once it knows the duration, so
 * a preset carrying all twelve values would count as twelve explicit overrides
 * and would drag a ten-minute file onto settings that need gigabytes. Storing
 * the differences means a preset composes with that choice instead of
 * replacing it.
 *
 * APPLYING RESETS FIRST
 * ---------------------
 * Selecting a preset resets every slider before applying it. Without that,
 * switching from a preset that changes five things to one that changes two
 * would leave the other three set — a combination neither preset describes,
 * under a name that claims otherwise.
 */

/** Two settings objects hold the same combination. */
function sameSettings(a, b) {
  const ka = Object.keys(a || {});
  const kb = Object.keys(b || {});
  if (ka.length !== kb.length) return false;
  return ka.every((k) => Object.is(Number(a[k]), Number(b[k])));
}

export class PresetBar {
  /**
   * @param {HTMLElement} host where to render (prepended into the panel)
   * @param {{values: Function, apply: Function, reset: Function}} panel
   * @param {{apiBase?: string, onError?: Function}} [deps]
   */
  constructor(host, panel, deps = {}) {
    this._host = host;
    this._panel = panel;
    this._api = deps.apiBase || '/api';
    this._onError = typeof deps.onError === 'function' ? deps.onError : () => {};

    /** @type {Array<{name:string,settings:object,description:string,builtin:boolean}>} */
    this._presets = [];
    /** Where the file lives, so the user can find and edit it. */
    this._file = '';
    /** Guards the delete button's two-step confirmation. */
    this._pendingDelete = false;
    /** True while a preset is being applied; see _applySelected. */
    this._applying = false;

    this._build();
  }

  // --- Rendering -----------------------------------------------------------

  _build() {
    const bar = document.createElement('div');
    bar.className = 'preset-bar';

    const label = document.createElement('label');
    label.className = 'preset-label';
    label.setAttribute('for', 'preset-select');
    label.textContent = 'Preset';

    this._select = document.createElement('select');
    this._select.id = 'preset-select';
    this._select.className = 'preset-select';
    this._select.addEventListener('change', () => this._applySelected());

    this._saveButton = document.createElement('button');
    this._saveButton.type = 'button';
    this._saveButton.className = 'ghost';
    this._saveButton.textContent = 'Save as…';
    this._saveButton.addEventListener('click', () => this._openSaveForm());

    this._deleteButton = document.createElement('button');
    this._deleteButton.type = 'button';
    this._deleteButton.className = 'ghost';
    this._deleteButton.textContent = 'Delete';
    this._deleteButton.hidden = true;
    this._deleteButton.addEventListener('click', () => this._deleteSelected());

    bar.appendChild(label);
    bar.appendChild(this._select);
    bar.appendChild(this._saveButton);
    bar.appendChild(this._deleteButton);

    // What the chosen preset is for. Built-ins carry a sentence explaining the
    // material they suit, which is most of how anyone learns what the settings
    // underneath actually do.
    this._description = document.createElement('p');
    this._description.className = 'preset-description';

    // The save form, inline rather than a prompt(): a modal dialog blocks the
    // page and cannot be styled, and this needs to sit next to the values it
    // is about to capture.
    this._saveForm = document.createElement('form');
    this._saveForm.className = 'preset-save';
    this._saveForm.hidden = true;

    this._nameInput = document.createElement('input');
    this._nameInput.type = 'text';
    this._nameInput.className = 'preset-name';
    this._nameInput.maxLength = 48;
    this._nameInput.placeholder = 'Name this combination';
    this._nameInput.setAttribute('aria-label', 'Preset name');

    const confirm = document.createElement('button');
    confirm.type = 'submit';
    confirm.textContent = 'Save';

    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'ghost';
    cancel.textContent = 'Cancel';
    cancel.addEventListener('click', () => this._closeSaveForm());

    this._saveForm.appendChild(this._nameInput);
    this._saveForm.appendChild(confirm);
    this._saveForm.appendChild(cancel);
    this._saveForm.addEventListener('submit', (event) => {
      event.preventDefault();
      this._save();
    });

    this._note = document.createElement('p');
    this._note.className = 'preset-note';
    this._note.hidden = true;

    this._host.prepend(this._note);
    this._host.prepend(this._saveForm);
    this._host.prepend(this._description);
    this._host.prepend(bar);
  }

  // --- Data ----------------------------------------------------------------

  /** Load the list from the server. Safe to call again after any change. */
  async refresh() {
    try {
      const response = await fetch(`${this._api}/presets`);
      if (!response.ok) throw new Error('could not load presets');
      const data = await response.json();
      this._presets = Array.isArray(data.presets) ? data.presets : [];
      this._file = data.file || '';
    } catch {
      // Presets are an enhancement on top of settings that already work. A
      // failure here must not take the panel with it.
      this._presets = [];
    }
    this._renderOptions();
  }

  _renderOptions() {
    const previous = this._select.value;
    this._select.textContent = '';

    // "Custom" is first and is what any hand-edited combination selects. It is
    // an option rather than a blank so the control always describes the state
    // truthfully — there is no moment where it names a preset you are not on.
    const custom = document.createElement('option');
    custom.value = '';
    custom.textContent = 'Custom';
    this._select.appendChild(custom);

    const groups = [
      ['Built in', this._presets.filter((p) => p.builtin)],
      ['Yours', this._presets.filter((p) => !p.builtin)],
    ];
    for (const [label, list] of groups) {
      if (!list.length) continue;
      const group = document.createElement('optgroup');
      group.label = label;
      for (const preset of list) {
        const option = document.createElement('option');
        option.value = preset.name;
        option.textContent = preset.name;
        group.appendChild(option);
      }
      this._select.appendChild(group);
    }

    if (previous && this._presets.some((p) => p.name === previous)) {
      this._select.value = previous;
    }
    this.syncToPanel();
  }

  // --- Behaviour -----------------------------------------------------------

  /**
   * Point the control at whatever the sliders currently say.
   *
   * Called whenever a slider moves, so that hand-editing a preset's values
   * drops the selection to "Custom" rather than leaving a name on screen that
   * no longer describes what is set.
   */
  syncToPanel() {
    const current = this._panel.values();
    const match = this._presets.find((p) => sameSettings(p.settings, current));
    this._select.value = match ? match.name : '';
    this._paintSelection(match || null);
  }

  _paintSelection(preset) {
    this._description.textContent = preset ? preset.description || '' : '';
    this._description.hidden = !preset || !preset.description;
    this._deleteButton.hidden = !preset || preset.builtin;
    this._resetDeleteConfirmation();
  }

  _applySelected() {
    const name = this._select.value;
    if (!name) return; // "Custom" is a description, not an action.

    const preset = this._presets.find((p) => p.name === name);
    if (!preset) return;

    // Reset first — see the note at the top of this file. The flag tells the
    // panel's own reset() not to sync the selection back to "Custom" while we
    // are mid-apply, which would erase the choice being made.
    this._applying = true;
    try {
      this._panel.reset();
      this._panel.apply(preset.settings);
    } finally {
      this._applying = false;
    }
    this._paintSelection(preset);
    this._say(`Loaded “${preset.name}”.`);
  }

  _openSaveForm() {
    if (!Object.keys(this._panel.values()).length) {
      this._say('Change a setting first, then save it as a preset.', true);
      return;
    }
    this._saveForm.hidden = false;
    // Pre-fill with the current preset's name, so "tweak and re-save" is the
    // path of least resistance rather than a retyping exercise.
    this._nameInput.value = this._select.value || '';
    this._nameInput.focus();
    this._nameInput.select();
  }

  _closeSaveForm() {
    this._saveForm.hidden = true;
    this._nameInput.value = '';
  }

  async _save() {
    const name = this._nameInput.value.trim();
    if (!name) {
      this._say('A preset needs a name.', true);
      this._nameInput.focus();
      return;
    }

    try {
      const response = await fetch(`${this._api}/presets/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: this._panel.values(), description: '' }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || 'The preset could not be saved.');
      }
      this._closeSaveForm();
      await this.refresh();
      this._select.value = name;
      this.syncToPanel();
      this._say(`Saved “${name}”.${this._file ? ` In ${this._file}` : ''}`);
    } catch (err) {
      this._say(err.message || 'The preset could not be saved.', true);
    }
  }

  async _deleteSelected() {
    const name = this._select.value;
    if (!name) return;

    // Two steps, not a modal. Deleting a preset destroys work that took several
    // analyses to arrive at, and a single mis-click should not be enough — but
    // a confirm() dialog for something this small is heavier than the action.
    if (!this._pendingDelete) {
      this._pendingDelete = true;
      this._deleteButton.textContent = 'Really delete?';
      this._deleteButton.classList.add('is-danger');
      return;
    }

    try {
      const response = await fetch(`${this._api}/presets/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || 'The preset could not be removed.');
      }
      await this.refresh();
      this._say(`Deleted “${name}”. The settings are still loaded.`);
    } catch (err) {
      this._say(err.message || 'The preset could not be removed.', true);
    } finally {
      this._resetDeleteConfirmation();
    }
  }

  _resetDeleteConfirmation() {
    this._pendingDelete = false;
    this._deleteButton.textContent = 'Delete';
    this._deleteButton.classList.remove('is-danger');
  }

  /** A short line under the controls. `isProblem` colours it as a warning. */
  _say(message, isProblem = false) {
    this._note.textContent = message;
    this._note.hidden = false;
    this._note.classList.toggle('is-problem', Boolean(isProblem));
    if (isProblem) this._onError(message);
  }
}

/** Build a preset bar into `host`, above an existing settings panel. */
export async function createPresetBar(host, panel, deps) {
  const bar = new PresetBar(host, panel, deps);
  await bar.refresh();
  return bar;
}
