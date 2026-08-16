# fourier_partial — Navigation Map

> Decompose a short audio phrase into individually-audible pitched tracks using sinusoidal partial tracking, then play, visualise, and export them.

## Architecture Overview

A hexagonal (ports-and-adapters) Python backend does the DSP; a vanilla-JS browser frontend does visualisation and sample-accurate multi-track playback. Every replaceable concern — audio decoding, peak finding, tracking, grouping, rendering, export, job storage — sits behind a `Protocol` in `backend/core/interfaces.py`, so swapping any one of them means writing a class, not editing a pipeline.

The system's central design decision is that the **residual track is defined as an exact time-domain subtraction** (`original − Σ partials`). This makes the reconstruction bit-exact by arithmetic rather than by approximation: every track can be individually imperfect and "play everything" still reproduces the upload sample for sample.

## Directory Map

| Directory | Layer/Role | Description |
|---|---|---|
| `backend/core/` | Domain | Value objects, entities, ports, errors. Depends on nothing else. |
| `backend/audio_io/` | Adapter | Audio file decoding and validation — the untrusted-input boundary |
| `backend/analysis/` | Business Logic | STFT → peaks → tracking → grouping. The Fourier work. |
| `backend/synthesis/` | Business Logic | Oscillator bank resynthesis; owns the reconstruction invariant |
| `backend/export/` | Adapter | `.wav` (lossless) and `.mid` (lossy, pitch-bent) writers |
| `backend/api/` | Interface | FastAPI routes, WebSocket progress, background job store |
| `frontend/js/` | Interface | Transport, canvas rendering, upload wiring |
| `tests/` | Testing | Unit tests plus the reconstruction invariant suite |
| `docs/` | Documentation | TDD, error log, session summaries |

## Key Entry Points

- **Application start:** `backend/main.py` (`python3 -m backend.main`)
- **API routes:** `backend/api/routes.py`
- **The domain model:** `backend/core/types.py` — read this first
- **The contracts:** `backend/core/interfaces.py`
- **Tunable DSP parameters:** `AnalysisParams` in `backend/core/types.py`
- **Config:** `.env` (template at `.env.example`), loaded by `backend/config.py`
- **Design document:** `docs/TDD_fourier_partial.html`

## Module Relationships

- `api` calls `analysis` and `export`; it never touches DSP primitives directly
- `analysis` and `synthesis` depend only on `core`
- `core` depends on nothing in the project — if you find an import from `analysis` or `api` in `core`, a dependency arrow has been reversed
- `pipeline.py` is the only module that knows the full stage sequence; every stage is injected
- Frontend talks to the backend only through the documented API surface in `routes.py`

## Pipeline Order

```
audio_io.loader     →  mono float32 @ 44.1 kHz
analysis.stft       →  magnitude + phase per frame
analysis.peaks      →  sub-bin-accurate spectral peaks
analysis.tracking   →  peaks linked into partials over time
analysis.grouping   →  partials clustered into playable tracks
synthesis.oscillator_bank
                    →  each track rendered to PCM
                    →  residual := original − Σ rendered   (exact)
```

## External Dependencies

| Dependency | Purpose | Swappable? |
|---|---|---|
| numpy / scipy | FFT, windowing, resampling, interpolation | No (foundational) |
| soundfile | Audio file read/write (bundles libsndfile 1.2.2) | Yes (`AudioLoader` / `TrackExporter` ports) |
| mido | MIDI file construction | Yes (`TrackExporter` port) |
| FastAPI + uvicorn | HTTP serving | Yes (interface layer only) |
| websockets | **Required** for the progress socket — bare uvicorn has no WebSocket implementation and the endpoint 404s without it | No |
| In-memory job store | Background job tracking | Yes (`JobStore` port — Redis drop-in if multi-worker is ever needed) |

> **librosa was deliberately removed.** It was only wanted for resampling and f0 estimation; `scipy.signal.resample_poly` covers the first and the grouper does its own harmonic search for the second. Dropping it removes the numba/llvmlite chain, which has no prebuilt wheels for macOS 12 / Intel / CPython 3.13 and would compile from source for hours.

## Build Phases

Implementation is staged; each module's docstring names its phase.

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, domain model, contracts, config, CI | **Complete** |
| 1 | STFT, peak picking, tracking, audio loading | **Complete** |
| 2 | Grouping, resynthesis, residual, pipeline | **Complete** |
| 3 | WAV and MIDI export | **Complete** |
| 4 | API routes, job store, progress socket | **Complete** |
| 5 | Frontend: transport, canvas, workbench | **Complete** |
| 6 | `FourierPartial.app` launcher + icon | **Complete** |

All 56 tests pass. Verified end to end: upload → analyse → tracks → audio → WAV/MIDI export, with the reconstruction invariant holding at 2.98e-08 across the HTTP boundary.

## How to run it

| Method | Command |
|---|---|
| Double-click | `FourierPartial.app` in Finder |
| Terminal | `./run.sh` (add `--no-open` to skip the browser) |
| Dev, auto-reload | `.venv/bin/uvicorn backend.main:app --reload --port 8420` |

The server picks a free port if 8420 is taken and prints `FOURIER_PARTIAL_URL=...`; the launchers parse that line rather than assuming a port. Logs go to `logs/`, deliberately **outside** `sessions/` — the server clears `sessions/` on shutdown, which would otherwise delete the log at exactly the moment it became useful.

## Long Audio (up to 10 minutes)

Measured on an Intel Mac, macOS 12, with synthetic music (notes that decay and stop):

| Duration | Settings | Time | Peak RSS | Tracks |
|---|---|---|---|---|
| 60 s | hop 1024, zp 2 | 8.9 s | 260 MB | 6 |
| 5 min | hop 2048, zp 1 | 23 s | 827 MB | ~30 |
| 10 min | hop 2048, zp 1 | 51 s | 1.6 GB | 41 |

Cost is roughly linear in duration and **strongly dependent on material**: notes that stop are cheap, dense sustained textures are much more expensive because every partial then spans the whole file.

Four things make it work — remove any one and 10 minutes stops being feasible:

1. `iter_stft_blocks` streams the spectrogram (the full arrays cost ~11 MB per second of audio; 10 min would be ~6.8 GB).
2. `AnalysisParams.for_duration` spends less per frame on long input.
3. Grouping requires **temporal overlap**, not just a harmonic relationship.
4. The quiet-track energy floor is measured against the **loudest track**, not the total, so it means the same thing at any length.

### Playback memory (resolved)

The browser used to fetch every track before the first Play — measured at **4,339 MB** for a 41-track 10-minute analysis. It now holds **105 MB**, a 41× reduction, because the tracks sum to the original exactly, so "play everything" is one buffer (`GET /api/jobs/{id}/mixdown`) rather than one per track. Muting uses `?exclude=ids`; individual tracks load on demand with LRU eviction. Short files keep the eager loader, so solo stays instant where it's affordable.

The rule that keeps this correct: **the mix and individual tracks are never both audible.** An individually-fetched track is registered `soloOnly`, so it cannot stack on top of a mix that already contains it.

### Track rows

Rows collapse by pitch — a 41-track 10-minute analysis renders as **7 top-level rows** (G4×15, D4×8, B3×6, E4×5, A3×4, G3×2, Residual), each expandable to its occurrences. A pitch occurring once stays a plain row.

### Remaining limitations

- The heavy stages are **single-threaded**, so extra cores sit idle. Rendering is parallel across tracks and is the obvious next win.
- Cost depends strongly on **material**: dense sustained textures are far more expensive than notes that stop, because every partial then spans the whole file.
- A long file surfaces its 40 most prominent notes and folds the rest into the residual (still audible and exportable — the UI says so).

## Two Bugs Worth Knowing About

Both shipped in the first implementation, both produced *plausible* output rather than a crash, and both are now pinned by `tests/test_windowing.py`:

1. **Phase reference.** The DFT reports phase relative to sample 0 of its buffer, but frames are timestamped at the window *centre*. Every rendered partial came out at exactly the right frequency and amplitude but **anti-phase**, so forming the residual *doubled* each tone instead of cancelling it. A pure sine reported 19.8% partial energy where the true answer is ~95%. Fixed by zero-phase windowing (rotate the padded frame so the window centre sits at index 0) plus a cosine→sine convention correction (`+ π/2`).

2. **Window sidelobes.** Hann's first sidelobe sits at −31.5 dB — *above* the −60 dB relative peak threshold. So every loud partial grew phantom neighbours about a semitone away, loud enough to threshold, consistent enough to track, and long enough to survive the duration filter. They arrived in the UI as real tracks with real note names: a pure 440 Hz sine produced **eleven** "partials", nine of them window artefacts. Switched to Blackman-Harris (−92.2 dB, comfortably below the threshold). The cost is a wider main lobe, bounded by a test asserting that two tones a major third apart still resolve.

A third, smaller one: noise throws off short partials that cluster into convincing-looking harmonic groups with ~0% energy. On a real recording these outnumbered the genuine notes five to one (18 tracks where 4 were meaningful). `MIN_TRACK_ENERGY_SHARE` in `grouping.py` folds them into the residual, where they remain audible and exportable but stop claiming to be notes.
