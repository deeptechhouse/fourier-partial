"""Short-Time Fourier Transform — the windowing stage.

WHAT THIS DOES
--------------
A Fourier transform tells you which frequencies are present in a signal, but
it tells you nothing about *when*. That is useless for music, where when is
the whole point. So instead of transforming the entire clip at once, we slide
a short window along it and transform each window separately. The result is a
2D picture: frequency against time.

THE COST
--------
The window has to be short enough that the sound does not change much within
it, but long enough to resolve close-together frequencies. You cannot have
both. A 4096-sample window at 44.1 kHz sees 93 ms of audio and can separate
tones ~10.8 Hz apart; halve the window and you see 46 ms but now need tones
~21.5 Hz apart to tell them apart.

That trade is not an artifact of this implementation or of any implementation.
It is the Heisenberg-Gabor limit, and TDD section 2 explains why chasing it is
futile. Choose a side deliberately.

STATUS: implemented (Phase 1). Contract unchanged from the scaffold.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from backend.core.errors import InvalidParameterError
from backend.core.types import AnalysisParams

# =============================================================================
# ADJUSTABLE — magnitude scaling convention
# =============================================================================
# A real sine of amplitude A, once windowed and transformed, splits its energy
# between the positive and negative frequency halves of the spectrum. `rfft`
# only returns the positive half, so the peak there holds only HALF the energy
# -- hence the factor of 2. The window itself also attenuates the signal by its
# "coherent gain" (which is exactly `window.sum()`), so we divide that out.
#
# Net effect: a pure sine of amplitude 0.5 produces a peak magnitude of ~0.5,
# i.e. magnitudes come back in the SAME UNITS as the input waveform. Every
# downstream stage (peak picking, dB thresholds, resynthesis amplitudes)
# assumes this, so changing it means changing all of them too.
#
# ADJUSTABLE: set to 1.0 for a "raw one-sided" convention, or to
# 2.0 / window.size for a power-preserving convention. Only do so deliberately.
_AMPLITUDE_CORRECTION: float = 2.0

# ADJUSTABLE: the analysis window shape. See `make_window` for the full
# reasoning -- in short, sidelobe suppression matters far more here than main
# lobe width, because a sidelobe becomes a phantom track with a note name.
_WINDOW_SHAPE: str = "blackmanharris"

# ADJUSTABLE: frames transformed per block when streaming (see
# `iter_stft_blocks`). 256 frames at the defaults is ~17 MB of working set --
# small enough that duration stops mattering, large enough that the per-block
# overhead disappears into the FFT cost.
DEFAULT_BLOCK_FRAMES: int = 256
# =============================================================================


def make_window(window_size: int) -> NDArray[np.float64]:
    """Build the analysis window.

    ADJUSTABLE: the window shape. Blackman-Harris is the default, and the
    reason is worth understanding because it is the single most consequential
    choice in this module.

    Every window trades MAIN LOBE WIDTH against SIDELOBE HEIGHT:

        hann              main lobe +/-2 bins    first sidelobe  -31.5 dB
        blackmanharris    main lobe +/-4 bins    first sidelobe  -92.2 dB

    (Those numbers are measured, not quoted -- see tests/test_windowing.py.)

    Hann resolves closer tones. But its sidelobes sit at -31.5 dB, and the
    peak picker's threshold is -60 dB relative, so EVERY loud partial grows a
    pair of phantom neighbours about a semitone away, loud enough to survive
    thresholding, consistent enough to survive tracking, and long enough to
    survive the duration filter. They arrive in the interface as real tracks
    with real note names. A pure 440 Hz sine analysed with Hann produces
    eleven "partials", nine of which are artefacts of the window.

    For a tool whose entire promise is "each track is a genuine tone you can
    solo", inventing tones is a far worse failure than merging two that sit
    within 40 Hz of each other. Blackman-Harris puts its sidelobes at -92 dB,
    which is 30 dB below the threshold -- they cannot be picked at all.

    Alternatives: 'hann' (best resolution, phantom partials as described),
    'blackman' (a middle setting, -58 dB), 'nuttall' (similar to
    blackmanharris). If you switch back to a low-suppression window, raise
    `peak_threshold_db` to about -28 to compensate, and expect to lose quiet
    real partials in the bargain.
    """
    return scipy_signal.get_window(_WINDOW_SHAPE, window_size, fftbins=True).astype(np.float64)


def frame_count(num_samples: int, params: AnalysisParams) -> int:
    """How many analysis frames a signal of this length will produce."""
    if num_samples < params.window_size:
        return 0
    return 1 + (num_samples - params.window_size) // params.hop_size


def frame_time(frame_index: int, params: AnalysisParams) -> float:
    """Timestamp of a frame, measured at the *centre* of its window.

    Centre, not start: the window's energy is concentrated in its middle, so
    the centre is the honest answer to "when did this happen". Using the start
    would shift every trajectory earlier by half a window.
    """
    centre_sample = frame_index * params.hop_size + params.window_size / 2.0
    return centre_sample / params.sample_rate


def compute_stft(
    samples: NDArray[np.float32], params: AnalysisParams
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the magnitude and phase spectra of every frame.

    Args:
        samples: Mono float32 audio.
        params: Window size, hop size, zero-padding factor.

    Returns:
        A tuple of (magnitude, phase), each shaped (n_frames, n_bins), where
        n_bins is `window_size * zero_pad_factor // 2 + 1`.

    Implementation notes for whoever builds this:
        - Zero-pad each windowed frame to `window_size * zero_pad_factor`
          before the FFT. This adds no true resolution but interpolates the
          spectrum so parabolic peak-finding lands closer to the truth.
        - Use `np.fft.rfft`; the input is real, so the negative half is
          redundant and computing it wastes half the time.
        - Normalise magnitudes by the window's coherent gain (`window.sum()`)
          so amplitudes come out in the same units as the input signal.

    Raises:
        InvalidParameterError: If `samples` is not a one-dimensional array.
            Stereo must be mixed to mono by the loader before it gets here --
            silently flattening it would interleave the channels and produce a
            spectrum of nonsense, which is far worse than failing loudly.
    """
    # --- 1. Validate and normalise the input -------------------------------
    # float64 throughout: the FFT is a long chain of additions, and float32
    # rounding accumulates enough to move interpolated peak positions. Audio is
    # stored as float32 to save memory; analysis is done in float64 for accuracy.
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 1:
        raise InvalidParameterError(
            f"compute_stft expects mono (1-D) audio, got shape {audio.shape}"
        )

    # --- 2. Work out the geometry of the output ----------------------------
    window_size = params.window_size
    hop_size = params.hop_size
    # The zero-padded transform length. Zero-padding does NOT add real
    # resolution (that is fixed by window_size); it INTERPOLATES the spectrum,
    # putting more sample points on the same underlying curve so that the
    # parabolic peak-finder downstream has a better-sampled shape to fit.
    fft_size = window_size * params.zero_pad_factor
    # rfft returns only the non-redundant half of a real signal's spectrum:
    # bins 0 (DC) through fft_size/2 (Nyquist) inclusive -- hence the "+ 1".
    n_bins = fft_size // 2 + 1
    n_frames = frame_count(audio.size, params)

    # --- 3. Degenerate case: audio shorter than one window -----------------
    # Return correctly-shaped empty arrays rather than raising. Callers loop
    # over frames, and looping zero times is the right behaviour for a clip too
    # short to analyse; shape correctness keeps their np code from breaking.
    if n_frames == 0:
        empty = np.zeros((0, n_bins), dtype=np.float64)
        return empty, empty.copy()

    window = make_window(window_size)

    # --- 4. Build every frame at once --------------------------------------
    # `sliding_window_view` produces a VIEW of all overlapping windows without
    # copying the audio; slicing it by `hop_size` then keeps only the ones we
    # actually want. This is the vectorised equivalent of:
    #     for f in range(n_frames):
    #         block = audio[f * hop_size : f * hop_size + window_size]
    # but it hands numpy one big array instead of n_frames small ones, which is
    # where nearly all of the speed comes from.
    all_windows = np.lib.stride_tricks.sliding_window_view(audio, window_size)
    frames = all_windows[::hop_size][:n_frames]  # shape (n_frames, window_size)

    # Apply the window to every frame in one broadcast multiply. This is the
    # first materialised copy -- everything above was a view.
    windowed = frames * window  # shape (n_frames, window_size)

    # --- 5. ZERO-PHASE WINDOWING, then one transform -----------------------
    # This block looks like a fiddly detail and is in fact load-bearing. Read
    # it before changing anything here.
    #
    # A DFT reports each sinusoid's phase relative to sample 0 of its input
    # buffer. If we simply zero-padded the windowed block at the start (which
    # is what `rfft(..., n=fft_size)` does), sample 0 would be the window's
    # FIRST sample -- but `frame_time()` timestamps every frame at the window's
    # CENTRE, and the renderer places the partial there. Those two reference
    # points are half a window apart (46 ms at the defaults), so every partial
    # would be resynthesised with a wrong starting phase.
    #
    # The symptom is spectacular and easy to misdiagnose: a rendered partial
    # comes out at exactly the right frequency and exactly the right amplitude,
    # but anti-phase with the original. Subtracting it to form the residual
    # then DOUBLES the tone instead of cancelling it, and a pure sine reports
    # ~20% of its energy as "partials" when the true answer is ~100%.
    #
    # The fix is standard practice in sinusoidal modelling: rotate the padded
    # buffer so the window's centre sample lands at index 0. Phase then comes
    # back referenced to the window centre -- exactly where frame_time says the
    # frame is, and exactly where the renderer starts the oscillator.
    padded = np.zeros((n_frames, fft_size), dtype=np.float64)
    padded[:, :window_size] = windowed
    # A 30-second clip at the defaults is ~2600 frames x 8193 bins; the arrays
    # in flight here run to hundreds of megabytes, so release the windowed
    # frames as soon as they have been copied.
    del windowed
    padded = np.roll(padded, -(window_size // 2), axis=1)

    # axis=1 => transform along the sample axis, one FFT per frame.
    spectrum = np.fft.rfft(padded, axis=1)  # (n_frames, n_bins)
    del padded

    # --- 6. Split into the two things callers actually want ----------------
    # Magnitude: how loud each frequency is, rescaled into input units (see the
    # _AMPLITUDE_CORRECTION note at the top of this module for the why).
    coherent_gain = float(window.sum())
    magnitude = np.abs(spectrum) * (_AMPLITUDE_CORRECTION / coherent_gain)
    # Phase: where in its cycle each sinusoid is at the CENTRE of this frame
    # (that is what the zero-phase rotation above bought us). Discarding it
    # would make resynthesis click at every frame boundary, so it is carried
    # all the way through to the renderer.
    #
    # The +pi/2 converts between two conventions. `np.angle` measures phase
    # against a COSINE basis, because that is what the real part of the DFT
    # kernel is. The renderer synthesises with `sin`. For a signal
    # A*sin(2*pi*f*t) the transform returns an angle of (true_phase - pi/2),
    # so adding it back hands the renderer a number it can feed straight to
    # `sin` and land on the original waveform.
    #
    # Verified, not assumed: tests/test_windowing.py reconstructs the centre
    # sample of a known sine from (magnitude, phase) alone and asserts it
    # matches. Delete the +pi/2 and that test fails immediately.
    phase = np.angle(spectrum) + (np.pi / 2.0)

    # `np.abs`/`np.angle` on complex128 already yield float64; the casts are
    # cheap no-ops that document (and enforce) the contract for future readers.
    return magnitude.astype(np.float64, copy=False), phase.astype(np.float64, copy=False)


def iter_stft_blocks(
    samples: NDArray[np.float32],
    params: AnalysisParams,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
) -> Iterator[tuple[int, NDArray[np.float64], NDArray[np.float64]]]:
    """Stream the spectrogram a block of frames at a time.

    WHY THIS EXISTS — it is what makes long audio possible at all.
    ------------------------------------------------------------
    `compute_stft` materialises the entire spectrogram. That is fine for a
    phrase and fatal for a piece of music, because the array scales with
    duration and it is enormous per second:

        n_frames x n_bins x 8 bytes, twice (magnitude AND phase)

    At the defaults that is 8193 bins and 86 frames per second, so roughly
    11 MB for every second of audio. Measured: 84 MB for a 15-second clip,
    which extrapolates to **6.8 GB for ten minutes** -- before a single
    partial has been tracked or rendered.

    Nothing downstream ever needs the whole thing. Peak picking reads exactly
    one frame at a time and keeps only the handful of peaks it finds, which is
    a reduction of about a thousand to one. So we hand out blocks and let each
    one be collected once its peaks have been extracted. Memory becomes a
    function of `block_frames` rather than of duration, and a ten-minute file
    costs the same as a ten-second one.

    Args:
        samples: Mono float32 audio.
        params: Window size, hop size, zero-padding factor.
        block_frames: How many frames to transform per iteration. ADJUSTABLE:
            larger is marginally faster (fewer, bigger FFT calls) and uses
            proportionally more memory.

    Yields:
        (first_frame_index, magnitude_block, phase_block), where each block is
        shaped (frames_in_this_block, n_bins) and carries the same scaling and
        phase convention as `compute_stft`.
    """
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 1:
        raise InvalidParameterError(
            f"iter_stft_blocks expects mono (1-D) audio, got shape {audio.shape}"
        )

    n_frames = frame_count(audio.size, params)
    if n_frames == 0:
        return

    window_size = params.window_size
    hop_size = params.hop_size
    fft_size = window_size * params.zero_pad_factor
    window = make_window(window_size)
    coherent_gain = float(window.sum())

    all_windows = np.lib.stride_tricks.sliding_window_view(audio, window_size)
    strided = all_windows[::hop_size][:n_frames]

    step = max(1, int(block_frames))
    for start in range(0, n_frames, step):
        stop = min(start + step, n_frames)

        # Same three operations as compute_stft, applied to a slice. Kept
        # deliberately identical rather than factored out: the phase
        # convention here is subtle (see compute_stft) and having the two
        # paths drift apart would be a bug nobody would find quickly.
        windowed = strided[start:stop] * window
        padded = np.zeros((stop - start, fft_size), dtype=np.float64)
        padded[:, :window_size] = windowed
        del windowed
        padded = np.roll(padded, -(window_size // 2), axis=1)

        spectrum = np.fft.rfft(padded, axis=1)
        del padded

        magnitude = np.abs(spectrum) * (_AMPLITUDE_CORRECTION / coherent_gain)
        phase = np.angle(spectrum) + (np.pi / 2.0)
        del spectrum

        yield start, magnitude, phase


def bin_to_hz(bin_index: float, params: AnalysisParams) -> float:
    """Convert a (possibly fractional) FFT bin index to frequency in Hz."""
    fft_size = params.window_size * params.zero_pad_factor
    return bin_index * params.sample_rate / fft_size
