"""Shared test fixtures.

The synthetic signals here are deliberately simple and exactly known. When a
test fails against a 440 Hz sine you built yourself, you know the analyser is
wrong; when it fails against a real recording, you are debugging two things at
once.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from backend.core.types import AnalysisParams, Partial, PartialFrame

SAMPLE_RATE = 44_100


@pytest.fixture
def params() -> AnalysisParams:
    """Default analysis parameters."""
    return AnalysisParams()


def make_sine(
    frequency_hz: float,
    duration_s: float,
    amplitude: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> NDArray[np.float32]:
    """Generate a pure sine wave — the simplest thing a tracker must get right."""
    t = np.arange(int(duration_s * sample_rate), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)


def make_harmonic_tone(
    fundamental_hz: float,
    n_harmonics: int,
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
) -> NDArray[np.float32]:
    """Generate a harmonic complex with 1/n amplitude rolloff (sawtooth-like).

    This is the canonical "should group into one voice" test signal.
    """
    signal = np.zeros(int(duration_s * sample_rate), dtype=np.float64)
    for n in range(1, n_harmonics + 1):
        signal += make_sine(fundamental_hz * n, duration_s, 1.0 / n, sample_rate)
    peak = np.max(np.abs(signal))
    return (signal / peak * 0.8).astype(np.float32) if peak > 0 else signal.astype(np.float32)


def make_noise_burst(
    duration_s: float, seed: int = 0, sample_rate: int = SAMPLE_RATE
) -> NDArray[np.float32]:
    """Generate white noise — content that provably cannot become few partials.

    Used to assert the uncertainty trade-off shows up in the metrics: noise
    should land overwhelmingly in the residual, not in tidy tonal tracks.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration_s * sample_rate)) * 0.2).astype(np.float32)


@pytest.fixture
def steady_partial() -> Partial:
    """A 1-second, unwavering 440 Hz partial at constant amplitude."""
    partial = Partial(partial_id=0)
    for i in range(87):  # ~1s at the default 512-sample hop
        partial.append_frame(
            PartialFrame(time_s=i * 512 / SAMPLE_RATE, frequency_hz=440.0, amplitude=0.5, phase_rad=0.0)
        )
    return partial


@pytest.fixture
def gliding_partial() -> Partial:
    """A partial sliding 440 -> 880 Hz, to exercise pitch-bend export."""
    partial = Partial(partial_id=1)
    n = 87
    for i in range(n):
        partial.append_frame(
            PartialFrame(
                time_s=i * 512 / SAMPLE_RATE,
                frequency_hz=440.0 * (2.0 ** (i / (n - 1))),
                amplitude=0.5,
                phase_rad=0.0,
            )
        )
    return partial


@pytest.fixture(autouse=True)
def _clear_rate_limit():
    """Reset the API rate limiter before every test.

    The limiter is process-global by design — it has to outlive individual
    requests to mean anything. That also makes it outlive individual *tests*,
    so without this a suite that uploads more than the per-minute allowance
    fails in whichever test happens to run last, pointing at the wrong thing
    entirely. (That is exactly how it was found.)
    """
    from backend.api.routes import reset_rate_limit

    reset_rate_limit()
    yield
