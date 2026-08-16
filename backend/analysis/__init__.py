"""Analysis: waveform in, partials out.

Pipeline order (see TDD section 4):
    stft.py     -> windowed spectra
    peaks.py    -> per-frame spectral peaks, sub-bin accurate
    tracking.py -> peaks linked into partials across time
    grouping.py -> partials clustered into playable tracks
    pipeline.py -> composes the above and enforces the invariant
"""
