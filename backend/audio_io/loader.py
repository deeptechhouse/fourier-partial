"""Audio loading — the system's boundary with untrusted input.

Everything that arrives here came from outside and is therefore hostile until
proven otherwise. Validation order matters:

    1. Size, before reading (a 4 GB "wav" should never reach memory).
    2. Magic bytes, not the file extension. An extension is a claim by the
       uploader; magic bytes are evidence.
    3. Duration, after decoding the header but before decoding the samples.
    4. Content, last: is there actually any audio in here?

STATUS: implemented in Phase 1. The contract is unchanged from the scaffold.

What libsndfile can actually decode (backend reality check)
-----------------------------------------------------------
We decode through the `soundfile` package, which is a thin ctypes wrapper over
libsndfile. This project pins libsndfile 1.2.2, and what that library supports
is NOT the same as what our `_MAGIC_SIGNATURES` table recognises:

    WAV / RF64  -- always supported (PCM 8/16/24/32-bit, float, plus common
                   ADPCM variants). This is the safe path.
    FLAC        -- always supported; libsndfile bundles libFLAC.
    AIFF / AIFC -- always supported (AIFC = compressed AIFF; the sub-format
                   inside it still has to be one libsndfile knows).
    OGG Vorbis  -- supported in every mainstream build (libvorbis is bundled).
                   OGG Opus arrived in libsndfile 1.2.x builds compiled with
                   libopus, so it is *probable* but not guaranteed.
    MP3         -- BUILD-DEPENDENT. MP3 decoding landed in libsndfile 1.1.0 via
                   bundled minimp3, but distributors routinely disable it, and
                   older system copies of libsndfile predate it entirely.

Because MP3 is the one format that can be recognised by its magic bytes and
still fail to open, `SoundFileLoader.load` catches that failure specifically
and re-raises it as `UnsupportedAudioError` carrying a user-facing message that
tells the user what to do about it (convert to WAV/FLAC) rather than leaking a
libsndfile error string to the browser. See TDD section 13.

You can check the running build at any time with:

    python -c "import soundfile as sf; print(sf.__libsndfile_version__, \
sorted(sf.available_formats()))"

Resampling
----------
Resampling is done with `scipy.signal.resample_poly`, NOT with a naive
index-picking or linear-interpolation approach. That choice is not cosmetic:
cheap resampling folds energy above the new Nyquist limit back down into the
audible band as *aliases*, and this tool's entire job is to point at frequency
peaks and tell the user "there is a tone here". An alias is a peak that looks
completely real and is not. `resample_poly` applies a proper anti-aliasing FIR
filter as part of the rational up/down conversion, so those ghosts never form.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from backend.core.errors import (
    AudioTooLongError,
    EmptyAudioError,
    UnsupportedAudioError,
)

# Magic byte signatures for the formats we accept. Checked against the file's
# actual leading bytes -- never against its name.
#
# `fmt: off` keeps black from collapsing the column-aligned trailing comments;
# they line up on purpose so the two containers that need a sub-type check are
# obvious at a glance. (types.py does the same thing with `fmt: skip`.)
# fmt: off
_MAGIC_SIGNATURES: dict[bytes, str] = {
    b"RIFF": "wav",       # bytes 8..12 must also be 'WAVE'
    b"fLaC": "flac",
    b"FORM": "aiff",      # bytes 8..12 must also be 'AIFF' or 'AIFC'
    b"OggS": "ogg",
    b"ID3": "mp3",        # MP3 with an ID3 tag
    b"\xff\xfb": "mp3",   # bare MPEG frame sync
    b"\xff\xf3": "mp3",
    b"\xff\xf2": "mp3",
}
# fmt: on

# The signature table above, ordered longest-first.
#
# WHY THIS EXISTS (junior devs, read this): dict iteration order in Python is
# insertion order, which is *not* the order we need. Short signatures are
# greedy -- if we happened to test a 2- or 3-byte prefix before a 4-byte one,
# a file whose real signature is longer could be claimed by the shorter entry
# and mislabelled. Testing longest-first makes the match unambiguous: the most
# specific signature that fits always wins.
_SIGNATURES_LONGEST_FIRST: tuple[tuple[bytes, str], ...] = tuple(
    sorted(_MAGIC_SIGNATURES.items(), key=lambda item: len(item[0]), reverse=True)
)

# How many leading bytes we need in hand to identify a container.
#
# NOT adjustable: 12 is forced by the file formats themselves. RIFF/WAVE and
# FORM/AIFF both put their *real* type identifier at bytes 8..12, after a
# 4-byte tag and a 4-byte length field. Anything less than 12 cannot tell a
# WAV from an AVI (both are RIFF) or an AIFF from a DjVu file (both are FORM).
_HEADER_BYTES: int = 12

# Container sub-type tags that must follow a 'RIFF' / 'FORM' chunk header.
# These live at bytes 8..12 and are what actually distinguishes audio from the
# other things that share the same outer container.
_RIFF_AUDIO_SUBTYPE: bytes = b"WAVE"
_FORM_AUDIO_SUBTYPES: frozenset[bytes] = frozenset({b"AIFF", b"AIFC"})


def _mp3_decoding_available() -> bool:
    """Whether the libsndfile actually linked at runtime can decode MP3.

    Asked once, at import, because `sf.available_formats()` walks libsndfile's
    format table and there is no point repeating that per upload. This is a
    property of the *build on this machine*, not of the code, which is exactly
    why it has to be discovered rather than assumed -- see the module docstring.
    """
    return "MP3" in sf.available_formats()


#: Cached at import; see `_mp3_decoding_available` for why it is not re-checked.
_MP3_SUPPORTED: bool = _mp3_decoding_available()

# ADJUSTABLE: full-scale ceiling. Samples are expected in [-1.0, 1.0]; anything
# hotter is scaled down so its peak lands exactly here. Raising this above 1.0
# would let clipped audio through to the WAV exporter and the browser, both of
# which hard-clip anyway -- so don't.
_FULL_SCALE: float = 1.0

# ADJUSTABLE: below this peak amplitude we call the file silent and reject it.
# 1e-9 is roughly -180 dBFS, i.e. far below the noise floor of any real
# recording and below the resolution of 24-bit audio (-144 dBFS). Raise it if
# you want to reject "technically not silent but useless" input more
# aggressively; lower it only if you have a genuine reason to analyse
# near-infinitesimal signals.
_SILENCE_THRESHOLD: float = 1e-9

# ADJUSTABLE: highest sample rate we will believe a file header, in Hz.
#
# WHY THIS EXISTS (junior devs, read this): the sample rate is a number a file
# *claims about itself*, and it is load-bearing -- the duration ceiling is
# computed as `frames / sample_rate`. A header that claims two billion hertz
# makes ten million frames look like five milliseconds of audio, so the
# ceiling waves it through and we then decode every one of those frames. A
# 10 MB upload measured that way costs gigabytes of RAM and minutes of CPU:
# the duration check is only a memory bound if the rate it divides by is real.
# 768 kHz is the top rate any converter on the market actually runs at (DXD
# and the fastest pro interfaces), so this rejects nothing a user can have.
_MAX_SAMPLE_RATE: int = 768_000

# ADJUSTABLE: largest polyphase factor we will hand to `resample_poly`.
#
# `resample_poly(x, up, down)` designs an anti-aliasing FIR whose length grows
# as 20 * max(up, down). Those factors come from the *reduced* ratio, so they
# are small for every rate pairing that occurs in real life (48000 -> 44100 is
# 147/160; the worst realistic case, 44056 -> 44100, is 11025/11014). A rate
# that shares no factors with the target, however, does not reduce at all: a
# 4 KB file claiming 1000003 Hz asks for a twenty-million-tap filter, which
# measured at ~1 GB of RAM and 6.9 s of CPU for 45 output samples.
# 100000 caps the filter at ~2 million taps (tens of MB, milliseconds) while
# sitting an order of magnitude above anything a real recording produces.
_MAX_RESAMPLE_FACTOR: int = 100_000


def sniff_format(path: Path) -> str:
    """Identify an audio format from its leading bytes.

    Returns:
        A short format name, e.g. 'wav'.

    Raises:
        UnsupportedAudioError: If no known signature matches.

    Implementation notes:
        We read exactly `_HEADER_BYTES` (12) bytes and nothing more. Reading a
        fixed, tiny prefix means a hostile 4 GB upload costs us one page of
        memory to reject, and it happens before any decoder is handed the file.

        A file shorter than 12 bytes cannot be valid audio in any format we
        accept, so "too short" and "unrecognised" collapse into the same
        answer for the caller: `UnsupportedAudioError`. I/O failures land there
        too -- from the user's point of view "I cannot read this file" and "I
        do not understand this file" are the same event, and distinguishing
        them in the response would only tell an attacker which paths exist.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(_HEADER_BYTES)
    except OSError as exc:
        # Missing file, permission denied, path is a directory, device error...
        # The detail goes in the exception text (which is logged); the
        # `user_message` the client sees stays deliberately generic.
        raise UnsupportedAudioError(f"could not read header of {path}: {exc}") from exc

    if len(header) < _HEADER_BYTES:
        raise UnsupportedAudioError(
            f"{path} is {len(header)} bytes; need at least {_HEADER_BYTES} to identify it"
        )

    # bytes 8..12 -- the sub-type tag for the two container formats that have one.
    subtype = header[8:12]

    for signature, format_name in _SIGNATURES_LONGEST_FIRST:
        if not header.startswith(signature):
            continue

        # RIFF and FORM are *containers*, not formats. A RIFF file may hold
        # AVI video; a FORM file may hold an image. The leading tag alone is
        # not evidence of audio -- the sub-type tag is, so demand it.
        if signature == b"RIFF" and subtype != _RIFF_AUDIO_SUBTYPE:
            raise UnsupportedAudioError(
                f"{path} is a RIFF container but its subtype is {subtype!r}, not "
                f"{_RIFF_AUDIO_SUBTYPE!r}"
            )
        if signature == b"FORM" and subtype not in _FORM_AUDIO_SUBTYPES:
            raise UnsupportedAudioError(
                f"{path} is a FORM container but its subtype is {subtype!r}, not "
                f"one of {sorted(_FORM_AUDIO_SUBTYPES)}"
            )
        return format_name

    raise UnsupportedAudioError(
        f"{path} matches no known audio signature (first bytes: {header[:4]!r})"
    )


class SoundFileLoader:
    """Loads audio via libsndfile (through the `soundfile` package).

    Implements the `AudioLoader` port. Swapping to a different decoder means
    writing one new class with the same two methods -- nothing downstream
    knows which decoder produced its samples.
    """

    __slots__ = ("_max_duration",)

    def __init__(self, max_duration_seconds: float) -> None:
        """Create a loader that rejects audio longer than the given duration."""
        self._max_duration = max_duration_seconds

    @property
    def max_duration_seconds(self) -> float:
        """The duration ceiling this loader enforces, in seconds.

        Read-only: the ceiling is a security boundary (it bounds how much
        memory one upload can cost us), so nothing outside this object gets to
        move it after construction.
        """
        return self._max_duration

    def load(self, path: Path, target_sample_rate: int) -> tuple[NDArray[np.float32], int]:
        """Load, validate, downmix to mono, and resample.

        Returns:
            A tuple of (samples, sample_rate) with samples in [-1.0, 1.0].

        Raises:
            UnsupportedAudioError: Format not recognised, or libsndfile could
                not decode it (most commonly MP3 on a build without MP3), or
                the header claims a sample rate no real recording carries, or
                the decoded samples are not all finite.
            AudioTooLongError: Duration exceeds this loader's ceiling.
            EmptyAudioError: File decoded but holds no samples, or is silent.

        Implementation notes:
            - Downmix to mono by averaging channels. Stereo analysis is a
              future extension; for now, analysing a mono sum keeps partial
              counts halved and avoids a phase-cancellation class of bug.
            - Resample with `librosa.resample` using the 'soxr_hq' backend --
              cheap resampling produces aliasing that shows up as spurious
              high-frequency partials, which is exactly the artefact this tool
              would otherwise present to the user as a real tone.
              SUPERSEDED IN PHASE 1: librosa is deliberately NOT a dependency
              of this project (it drags in numba, llvmlite, and a large chunk
              of scikit-learn for one function). The *reason* above still
              stands in full, so we satisfy it with `scipy.signal.resample_poly`,
              which is polyphase rational resampling with a built-in
              anti-aliasing FIR -- the same anti-alias guarantee, from a
              dependency we already have. See the module docstring.
            - Peak-normalise only if the input is hotter than full scale.
              Do not normalise quiet input: relative amplitudes between
              partials are meaningful and rescaling would distort the
              energy-ratio diagnostics.

        Order of operations, and why it is this order:
            magic bytes -> header -> sample rate -> duration -> decode ->
            downmix -> finite check -> resample-ratio check -> resample
            -> normalise -> silence check. Every check that can be made cheaply
            happens before the expensive one it protects. We never decode
            samples from a file we have not already agreed is short enough --
            which is why the sample rate is vetted *before* the duration it is
            used to compute, and the resampling ratio *before* the filter it
            sizes.
        """
        if target_sample_rate <= 0:
            # Programmer error, not user error -- caught here so it surfaces at
            # the call site instead of as a bizarre gcd/resample failure later.
            raise ValueError(f"target_sample_rate must be positive, got {target_sample_rate}")

        # --- 1. Magic bytes ---------------------------------------------------
        # The extension is a claim by the uploader; these bytes are evidence.
        # Raises UnsupportedAudioError before libsndfile ever sees the file.
        detected_format = sniff_format(path)

        # --- 2. Header, then duration, then emptiness -------------------------
        # Opening a SoundFile parses the header only; no sample data is read
        # yet. That is exactly what we want: we learn frames and samplerate for
        # the cost of a header parse, and can bail out before allocating an
        # array big enough to hurt.
        try:
            with sf.SoundFile(str(path)) as handle:
                frame_count = int(handle.frames)
                source_sample_rate = int(handle.samplerate)

                if source_sample_rate <= 0:
                    raise UnsupportedAudioError(
                        f"{path} reports a non-positive sample rate ({source_sample_rate})"
                    )
                # An absurdly *high* rate is the dangerous direction, and it
                # has to be caught here -- before the division below -- because
                # the duration ceiling divides by this number. See
                # `_MAX_SAMPLE_RATE`: an inflated rate turns the ceiling into
                # no ceiling at all and lets a small file decode to gigabytes.
                if source_sample_rate > _MAX_SAMPLE_RATE:
                    raise UnsupportedAudioError(
                        f"{path} reports an implausible sample rate "
                        f"({source_sample_rate} Hz > {_MAX_SAMPLE_RATE} Hz)"
                    )

                duration_s = frame_count / source_sample_rate
                if duration_s > self._max_duration:
                    raise AudioTooLongError(
                        f"{path} is {duration_s:.2f}s, ceiling is {self._max_duration:.2f}s"
                    )
                if frame_count == 0:
                    raise EmptyAudioError(f"{path} decoded to zero frames")

                # --- 3. Decode ------------------------------------------------
                # float64 for the intermediate maths: downmixing and resampling
                # both accumulate, and doing that in float32 lets rounding
                # error creep into the very amplitudes the analysis measures.
                # always_2d gives a uniform (frames, channels) shape so the
                # downmix below needs no special case for mono files.
                samples = handle.read(dtype="float64", always_2d=True)
        except (AudioTooLongError, EmptyAudioError, UnsupportedAudioError):
            # Our own validation errors are already correctly typed and
            # carry safe user messages -- let them straight through rather
            # than re-wrapping them as decode failures below.
            raise
        except (sf.LibsndfileError, RuntimeError) as exc:
            # libsndfile recognised nothing it could decode. The one case a
            # user can act on is MP3, so say something useful instead of the
            # generic message -- and distinguish the two very different reasons
            # an MP3 can land here, because the advice differs:
            #   * this build has no MP3 decoder at all -> convert the file
            #   * this build has one and still failed  -> the file is damaged
            error = UnsupportedAudioError(f"libsndfile could not decode {path}: {exc}")
            if detected_format == "mp3":
                error.user_message = (
                    "That MP3 could not be decoded -- the file appears to be damaged or "
                    "truncated. Try re-exporting it, or upload WAV, FLAC, or AIFF instead."
                    if _MP3_SUPPORTED
                    else "This server cannot decode MP3 files. "
                    "Please convert your audio to WAV, FLAC, or AIFF and try again."
                )
            raise error from exc

        # --- 4. Downmix to mono ----------------------------------------------
        # Mean, not sum: summing two correlated channels doubles the amplitude
        # and would clip a file that was perfectly in range to begin with.
        mono = samples.mean(axis=1)

        # --- 4b. Reject non-finite samples -----------------------------------
        # Float WAV/AIFF carry raw IEEE-754, so a file can legally contain NaN
        # or +/-Inf -- a broken encoder, a truncated render, or a deliberate
        # poisoning of the input. Neither survives contact with the rest of the
        # system: NaN smears across every FFT frame that overlaps it (one bad
        # sample becomes a whole silent-looking analysis), and Inf is worse
        # still -- `peak` below becomes Inf, `mono * (1.0 / Inf)` multiplies
        # the entire signal by zero, and the file sails through the silence
        # check because `peak` was assigned full scale by hand. The user gets a
        # successful analysis of pure digital silence with no explanation.
        #
        # Checked on `mono` rather than on `samples` because it cannot hide
        # there: averaging can only turn a non-finite channel into another
        # non-finite value, never into a finite one, and `mono` is half the
        # array to scan. Rejecting beats sanitising -- silently rewriting a
        # user's samples would corrupt the very amplitudes we then measure.
        if not bool(np.isfinite(mono).all()):
            error = UnsupportedAudioError(
                f"{path} contains non-finite samples "
                f"({int(np.count_nonzero(~np.isfinite(mono)))} of {mono.size} are NaN or Inf)"
            )
            error.user_message = (
                "That file contains invalid sample values (NaN or infinity) and cannot be "
                "analysed. Try re-exporting it, ideally as 16- or 24-bit WAV or FLAC."
            )
            raise error

        # --- 5. Resample -----------------------------------------------------
        if source_sample_rate != target_sample_rate:
            # resample_poly works on an integer ratio up/down. Reducing that
            # ratio by its GCD is what keeps it cheap: 48000 -> 44100 is
            # 44100/48000, which reduces to 147/160. Unreduced it would build a
            # polyphase filter with 44100 phases instead of 147 -- same result,
            # hundreds of times the work and memory.
            common = math.gcd(source_sample_rate, target_sample_rate)
            up = target_sample_rate // common
            down = source_sample_rate // common
            # ...but a ratio that does not reduce stays enormous, and the
            # filter `resample_poly` builds grows with it. That cost is paid
            # before a single output sample exists, so it has to be refused
            # rather than survived. See `_MAX_RESAMPLE_FACTOR`.
            if max(up, down) > _MAX_RESAMPLE_FACTOR:
                error = UnsupportedAudioError(
                    f"{path}: converting {source_sample_rate} Hz to {target_sample_rate} Hz "
                    f"reduces to {up}/{down}, above the {_MAX_RESAMPLE_FACTOR} factor limit"
                )
                error.user_message = (
                    f"That file's sample rate ({source_sample_rate} Hz) cannot be converted "
                    "for analysis. Please re-export it at 44.1 kHz or 48 kHz."
                )
                raise error
            mono = resample_poly(mono, up, down)

        # --- 6. Peak-normalise ONLY if hotter than full scale -----------------
        # Note this happens *after* resampling on purpose: a polyphase filter's
        # ringing can push a sample slightly past the original peak, so
        # normalising earlier would leave us over full scale again.
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > _FULL_SCALE:
            # Scale so the loudest sample sits exactly at full scale. We do NOT
            # touch quiet audio: the ratios between partial amplitudes are the
            # measurement, and rescaling them would corrupt the energy-ratio
            # diagnostics in ReconstructionReport.
            mono = mono * (_FULL_SCALE / peak)
            peak = _FULL_SCALE

        # --- 7. Silence check ------------------------------------------------
        # A file full of zeros decodes perfectly and analyses to nothing, which
        # would show the user an empty interface with no explanation. Catch it
        # here where we can still say why.
        if peak < _SILENCE_THRESHOLD:
            raise EmptyAudioError(
                f"{path} decoded to silence (peak amplitude {peak:.3e} "
                f"< {_SILENCE_THRESHOLD:.0e})"
            )

        # --- 8. Hand back a fresh float32 array -------------------------------
        # float32 downstream: it halves the memory of every STFT buffer and is
        # far more precision than 24-bit audio carries. astype() always copies,
        # so the caller owns this array outright -- no aliasing back into any
        # buffer libsndfile or scipy still holds a reference to.
        return mono.astype(np.float32), target_sample_rate
