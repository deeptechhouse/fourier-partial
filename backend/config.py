"""Runtime configuration, loaded from environment with safe defaults.

Everything here can be overridden by an environment variable prefixed `FP_`.
Copy `.env.example` to `.env` to set them locally.
"""

from __future__ import annotations

import socket
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Validated at startup so a misconfiguration fails immediately and loudly
    rather than halfway through someone's first upload.
    """

    model_config = SettingsConfigDict(env_prefix="FP_", env_file=".env", extra="forbid")

    # --- Server --------------------------------------------------------------
    host: str = "127.0.0.1"

    # ADJUSTABLE: preferred port. If busy, `resolve_port()` picks another.
    port: int = 8420

    # --- Upload limits (security boundary; see TDD section 13) ---------------
    # ADJUSTABLE: hard ceiling on upload size. 256 MB clears ten minutes of
    # 24-bit stereo WAV (~158 MB), which is what the duration ceiling below
    # implies for uncompressed input. Raising the duration without raising
    # this would make long WAVs fail on size with a confusing message.
    max_upload_bytes: int = 268_435_456

    # ADJUSTABLE: reject audio longer than this.
    #
    # 600 s (ten minutes) is the supported ceiling. It is reachable because the
    # spectrogram is streamed rather than materialised (see iter_stft_blocks)
    # and because AnalysisParams.for_duration spends less per frame on long
    # input. Both are required: at phrase settings a ten-minute file needs
    # ~6.8 GB for the spectrogram alone.
    #
    # Cost grows a little faster than linearly, so raising this further is not
    # free -- measure before you do.
    max_duration_seconds: float = 600.0

    # --- Presets -------------------------------------------------------------
    # ADJUSTABLE: where saved settings combinations live. A plain JSON file in
    # the project folder rather than browser storage, so it survives clearing a
    # cache, can be edited by hand, copied to another machine, or committed.
    presets_file: Path = Field(default=Path("./presets.json"))

    # --- Session storage -----------------------------------------------------
    session_dir: Path = Field(default=Path("./sessions"))

    # ADJUSTABLE: how long analysis artifacts survive before purging.
    session_ttl_seconds: int = 3600

    # --- CORS ----------------------------------------------------------------
    # Explicit origins only. A wildcard here would let any website drive this
    # API using a visitor's browser.
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:8420"])

    # --- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("allowed_origins")
    @classmethod
    def _reject_wildcard_origin(cls, value: list[str]) -> list[str]:
        """Refuse a wildcard CORS origin outright."""
        if "*" in value:
            raise ValueError(
                "Wildcard CORS origin is not permitted. List explicit origins instead."
            )
        return value

    @field_validator("max_duration_seconds")
    @classmethod
    def _positive_duration(cls, value: float) -> float:
        """Duration ceiling must be positive."""
        if value <= 0:
            raise ValueError("max_duration_seconds must be positive")
        return value


def is_port_available(host: str, port: int) -> bool:
    """Check whether a TCP port can be bound on `host`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def resolve_port(host: str, preferred: int, max_attempts: int = 20) -> int:
    """Return `preferred` if free, otherwise the next available port.

    Auto-selecting beats failing with "address already in use" when a previous
    run has not released the socket yet, and beats prompting the user for
    something they do not care about.

    Raises:
        RuntimeError: If no free port is found within `max_attempts`.
    """
    for offset in range(max_attempts):
        candidate = preferred + offset
        if is_port_available(host, candidate):
            return candidate
    raise RuntimeError(
        f"No free port found in range {preferred}-{preferred + max_attempts - 1} on {host}"
    )


#: Module-level singleton, imported by the app and by services.
settings = Settings()
