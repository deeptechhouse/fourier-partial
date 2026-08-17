"""Application entry point.

Run locally with:

    python3 -m backend.main

or, for auto-reload during development:

    uvicorn backend.main:app --reload --port 8420

The double-clickable FourierPartial.app launches the first form and waits for
/api/health before opening a browser.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import cleanup_sessions
from backend.api.routes import router as api_router
from backend.config import resolve_port, settings

logging.basicConfig(
    level=settings.log_level,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

#: Absolute path to the frontend directory.
#: Absolute, not relative: the .app launcher may start the server from any
#: working directory, and a relative "frontend" would then silently resolve to
#: nothing and serve 404s for the entire UI.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shut-down hooks."""
    logger.info("fourier_partial starting; frontend at %s", _FRONTEND_DIR)
    if not _FRONTEND_DIR.is_dir():
        logger.error("frontend directory missing at %s -- UI will not load", _FRONTEND_DIR)
    yield
    # Uploaded audio and rendered tracks are user data with no reason to
    # outlive the process. See design document section 13.
    cleanup_sessions()
    logger.info("fourier_partial stopped; session data cleaned up")


def create_app() -> FastAPI:
    """Build and configure the ASGI application.

    A factory rather than a module-level singleton so tests can construct an
    app with different settings without reimporting the module.
    """
    app = FastAPI(
        title="fourier_partial",
        version="0.1.0",
        description="Decompose audio into individually-audible pitched tracks.",
        lifespan=_lifespan,
    )

    # --- CORS: explicit origins only ----------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # --- Security headers ---------------------------------------------------
    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach hardening headers to every response. See section 13.3."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"

        # Always revalidate the interface.
        #
        # StaticFiles sends ETags, so without this a browser may serve the page
        # from its own cache without asking — and since the launcher opens the
        # app by URL, an already-open tab is simply focused rather than
        # reloaded. The result is a user looking at a version of the interface
        # from before the last change, wondering where a new control went. That
        # happened, and it cost a diagnosis of code that was working perfectly.
        #
        # `no-cache` means "revalidate", not "do not store": the ETag still
        # answers with a 304 and nothing is re-downloaded when nothing changed.
        # Costs one conditional request per asset; buys never debugging a stale
        # page again.
        #
        # Deliberately not applied to /api — the audio endpoints set their own
        # caching, and a track's rendered PCM is genuinely immutable for the
        # life of its job.
        if not request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-cache"
        # ws: is listed explicitly for the progress socket. Some browsers do
        # not treat a same-origin WebSocket as covered by connect-src 'self',
        # and the failure mode is a progress bar that silently never moves.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "media-src 'self' blob:; "
            "connect-src 'self' ws://127.0.0.1:* ws://localhost:*; "
            "img-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'none'; "
            "frame-ancestors 'none'"
        )
        return response

    # API first, then the static mount. Order matters: StaticFiles at "/" is a
    # catch-all, so mounting it before the router would shadow every endpoint.
    app.include_router(api_router)

    if _FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()


def main() -> None:
    """Start the development server on an available port."""
    import uvicorn

    port = resolve_port(settings.host, settings.port)
    if port != settings.port:
        logger.warning("Port %d busy; using %d instead", settings.port, port)

    # The .app launcher greps stdout for this exact line to learn which port
    # was actually chosen. Keep the "http://host:port" shape if you edit it.
    url = f"http://{settings.host}:{port}"
    logger.info("fourier_partial listening on %s", url)
    print(f"FOURIER_PARTIAL_URL={url}", flush=True)

    uvicorn.run(app, host=settings.host, port=port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
