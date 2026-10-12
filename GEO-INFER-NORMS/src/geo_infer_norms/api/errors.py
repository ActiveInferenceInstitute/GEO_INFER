"""Shared API error handling for GEO-INFER-NORMS routers.

Domain errors surfaced by request handling as ``ValueError`` are client faults
and map to HTTP 400. Any other exception escaping a handler is a server fault
and must return HTTP 500 with a generic message (never the internal exception
text), mirroring the ``ErrorHandlerMiddleware`` pattern in GEO-INFER-LOG.
"""

import logging
from typing import cast
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Convert unexpected (non-domain) exceptions to HTTP 500.

    Handlers are responsible for mapping domain ``ValueError``s to HTTP 400;
    this middleware only catches what escapes them, logs the full traceback,
    and returns a generic 500 without leaking internal exception details.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Catch unexpected exceptions and return a generic HTTP 500."""
        try:
            return cast("Response", await call_next(request))
        except Exception:
            logger.exception(
                "Unexpected error processing %s %s", request.method, request.url.path
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred",
                        "status_code": 500,
                    }
                },
            )


def register_error_handlers(app: FastAPI) -> None:
    """Attach shared NORMS error handling to a FastAPI app hosting NORMS routers."""
    app.add_middleware(ErrorHandlerMiddleware)
