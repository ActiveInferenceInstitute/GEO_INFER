"""Shared API error handling for GEO-INFER-DATA routers.

Domain errors surfaced by core logic as ``ValueError`` (and request-driven
``KeyError``s for missing datasets) are client faults and map to HTTP 4xx.
Any other exception escaping a handler is a server fault and must return
HTTP 500 with a generic message (never the internal exception text),
mirroring the ``ErrorHandlerMiddleware`` pattern in GEO-INFER-LOG
(LOG-EXC-01).
"""

import logging
from typing import Any, cast
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Convert unexpected (non-domain) exceptions to HTTP 500.

    Handlers are responsible for mapping domain errors to HTTP 4xx; this
    middleware only catches what escapes them, logs the full traceback, and
    returns a generic 500 without leaking internal exception details.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Any], Any]
    ) -> Response:
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
    """Attach shared DATA error handling to a FastAPI app hosting DATA routers."""
    app.add_middleware(ErrorHandlerMiddleware)
