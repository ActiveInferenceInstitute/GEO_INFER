"""
Middleware for the GEO-INFER-API.
"""

import logging
import time
from typing import cast
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .exceptions import APIError

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for handling API errors consistently."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle requests and convert API exceptions to proper HTTP responses."""
        try:
            response = cast("Response", await call_next(request))
            return response
        except APIError as e:
            return JSONResponse(status_code=e.status_code, content=e.to_dict())
        except Exception:
            logger.exception(
                "Unexpected error processing %s %s", request.method, request.url
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


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log API requests with timing."""
        start_time = time.time()

        logger.info("API Request: %s %s", request.method, request.url)

        response = cast("Response", await call_next(request))

        process_time = time.time() - start_time

        logger.info(
            "API Response: %s for %s %s in %.3fs",
            response.status_code,
            request.method,
            request.url,
            process_time,
        )

        response.headers["X-Process-Time"] = str(process_time)

        return response
