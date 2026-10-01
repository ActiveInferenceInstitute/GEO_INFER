"""
Main application entry point for GEO-INFER-API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from geo_infer_api.core.config import get_settings
from geo_infer_api.core.middleware import (
    ErrorHandlerMiddleware,
    RequestLoggingMiddleware,
)
from geo_infer_api.endpoints import (
    algorithms_router,
    geojson_router,
    health_router,
)


def cors_allow_credentials(origins: list[str]) -> bool:
    """Decide whether CORS may send credentials.

    Credentialed CORS is only safe for an explicit, finite origin list.
    A wildcard (``"*"``) combined with ``allow_credentials=True`` would
    let any origin make credentialed requests, so credentials are
    enabled only when at least one explicit origin is configured and no
    wildcard is present.
    """
    return bool(origins) and "*" not in origins


# Application construction is lazy (PEP 562 ``__getattr__`` below): importing
# ``geo_infer_api`` must not require SECRET_KEY; the fail-closed settings
# check fires when the app instance is actually built (uvicorn target
# ``geo_infer_api.app:main_app`` or the first attribute access).
def create_app() -> FastAPI:
    """Build the FastAPI application (requires SECRET_KEY to be set)."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="GEO-INFER API - Standardized Interfaces for Geospatial Interoperability",
        version=settings.app_version,
    )
    # Add middleware (order matters — outermost first)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    # Configure CORS via FastAPI's built-in middleware; credentials are only
    # enabled for an explicit, wildcard-free origin list (see
    # cors_allow_credentials).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=cors_allow_credentials(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Include routers
    app.include_router(health_router.router, tags=["Health"])
    app.include_router(geojson_router.router, prefix="/api/v1", tags=["GeoJSON"])
    app.include_router(algorithms_router.router, prefix="/api/v1", tags=["Algorithms"])
    return app


def __getattr__(name: str):
    """Construct ``main_app`` lazily on first attribute access."""
    if name == "main_app":
        module = __import__(__name__)
        app = create_app()
        module.main_app = app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("geo_infer_api.app:main_app", host="0.0.0.0", port=8000, reload=True)
