"""
Main application entry point for GEO-INFER-OPS.

This module initializes the FastAPI application, sets up logging,
and configures all routes and middleware.

Config loader note: the app entrypoint uses the dict-based
``geo_infer_ops.utils.config.load_config`` because app-level settings
(``service``, ``development`` in config/local.yaml) are not part of the
pydantic domain schema. The pydantic ``geo_infer_ops.core.config.load_config``
remains the loader for domain components (orchestrator, cache, security).
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from geo_infer_ops import __version__
from geo_infer_ops.utils import load_config, configure_logging, get_logger

# Initialize logger
logger = get_logger("geo_infer_ops.app")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Load configuration
    try:
        config = load_config()
        # Configure logging based on config
        configure_logging(
            log_level=config["logging"]["level"],
            json_format=config["logging"]["format"] == "json",
            log_file=config["logging"]["file"],
        )
    except Exception as e:
        # Fall back to default logging if config fails
        configure_logging()
        logger.error("Failed to load configuration", error=str(e))
        config = {
            "service": {"host": "0.0.0.0", "port": 8000},
            "security": {"cors_origins": []},
            "monitoring": {"enabled": True, "metrics_path": "/metrics"},
            "logging": {"level": "INFO", "format": "json", "file": None},
            "development": {"hot_reload": False},
        }

    # Create FastAPI app
    app = FastAPI(
        title="GEO-INFER-OPS",
        description="Operational kernel for system orchestration, logging, testing, and architecture",
        version=__version__,
    )

    # Configure CORS
    origins = config["security"].get("cors_origins", [])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Prometheus metrics endpoint
    if config["monitoring"].get("enabled", False):
        metrics_app = make_asgi_app()
        app.mount(config["monitoring"]["metrics_path"], metrics_app)

    # Health check endpoint
    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    # Version endpoint
    @app.get("/version")
    def version() -> dict[str, str]:
        return {"version": __version__}

    # Expose the resolved configuration on the ASGI app state
    app.state.config = config
    return app


if __name__ == "__main__":
    """Run the application when executed as a script."""
    app = create_app()
    config = app.state.config
    logger.info(
        "Starting GEO-INFER-OPS",
        host=config["service"]["host"],
        port=config["service"]["port"],
    )

    reload_enabled = config["development"].get("hot_reload", False)
    # With reload enabled uvicorn must re-import the module itself, so hand it
    # the factory string; otherwise pass the already-built app and skip the
    # redundant second construction.
    uvicorn.run(
        "geo_infer_ops.app:create_app" if reload_enabled else app,
        factory=reload_enabled,
        host=config["service"]["host"],
        port=config["service"]["port"],
        reload=reload_enabled,
    )
