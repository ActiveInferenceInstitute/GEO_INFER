"""
Configuration settings for the GEO-INFER-API.
"""

import os
import json
from functools import lru_cache

from importlib.metadata import PackageNotFoundError, version as _distribution_version

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _installed_version(fallback: str) -> str:
    """The installed ``geo-infer-api`` distribution version, else ``fallback``."""
    try:
        return _distribution_version("geo-infer-api")
    except PackageNotFoundError:
        return fallback


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Security invariants:

    - ``secret_key`` has **no default**: it must be provided via the
      ``SECRET_KEY`` environment variable (or an explicit constructor
      argument). :func:`get_settings` raises ``RuntimeError`` when
      ``SECRET_KEY`` is unset, so the application fails closed instead of
      silently signing tokens with a well-known development secret.
    - ``cors_origins`` defaults to an **empty list**. A wildcard (``"*"``)
      must never be combined with credentialed CORS; callers wiring
      CORSMiddleware must enable ``allow_credentials`` only when the
      origin list is non-empty and does not contain ``"*"``.
    """

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # Application metadata
    app_name: str = "GEO-INFER-API"
    # Derived from the installed distribution so the version surface tracks
    # releases without a code edit; the literal is only the source-checkout
    # fallback when the package was never installed.
    app_version: str = _installed_version("0.4.0")

    # API settings
    api_prefix: str = "/api/v1"

    # CORS settings
    cors_origins: list[str] = []

    # Security settings
    secret_key: str
    # NOTE: JWT/auth settings (algorithm, token expiry) are intentionally
    # absent — no auth middleware exists yet. Add them together with the code
    # that consumes them.

    # Database settings
    database_url: str | None = None

    # OGC API settings
    ogc_api_features_enabled: bool = True
    ogc_api_processes_enabled: bool = True

    # In-memory polygon store settings
    polygon_store_max_size: int = 10_000

    @field_validator("polygon_store_max_size")
    @classmethod
    def check_polygon_store_max_size(cls, v: int) -> int:
        """Reject non-positive caps; an unbounded or zero-cap store is a
        configuration error, not something to fall back on."""
        if v < 1:
            raise ValueError("polygon_store_max_size must be >= 1")
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from a JSON array string, comma list, or list."""
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return json.loads(v)
            return [i.strip() for i in v.split(",")]
        return list(v)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings to avoid reloading from env every time.

    Raises:
        RuntimeError: If the ``SECRET_KEY`` environment variable is unset.
            There is deliberately no default secret; running without an
            explicitly configured key is a deployment error.
    """
    secret = os.getenv("SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "SECRET_KEY environment variable is not set; refusing to start "
            "with an insecure default signing key"
        )
    return Settings(secret_key=secret)
