"""
API client for an externally deployed GEO-INFER-ACT model service.
"""

from typing import Any, cast
import math
from urllib.parse import quote, urlsplit
import requests


class Client:
    """REST client for an **external** ACT ``/models`` service deployment.

    This module ships no HTTP server: no FastAPI app or router exists in
    GEO-INFER-ACT, and the endpoint map in
    :mod:`geo_infer_act.api.endpoints` describes the same external surface.
    Point ``base_url`` at the deployed service (e.g. a separate FastAPI app
    exposing ``POST /models`` and ``GET /models/{model_id}``); the client
    speaks HTTP to it and never constructs one locally.
    """

    def __init__(
        self, base_url: str = "http://localhost:8000", timeout: float = 10.0
    ) -> None:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and greater than zero")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTP(S) URL")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) URL without credentials, query or fragment"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a bounded request and raise for non-success responses."""
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("ACT service must return a JSON object")
        return cast(dict[str, Any], result)

    def create_model(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Create a new model via API."""
        return self._request("POST", "/models", json=model_config)

    def get_model(self, model_id: str) -> dict[str, Any]:
        """Get model details via API."""
        return self._request("GET", f"/models/{quote(str(model_id), safe='')}")
