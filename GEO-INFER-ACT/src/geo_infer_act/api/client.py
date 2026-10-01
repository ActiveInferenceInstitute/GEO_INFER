"""
API client for an externally deployed GEO-INFER-ACT model service.
"""

from typing import Any, cast
from urllib.parse import quote
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
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
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
        return cast(dict[str, Any], response.json())

    def create_model(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Create a new model via API."""
        return self._request("POST", "/models", json=model_config)

    def get_model(self, model_id: str) -> dict[str, Any]:
        """Get model details via API."""
        return self._request("GET", f"/models/{quote(str(model_id), safe='')}")
