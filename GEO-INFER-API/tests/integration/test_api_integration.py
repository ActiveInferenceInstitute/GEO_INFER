"""
Integration tests for GEO-INFER-API: FastAPI application, config, middleware, and exception hierarchy.

Tests the API configuration, custom exception classes, and middleware components
working together. Uses FastAPI's TestClient for HTTP-level integration testing.
"""

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def settings():
    """Create test settings."""
    from geo_infer_api.core.config import Settings

    return Settings(
        app_name="GEO-INFER-API-Test",
        app_version="0.0.1-test",
        secret_key="test_secret_key",
    )


@pytest.fixture
def test_app():
    """Create a minimal FastAPI app with middleware and routers for testing."""
    from geo_infer_api.core.middleware import (
        ErrorHandlerMiddleware,
        RequestLoggingMiddleware,
    )
    from geo_infer_api.endpoints import health_router

    app = FastAPI(title="Test App", version="0.0.1")
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(health_router.router, tags=["Health"])

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the app."""
    return TestClient(test_app)


class TestConfigIntegration:
    """Test configuration system integration."""

    def test_settings_defaults(self, settings):
        """Test that settings have sensible defaults."""
        assert settings.app_name == "GEO-INFER-API-Test"
        assert settings.app_version == "0.0.1-test"
        assert settings.api_prefix == "/api/v1"
        assert settings.cors_origins == []

    def test_settings_ogc_flags(self, settings):
        """Test OGC API feature flags."""
        assert settings.ogc_api_features_enabled is True
        assert settings.ogc_api_processes_enabled is True

    def test_cached_settings_singleton(self):
        """Test that get_settings returns cached instance."""
        from geo_infer_api.core.config import get_settings

        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2, "get_settings should return the same cached instance"
        get_settings.cache_clear()


class TestExceptionHierarchy:
    """Test custom exception classes work together."""

    def test_api_error_to_dict(self):
        """Test APIError serialization."""
        from geo_infer_api.core.exceptions import APIError

        error = APIError(status_code=400, detail="Bad request", error_code="TEST_ERROR")
        d = error.to_dict()
        assert d["error"]["code"] == "TEST_ERROR"
        assert d["error"]["message"] == "Bad request"
        assert d["error"]["status_code"] == 400

    def test_not_found_error(self):
        """Test NotFoundError with resource and identifier."""
        from geo_infer_api.core.exceptions import NotFoundError

        error = NotFoundError("Polygon", "poly_123")
        d = error.to_dict()
        assert "not found" in d["error"]["message"]
        assert d["error"]["resource"] == "Polygon"
        assert d["error"]["identifier"] == "poly_123"
        assert error.status_code == 404

    def test_validation_error(self):
        """Test ValidationError with field information."""
        from geo_infer_api.core.exceptions import ValidationError

        error = ValidationError("Invalid value", field="bbox", value="bad_data")
        d = error.to_dict()
        assert d["error"]["code"] == "VALIDATION_ERROR"
        assert d["error"]["field"] == "bbox"
        assert error.status_code == 422

    def test_geometry_error(self):
        """Test GeometryError with geometry context."""
        from geo_infer_api.core.exceptions import GeometryError

        error = GeometryError(
            "Self-intersecting polygon",
            geometry_type="Polygon",
            operation="validate",
        )
        d = error.to_dict()
        assert d["error"]["code"] == "GEOMETRY_ERROR"
        assert d["error"]["geometry_type"] == "Polygon"
        assert error.status_code == 400

    def test_conflict_error(self):
        """Test ConflictError for duplicate resources."""
        from geo_infer_api.core.exceptions import ConflictError

        error = ConflictError("Feature", "already exists", "feat_001")
        assert error.status_code == 409
        assert "already exists" in error.to_dict()["error"]["message"]


class TestHealthEndpoints:
    """Test health check endpoints via middleware stack."""

    def test_health_check(self, client):
        """Test basic health check endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_detailed_health_check_structure(self, client):
        """Test detailed health check returns component info."""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "components" in data
        assert "geojson_service" in data["components"]
        assert "memory_usage" in data["components"]
        assert "uptime" in data["components"]
        # Verify live values are present
        assert "value_mb" in data["components"]["memory_usage"]
        assert "seconds" in data["components"]["uptime"]
        assert "human" in data["components"]["uptime"]

    def test_process_time_header(self, client):
        """Test that RequestLoggingMiddleware adds X-Process-Time header."""
        response = client.get("/health")
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0, "Process time should be non-negative"

    def test_nonexistent_endpoint_returns_404(self, client):
        """Test that non-existent endpoints return proper 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404


class TestMiddlewareIntegration:
    """Test middleware components working together."""

    def test_error_handler_catches_unexpected_errors(self):
        """Test ErrorHandlerMiddleware catches unexpected (non-HTTP) errors."""
        from geo_infer_api.core.middleware import (
            ErrorHandlerMiddleware,
            RequestLoggingMiddleware,
        )

        app = FastAPI()
        app.add_middleware(ErrorHandlerMiddleware)
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/trigger-unexpected-error")
        async def trigger_unexpected_error():
            raise RuntimeError("Something broke")

        c = TestClient(app, raise_server_exceptions=False)
        response = c.get("/trigger-unexpected-error")
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert data["error"]["message"] == "An unexpected error occurred"

    def test_api_error_returns_proper_status(self):
        """Test that APIError (HTTPException subclass) returns correct status via FastAPI handler."""
        from geo_infer_api.core.middleware import (
            ErrorHandlerMiddleware,
            RequestLoggingMiddleware,
        )
        from geo_infer_api.core.exceptions import APIError

        app = FastAPI()
        app.add_middleware(ErrorHandlerMiddleware)
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/trigger-api-error")
        async def trigger_api_error():
            raise APIError(status_code=400, detail="Test error", error_code="TEST")

        c = TestClient(app, raise_server_exceptions=False)
        response = c.get("/trigger-api-error")
        assert response.status_code == 400
        data = response.json()
        # APIError extends HTTPException; FastAPI's built-in handler returns {"detail": ...}
        assert data["detail"] == "Test error"

    def test_cors_headers_present(self):
        """Test that CORSMiddleware adds appropriate headers."""
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        c = TestClient(app)
        response = c.get("/test", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestFullAPIWorkflow:
    """End-to-end workflow tests using the main application."""

    @pytest.fixture
    def main_client(self):
        """TestClient wrapping the full main_app."""
        from geo_infer_api.app import main_app
        from geo_infer_api.endpoints.geojson_router import POLYGON_FEATURES

        POLYGON_FEATURES.clear()
        yield TestClient(main_app)
        POLYGON_FEATURES.clear()

    POLYGON_BODY = {
        "type": "Feature",
        "id": "workflow-poly-1",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.51, 37.77],
                    [-122.42, 37.81],
                    [-122.37, 37.73],
                    [-122.51, 37.77],
                ]
            ],
        },
        "properties": {"name": "Workflow Test Polygon"},
    }

    def test_create_read_update_delete_workflow(self, main_client):
        """Full CRUD lifecycle for a polygon feature."""
        # CREATE
        resp = main_client.post(
            "/api/v1/collections/polygons/items", json=self.POLYGON_BODY
        )
        assert resp.status_code == 201
        feature_id = resp.json()["id"]

        # READ
        resp = main_client.get(f"/api/v1/collections/polygons/items/{feature_id}")
        assert resp.status_code == 200
        assert resp.json()["properties"]["name"] == "Workflow Test Polygon"

        # UPDATE
        updated = {**self.POLYGON_BODY, "properties": {"name": "Updated Polygon"}}
        resp = main_client.put(
            f"/api/v1/collections/polygons/items/{feature_id}", json=updated
        )
        assert resp.status_code == 200
        assert resp.json()["properties"]["name"] == "Updated Polygon"

        # DELETE
        resp = main_client.delete(f"/api/v1/collections/polygons/items/{feature_id}")
        assert resp.status_code == 204

        # VERIFY GONE
        resp = main_client.get(f"/api/v1/collections/polygons/items/{feature_id}")
        assert resp.status_code == 404

    def test_area_operation_on_feature(self, main_client):
        """Area calculation operation returns positive area_sq_km."""
        resp = main_client.post(
            "/api/v1/operations/polygon/area", json=self.POLYGON_BODY
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["area_sq_km"] > 0
        assert data["method"] == "planar"

    def test_buffer_operation_on_feature(self, main_client):
        """Buffer operation returns a larger polygon."""
        resp = main_client.post(
            "/api/v1/operations/polygon/buffer?distance=5&unit=kilometers",
            json=self.POLYGON_BODY,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["geometry"]["type"] == "Polygon"
        assert data["properties"]["buffer_distance"] == 5
