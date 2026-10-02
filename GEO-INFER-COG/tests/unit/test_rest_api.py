"""
Unit tests for the GEO-INFER-COG REST API surface.

Covers the Flask application factory and its HTTP endpoints via
app.test_client(): happy paths for text analysis, entity extraction and
decision analysis; the per-endpoint JSON error shape on an internal
failure; and partial-initialization degradation observable through
/system/status.
"""

import pytest

from geo_infer_cog.api.rest_api import create_cog_api_app


@pytest.fixture()
def api_app():
    """Fully initialized COG REST API application with a test client."""
    app = create_cog_api_app()
    assert app is not None
    return app


class TestCreateCogApiApp:
    """Test the application factory."""

    def test_create_returns_flask_app(self, api_app):
        """create_cog_api_app returns a configured Flask application."""
        import flask

        assert isinstance(api_app, flask.Flask)
        assert api_app.config["PROPAGATE_EXCEPTIONS"] is True
        assert api_app.config["MAX_CONTENT_LENGTH"] == 50 * 1024 * 1024

    def test_health_reports_component_status(self, api_app):
        """Health endpoint reports every initialized component as present."""
        client = api_app.test_client()
        response = client.get("/health")

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "healthy"
        assert all(body["components"].values()), body["components"]


class TestTextAnalysisEndpoints:
    """Happy paths for the NLP endpoints."""

    def test_analyze_text(self, api_app):
        """/nlp/analyze returns extracted spatial entities for the text."""
        client = api_app.test_client()
        response = client.post(
            "/nlp/analyze",
            json={"text": "Meeting at 123 Main Street near Central Park"},
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["text"] == "Meeting at 123 Main Street near Central Park"
        assert body["entity_count"] == len(body["entities"])
        assert body["entity_count"] >= 1
        entity_texts = {entity["text"] for entity in body["entities"]}
        assert "123 Main Street" in entity_texts
        assert any("Central Park" in text for text in entity_texts)

    def test_analyze_text_requires_text_field(self, api_app):
        """/nlp/analyze rejects a body without the text field."""
        client = api_app.test_client()
        response = client.post("/nlp/analyze", json={})

        assert response.status_code == 400
        assert response.get_json()["error"] == "Missing text field"

    def test_extract_entities(self, api_app):
        """/nlp/entities/extract returns entities with confidence scores."""
        client = api_app.test_client()
        response = client.post(
            "/nlp/entities/extract",
            json={"text": "Walk from 12 Oak Avenue to the Harbor Station"},
        )

        assert response.status_code == 200
        body = response.get_json()
        assert len(body["entities"]) >= 1
        for entity in body["entities"]:
            assert 0.0 <= body["confidence_scores"][entity["text"]] <= 1.0

    def test_extract_entities_filters_by_type(self, api_app):
        """/nlp/entities/extract filters entities when types are requested."""
        client = api_app.test_client()
        response = client.post(
            "/nlp/entities/extract",
            json={
                "text": "123 Main Street near Central Park",
                "entity_types": ["street"],
            },
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["entities"], "expected at least one street entity"
        assert {entity["entity_type"] for entity in body["entities"]} == {"street"}


class TestDecisionSupportEndpoint:
    """Happy path for the decision analysis endpoint."""

    def test_analyze_decision(self, api_app):
        """/decision-support/analyze evaluates alternatives and ranks them."""
        client = api_app.test_client()
        response = client.post(
            "/decision-support/analyze",
            json={
                "scenario": {
                    "name": "evacuation site selection",
                    "description": "Choose a staging area for flood evacuation",
                },
                "criteria": ["capacity", "accessibility"],
                "alternatives": [
                    {
                        "id": "site_north",
                        "description": "North school gymnasium",
                        "spatial_context": {"lat": 47.6, "lon": -122.3},
                        "uncertainty_measures": {"capacity_variance": 0.1},
                    },
                    {
                        "id": "site_south",
                        "description": "South community center",
                        "spatial_context": {"lat": 47.5, "lon": -122.2},
                        "uncertainty_measures": {"capacity_variance": 0.4},
                    },
                ],
            },
        )

        assert response.status_code == 200
        body = response.get_json()
        assert body["scenario"]["name"] == "evacuation site selection"
        assert set(body["evaluation_scores"]) == {"site_north", "site_south"}
        assert len(body["recommendations"]) == 2
        assert isinstance(body["uncertainty_assessment"], dict)


class TestEndpointErrorHandling:
    """The per-endpoint except handler returns the JSON error shape."""

    def test_internal_failure_returns_json_500_shape(self, api_app, monkeypatch):
        """A failing component surfaces as 500 with {'error': <message>}."""

        class FailingProcessor:
            def extract_spatial_entities(self, text):
                raise RuntimeError("processor exploded")

        api_app.language_processor = FailingProcessor()
        client = api_app.test_client()
        response = client.post("/nlp/analyze", json={"text": "123 Main Street"})

        assert response.status_code == 500
        assert response.get_json() == {"error": "processor exploded"}


class TestPartialInitializationDegradation:
    """Component-init failures degrade the app instead of crashing it."""

    def test_partial_init_visible_via_system_status(self, monkeypatch):
        """A component that fails to initialize is reported absent by /system/status."""
        import geo_infer_cog.api.rest_api as rest_api

        def failing_visualizer(*args, **kwargs):
            raise RuntimeError("visualizer unavailable")

        monkeypatch.setattr(rest_api, "HumanCenteredVisualizer", failing_visualizer)
        app = create_cog_api_app()
        assert app is not None

        client = app.test_client()
        response = client.get("/system/status")

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "operational"
        assert body["components"]["visualizer"] is False
        assert body["components"]["cognitive_engine"] is True
        assert body["components"]["language_processor"] is True


@pytest.mark.parametrize("dependency", ["flask", "flask_cors"])
@pytest.mark.parametrize(
    "failure", ["native_import_error", "transitive_module_missing"]
)
@pytest.mark.parametrize("target", ["geo_infer_cog", "geo_infer_cog.api.rest_api"])
def test_installed_api_dependency_failure_propagates(
    tmp_path, dependency, failure, target
):
    """An installed but broken API extra must not masquerade as an absent extra."""
    import subprocess
    import sys

    script = f"""
import builtins, importlib
original = builtins.__import__
def import_with_failure(name, *args, **kwargs):
    if name == {dependency!r}:
        if {failure!r} == 'native_import_error':
            raise ImportError('installed dependency failed its native import')
        raise ModuleNotFoundError('installed dependency has a missing transitive module', name='broken_transitive_dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = import_with_failure
try:
    importlib.import_module({target!r})
except ImportError as error:
    assert str(error).startswith('installed dependency'), str(error)
    if {failure!r} == 'transitive_module_missing':
        assert isinstance(error, ModuleNotFoundError)
        assert error.name == 'broken_transitive_dependency'
    else:
        assert type(error) is ImportError
else:
    raise AssertionError('Installed API dependency failure was suppressed')
print('original installed-dependency failure propagated')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "original installed-dependency failure propagated" in result.stdout
