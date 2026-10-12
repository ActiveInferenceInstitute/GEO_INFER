"""Regression tests for DATA API error handling (mirrors LOG-EXC-01).

The pre-LOG-EXC-01 handlers caught every ``Exception`` and returned
``HTTP 400 str(e)``, leaking internal text on server faults. Domain
``ValueError``s must map to HTTP 400 with their message, unregistered
datasets to HTTP 404, and any other escaping exception to a generic
HTTP 500 ``INTERNAL_ERROR`` body produced by ``ErrorHandlerMiddleware``.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from geo_infer_data.api.rest_api import DataAPI


class _StubService:
    """Stub replacing a DataAPI service whose call raises or returns."""

    def __init__(self, result: Any = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc

    async def ingest_multi_source(self, **kwargs: Any) -> Any:
        return self._invoke()

    async def execute_workflow(self, **kwargs: Any) -> Any:
        return self._invoke()

    async def validate_dataset(self, dataset_id: str) -> Any:
        return self._invoke()

    def _invoke(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._result


def _client(stubs: dict[str, Any]) -> TestClient:
    """Build a DataAPI-backed TestClient with stubbed core services."""
    api = DataAPI()
    for attr, service in stubs.items():
        setattr(api, attr, service)
    return TestClient(api.app)


_SUCCESS = {"status": "ok"}


class TestIngestEndpointErrorMapping:
    """/data/ingest/multi-source must distinguish domain errors from faults."""

    def test_domain_value_error_maps_to_400(self) -> None:
        client = _client(
            {
                "ingestion_service": _StubService(
                    exc=ValueError("Unsupported data source(s): foo")
                )
            }
        )
        response = client.post("/data/ingest/multi-source", json={})
        assert response.status_code == 400
        assert "Unsupported data source(s)" in response.json()["detail"]

    def test_connection_error_maps_to_400(self) -> None:
        client = _client(
            {
                "ingestion_service": _StubService(
                    exc=ConnectionError("Failed to connect")
                )
            }
        )
        response = client.post("/data/ingest/multi-source", json={})
        assert response.status_code == 400

    def test_type_error_surfaces_as_500(self) -> None:
        client = _client(
            {
                "ingestion_service": _StubService(
                    exc=TypeError("'int' object is not subscriptable")
                )
            }
        )
        response = client.post("/data/ingest/multi-source", json={})
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "not subscriptable" not in response.text

    def test_success_path_returns_result(self) -> None:
        client = _client({"ingestion_service": _StubService(result=_SUCCESS)})
        response = client.post("/data/ingest/multi-source", json={})
        assert response.status_code == 200
        assert response.json() == _SUCCESS


class TestETLEndpointErrorMapping:
    """/data/etl/execute must distinguish domain errors from server faults."""

    def test_domain_value_error_maps_to_400(self) -> None:
        client = _client(
            {
                "pipeline_service": _StubService(
                    exc=ValueError("Unknown transformation type: foo")
                )
            }
        )
        response = client.post("/data/etl/execute", json={})
        assert response.status_code == 400
        assert "Unknown transformation type" in response.json()["detail"]

    def test_type_error_surfaces_as_500(self) -> None:
        client = _client(
            {
                "pipeline_service": _StubService(
                    exc=TypeError("clean transformation requires a pandas DataFrame")
                )
            }
        )
        response = client.post("/data/etl/execute", json={})
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "pandas DataFrame" not in response.text

    def test_success_path_returns_result(self) -> None:
        client = _client({"pipeline_service": _StubService(result=_SUCCESS)})
        response = client.post("/data/etl/execute", json={})
        assert response.status_code == 200
        assert response.json() == _SUCCESS


class TestQualityEndpointErrorMapping:
    """/quality/validate/{id} must distinguish domain errors from faults."""

    def test_unregistered_dataset_maps_to_404(self) -> None:
        client = _client({"quality_service": _StubService(exc=KeyError("ds-1"))})
        response = client.post("/quality/validate/ds-1")
        assert response.status_code == 404
        assert "not registered" in response.json()["detail"]

    def test_domain_value_error_maps_to_400(self) -> None:
        client = _client(
            {
                "quality_service": _StubService(
                    exc=ValueError("Unknown validation rule(s): foo")
                )
            }
        )
        response = client.post("/quality/validate/ds-1")
        assert response.status_code == 400
        assert "Unknown validation rule(s)" in response.json()["detail"]

    def test_type_error_surfaces_as_500(self) -> None:
        client = _client(
            {
                "quality_service": _StubService(
                    exc=TypeError("'NoneType' object is not iterable")
                )
            }
        )
        response = client.post("/quality/validate/ds-1")
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "not iterable" not in response.text

    def test_success_path_returns_report(self) -> None:
        client = _client({"quality_service": _StubService(result=_SUCCESS)})
        response = client.post("/quality/validate/ds-1")
        assert response.status_code == 200
        assert response.json() == _SUCCESS


@pytest.mark.parametrize(
    "attr", ["ingestion_service", "pipeline_service", "quality_service"]
)
def test_no_internal_text_on_server_fault(attr: str) -> None:
    """A server fault must never echo the internal exception text."""
    client = _client({attr: _StubService(exc=RuntimeError("secret internal detail"))})
    method = client.post
    if attr == "quality_service":
        response = method("/quality/validate/ds-1")
    elif attr == "ingestion_service":
        response = method("/data/ingest/multi-source", json={})
    else:
        response = method("/data/etl/execute", json={})
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
