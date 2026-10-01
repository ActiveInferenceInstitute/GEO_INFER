"""Regression tests for NORMS API error handling (generic-500 contract).

Domain errors surfaced by handlers as ``ValueError`` must map to HTTP 400
with the domain message, and any non-domain exception must return the
generic INTERNAL_ERROR 500 without leaking internal exception text to the
client, mirroring
GEO-INFER-LOG/tests/unit/test_api_error_handling.py (GS-223).
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from geo_infer_norms.api import register_error_handlers
from geo_infer_norms.api.compliance_api import ComplianceAPI
from geo_infer_norms.api.legal_api import LegalAPI
from geo_infer_norms.api.normative_api import NormativeAPI
from geo_infer_norms.api.policy_api import PolicyAPI
from geo_infer_norms.api.zoning_api import ZoningAPI
from geo_infer_norms.core.compliance_tracking import ComplianceTracker
from geo_infer_norms.core.legal_frameworks import LegalFramework
from geo_infer_norms.core.normative_inference import SocialNormDiffusion
from geo_infer_norms.core.zoning_analysis import ZoningAnalyzer


def _client(router: APIRouter) -> TestClient:
    """Build a TestClient hosting one router with shared error handling."""
    app = FastAPI()
    app.include_router(router)
    register_error_handlers(app)
    return TestClient(app)


def _assert_generic_500(response: Any) -> None:
    """Assert the response is the generic INTERNAL_ERROR 500 with no leak."""
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "An unexpected error occurred"
    assert body["error"]["status_code"] == 500
    assert "boom" not in response.text


class TestComplianceRouterErrorMapping:
    """Compliance handlers must distinguish domain errors from server faults."""

    def test_domain_error_maps_to_400(self) -> None:
        client = _client(ComplianceAPI().router)
        response = client.post(
            "/reports/export",
            json={"report_type": "bogus", "params": {}},
        )
        assert response.status_code == 400
        assert "Invalid report type" in response.json()["detail"]

    def test_server_fault_stays_generic(self) -> None:
        class _FailingTracker:
            """Stub tracker whose add_compliance_status raises a server fault."""

            def add_compliance_status(self, status: Any) -> None:
                raise RuntimeError("boom")

        client = _client(
            ComplianceAPI(
                compliance_tracker=cast("ComplianceTracker", _FailingTracker())
            ).router
        )
        response = client.post(
            "/status",
            json={
                "entity_id": "entity-123",
                "regulation_id": "reg-456",
                "is_compliant": True,
                "compliance_level": 0.85,
            },
        )
        _assert_generic_500(response)


class TestLegalRouterErrorMapping:
    """Legal handlers must distinguish domain errors from server faults."""

    def test_domain_error_maps_to_400(self) -> None:
        client = _client(LegalAPI().router)
        response = client.post(
            "/jurisdictions",
            json={
                "name": "Sample City",
                "level": "local",
                "geometry": {"type": "Bogus", "coordinates": []},
            },
        )
        assert response.status_code == 400
        assert "Invalid geometry" in response.json()["detail"]

    def test_server_fault_stays_generic(self) -> None:
        class _FailingFramework:
            """Stub framework whose export raises a server fault."""

            def export_to_geodataframe(self) -> Any:
                raise RuntimeError("boom")

        client = _client(
            LegalAPI(legal_framework=cast("LegalFramework", _FailingFramework())).router
        )
        response = client.get("/spatial/export")
        _assert_generic_500(response)


class TestNormativeRouterErrorMapping:
    """Normative handlers must distinguish domain errors from server faults."""

    def test_domain_error_maps_to_400(self) -> None:
        client = _client(NormativeAPI().router)
        response = client.post(
            "/inference/spatial-patterns",
            json={
                "data_source": "observations.csv",
                "inference_type": "behavioral",
                "spatial_extent": {"type": "Bogus", "coordinates": []},
            },
        )
        assert response.status_code == 400
        assert "Invalid geometry" in response.json()["detail"]

    def test_server_fault_stays_generic(self) -> None:
        class _FailingDiffusion:
            """Stub diffusion engine whose simulate raises a server fault."""

            entities: dict[str, Any] = {}
            adoption_state: dict[str, dict[str, Any]] = {}

            def add_entity(self, *args: Any, **kwargs: Any) -> None:
                return None

            def add_norm(self, *args: Any, **kwargs: Any) -> None:
                return None

            def simulate(self, time_steps: int) -> Any:
                raise RuntimeError("boom")

            def get_adoption_history(self) -> dict[str, Any]:
                return {}

            def get_adoption_summary(self) -> dict[str, Any]:
                return {}

        api = NormativeAPI(
            social_norm_diffusion=cast("SocialNormDiffusion", _FailingDiffusion())
        )
        client = _client(api.router)
        created = client.post(
            "/norms",
            json={
                "name": "Recycling Behavior",
                "category": "environmental",
                "strength": 0.5,
                "jurisdiction_ids": ["city-001"],
            },
        ).json()
        response = client.post(
            "/diffusion/simulate",
            json={"norm_id": created["norm_id"], "time_steps": 3},
        )
        _assert_generic_500(response)


class TestPolicyRouterErrorMapping:
    """Policy handlers must distinguish domain errors from server faults."""

    def _create_policy(self, client: TestClient) -> str:
        created = client.post(
            "/policies",
            json={
                "name": "Green Infrastructure Policy",
                "category": "environmental",
                "jurisdiction_ids": ["city-001"],
            },
        )
        return str(created.json()["policy_id"])

    def test_domain_error_maps_to_400(self) -> None:
        client = _client(PolicyAPI().router)
        policy_id = self._create_policy(client)
        response = client.post(
            "/implementations",
            json={
                "policy_id": policy_id,
                "name": "Downtown Program",
                "jurisdiction_id": "city-001",
                "geometry": {"type": "Bogus", "coordinates": []},
            },
        )
        assert response.status_code == 400
        assert "Invalid geometry" in response.json()["detail"]

    def test_server_fault_stays_generic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from geo_infer_norms.api import policy_api as policy_api_module

        class _ExplodingGeoDataFrame:
            """Stub GeoDataFrame whose construction raises a server fault."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("boom")

        api = PolicyAPI()
        monkeypatch.setattr(
            policy_api_module,
            "gpd",
            SimpleNamespace(GeoDataFrame=_ExplodingGeoDataFrame),
        )
        client = _client(api.router)
        policy_id = self._create_policy(client)
        created = client.post(
            "/implementations",
            json={
                "policy_id": policy_id,
                "name": "Downtown Program",
                "jurisdiction_id": "city-001",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
        )
        assert created.status_code == 200
        response = client.get("/export/geojson")
        _assert_generic_500(response)


class TestZoningRouterErrorMapping:
    """Zoning handlers must distinguish domain errors from server faults."""

    def test_domain_error_maps_to_400(self) -> None:
        client = _client(ZoningAPI().router)
        response = client.post(
            "/districts",
            json={
                "name": "North Heights R-1 District",
                "zoning_code": "R-1",
                "geometry": {"type": "Bogus", "coordinates": []},
            },
        )
        assert response.status_code == 400
        assert "Invalid geometry" in response.json()["detail"]

    def test_server_fault_stays_generic(self) -> None:
        class _FailingAnalyzer:
            """Stub analyzer whose compatibility check raises a server fault."""

            def calculate_compatibility(self, code1: str, code2: str) -> float:
                raise RuntimeError("boom")

        client = _client(
            ZoningAPI(zoning_analyzer=cast("ZoningAnalyzer", _FailingAnalyzer())).router
        )
        response = client.get("/compatibility", params={"code1": "R-1", "code2": "C-1"})
        _assert_generic_500(response)
