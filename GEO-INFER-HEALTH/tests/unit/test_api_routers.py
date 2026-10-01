"""FastAPI TestClient tests for the GEO-INFER-HEALTH API routers."""

from datetime import datetime, timedelta, UTC

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from geo_infer_health.api import router
from geo_infer_health.api import (
    api_disease_surveillance,
    api_environmental_health,
    api_healthcare_accessibility,
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    api_disease_surveillance.reset_stores()
    api_environmental_health.reset_stores()
    api_healthcare_accessibility.reset_stores()
    with TestClient(app) as test_client:
        yield test_client
    api_disease_surveillance.reset_stores()
    api_environmental_health.reset_stores()
    api_healthcare_accessibility.reset_stores()


def _report(report_id: str, lat: float, lon: float, case_count: int) -> dict:
    return {
        "report_id": report_id,
        "disease_code": "FLU",
        "location": {"latitude": lat, "longitude": lon},
        "report_date": datetime.now(UTC).isoformat(),
        "case_count": case_count,
        "source": "Hospital A",
    }


# --- Surveillance router -------------------------------------------------


def test_submit_and_list_reports(client):
    response = client.post(
        "/api/v1/surveillance/reports/", json=_report("r1", 34.05, -118.24, 5)
    )
    assert response.status_code == 201
    assert response.json()["report_id"] == "r1"

    listing = client.get("/api/v1/surveillance/reports/")
    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_hotspots_require_reports(client):
    response = client.post("/api/v1/surveillance/hotspots/identify")
    assert response.status_code == 404


def test_identify_hotspots(client):
    for i in range(5):
        client.post(
            "/api/v1/surveillance/reports/",
            json=_report(f"r{i}", 34.05 + i * 1e-4, -118.24, 3),
        )
    client.post(
        "/api/v1/surveillance/population_data/",
        json={"area_id": "la", "population_count": 10000},
    )
    response = client.post(
        "/api/v1/surveillance/hotspots/identify?threshold_case_count=5"
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_local_incidence_rate(client):
    client.post("/api/v1/surveillance/reports/", json=_report("r1", 34.05, -118.24, 10))
    client.post(
        "/api/v1/surveillance/population_data/",
        json={"area_id": "la", "population_count": 10000},
    )
    response = client.post(
        "/api/v1/surveillance/incidence_rate/local"
        "?latitude=34.05&longitude=-118.24&radius_km=5&time_window_days=30"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_cases_in_area"] == 10
    assert body["population_estimated"] is True

    assert body["incidence_rate_per_100k"] == pytest.approx(100.0)


def test_environmental_reading_roundtrip(client):
    payload = {
        "data_id": "e1",
        "parameter_name": "PM2.5",
        "value": 35.0,
        "unit": "ug/m3",
        "location": {"latitude": 34.05, "longitude": -118.24},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    post = client.post("/api/v1/environment/readings/", json=payload)
    assert post.status_code == 201

    listing = client.get("/api/v1/environment/readings/?parameter_name=pm2.5")
    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_average_exposure_endpoints(client):
    now = datetime.now(UTC)
    for i, value in enumerate([10.0, 20.0]):
        client.post(
            "/api/v1/environment/readings/",
            json={
                "data_id": f"e{i}",
                "parameter_name": "PM2.5",
                "value": value,
                "unit": "ug/m3",
                "location": {"latitude": 34.05, "longitude": -118.24},
                "timestamp": (now - timedelta(days=1)).isoformat(),
            },
        )
    response = client.post(
        "/api/v1/environment/exposure/average"
        "?radius_km=5&parameter_name=PM2.5&time_window_days=7",
        json=[{"latitude": 34.05, "longitude": -118.24}],
    )
    assert response.status_code == 200
    values = list(response.json().values())
    assert values[0] == pytest.approx(15.0)


def test_exposure_uses_latest_reading_anchor(client):
    """Historical data older than the window from *today* still yields a value."""
    old = datetime(2020, 1, 1, tzinfo=UTC)
    for i, value in enumerate([4.0, 6.0]):
        client.post(
            "/api/v1/environment/readings/",
            json={
                "data_id": f"old{i}",
                "parameter_name": "PM2.5",
                "value": value,
                "unit": "ug/m3",
                "location": {"latitude": 34.05, "longitude": -118.24},
                "timestamp": (old + timedelta(hours=i)).isoformat(),
            },
        )
    response = client.post(
        "/api/v1/environment/exposure/average"
        "?radius_km=5&parameter_name=PM2.5&time_window_days=7",
        json=[{"latitude": 34.05, "longitude": -118.24}],
    )
    assert response.status_code == 200
    assert list(response.json().values())[0] == pytest.approx(5.0)


# --- Accessibility router ------------------------------------------------


def test_accessibility_facility_roundtrip(client):
    payload = {
        "facility_id": "f1",
        "name": "General Hospital",
        "location": {"latitude": 34.05, "longitude": -118.24},
        "facility_type": "hospital",
        "capacity": 200,
    }
    post = client.post("/api/v1/accessibility/facilities/", json=payload)
    assert post.status_code == 201

    listing = client.get("/api/v1/accessibility/facilities/")
    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_reset_stores_clears_state(client):
    client.post("/api/v1/surveillance/reports/", json=_report("r1", 34.05, -118.24, 5))
    api_disease_surveillance.reset_stores()
    assert client.get("/api/v1/surveillance/reports/").json() == []
