"""
Tests for the GeoJSON polygon endpoints.

These tests verify the functionality of the GeoJSON polygon API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from geo_infer_api.app import main_app
from geo_infer_api.endpoints.geojson_router import POLYGON_FEATURES
from geo_infer_api.models.geojson import GeoJSONType, Polygon, PolygonFeature

# Create a test client
client = TestClient(main_app)

# Sample polygon data for tests
SAMPLE_POLYGON_COORDS = [
    [
        [-122.51, 37.77],
        [-122.42, 37.81],
        [-122.37, 37.73],
        [-122.51, 37.77],  # Close the polygon
    ]
]

SAMPLE_POLYGON_FEATURE = {
    "type": "Feature",
    "id": "test-polygon-1",
    "geometry": {"type": "Polygon", "coordinates": SAMPLE_POLYGON_COORDS},
    "properties": {"name": "Test Polygon", "description": "A polygon for testing"},
}

INVALID_POLYGON_FEATURE = {
    "type": "Feature",
    "id": "invalid-polygon",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.51, 37.77],
                [-122.42, 37.81],
                # Missing the closing point
            ]
        ],
    },
    "properties": {"name": "Invalid Polygon"},
}


ZERO_ID_FEATURE = {
    "type": "Feature",
    "id": 0,
    "geometry": {"type": "Polygon", "coordinates": SAMPLE_POLYGON_COORDS},
    "properties": {"name": "Zero ID"},
}


# Fixtures
@pytest.fixture(autouse=True)
def clear_polygon_features():
    """Clear the polygon features dictionary before each test."""
    POLYGON_FEATURES.clear()
    yield
    POLYGON_FEATURES.clear()


@pytest.fixture
def sample_polygon_feature():
    """Create a sample polygon feature for testing."""
    return PolygonFeature(
        type=GeoJSONType.FEATURE,
        id="test-polygon-1",
        geometry=Polygon(type=GeoJSONType.POLYGON, coordinates=SAMPLE_POLYGON_COORDS),
        properties={"name": "Test Polygon", "description": "A polygon for testing"},
    )


def add_sample_feature():
    """Add a sample feature to the POLYGON_FEATURES dictionary."""
    response = client.post(
        "/api/v1/collections/polygons/items", json=SAMPLE_POLYGON_FEATURE
    )
    assert response.status_code == 201
    return response.json()["id"]


# Tests
def test_list_collections():
    """Test listing available feature collections."""
    response = client.get("/api/v1/collections")
    assert response.status_code == 200
    data = response.json()
    assert "collections" in data
    assert len(data["collections"]) == 1
    assert data["collections"][0]["id"] == "polygons"


def test_get_polygon_collection():
    """Test getting polygon collection metadata."""
    response = client.get("/api/v1/collections/polygons")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "polygons"
    assert "extent" in data
    assert "links" in data


def test_list_polygon_features_empty():
    """Test listing polygon features when none exist."""
    response = client.get("/api/v1/collections/polygons/items")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 0


def test_create_polygon_feature():
    """Test creating a new polygon feature."""
    response = client.post(
        "/api/v1/collections/polygons/items", json=SAMPLE_POLYGON_FEATURE
    )
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "test-polygon-1"
    assert data["type"] == "Feature"
    assert data["geometry"]["type"] == "Polygon"
    assert len(POLYGON_FEATURES) == 1


def test_create_invalid_polygon_feature():
    """Test creating an invalid polygon feature."""
    response = client.post(
        "/api/v1/collections/polygons/items", json=INVALID_POLYGON_FEATURE
    )
    assert response.status_code == 422  # Validation error


def test_create_duplicate_polygon_feature():
    """Test creating a polygon feature with an ID that already exists."""
    # Add the first feature
    add_sample_feature()

    # Try to add a duplicate
    response = client.post(
        "/api/v1/collections/polygons/items", json=SAMPLE_POLYGON_FEATURE
    )
    assert response.status_code == 409  # Conflict


def test_get_polygon_feature():
    """Test getting a specific polygon feature."""
    # Add a feature
    feature_id = add_sample_feature()

    # Get the feature
    response = client.get(f"/api/v1/collections/polygons/items/{feature_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == feature_id
    assert data["properties"]["name"] == "Test Polygon"


def test_get_nonexistent_polygon_feature():
    """Test getting a polygon feature that doesn't exist."""
    response = client.get("/api/v1/collections/polygons/items/nonexistent")
    assert response.status_code == 404


def test_update_polygon_feature():
    """Test updating a polygon feature."""
    # Add a feature
    feature_id = add_sample_feature()

    # Update the feature
    updated_feature = SAMPLE_POLYGON_FEATURE.copy()
    updated_feature["properties"] = {
        "name": "Updated Test Polygon",
        "description": "An updated polygon for testing",
    }

    response = client.put(
        f"/api/v1/collections/polygons/items/{feature_id}", json=updated_feature
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == feature_id
    assert data["properties"]["name"] == "Updated Test Polygon"


def test_update_nonexistent_polygon_feature():
    """Test updating a polygon feature that doesn't exist."""
    response = client.put(
        "/api/v1/collections/polygons/items/nonexistent", json=SAMPLE_POLYGON_FEATURE
    )
    assert response.status_code == 404


def test_delete_polygon_feature():
    """Test deleting a polygon feature."""
    # Add a feature
    feature_id = add_sample_feature()
    assert len(POLYGON_FEATURES) == 1

    # Delete the feature
    response = client.delete(f"/api/v1/collections/polygons/items/{feature_id}")
    assert response.status_code == 204
    assert len(POLYGON_FEATURES) == 0


def test_delete_nonexistent_polygon_feature():
    """Test deleting a polygon feature that doesn't exist."""
    response = client.delete("/api/v1/collections/polygons/items/nonexistent")
    assert response.status_code == 404


def test_list_polygon_features_with_bbox():
    """Test listing polygon features with a bounding box filter."""
    # Add a feature
    add_sample_feature()

    # List features with a bounding box that contains the feature
    response = client.get("/api/v1/collections/polygons/items?bbox=-123,37,-122,38")
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) == 1

    # List features with a bounding box that doesn't contain the feature
    response = client.get("/api/v1/collections/polygons/items?bbox=-120,30,-119,31")
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) == 0


def test_list_polygon_features_with_invalid_bbox():
    """Test listing polygon features with an invalid bounding box."""
    response = client.get("/api/v1/collections/polygons/items?bbox=invalid")
    assert response.status_code == 400


def test_bbox_filter_matches_polygon_containing_query():
    """A bbox inside a polygon must match without a vertex inside it."""
    feature = {
        "type": "Feature",
        "id": "large-polygon",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-130, 30],
                    [-110, 30],
                    [-110, 45],
                    [-130, 45],
                    [-130, 30],
                ]
            ],
        },
        "properties": {},
    }
    add_response = client.post("/api/v1/collections/polygons/items", json=feature)
    assert add_response.status_code == 201

    response = client.get("/api/v1/collections/polygons/items?bbox=-123,37,-122,38")
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["features"]] == ["large-polygon"]


def test_polygon_coordinates_must_be_in_wgs84_bounds():
    invalid = {
        **SAMPLE_POLYGON_FEATURE,
        "id": "out-of-range",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[200, 37], [201, 37], [201, 38], [200, 37]]],
        },
    }
    response = client.post("/api/v1/collections/polygons/items", json=invalid)
    assert response.status_code == 422


def test_calculate_polygon_area():
    """Test calculating the area of a polygon."""
    response = client.post(
        "/api/v1/operations/polygon/area", json=SAMPLE_POLYGON_FEATURE
    )
    assert response.status_code == 200
    data = response.json()
    assert "area_sq_km" in data
    assert isinstance(data["area_sq_km"], float)
    assert data["area_sq_km"] > 0


def test_simplify_polygon():
    """Test simplifying a polygon."""
    response = client.post(
        "/api/v1/operations/polygon/simplify?tolerance=0.1", json=SAMPLE_POLYGON_FEATURE
    )
    assert response.status_code == 200
    data = response.json()
    assert data["geometry"]["type"] == "Polygon"
    # The simplified polygon should still have at least 4 points (triangle + closing point)
    assert len(data["geometry"]["coordinates"][0]) >= 4


def test_check_polygon_contains_point():
    """Test checking if a polygon contains a point."""
    # Add a feature
    add_sample_feature()

    # Check if the polygon contains a point inside it
    response = client.post(
        "/api/v1/operations/polygon/contains?lon=-122.42&lat=37.78",
        json=SAMPLE_POLYGON_FEATURE,
    )
    assert response.status_code == 200
    data = response.json()
    assert "contains" in data

    # Check a point outside the polygon
    response = client.post(
        "/api/v1/operations/polygon/contains?lon=-123.0&lat=38.0",
        json=SAMPLE_POLYGON_FEATURE,
    )
    assert response.status_code == 200
    data = response.json()
    assert "contains" in data
    assert data["contains"] is False


def test_create_feature_with_integer_zero_id():
    """Regression: integer id 0 is a valid feature id and must be accepted."""
    response = client.post("/api/v1/collections/polygons/items", json=ZERO_ID_FEATURE)
    assert response.status_code == 201
    stored = client.get("/api/v1/collections/polygons/items/0")
    assert stored.status_code == 200
    assert stored.json()["id"] == 0


@pytest.mark.parametrize(
    "first_id, duplicate_id",
    [(0, 0), (1, 1), (0, "0"), ("0", 0), (1, "1"), ("1", 1), (-1, "-1")],
)
def test_duplicate_ids_use_same_identity_as_storage(first_id, duplicate_id):
    original = {**SAMPLE_POLYGON_FEATURE, "id": first_id}
    assert (
        client.post("/api/v1/collections/polygons/items", json=original).status_code
        == 201
    )
    duplicate = {**original, "id": duplicate_id, "properties": {"name": "overwrite"}}
    response = client.post("/api/v1/collections/polygons/items", json=duplicate)
    assert response.status_code == 409
    assert len(POLYGON_FEATURES) == 1
    stored = client.get(f"/api/v1/collections/polygons/items/{first_id}")
    assert stored.json() == original


@pytest.mark.parametrize("body_id", [0, "", "different"])
def test_update_rejects_falsy_mismatched_ids_without_mutating_store(body_id):
    original = {**SAMPLE_POLYGON_FEATURE, "id": "1"}
    assert (
        client.post("/api/v1/collections/polygons/items", json=original).status_code
        == 201
    )
    response = client.put(
        "/api/v1/collections/polygons/items/1", json={**original, "id": body_id}
    )
    assert response.status_code == 422
    assert client.get("/api/v1/collections/polygons/items/1").json() == original


def test_update_accepts_matching_integer_zero_id():
    assert (
        client.post(
            "/api/v1/collections/polygons/items", json=ZERO_ID_FEATURE
        ).status_code
        == 201
    )
    response = client.put("/api/v1/collections/polygons/items/0", json=ZERO_ID_FEATURE)
    assert response.status_code == 200 and response.json()["id"] == "0"


def test_buffer_preserves_integer_zero_feature_identity():
    response = client.post(
        "/api/v1/operations/polygon/buffer?distance=1", json=ZERO_ID_FEATURE
    )
    assert response.status_code == 200
    assert response.json()["id"] == "0_buffer"


def test_create_polygon_feature_rejects_beyond_store_cap(monkeypatch):
    """Regression: the in-memory polygon store must be bounded.

    Creating more features than ``polygon_store_max_size`` is rejected with
    HTTP 409 instead of the store growing without limit (GS-211).
    """
    from geo_infer_api.core.config import get_settings

    cap = 2
    monkeypatch.setattr(get_settings(), "polygon_store_max_size", cap)

    def feature_payload(fid):
        return {
            "type": "Feature",
            "id": fid,
            "geometry": {"type": "Polygon", "coordinates": SAMPLE_POLYGON_COORDS},
            "properties": {"name": f"cap-{fid}"},
        }

    for i in range(cap):
        response = client.post(
            "/api/v1/collections/polygons/items", json=feature_payload(f"cap-{i}")
        )
        assert response.status_code == 201

    overflow = client.post(
        "/api/v1/collections/polygons/items", json=feature_payload("cap-overflow")
    )
    assert overflow.status_code == 409
    assert len(POLYGON_FEATURES) == cap
