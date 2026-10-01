#!/usr/bin/env python3
"""
Example client for the GEO-INFER-API GeoJSON polygon endpoints.

This script demonstrates how to interact with the GEO-INFER-API
for working with GeoJSON polygon features.
"""

import uuid

import requests


# API base URL (change this to match your deployment)
API_BASE_URL = "http://localhost:8000/api/v1"


def list_collections():
    """List available feature collections."""
    response = requests.get(f"{API_BASE_URL}/collections")
    response.raise_for_status()
    return response.json()


def get_polygon_collection():
    """Get metadata about the polygon collection."""
    response = requests.get(f"{API_BASE_URL}/collections/polygons")
    response.raise_for_status()
    return response.json()


def list_polygon_features(bbox=None, limit=10):
    """
    List polygon features with optional filtering.

    Args:
        bbox: Optional bounding box in format "minLon,minLat,maxLon,maxLat"
        limit: Maximum number of features to return
    """
    params = {"limit": limit}
    if bbox:
        params["bbox"] = bbox

    response = requests.get(f"{API_BASE_URL}/collections/polygons/items", params=params)
    response.raise_for_status()
    return response.json()


def get_polygon_feature(feature_id):
    """
    Get a specific polygon feature by ID.

    Args:
        feature_id: ID of the feature to retrieve
    """
    response = requests.get(f"{API_BASE_URL}/collections/polygons/items/{feature_id}")
    response.raise_for_status()
    return response.json()


def create_polygon_feature(coords, properties=None, feature_id=None):
    """
    Create a new polygon feature.

    Args:
        coords: List of rings where each ring is a list of [lon, lat] coordinates
        properties: Optional properties dict to attach to the feature
        feature_id: Optional feature ID (if not provided, a UUID will be generated)
    """
    # Generate a feature ID if none provided
    if not feature_id:
        feature_id = str(uuid.uuid4())

    # Create the GeoJSON Feature
    feature = {
        "type": "Feature",
        "id": feature_id,
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": properties or {},
    }

    # Send the request
    response = requests.post(f"{API_BASE_URL}/collections/polygons/items", json=feature)
    response.raise_for_status()
    return response.json()


def update_polygon_feature(feature_id, coords, properties=None):
    """
    Update an existing polygon feature.

    Args:
        feature_id: ID of the feature to update
        coords: New coordinates for the polygon
        properties: New properties (or None to keep existing)
    """
    # Create the GeoJSON Feature
    feature = {
        "type": "Feature",
        "id": feature_id,
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": properties or {},
    }

    # Send the request
    response = requests.put(
        f"{API_BASE_URL}/collections/polygons/items/{feature_id}", json=feature
    )
    response.raise_for_status()
    return response.json()


def delete_polygon_feature(feature_id):
    """
    Delete a polygon feature.

    Args:
        feature_id: ID of the feature to delete
    """
    response = requests.delete(
        f"{API_BASE_URL}/collections/polygons/items/{feature_id}"
    )
    response.raise_for_status()
    return None


def calculate_polygon_area(feature):
    """
    Calculate the area of a polygon.

    Args:
        feature: GeoJSON Feature with Polygon geometry
    """
    response = requests.post(f"{API_BASE_URL}/operations/polygon/area", json=feature)
    response.raise_for_status()
    return response.json()


def simplify_polygon(feature, tolerance=0.01):
    """
    Simplify a polygon.

    Args:
        feature: GeoJSON Feature with Polygon geometry
        tolerance: Simplification tolerance
    """
    response = requests.post(
        f"{API_BASE_URL}/operations/polygon/simplify",
        json=feature,
        params={"tolerance": tolerance},
    )
    response.raise_for_status()
    return response.json()


def check_point_in_polygon(feature, lon, lat):
    """
    Check if a point is inside a polygon.

    Args:
        feature: GeoJSON Feature with Polygon geometry
        lon: Longitude of the point
        lat: Latitude of the point
    """
    response = requests.post(
        f"{API_BASE_URL}/operations/polygon/contains",
        json=feature,
        params={"lon": lon, "lat": lat},
    )
    response.raise_for_status()
    return response.json()


def main():
    """Run example client operations."""
    print("GEO-INFER-API Python Client Example")
    print("-" * 40)

    # Define a sample polygon (triangle around the San Francisco area)
    sf_polygon = [
        [
            [-122.51, 37.77],
            [-122.42, 37.81],
            [-122.37, 37.73],
            [-122.51, 37.77],  # Close the polygon
        ]
    ]

    # Define properties
    sf_properties = {
        "name": "San Francisco Triangle",
        "description": "A triangular area in San Francisco",
        "tags": ["example", "demo", "triangle"],
    }

    try:
        # List collections
        print("\n1. Listing collections...")
        collections = list_collections()
        print(f"Found {len(collections['collections'])} collections")

        # Create a polygon feature
        print("\n2. Creating a polygon feature...")
        feature_id = "sf-triangle-demo"
        created_feature = create_polygon_feature(sf_polygon, sf_properties, feature_id)
        print(f"Created feature with ID: {created_feature['id']}")

        # Get the feature
        print("\n3. Retrieving the feature...")
        retrieved_feature = get_polygon_feature(feature_id)
        print(f"Retrieved feature: {retrieved_feature['id']}")

        # Calculate area
        print("\n4. Calculating polygon area...")
        area_result = calculate_polygon_area(retrieved_feature)
        print(f"Area: {area_result['area_sq_km']:.2f} square kilometers")

        # Check if a point is inside
        print("\n5. Checking if a point is inside the polygon...")
        # Point in downtown San Francisco
        contains_result = check_point_in_polygon(retrieved_feature, -122.42, 37.78)
        print(f"Contains point: {contains_result['contains']}")

        # Simplify the polygon
        print("\n6. Simplifying the polygon...")
        simplified = simplify_polygon(retrieved_feature, tolerance=0.05)
        outer_ring = simplified["geometry"]["coordinates"][0]
        print(f"Simplified polygon outer ring: {len(outer_ring)} positions")

        # Update the feature
        print("\n7. Updating the feature...")
        updated_properties = sf_properties.copy()
        updated_properties["updated"] = True
        updated_feature = update_polygon_feature(
            feature_id, sf_polygon, updated_properties
        )
        print(f"Updated feature: {updated_feature['id']}")

        # List features
        print("\n8. Listing all polygon features...")
        features = list_polygon_features()
        print(f"Found {len(features['features'])} features")

        # Delete the feature
        print("\n9. Deleting the feature...")
        delete_polygon_feature(feature_id)
        print(f"Deleted feature with ID: {feature_id}")

        print("\nAll operations completed successfully!")

    except requests.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")


if __name__ == "__main__":
    main()
