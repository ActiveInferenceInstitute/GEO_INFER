"""
Tests for GeoJSON visualization capabilities.

Verifies export of GeoJSON data to formats usable for visualization.
"""

import json
import os
from tempfile import NamedTemporaryFile

from geo_infer_api.models.geojson import (
    GeoJSONType,
    Polygon,
    PolygonFeature,
    PolygonFeatureCollection,
)


# Test data
SAMPLE_POLYGONS = [
    {
        "id": "sf-triangle",
        "name": "San Francisco Triangle",
        "coordinates": [
            [
                [-122.51, 37.77],
                [-122.42, 37.81],
                [-122.37, 37.73],
                [-122.51, 37.77],
            ]
        ],
    },
    {
        "id": "nyc-square",
        "name": "New York Square",
        "coordinates": [
            [
                [-74.01, 40.70],
                [-73.96, 40.70],
                [-73.96, 40.75],
                [-74.01, 40.75],
                [-74.01, 40.70],
            ]
        ],
    },
]


def create_feature_collection() -> PolygonFeatureCollection:
    """Create a sample PolygonFeatureCollection for testing."""
    features = []
    for poly in SAMPLE_POLYGONS:
        polygon = Polygon(type=GeoJSONType.POLYGON, coordinates=poly["coordinates"])
        feature = PolygonFeature(
            type=GeoJSONType.FEATURE,
            id=poly["id"],
            geometry=polygon,
            properties={"name": poly["name"]},
        )
        features.append(feature)
    return PolygonFeatureCollection(
        type=GeoJSONType.FEATURE_COLLECTION,
        features=features,
    )


def test_feature_collection_to_geojson():
    """Test conversion of PolygonFeatureCollection to GeoJSON string."""
    fc = create_feature_collection()

    geojson_str = fc.model_dump_json(exclude_none=True)

    geojson_data = json.loads(geojson_str)
    assert geojson_data["type"] == "FeatureCollection"
    assert len(geojson_data["features"]) == 2
    assert geojson_data["features"][0]["id"] == "sf-triangle"
    assert geojson_data["features"][1]["id"] == "nyc-square"


def test_feature_collection_to_file():
    """Test writing PolygonFeatureCollection to a GeoJSON file."""
    fc = create_feature_collection()

    with NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(fc.model_dump_json(exclude_none=True).encode("utf-8"))

    try:
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        with open(tmp_path) as f:
            geojson_data = json.load(f)

        assert geojson_data["type"] == "FeatureCollection"
        assert len(geojson_data["features"]) == 2
        assert geojson_data["features"][0]["id"] == "sf-triangle"
        assert geojson_data["features"][1]["id"] == "nyc-square"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_feature_dict_conversion():
    """Test conversion between PolygonFeature and dictionary."""
    polygon = Polygon(
        type=GeoJSONType.POLYGON,
        coordinates=SAMPLE_POLYGONS[0]["coordinates"],
    )
    feature = PolygonFeature(
        type=GeoJSONType.FEATURE,
        id=SAMPLE_POLYGONS[0]["id"],
        geometry=polygon,
        properties={"name": SAMPLE_POLYGONS[0]["name"]},
    )

    feature_dict = feature.model_dump(exclude_none=True)

    assert feature_dict["type"] == "Feature"
    assert feature_dict["id"] == "sf-triangle"
    assert feature_dict["geometry"]["type"] == "Polygon"

    coords_from_dict = feature_dict["geometry"]["coordinates"]
    coords_from_sample = SAMPLE_POLYGONS[0]["coordinates"]
    assert len(coords_from_dict) == len(coords_from_sample)
    assert len(coords_from_dict[0]) == len(coords_from_sample[0])

    for i in range(len(coords_from_dict)):
        for j in range(len(coords_from_dict[i])):
            assert coords_from_dict[i][j][0] == coords_from_sample[i][j][0]
            assert coords_from_dict[i][j][1] == coords_from_sample[i][j][1]

    assert feature_dict["properties"]["name"] == "San Francisco Triangle"

    new_feature = PolygonFeature(**feature_dict)
    assert new_feature.type == feature.type
    assert new_feature.id == feature.id
    assert new_feature.geometry.type == feature.geometry.type

    for i in range(len(new_feature.geometry.coordinates)):
        for j in range(len(new_feature.geometry.coordinates[i])):
            assert (
                new_feature.geometry.coordinates[i][j][0]
                == feature.geometry.coordinates[i][j][0]
            )
            assert (
                new_feature.geometry.coordinates[i][j][1]
                == feature.geometry.coordinates[i][j][1]
            )

    assert new_feature.properties == feature.properties
