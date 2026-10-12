"""Native H3 v4 conformance tests for the ACT adapter."""

import h3
import pytest

from geo_infer_act.utils.h3_adapter import get_h3_adapter
from geo_infer_act.utils.integration import create_h3_spatial_model


def test_adapter_enforces_the_supported_h3_version() -> None:
    adapter = get_h3_adapter(prefer_space=False)
    assert adapter.h3.__version__ == "4.5.0"


def test_direct_adapter_preserves_geojson_holes_and_multipolygons() -> None:
    adapter = get_h3_adapter(prefer_space=False)
    polygon_with_hole = {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.50, 37.70],
                [-122.30, 37.70],
                [-122.30, 37.80],
                [-122.50, 37.80],
                [-122.50, 37.70],
            ],
            [
                [-122.46, 37.74],
                [-122.34, 37.74],
                [-122.34, 37.76],
                [-122.46, 37.76],
                [-122.46, 37.74],
            ],
        ],
    }
    multipolygon = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [-122.50, 37.70],
                    [-122.45, 37.70],
                    [-122.45, 37.75],
                    [-122.50, 37.75],
                    [-122.50, 37.70],
                ]
            ],
            [
                [
                    [-122.35, 37.75],
                    [-122.30, 37.75],
                    [-122.30, 37.80],
                    [-122.35, 37.80],
                    [-122.35, 37.75],
                ]
            ],
        ],
    }

    assert set(adapter.polygon_to_cells(polygon_with_hole, 8)) == set(
        h3.geo_to_cells(polygon_with_hole, 8)
    )
    assert set(adapter.polygon_to_cells(multipolygon, 8)) == set(
        h3.geo_to_cells(multipolygon, 8)
    )


def test_adapter_grid_ring_is_exact_and_pentagon_safe() -> None:
    adapter = get_h3_adapter(prefer_space=False)
    pentagon = h3.get_pentagons(0)[0]

    assert set(adapter.grid_ring(pentagon, 1)) == set(h3.grid_disk(pentagon, 1)) - {
        pentagon
    }


def test_h3_spatial_model_accepts_feature_boundaries() -> None:
    result = create_h3_spatial_model(
        {},
        8,
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.50, 37.70],
                        [-122.30, 37.70],
                        [-122.30, 37.80],
                        [-122.50, 37.80],
                        [-122.50, 37.70],
                    ]
                ],
            },
        },
    )

    assert result["status"] == "success"
    assert result["model_config"]["boundary_cells"]


@pytest.mark.parametrize(
    "boundary",
    [
        {"type": "Point", "coordinates": [-122.4, 37.7]},
        {"type": "Polygon", "coordinates": []},
        {"type": "Polygon", "coordinates": [[[]]]},
        # Untyped coordinate-only mappings are rejected, not coerced to Polygon.
        {
            "coordinates": [
                [
                    [-122.50, 37.70],
                    [-122.30, 37.70],
                    [-122.30, 37.80],
                    [-122.50, 37.70],
                ]
            ]
        },
    ],
)
def test_h3_spatial_model_rejects_invalid_boundaries(boundary) -> None:
    result = create_h3_spatial_model({}, 8, boundary)

    assert result["status"] == "error"
    assert "boundary" in result["message"].lower()
