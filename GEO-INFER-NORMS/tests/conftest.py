"""
Pytest fixtures for GEO-INFER-NORMS tests.

Provides normative rules, compliance data, norms configurations,
and standard spatial fixtures.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from typing import Any


@pytest.fixture(scope="session")
def sample_coordinates() -> list[tuple[float, float]]:
    """Standard (lat, lng) coordinate pairs for spatial tests."""
    return [
        (47.6062, -122.3321),
        (37.7749, -122.4194),
        (40.7128, -74.0060),
        (51.5074, -0.1278),
        (35.6762, 139.6503),
    ]


@pytest.fixture(scope="function")
def sample_geodataframe() -> gpd.GeoDataFrame:
    """Standard GeoDataFrame with EPSG:4326 for spatial tests."""
    return gpd.GeoDataFrame(
        {"id": range(5), "value": np.random.default_rng(0).uniform(0, 100, 5)},
        geometry=[Point(-122.33 + i * 0.01, 47.61 + i * 0.01) for i in range(5)],
        crs="EPSG:4326",
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def normative_rules() -> list[dict[str, Any]]:
    """List of normative rule dicts for compliance testing.

    Contains 5 rules spanning environmental, zoning, and safety
    domains with jurisdiction, enforcement level, and spatial
    applicability metadata.
    """
    return [
        {
            "rule_id": "ENV_001",
            "type": "environmental",
            "jurisdiction": "state",
            "description": "Riparian buffer zone minimum 30m from waterways",
            "enforcement": "mandatory",
            "buffer_m": 30,
        },
        {
            "rule_id": "ZON_001",
            "type": "zoning",
            "jurisdiction": "municipal",
            "description": "Maximum building height 12m in residential zones",
            "enforcement": "mandatory",
            "max_height_m": 12,
        },
        {
            "rule_id": "SAF_001",
            "type": "safety",
            "jurisdiction": "federal",
            "description": "No construction within 100m of fault lines",
            "enforcement": "mandatory",
            "exclusion_zone_m": 100,
        },
        {
            "rule_id": "ENV_002",
            "type": "environmental",
            "jurisdiction": "regional",
            "description": "Impervious surface limit 35% in watershed protection zones",
            "enforcement": "guideline",
            "max_impervious_pct": 35,
        },
        {
            "rule_id": "ZON_002",
            "type": "zoning",
            "jurisdiction": "municipal",
            "description": "Commercial activity permitted only in designated zones",
            "enforcement": "mandatory",
            "permitted_zones": ["C1", "C2", "MU"],
        },
    ]


@pytest.fixture
def compliance_data() -> dict[str, Any]:
    """Compliance assessment results for a set of parcels.

    Contains parcel-level compliance status against multiple rules,
    including violation details and remediation timelines.
    """
    return {
        "assessment_date": "2024-06-15",
        "total_parcels": 120,
        "compliant": 95,
        "non_compliant": 20,
        "under_review": 5,
        "violations": [
            {
                "parcel_id": "P_042",
                "rule_id": "ENV_001",
                "severity": "major",
                "remediation_deadline": "2024-12-31",
            },
            {
                "parcel_id": "P_078",
                "rule_id": "ZON_001",
                "severity": "minor",
                "remediation_deadline": "2025-03-01",
            },
        ],
    }


@pytest.fixture
def norms_config() -> dict[str, Any]:
    """Configuration dict for normative analysis operations.

    Specifies rule evaluation parameters, spatial overlay methods,
    and reporting formats.
    """
    return {
        "evaluation_method": "spatial_overlay",
        "buffer_precision_m": 1.0,
        "crs": "EPSG:4326",
        "conflict_resolution": "stricter_rule_wins",
        "reporting": {
            "format": "json",
            "include_geometry": True,
            "include_remediation": True,
        },
    }
