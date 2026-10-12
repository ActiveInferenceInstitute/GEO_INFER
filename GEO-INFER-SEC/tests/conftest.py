"""
Pytest fixtures for GEO-INFER-SEC tests.

Provides sample credentials, security configurations, audit log
entries, and standard spatial fixtures.
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
        {"id": range(5), "value": np.random.uniform(0, 100, 5)},
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
def sample_credentials() -> dict[str, str]:
    """Test credential tokens for authentication tests.

    WARNING: These are synthetic test-only tokens. They are NOT real
    credentials and must never be used in production. They follow the
    expected token format for validation tests.
    """
    return {
        "api_key": "test_key_abc123def456ghi789",
        "jwt_token": (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiJ0ZXN0X3VzZXIiLCJpYXQiOjE3MDAwMDAwMDB9."
            "fake_signature_for_testing_only"
        ),
        "refresh_token": "test_refresh_token_xyz987wvu654",
        "client_id": "test_client_geo_infer",
        "client_secret": "test_secret_not_real_s3cr3t",
    }


@pytest.fixture
def security_config() -> dict[str, Any]:
    """Configuration dict for security module settings.

    Specifies authentication, authorization, encryption, and rate
    limiting parameters for security policy tests.
    """
    return {
        "authentication": {
            "method": "jwt",
            "token_expiry_seconds": 3600,
            "refresh_enabled": True,
            "max_refresh_count": 10,
        },
        "authorization": {
            "model": "rbac",
            "roles": ["admin", "analyst", "viewer"],
            "default_role": "viewer",
        },
        "encryption": {
            "algorithm": "AES-256-GCM",
            "key_rotation_days": 90,
            "at_rest": True,
            "in_transit": True,
        },
        "rate_limiting": {
            "requests_per_minute": 60,
            "burst_limit": 10,
            "window_seconds": 60,
        },
    }


@pytest.fixture
def audit_log_entries() -> list[dict[str, Any]]:
    """List of audit log entry dicts for security logging tests.

    Contains 6 entries representing login, data access, permission
    change, and failed authentication events with timestamps and
    actor metadata.
    """
    return [
        {
            "event_id": "EVT_001",
            "timestamp": "2024-06-15T08:00:00Z",
            "event_type": "login",
            "actor": "user_alice",
            "ip_address": "192.168.1.100",
            "success": True,
            "details": "Standard login via SSO",
        },
        {
            "event_id": "EVT_002",
            "timestamp": "2024-06-15T08:05:00Z",
            "event_type": "data_access",
            "actor": "user_alice",
            "ip_address": "192.168.1.100",
            "success": True,
            "details": "Read access to dataset DS_042",
        },
        {
            "event_id": "EVT_003",
            "timestamp": "2024-06-15T09:00:00Z",
            "event_type": "login_failed",
            "actor": "unknown",
            "ip_address": "10.0.0.55",
            "success": False,
            "details": "Invalid credentials, attempt 3 of 5",
        },
        {
            "event_id": "EVT_004",
            "timestamp": "2024-06-15T10:30:00Z",
            "event_type": "permission_change",
            "actor": "admin_bob",
            "ip_address": "192.168.1.50",
            "success": True,
            "details": "Elevated user_charlie to analyst role",
        },
        {
            "event_id": "EVT_005",
            "timestamp": "2024-06-15T11:00:00Z",
            "event_type": "data_export",
            "actor": "user_alice",
            "ip_address": "192.168.1.100",
            "success": True,
            "details": "Exported 500 records from spatial dataset",
        },
        {
            "event_id": "EVT_006",
            "timestamp": "2024-06-15T12:00:00Z",
            "event_type": "login_failed",
            "actor": "unknown",
            "ip_address": "10.0.0.55",
            "success": False,
            "details": "Account locked after 5 failed attempts",
        },
    ]
