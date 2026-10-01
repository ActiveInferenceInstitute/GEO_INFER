"""
Integration tests for GEO-INFER-SEC: anonymization and encryption pipeline end-to-end.

Tests GeospatialAnonymizer and GeospatialEncryption working together in a full
data protection pipeline using real geospatial data.
"""

import importlib.util

import pytest
import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    HAS_GEO_DEPS = True
except ImportError:
    HAS_GEO_DEPS = False

HAS_CRYPTO = importlib.util.find_spec("cryptography") is not None


pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def sample_geodataframe():
    """Create a GeoDataFrame with realistic point data."""
    np.random.seed(42)
    n_points = 20
    data = {
        "name": [f"Location_{i}" for i in range(n_points)],
        "population": np.random.randint(100, 10000, n_points),
        "income": np.random.uniform(30000, 150000, n_points).round(2),
        "geometry": [
            Point(
                -118.24 + np.random.uniform(-0.1, 0.1),
                34.05 + np.random.uniform(-0.1, 0.1),
            )
            for _ in range(n_points)
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def admin_boundaries():
    """Create administrative boundary polygons for geographic masking."""
    boundaries = gpd.GeoDataFrame(
        {
            "admin_id": ["zone_A", "zone_B"],
            "geometry": [
                Polygon(
                    [
                        (-118.35, 33.95),
                        (-118.35, 34.05),
                        (-118.24, 34.05),
                        (-118.24, 33.95),
                        (-118.35, 33.95),
                    ]
                ),
                Polygon(
                    [
                        (-118.24, 34.05),
                        (-118.24, 34.15),
                        (-118.13, 34.15),
                        (-118.13, 34.05),
                        (-118.24, 34.05),
                    ]
                ),
            ],
        },
        crs="EPSG:4326",
    )
    return boundaries


class TestAnonymizationPipeline:
    """Test the full anonymization pipeline end-to-end."""

    def test_location_perturbation_preserves_structure(self, sample_geodataframe):
        """Test that perturbation changes coordinates but preserves data structure."""
        from geo_infer_sec.core.anonymization import GeospatialAnonymizer

        anonymizer = GeospatialAnonymizer(seed=42)
        result = anonymizer.location_perturbation(sample_geodataframe, epsilon=500.0)

        # Structure preserved
        assert len(result) == len(sample_geodataframe)
        assert list(result.columns) == list(sample_geodataframe.columns)

        # Coordinates actually changed
        original_coords = [(p.x, p.y) for p in sample_geodataframe.geometry]
        perturbed_coords = [(p.x, p.y) for p in result.geometry]
        coords_changed = sum(
            1
            for o, p in zip(original_coords, perturbed_coords)
            if abs(o[0] - p[0]) > 1e-10 or abs(o[1] - p[1]) > 1e-10
        )
        assert coords_changed == len(sample_geodataframe), (
            "All coordinates should be perturbed"
        )

        # Perturbation within expected bounds (500m ~ 0.0045 degrees)
        max_displacement_deg = 500.0 / 111000.0  # meters to degrees approximation
        for orig, pert in zip(original_coords, perturbed_coords):
            dx = abs(orig[0] - pert[0])
            dy = abs(orig[1] - pert[1])
            assert dx < max_displacement_deg * 2, (
                f"X displacement {dx} exceeds expected range"
            )
            assert dy < max_displacement_deg * 2, (
                f"Y displacement {dy} exceeds expected range"
            )

    def test_perturbation_reproducibility_with_seed(self, sample_geodataframe):
        """Test that same seed produces identical perturbation."""
        from geo_infer_sec.core.anonymization import GeospatialAnonymizer

        result1 = GeospatialAnonymizer(seed=99).location_perturbation(
            sample_geodataframe, epsilon=200.0
        )
        result2 = GeospatialAnonymizer(seed=99).location_perturbation(
            sample_geodataframe, epsilon=200.0
        )

        for p1, p2 in zip(result1.geometry, result2.geometry):
            assert abs(p1.x - p2.x) < 1e-12
            assert abs(p1.y - p2.y) < 1e-12

    def test_spatial_k_anonymity_groups_points(self, sample_geodataframe):
        """Test that k-anonymity groups points into cells meeting the k threshold."""
        from geo_infer_sec.core.anonymization import GeospatialAnonymizer

        anonymizer = GeospatialAnonymizer(seed=42)
        k = 3
        result = anonymizer.spatial_k_anonymity(
            sample_geodataframe, k=k, h3_resolution=7
        )

        # All points should be snapped to cell centroids
        assert len(result) == len(sample_geodataframe)

        # Count unique locations (centroids) and verify at least k points per centroid
        coords = [(round(p.x, 8), round(p.y, 8)) for p in result.geometry]
        from collections import Counter

        coord_counts = Counter(coords)
        for coord, count in coord_counts.items():
            assert count >= k, f"Cell at {coord} has {count} points, below k={k}"

    def test_geographic_masking_aggregates_to_boundaries(
        self, sample_geodataframe, admin_boundaries
    ):
        """Test that geographic masking aggregates data to admin boundaries."""
        from geo_infer_sec.core.anonymization import GeospatialAnonymizer

        anonymizer = GeospatialAnonymizer(seed=42)
        result = anonymizer.geographic_masking(
            sample_geodataframe,
            attribute_cols=["population", "income"],
            admin_boundaries=admin_boundaries,
            admin_id_col="admin_id",
        )

        # Result should have same number of rows as admin boundaries
        assert len(result) == len(admin_boundaries)
        # Should contain aggregated attribute columns
        assert "population" in result.columns
        assert "income" in result.columns


class TestEncryptionPipeline:
    """Test encryption and decryption of geospatial data."""

    def test_text_encrypt_decrypt_roundtrip(self):
        """Test text encryption and decryption produce identical output."""
        from geo_infer_sec.core.encryption import GeospatialEncryption

        encryptor = GeospatialEncryption()
        original = "Sensitive location data: 34.0522, -118.2437"

        encrypted = encryptor.encrypt_text(original)
        assert encrypted != original.encode(), (
            "Encrypted data should differ from original"
        )

        decrypted = encryptor.decrypt_text(encrypted)
        assert decrypted == original, "Decrypted text should match original"

    def test_coordinate_encrypt_decrypt_roundtrip(self):
        """Test coordinate encryption preserves precision."""
        from geo_infer_sec.core.encryption import GeospatialEncryption

        encryptor = GeospatialEncryption()
        lat, lon = 34.052235, -118.243683

        encrypted = encryptor.encrypt_coordinates(lat, lon)
        assert isinstance(encrypted, str)

        dec_lat, dec_lon = encryptor.decrypt_coordinates(encrypted)
        assert abs(dec_lat - lat) < 1e-6
        assert abs(dec_lon - lon) < 1e-6

    def test_json_encrypt_decrypt_roundtrip(self):
        """Test JSON dict encryption and decryption."""
        from geo_infer_sec.core.encryption import GeospatialEncryption

        encryptor = GeospatialEncryption()
        data = {
            "location": "sensitive",
            "coordinates": [34.05, -118.24],
            "population": 3900000,
        }

        encrypted = encryptor.encrypt_json(data)
        decrypted = encryptor.decrypt_json(encrypted)
        assert decrypted == data

    def test_password_derived_key_consistency(self):
        """Test that same password + salt produces same key."""
        from geo_infer_sec.core.encryption import GeospatialEncryption
        import os

        salt = os.urandom(16)
        enc1 = GeospatialEncryption.from_password("test_password_123", salt=salt)
        enc2 = GeospatialEncryption.from_password("test_password_123", salt=salt)

        original = "secret geospatial data"
        encrypted = enc1.encrypt_text(original)
        decrypted = enc2.decrypt_text(encrypted)
        assert decrypted == original


class TestAnonymizationEncryptionPipeline:
    """Test anonymization followed by encryption in a combined pipeline."""

    def test_anonymize_then_encrypt_pipeline(self, sample_geodataframe):
        """Test full pipeline: anonymize locations, then encrypt sensitive columns."""
        from geo_infer_sec.core.anonymization import GeospatialAnonymizer
        from geo_infer_sec.core.encryption import GeospatialEncryption

        # Step 1: Anonymize locations
        anonymizer = GeospatialAnonymizer(seed=42)
        anonymized = anonymizer.location_perturbation(
            sample_geodataframe, epsilon=300.0
        )

        # Step 2: Encrypt sensitive attributes
        encryptor = GeospatialEncryption()
        encrypted = encryptor.encrypt_geodataframe(
            anonymized,
            sensitive_columns=["name", "income"],
            encrypt_coordinates=True,
        )

        # Verify sensitive columns are encrypted (not plaintext)
        assert encrypted["name"].iloc[0] != sample_geodataframe["name"].iloc[0]
        assert "encrypted_geometry" in encrypted.columns

        # Step 3: Decrypt and verify structure
        decrypted = encryptor.decrypt_geodataframe(
            encrypted,
            encrypted_columns=["name", "income"],
        )
        assert decrypted["name"].iloc[0] == anonymized["name"].iloc[0]
