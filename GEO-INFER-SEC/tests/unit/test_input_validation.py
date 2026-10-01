"""Tests for SecurityUtils input normalisation and validation helpers."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geo_infer_sec.utils.security_utils import (
    SecurityUtils,
    hash_password_simple,
    validate_spatial_bounds,
    verify_password_simple,
)


@pytest.fixture
def utils() -> SecurityUtils:
    return SecurityUtils()


class TestStripDangerousChars:
    """``strip_dangerous_chars`` removes the configured character set only."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("'; DROP TABLE users; --", " DROP TABLE users --"),
            ("<script>alert('xss')</script>", "scriptalertxss/script"),
            ("a | b && `c` $(d)", "a  b  c d"),
            ("plain text 123", "plain text 123"),
        ],
    )
    def test_strips_characters(
        self, utils: SecurityUtils, raw: str, expected: str
    ) -> None:
        assert utils.strip_dangerous_chars(raw) == expected

    def test_deprecated_sanitize_input_alias_removed(self) -> None:
        assert not hasattr(SecurityUtils, "sanitize_input")


class TestFileUploadValidation:
    """``validate_file_upload`` enforces extension and size limits."""

    def test_allowed_extension_within_size(
        self, utils: SecurityUtils, tmp_path: Path
    ) -> None:
        upload = tmp_path / "data.geojson"
        upload.write_text("{}", encoding="utf-8")
        assert utils.validate_file_upload(str(upload), [".geojson"]) == (
            True,
            "File upload valid",
        )

    def test_rejects_extension(self, utils: SecurityUtils, tmp_path: Path) -> None:
        upload = tmp_path / "payload.exe"
        upload.write_bytes(b"MZ")
        valid, message = utils.validate_file_upload(str(upload), [".geojson"])
        assert valid is False
        assert ".exe" in message

    def test_rejects_oversized_file(self, utils: SecurityUtils, tmp_path: Path) -> None:
        upload = tmp_path / "big.csv"
        upload.write_bytes(b"0" * (2 * 1024 * 1024))
        valid, message = utils.validate_file_upload(
            str(upload), [".csv"], max_size_mb=1
        )
        assert valid is False
        assert "exceeds maximum 1MB" in message


class TestSpatialBounds:
    """``validate_spatial_bounds`` accepts only WGS84-range geometries."""

    def test_valid_bounds(self) -> None:
        gdf = gpd.GeoDataFrame(
            geometry=[Point(-180, -90), Point(180, 90)], crs="EPSG:4326"
        )
        assert validate_spatial_bounds(gdf) is True

    @pytest.mark.parametrize("point", [Point(0, 91), Point(181, 0), Point(-200, 0)])
    def test_out_of_range_bounds(self, point: Point) -> None:
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), point])
        assert validate_spatial_bounds(gdf) is False

    def test_empty_or_non_geodataframe(self) -> None:
        assert validate_spatial_bounds(gpd.GeoDataFrame(geometry=[])) is False
        assert validate_spatial_bounds([Point(0, 0)]) is False


class TestSimplePasswordHashing:
    """``hash_password_simple``/``verify_password_simple`` round-trip."""

    def test_round_trip(self) -> None:
        stored = hash_password_simple("SecureP@ss1")
        assert verify_password_simple("SecureP@ss1", stored) is True
        assert verify_password_simple("wrong", stored) is False

    def test_malformed_hash_is_rejected(self) -> None:
        assert verify_password_simple("anything", "not-base64!") is False
