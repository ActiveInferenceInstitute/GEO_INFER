"""
Unit tests for utils/data_processing.py.

Tests cover spatial data preparation, validation, grid creation,
sampling methods, and file format detection.
"""

import numpy as np
import pandas as pd
import pytest


from geo_infer_bayes.utils.data_processing import (
    prepare_spatial_data,
    validate_spatial_data,
    create_spatial_grid,
    sample_spatial_data,
    _detect_file_format,
)
from pathlib import Path


class TestPrepareSpatialData:
    def test_prepare_from_dataframe(self) -> None:
        df = pd.DataFrame(
            {
                "lat": [40.0, 41.0, 42.0],
                "lon": [-74.0, -73.0, -72.0],
                "value": [1.0, 2.0, 3.0],
            }
        )
        coords, values, temporal, metadata = prepare_spatial_data(df, value_col="value")
        assert coords.shape == (3, 2)
        assert values.shape == (3,)
        assert temporal is None
        assert metadata["n_samples"] == 3
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])

    def test_prepare_from_ndarray(self) -> None:
        arr = np.array(
            [
                [40.0, -74.0, 10.0],
                [41.0, -73.0, 20.0],
            ]
        )
        coords, values, temporal, metadata = prepare_spatial_data(arr)
        assert coords.shape == (2, 2)
        assert values.shape == (2,)

    def test_prepare_auto_selects_value_col(self) -> None:
        df = pd.DataFrame(
            {
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
                "temperature": [15.0, 20.0],
            }
        )
        coords, values, temporal, metadata = prepare_spatial_data(df)
        np.testing.assert_allclose(values, [15.0, 20.0])

    def test_prepare_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"x": [0], "y": [0]})
        with pytest.raises(ValueError, match="must be present"):
            prepare_spatial_data(df, lat_col="lat", lon_col="lon")

    def test_prepare_with_temporal(self) -> None:
        df = pd.DataFrame(
            {
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
                "value": [1.0, 2.0],
                "time": [0, 1],
            }
        )
        coords, values, temporal, metadata = prepare_spatial_data(
            df, value_col="value", time_col="time"
        )
        assert temporal is not None
        assert len(temporal) == 2

    def test_metadata_contains_bounds(self) -> None:
        df = pd.DataFrame(
            {
                "lat": [10.0, 20.0, 30.0],
                "lon": [50.0, 60.0, 70.0],
                "val": [1.0, 2.0, 3.0],
            }
        )
        _, _, _, metadata = prepare_spatial_data(df, value_col="val")
        bounds = metadata["spatial_bounds"]
        assert bounds["lat_min"] == 10.0
        assert bounds["lat_max"] == 30.0
        assert bounds["lon_min"] == 50.0
        assert bounds["lon_max"] == 70.0


class TestValidateSpatialData:
    def test_valid_data(self) -> None:
        coords = np.array([[40.0, -74.0], [41.0, -73.0]])
        values = np.array([1.0, 2.0])
        result = validate_spatial_data(coords, values)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_invalid_lat_range(self) -> None:
        coords = np.array([[100.0, -74.0]])
        values = np.array([1.0])
        result = validate_spatial_data(coords, values)
        assert result["is_valid"] is False
        assert any("Latitude" in e for e in result["errors"])

    def test_nan_warning(self) -> None:
        coords = np.array([[np.nan, -74.0]])
        values = np.array([1.0])
        result = validate_spatial_data(coords, values)
        assert any("NaN" in w for w in result["warnings"])

    def test_length_mismatch_error(self) -> None:
        coords = np.array([[40.0, -74.0], [41.0, -73.0]])
        values = np.array([1.0])
        result = validate_spatial_data(coords, values)
        assert result["is_valid"] is False

    def test_invalid_lon_range_is_error(self) -> None:
        """Out-of-range longitude must be an error, symmetric with latitude."""
        coords = np.array([[40.0, -200.0]])
        values = np.array([1.0])
        result = validate_spatial_data(coords, values)
        assert result["is_valid"] is False
        assert any("Longitude" in e for e in result["errors"])

    def test_duplicate_coordinates_warning(self) -> None:
        coords = np.array([[40.0, -74.0], [40.0, -74.0]])
        values = np.array([1.0, 2.0])
        result = validate_spatial_data(coords, values)
        assert any("duplicate" in w for w in result["warnings"])


class TestCreateSpatialGrid:
    def test_regular_grid(self) -> None:
        bounds = {
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        }
        grid, metadata = create_spatial_grid(bounds, resolution=0.5)
        assert grid.ndim == 2
        assert grid.shape[1] == 2
        assert metadata["grid_type"] == "regular"
        assert metadata["resolution"] == 0.5
        assert metadata["n_points"] == grid.shape[0]

    def test_unsupported_grid_type_raises(self) -> None:
        bounds = {
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        }
        with pytest.raises(ValueError, match="Unsupported grid type"):
            create_spatial_grid(bounds, grid_type="hexagonal")


class TestSampleSpatialData:
    def test_random_sampling(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.standard_normal((100, 2))
        values = rng.standard_normal(100)
        sampled_coords, sampled_values = sample_spatial_data(
            coords, values, n_samples=20, method="random"
        )
        assert sampled_coords.shape == (20, 2)
        assert sampled_values.shape == (20,)

    def test_systematic_sampling(self) -> None:
        coords = np.arange(100).reshape(50, 2).astype(float)
        values = np.arange(50, dtype=float)
        sampled_coords, sampled_values = sample_spatial_data(
            coords, values, n_samples=10, method="systematic"
        )
        assert sampled_coords.shape[0] == 10

    def test_request_too_many_samples_returns_all(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        values = np.array([1.0, 2.0])
        sc, sv = sample_spatial_data(coords, values, n_samples=100)
        assert len(sc) == 2

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported sampling"):
            sample_spatial_data(
                np.zeros((10, 2)), np.zeros(10), n_samples=5, method="magic"
            )


class TestDetectFileFormat:
    def test_csv(self) -> None:
        assert _detect_file_format(Path("data.csv")) == "csv"

    def test_json(self) -> None:
        assert _detect_file_format(Path("data.json")) == "json"

    def test_parquet(self) -> None:
        assert _detect_file_format(Path("data.parquet")) == "parquet"

    def test_unknown_defaults_to_csv(self) -> None:
        assert _detect_file_format(Path("data.xyz")) == "csv"
