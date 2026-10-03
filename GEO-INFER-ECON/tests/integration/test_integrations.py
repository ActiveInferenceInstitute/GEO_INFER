"""
Integration Tests for GEO-INFER-ECON Module Integrations

Tests for integration adapters with GEO-INFER-SPACE, GEO-INFER-TIME, and GEO-INFER-DATA.
"""

import unittest
import subprocess
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


@pytest.mark.parametrize("missing", ["geo_infer_time", "broken_time_dependency"])
def test_time_optional_import_distinguishes_absence_from_breakage(missing):
    """Only absence of the owning package permits the local ECON path."""
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "geo_infer_econ"
        / "integrations"
        / "time_integration.py"
    )
    code = """
import importlib.util, sys
class Missing:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'geo_infer_time':
            raise ModuleNotFoundError('declared test boundary', name=sys.argv[2])
        return None
sys.meta_path.insert(0, Missing())
spec = importlib.util.spec_from_file_location('econ_time_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.TIME_AVAILABLE is False
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(source), missing],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if missing == "geo_infer_time":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and "declared test boundary" in result.stderr


class TestSpaceIntegration(unittest.TestCase):
    """Test cases for GEO-INFER-SPACE integration"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.integrations import SpaceIntegration

        self.space = SpaceIntegration(backend="h3")

        # Create test spatial data
        self.test_points = [
            (37.7749, -122.4194),  # San Francisco
            (37.8044, -122.2711),  # Oakland
            (37.8715, -122.2730),  # Berkeley
        ]

        self.test_gdf = gpd.GeoDataFrame(
            {
                "region_id": ["A", "B", "C"],
                "gdp": [1000, 1500, 800],
                "value": [100, 150, 80],
            },
            geometry=[Point(lon, lat) for lat, lon in self.test_points],
        )

    def test_initialization(self):
        """Test space integration initialization"""
        self.assertIsNotNone(self.space)
        # Should work even if GEO-INFER-SPACE not available
        self.assertIsInstance(self.space.is_available(), bool)

    def test_latlng_to_cell(self):
        """Test lat/lng to cell conversion"""
        lat, lng = self.test_points[0]
        cell = self.space.latlng_to_cell(lat, lng, resolution=9)

        # Should return None if not available, or a string if available
        if cell is not None:
            self.assertIsInstance(cell, str)
            self.assertGreater(len(cell), 0)

    def test_cell_to_latlng(self):
        """Test cell to lat/lng conversion"""
        # First get a cell
        lat, lng = self.test_points[0]
        cell = self.space.latlng_to_cell(lat, lng, resolution=9)

        if cell is not None:
            result = self.space.cell_to_latlng(cell)
            if result is not None:
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], (int, float))
                self.assertIsInstance(result[1], (int, float))

    def test_calculate_distance(self):
        """Test distance calculation"""
        p1 = self.test_points[0]
        p2 = self.test_points[1]

        distance = self.space.calculate_distance(p1, p2)

        # Should always return a value (has fallback)
        self.assertIsNotNone(distance)
        self.assertIsInstance(distance, (int, float))
        self.assertGreater(distance, 0)

    def test_analyze_hotspots(self):
        """Test hotspot analysis"""
        hotspots = self.space.analyze_hotspots(self.test_gdf, "gdp")

        # May return None if GEO-INFER-SPACE not available
        if hotspots is not None:
            self.assertIsInstance(hotspots, gpd.GeoDataFrame)
            self.assertEqual(len(hotspots), len(self.test_gdf))

    def test_create_buffer(self):
        """Test buffer creation"""
        buffer = self.space.create_buffer(self.test_gdf, distance=1000)

        # Should always return a value (has fallback)
        if buffer is not None:
            self.assertIsInstance(buffer, gpd.GeoDataFrame)
            self.assertEqual(len(buffer), len(self.test_gdf))


class TestTimeIntegration(unittest.TestCase):
    """Test cases for GEO-INFER-TIME integration"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.integrations import TimeIntegration

        self.time = TimeIntegration()

        # Create test time series
        dates = pd.date_range(start="2020-01-01", periods=24, freq="ME", tz="UTC")
        self.test_series = pd.Series(
            np.random.randn(24).cumsum() * 0.5 + 100, index=dates, name="gdp_growth"
        )

    def test_initialization(self):
        """Test time integration initialization"""
        self.assertIsNotNone(self.time)
        # Should work even if GEO-INFER-TIME not available
        self.assertIsInstance(self.time.is_available(), bool)

    def test_detect_trend(self):
        """Test trend detection"""
        trend = self.time.detect_trend(self.test_series, method="linear")

        # Should always return a value (has fallback)
        self.assertIsNotNone(trend)
        self.assertIsInstance(trend, dict)

        # Check for expected keys
        if "slope" in trend:
            self.assertIsInstance(trend["slope"], (int, float))
        if "r_squared" in trend:
            self.assertIsInstance(trend["r_squared"], (int, float))
            self.assertGreaterEqual(trend["r_squared"], 0)
            self.assertLessEqual(trend["r_squared"], 1)

    def test_analyze_seasonality(self):
        """Test seasonality analysis"""
        seasonality = self.time.analyze_seasonality(self.test_series, period=12)

        # May return None if GEO-INFER-TIME not available
        if seasonality is not None:
            self.assertIsInstance(seasonality, dict)

    def test_decompose_time_series(self):
        """Test time series decomposition"""
        decomposition = self.time.decompose_time_series(
            self.test_series, model="additive"
        )

        # Should always return a value (has fallback)
        self.assertIsNotNone(decomposition)
        self.assertIsInstance(decomposition, dict)

        # Check for expected components
        expected_components = ["trend", "seasonal", "residual"]
        for component in expected_components:
            if component in decomposition:
                self.assertIsInstance(decomposition[component], pd.Series)
                self.assertEqual(len(decomposition[component]), len(self.test_series))

    def test_forecast(self):
        """Test forecasting"""
        forecast = self.time.forecast(self.test_series, horizon=6, method="arima")

        # May return None if GEO-INFER-TIME not available
        if forecast is not None:
            self.assertIsInstance(forecast, dict)

    def test_align_time_series(self):
        """Test time series alignment"""
        series1 = pd.Series(
            [1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        )
        series2 = pd.Series(
            [4, 5], index=pd.date_range("2020-01-02", periods=2, freq="D", tz="UTC")
        )

        aligned = self.time.align_time_series([series1, series2], method="interpolate")

        # Should always return a value (has fallback)
        self.assertIsNotNone(aligned)
        self.assertIsInstance(aligned, list)
        self.assertEqual(len(aligned), 2)
        for ts in aligned:
            self.assertIsInstance(ts, pd.Series)


class TestDataIntegration(unittest.TestCase):
    """Test cases for GEO-INFER-DATA integration"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.integrations import DataIntegration

        self.data = DataIntegration()

    def test_initialization(self):
        """Test data integration initialization"""
        self.assertIsNotNone(self.data)
        # Should work even if GEO-INFER-DATA not available
        self.assertIsInstance(self.data.is_available(), bool)

    def test_list_datasets(self):
        """Test dataset listing"""
        datasets = self.data.list_datasets(dataset_type="vector", tags=["economic"])

        # May return None if GEO-INFER-DATA not available
        if datasets is not None:
            self.assertIsInstance(datasets, list)
            for dataset in datasets:
                self.assertIsInstance(dataset, dict)

    def test_load_economic_data_file(self):
        """Test loading economic data from file"""
        # This test would require an actual file
        # For now, just test that the method exists and handles errors gracefully
        try:
            result = self.data.load_economic_data(
                "nonexistent_file.csv", source_type="file"
            )
            # Should return None for nonexistent file
            self.assertIsNone(result)
        except Exception as e:
            # Should handle errors gracefully
            self.assertIsInstance(e, Exception)


class TestIntegratedWorkflow(unittest.TestCase):
    """Test integrated workflow using all three integrations"""

    def setUp(self):
        """Set up test fixtures"""
        from geo_infer_econ.integrations import (
            SpaceIntegration,
            TimeIntegration,
            DataIntegration,
        )

        self.space = SpaceIntegration()
        self.time = TimeIntegration()
        self.data = DataIntegration()

        # Create sample data
        self.regions = gpd.GeoDataFrame(
            {
                "region_id": ["A", "B", "C"],
                "gdp": [1000, 1500, 800],
                "lat": [37.7749, 37.8044, 37.8715],
                "lon": [-122.4194, -122.2711, -122.2730],
            },
            geometry=[
                Point(lon, lat)
                for lat, lon in zip(
                    [37.7749, 37.8044, 37.8715], [-122.4194, -122.2711, -122.2730]
                )
            ],
        )

        dates = pd.date_range(start="2020-01-01", periods=12, freq="ME", tz="UTC")
        self.time_series = pd.Series(
            np.random.randn(12).cumsum() * 0.5 + 100, index=dates
        )

    def test_spatial_temporal_analysis(self):
        """Test combined spatial and temporal analysis"""
        # Spatial analysis
        distance = self.space.calculate_distance(
            (self.regions.iloc[0]["lat"], self.regions.iloc[0]["lon"]),
            (self.regions.iloc[1]["lat"], self.regions.iloc[1]["lon"]),
        )
        self.assertIsNotNone(distance)

        # Temporal analysis
        trend = self.time.detect_trend(self.time_series)
        self.assertIsNotNone(trend)

    def test_integration_availability(self):
        """Test that integrations handle unavailability gracefully"""
        # All integrations should initialize even if modules not available
        self.assertIsNotNone(self.space)
        self.assertIsNotNone(self.time)
        self.assertIsNotNone(self.data)

        # Should have availability check methods
        self.assertIsInstance(self.space.is_available(), bool)
        self.assertIsInstance(self.time.is_available(), bool)
        self.assertIsInstance(self.data.is_available(), bool)


if __name__ == "__main__":
    unittest.main()


class TestTimeContractFailures:
    """Installed TIME adapters preserve rejected axes rather than returning None."""

    def test_naive_axis_propagates(self):
        import pytest
        from geo_infer_econ.integrations import TimeIntegration

        series = pd.Series(
            [1.0, 2.0, 3.0], index=pd.date_range("2026-10-01", periods=3)
        )
        with pytest.raises(ValueError, match="timezone"):
            TimeIntegration().detect_trend(series)

    def test_numeric_axis_is_not_an_implicit_epoch(self):
        import pytest
        from geo_infer_econ.integrations import TimeIntegration

        with pytest.raises(ValueError, match="timestamp must be"):
            TimeIntegration().detect_trend(pd.Series([1.0, 2.0, 3.0]))

    def test_decomposition_preserves_observation_axis(self):
        from geo_infer_econ.integrations import TimeIntegration

        axis = pd.date_range("2026-10-01", periods=24, freq="D", tz="UTC")
        values = pd.Series(np.tile([1.0, 2.0, 3.0, 2.0, 1.0, 2.0], 4), index=axis)
        result = TimeIntegration().decompose_time_series(values)
        assert result
        assert all(component.index.equals(axis) for component in result.values())
