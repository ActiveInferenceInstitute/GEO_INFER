"""
Unit tests for GEO-INFER-TIME core functionality.
"""

import pandas as pd
import numpy as np


from geo_infer_time import __version__
from geo_infer_time.core.analysis import TemporalAnalyzer
from geo_infer_time.models.timeseries import TimeSeries


class TestTimeModule:
    """Test basic module functionality."""

    def test_module_import(self) -> None:
        """Test that the module can be imported."""
        import geo_infer_time

        assert geo_infer_time is not None

    def test_module_version(self) -> None:
        """Test that module has a version."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_temporal_analyzer_initialization(self) -> None:
        """Test TemporalAnalyzer initialization."""
        analyzer = TemporalAnalyzer()
        assert analyzer is not None

    def test_timeseries_creation(self) -> None:
        """Test TimeSeries creation."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D", tz="UTC")
        values = np.random.randn(100)
        ts = TimeSeries(values, dates)
        assert ts is not None
        assert len(ts) == 100

    def test_temporal_analysis_trend_detection(self) -> None:
        """Test trend detection functionality."""
        analyzer = TemporalAnalyzer()
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D", tz="UTC")
        # Create data with a clear trend
        values = np.linspace(0, 10, 100) + np.random.randn(100) * 0.1
        ts = TimeSeries(values, dates)

        result = analyzer.detect_trend(ts, method="linear")
        assert result is not None
        assert "trend_direction" in result
        assert "trend_values" in result
