"""
Tests for Temporal Visualization Module.

Tests for time series visualization methods including plots for
decomposition, forecasts, anomalies, and dashboards.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from geo_infer_time.core.visualization import TemporalVisualization


@pytest.fixture
def viz():
    """Create a TemporalVisualization instance."""
    return TemporalVisualization()


@pytest.fixture
def sample_values():
    """Generate sample time series values."""
    np.random.seed(42)
    n = 100
    trend = np.arange(n) * 0.5
    seasonal = 10 * np.sin(np.arange(n) * 2 * np.pi / 12)
    noise = np.random.randn(n) * 3
    return list(trend + seasonal + noise + 50)


@pytest.fixture
def sample_timestamps():
    """Generate sample timestamps."""
    return list(pd.date_range(start="2024-01-01", periods=100, freq="D", tz="UTC"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for saving plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPlotTimeseries:
    """Tests for basic time series plotting."""

    def test_plot_basic(self, viz, sample_values):
        """Test basic time series plot."""
        fig = viz.plot_timeseries(sample_values)

        assert fig is not None
        # Close figure to prevent memory issues
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_timestamps(self, viz, sample_values, sample_timestamps):
        """Test plot with timestamps."""
        fig = viz.plot_timeseries(sample_values, timestamps=sample_timestamps)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_customization(self, viz, sample_values):
        """Test plot with custom options."""
        fig = viz.plot_timeseries(
            sample_values,
            title="Custom Title",
            ylabel="Custom Y",
            xlabel="Custom X",
            color="#FF5733",
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_save(self, viz, sample_values, temp_dir):
        """Test saving plot to file."""
        save_path = temp_dir / "timeseries.png"
        fig = viz.plot_timeseries(sample_values, save_path=save_path)

        assert save_path.exists()
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotDecomposition:
    """Tests for decomposition plotting."""

    def test_decomposition_basic(self, viz, sample_values):
        """Test basic decomposition plot."""
        n = len(sample_values)
        trend = list(np.linspace(50, 100, n))
        seasonal = list(10 * np.sin(np.arange(n) * 2 * np.pi / 12))
        residual = list(np.random.randn(n) * 3)

        fig = viz.plot_decomposition(trend, seasonal, residual)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_decomposition_with_original(self, viz, sample_values):
        """Test decomposition plot with original series."""
        n = len(sample_values)
        trend = list(np.linspace(50, 100, n))
        seasonal = list(10 * np.sin(np.arange(n) * 2 * np.pi / 12))
        residual = list(np.random.randn(n) * 3)

        fig = viz.plot_decomposition(trend, seasonal, residual, original=sample_values)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotForecast:
    """Tests for forecast plotting."""

    def test_forecast_basic(self, viz, sample_values):
        """Test basic forecast plot."""
        forecast = list(np.linspace(sample_values[-1], sample_values[-1] + 20, 20))

        fig = viz.plot_forecast(sample_values, forecast)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_forecast_with_confidence(self, viz, sample_values):
        """Test forecast plot with confidence intervals."""
        forecast = list(np.linspace(sample_values[-1], sample_values[-1] + 20, 20))
        lower = [f - 5 for f in forecast]
        upper = [f + 5 for f in forecast]

        fig = viz.plot_forecast(
            sample_values, forecast, confidence_lower=lower, confidence_upper=upper
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_forecast_validates_confidence_bounds(self, viz, sample_values):
        forecast = list(np.linspace(sample_values[-1], sample_values[-1] + 20, 20))
        with pytest.raises(ValueError, match="provided together"):
            viz.plot_forecast(sample_values, forecast, confidence_lower=[0.0] * 20)
        with pytest.raises(ValueError, match="match forecast length"):
            viz.plot_forecast(
                sample_values, forecast, confidence_lower=[0.0], confidence_upper=[1.0]
            )


class TestPlotAcfPacf:
    """Tests for ACF/PACF plotting."""

    def test_acf_only(self, viz):
        """Test ACF-only plot."""
        acf_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02]

        fig = viz.plot_acf_pacf(acf_values)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_acf_pacf_both(self, viz):
        """Test ACF and PACF plot."""
        acf_values = [1.0, 0.8, 0.6, 0.4, 0.2]
        pacf_values = [1.0, 0.7, 0.1, -0.1, -0.2]

        fig = viz.plot_acf_pacf(acf_values, pacf_values, confidence_bound=0.2)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotAnomalies:
    """Tests for anomaly plotting."""

    def test_anomalies_basic(self, viz, sample_values):
        """Test basic anomaly plot."""
        anomaly_indices = [10, 25, 50, 75]

        fig = viz.plot_anomalies(sample_values, anomaly_indices)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_anomalies_none(self, viz, sample_values):
        """Test anomaly plot with no anomalies."""
        fig = viz.plot_anomalies(sample_values, [])

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_anomalies_reject_out_of_range_indices(self, viz, sample_values):
        with pytest.raises(ValueError, match="valid integer"):
            viz.plot_anomalies(sample_values, [len(sample_values)])


class TestPlotRollingStatistics:
    """Tests for rolling statistics plotting."""

    def test_rolling_mean_only(self, viz, sample_values):
        """Test rolling mean plot."""
        window = 10
        rolling_mean = list(
            pd.Series(sample_values).rolling(window=window).mean().dropna()
        )

        fig = viz.plot_rolling_statistics(sample_values, rolling_mean, window=window)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_rolling_with_std(self, viz, sample_values):
        """Test rolling plot with std bands."""
        window = 10
        series = pd.Series(sample_values)
        rolling_mean = list(series.rolling(window=window).mean().dropna())
        rolling_std = list(series.rolling(window=window).std().dropna())

        fig = viz.plot_rolling_statistics(
            sample_values, rolling_mean, rolling_std=rolling_std, window=window
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotSeasonality:
    """Tests for seasonality plotting."""

    def test_seasonality_basic(self, viz, sample_values):
        """Test basic seasonality plot."""
        fig = viz.plot_seasonality(sample_values, period=12)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_seasonality_weekly(self, viz, sample_values):
        """Test weekly seasonality plot."""
        fig = viz.plot_seasonality(sample_values, period=7)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestCreateDashboard:
    """Tests for dashboard creation."""

    def test_dashboard_basic(self, viz, sample_values):
        """Test basic dashboard creation."""
        fig = viz.create_dashboard(sample_values)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dashboard_with_decomposition(self, viz, sample_values):
        """Test dashboard with decomposition."""
        n = len(sample_values)
        decomposition = {
            "trend": list(np.linspace(50, 100, n)),
            "seasonal": list(10 * np.sin(np.arange(n) * 2 * np.pi / 12)),
            "residual": list(np.random.randn(n) * 3),
        }

        fig = viz.create_dashboard(sample_values, decomposition=decomposition)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dashboard_with_anomalies(self, viz, sample_values):
        """Test dashboard with anomalies."""
        anomalies = [10, 25, 50, 75]

        fig = viz.create_dashboard(sample_values, anomalies=anomalies)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dashboard_with_forecast(self, viz, sample_values):
        """Test dashboard with forecast."""
        forecast = {
            "values": list(np.linspace(sample_values[-1], sample_values[-1] + 20, 10)),
            "lower": list(
                np.linspace(sample_values[-1] - 5, sample_values[-1] + 15, 10)
            ),
            "upper": list(
                np.linspace(sample_values[-1] + 5, sample_values[-1] + 25, 10)
            ),
        }

        fig = viz.create_dashboard(sample_values, forecast=forecast)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dashboard_save(self, viz, sample_values, temp_dir):
        """Test saving dashboard."""
        save_path = temp_dir / "dashboard.png"
        fig = viz.create_dashboard(sample_values, save_path=save_path)

        assert save_path.exists()
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestIntegration:
    """Integration tests for visualization module."""

    def test_multiple_plots(self, viz, sample_values, temp_dir):
        """Test creating multiple plot types."""
        import matplotlib.pyplot as plt

        # Basic plot
        fig1 = viz.plot_timeseries(sample_values)
        plt.close(fig1)

        # Decomposition
        n = len(sample_values)
        trend = list(np.linspace(50, 100, n))
        seasonal = list(10 * np.sin(np.arange(n) * 2 * np.pi / 12))
        residual = list(np.random.randn(n) * 3)
        fig2 = viz.plot_decomposition(trend, seasonal, residual)
        plt.close(fig2)

        # Anomalies
        fig3 = viz.plot_anomalies(sample_values, [10, 50])
        plt.close(fig3)

        # Dashboard
        fig4 = viz.create_dashboard(sample_values)
        plt.close(fig4)

        # All should complete without error
        assert True
