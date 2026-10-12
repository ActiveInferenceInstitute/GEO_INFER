"""Unit tests for the Seasonal Analysis core functionality."""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from geo_infer_ag.core.seasonal_analysis import SeasonalAnalysis


class TestSeasonalAnalysis:
    """Test suite for SeasonalAnalysis class."""

    def test_initialization(self):
        """Test initialization of SeasonalAnalysis."""
        # Test basic initialization
        sa = SeasonalAnalysis()
        assert sa.time_series_data is None
        assert sa.spatial_data is None
        assert isinstance(sa.config, dict)
        assert sa.growing_season == {}

        # Test with data
        time_series = pd.DataFrame({"ndvi": [0.2, 0.3, 0.4]})
        sa = SeasonalAnalysis(time_series_data=time_series)
        assert sa.time_series_data is time_series

    def test_detect_growing_season_threshold(self, sample_time_series_data):
        """Test growing season detection using threshold method."""
        sa = SeasonalAnalysis(time_series_data=sample_time_series_data)

        # Test detection with threshold method
        growing_season = sa.detect_growing_season(
            variable="ndvi", method="threshold", threshold=0.3
        )

        assert "seasons" in growing_season
        assert len(growing_season["seasons"]) >= 1
        assert "variable" in growing_season
        assert growing_season["variable"] == "ndvi"
        assert "method" in growing_season
        assert growing_season["method"] == "threshold"

        # Check season structure
        season = growing_season["seasons"][0]
        assert "start_date" in season
        assert "peak_date" in season
        assert "end_date" in season
        assert "length_days" in season
        assert "peak_value" in season
        assert "mean_value" in season

        # Check logical relationships
        assert season["start_date"] < season["peak_date"]
        assert season["peak_date"] < season["end_date"]
        assert season["length_days"] > 30  # We set min_length_days=30

        # Ensure peak value is the maximum in the season
        season_data = sample_time_series_data.loc[
            season["start_date"] : season["end_date"]
        ]["ndvi"]
        assert abs(season["peak_value"] - season_data.max()) < 1e-6

        # Test with no variable in the data
        with pytest.raises(ValueError):
            sa.detect_growing_season(variable="nonexistent")

        # Test with external time series
        custom_dates = pd.date_range(
            start=datetime.now() - timedelta(days=90), end=datetime.now(), freq="D"
        )
        custom_values = (
            0.2 + 0.6 * np.sin(np.pi * np.arange(len(custom_dates)) / 45) ** 2
        )
        custom_series = pd.Series(custom_values, index=custom_dates)

        growing_season = sa.detect_growing_season(
            time_series=custom_series, method="threshold", threshold=0.3
        )

        assert "seasons" in growing_season
        assert len(growing_season["seasons"]) >= 1

    def test_detect_growing_season_derivative(self, sample_time_series_data):
        """Test growing season detection using derivative method."""
        sa = SeasonalAnalysis(time_series_data=sample_time_series_data)

        # Test detection with derivative method
        growing_season = sa.detect_growing_season(
            variable="ndvi", method="derivative", smoothing_window=7
        )

        assert "seasons" in growing_season
        assert "variable" in growing_season
        assert growing_season["variable"] == "ndvi"
        assert "method" in growing_season
        assert growing_season["method"] == "derivative"

        # If seasons were detected, check their structure
        if growing_season["seasons"]:
            season = growing_season["seasons"][0]
            assert "start_date" in season
            assert "peak_date" in season
            assert "end_date" in season
            assert "length_days" in season

            # Check logical relationships
            assert season["start_date"] < season["peak_date"]
            assert season["peak_date"] < season["end_date"]

        # Test with unsupported method
        with pytest.raises(ValueError):
            sa.detect_growing_season(variable="ndvi", method="unsupported_method")

    def test_identify_phenological_stages(self, sample_time_series_data):
        """Test identification of phenological stages."""
        sa = SeasonalAnalysis(time_series_data=sample_time_series_data)

        # First detect growing season
        sa.detect_growing_season(variable="ndvi", method="threshold", threshold=0.3)

        # Test identification of phenological stages for corn
        phenology = sa.identify_phenological_stages(crop_type="corn", variable="ndvi")

        assert "crop_type" in phenology
        assert phenology["crop_type"] == "corn"
        assert "variable" in phenology
        assert phenology["variable"] == "ndvi"
        assert "season" in phenology
        assert "stages" in phenology

        # Check that some stages were identified
        assert len(phenology["stages"]) > 0

        # Check stage structure
        for stage_data in phenology["stages"].values():
            assert "start_date" in stage_data
            assert "end_date" in stage_data
            assert "length_days" in stage_data
            assert "mean_value" in stage_data
            assert stage_data["start_date"] <= stage_data["end_date"]

        # Test with custom reference stages
        custom_stages = {"early": (0.2, 0.4), "middle": (0.4, 0.7), "late": (0.3, 0.5)}

        phenology = sa.identify_phenological_stages(
            crop_type="custom", variable="ndvi", reference_stages=custom_stages
        )

        assert "stages" in phenology
        for stage_name in custom_stages.keys():
            if stage_name in phenology["stages"]:
                assert "start_date" in phenology["stages"][stage_name]
                assert "end_date" in phenology["stages"][stage_name]

        # Test without prior growing season detection
        sa = SeasonalAnalysis(time_series_data=sample_time_series_data)
        with pytest.raises(ValueError):
            sa.identify_phenological_stages(crop_type="corn", variable="ndvi")

    def test_analyze_temporal_trends(self, sample_time_series_data):
        """Test analysis of temporal trends."""
        sa = SeasonalAnalysis(time_series_data=sample_time_series_data)

        # Test trend analysis with default settings
        trends = sa.analyze_temporal_trends(variable="ndvi")

        assert "variable" in trends
        assert trends["variable"] == "ndvi"
        assert "period" in trends
        assert "statistics" in trends
        assert "data" in trends
        assert "original" in trends["data"]
        assert "moving_avg" in trends["data"]

        # Check that statistics were calculated
        assert "mean" in trends["statistics"]
        assert "std" in trends["statistics"]
        assert "min" in trends["statistics"]
        assert "max" in trends["statistics"]

        # Check trend analysis (if there are enough data points)
        if "trend_analysis" in trends:
            assert "slope" in trends["trend_analysis"]
            assert "intercept" in trends["trend_analysis"]
            assert "r_squared" in trends["trend_analysis"]
            assert "p_value" in trends["trend_analysis"]

        # Test with resampling to weekly
        trends = sa.analyze_temporal_trends(variable="ndvi", period="weekly")

        assert trends["period"] == "weekly"
        assert len(trends["data"]["original"]) < len(sample_time_series_data)

        # Test with detrending
        trends = sa.analyze_temporal_trends(variable="ndvi", detrend=True)

        if "trend_analysis" in trends:
            assert "detrended" in trends["data"]

        # Test with invalid variable
        with pytest.raises(ValueError):
            sa.analyze_temporal_trends(variable="nonexistent")

        # Test with external time series
        custom_dates = pd.date_range(
            start=datetime.now() - timedelta(days=90), end=datetime.now(), freq="D"
        )
        custom_values = (
            0.2 + 0.6 * np.sin(np.pi * np.arange(len(custom_dates)) / 45) ** 2
        )
        custom_series = pd.Series(custom_values, index=custom_dates)

        trends = sa.analyze_temporal_trends(time_series=custom_series, window_size=5)

        assert "data" in trends
        assert "moving_avg" in trends["data"]
        assert len(trends["data"]["moving_avg"]) == len(custom_series)

    def test_analyze_spatial_temporal_patterns(self, mocker):
        """Test analysis of spatial-temporal patterns."""
        # Create mock xarray Dataset
        mock_dataset = mocker.MagicMock()
        mock_data_array = mocker.MagicMock()
        mock_dataset.__getitem__.return_value = mock_data_array
        mock_dataset.data_vars = ["ndvi"]

        # Configure data array mocks
        mock_mean = mocker.MagicMock()
        mock_std = mocker.MagicMock()
        mock_data_array.mean.return_value = mock_mean
        mock_data_array.std.return_value = mock_std

        mock_mean.values = np.array([0.5])
        mock_std.values = np.array([0.1])
        mock_data_array.mean.return_value.values = np.array([0.5])
        mock_data_array.std.return_value.values = np.array([0.1])
        mock_data_array.min.return_value.values = np.array([0.2])
        mock_data_array.max.return_value.values = np.array([0.8])

        # Create seasonal analysis object
        sa = SeasonalAnalysis()

        # Test spatial-temporal analysis
        results = sa.analyze_spatial_temporal_patterns(
            dataset=mock_dataset, variable="ndvi"
        )

        assert "variable" in results
        assert results["variable"] == "ndvi"
        assert "temporal_stats" in results
        assert "spatial_series" in results
        assert "dataset_stats" in results

        # Check global statistics
        assert "global_mean" in results["dataset_stats"]
        assert "global_std" in results["dataset_stats"]
        assert "global_min" in results["dataset_stats"]
        assert "global_max" in results["dataset_stats"]

        # Check that dataset methods were called correctly
        mock_dataset.__getitem__.assert_called_with("ndvi")
        mock_data_array.mean.assert_any_call(dim="time")
        mock_data_array.std.assert_any_call(dim="time")
        mock_data_array.mean.assert_any_call(dim=["lat", "lon"])

        # Test with missing variable
        mock_dataset.data_vars = ["evi"]
        with pytest.raises(ValueError):
            sa.analyze_spatial_temporal_patterns(dataset=mock_dataset, variable="ndvi")

        # Test without dataset
        with pytest.raises(ValueError):
            sa.analyze_spatial_temporal_patterns(variable="ndvi")

    def test_plot_growing_season(self, sample_time_series_data):
        """Test plotting of growing season."""
        sa = SeasonalAnalysis(time_series_data=sample_time_series_data)

        # First detect growing season
        sa.detect_growing_season(variable="ndvi", method="threshold", threshold=0.3)

        # Test plotting
        fig, ax = plt.subplots()
        result_ax = sa.plot_growing_season(ax=ax)

        # Check that plotting was successful
        assert result_ax is ax
        assert len(ax.lines) > 0  # Should have at least the NDVI line
        assert len(ax.texts) > 0  # Should have season labels

        # Clean up
        plt.close(fig)

        # Test without growing season
        sa = SeasonalAnalysis(time_series_data=sample_time_series_data)
        with pytest.raises(ValueError):
            sa.plot_growing_season()

        # Test with missing time series data
        sa = SeasonalAnalysis()
        sa.growing_season = {
            "variable": "ndvi",
            "seasons": [
                {
                    "start_date": datetime.now(),
                    "peak_date": datetime.now(),
                    "end_date": datetime.now(),
                }
            ],
        }
        with pytest.raises(ValueError):
            sa.plot_growing_season()

    def test_identify_phenological_stages_constant_series(self):
        """A constant series must not produce NaN-normalized empty stages."""
        dates = pd.date_range("2024-04-01", periods=120, freq="D")
        sa = SeasonalAnalysis()
        # Inject a detected season so the constant series can be analyzed
        # directly (detect_growing_season cannot fire on a flat series).
        sa.growing_season = {
            "seasons": [
                {"start_date": dates[0], "peak_date": dates[60], "end_date": dates[-1]}
            ]
        }
        constant_series = pd.Series(0.5, index=dates)
        result = sa.identify_phenological_stages(
            crop_type="corn", time_series=constant_series
        )

        # Constant series maps to normalized 0.5, so stages whose window
        # includes 0.5 are populated instead of every stage being dropped.
        assert result["stages"], "expected at least one phenological stage"

    def test_analyze_temporal_trends_resample_no_futurewarning(self):
        """Monthly/annual resampling must not emit pandas FutureWarnings."""
        import warnings

        dates = pd.date_range("2024-01-01", periods=730, freq="D")
        series = pd.Series(np.linspace(0.1, 0.9, len(dates)), index=dates)
        sa = SeasonalAnalysis()

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            monthly = sa.analyze_temporal_trends(time_series=series, period="monthly")
            annual = sa.analyze_temporal_trends(time_series=series, period="annual")

        assert len(monthly["data"]["original"]) > 12
        assert len(annual["data"]["original"]) >= 2
