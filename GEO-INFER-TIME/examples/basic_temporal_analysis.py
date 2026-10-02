"""
Basic temporal analysis example using GEO-INFER-TIME.

This example demonstrates:
- Creating time series data
- Temporal trend detection
- Seasonality analysis
- Time series forecasting
- Event detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, UTC

from geo_infer_time.models.timeseries import TimeSeries
from geo_infer_time.core.analysis import TemporalAnalyzer
from geo_infer_time.core.forecasting import ForecastingEngine
from geo_infer_time.core.event_detection import EventDetector


def generate_sample_timeseries(n_points=100, trend=0.1, seasonality=True, noise=0.1):
    """Generate sample time series data with trend and seasonality."""
    rng = np.random.default_rng(42)

    # Create date range
    start_date = datetime(2020, 1, 1, tzinfo=UTC)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]

    # Generate values with trend
    time_points = np.arange(n_points)
    values = trend * time_points

    # Add seasonality (annual cycle)
    if seasonality:
        seasonal = 5 * np.sin(2 * np.pi * time_points / 365.25)
        values += seasonal

    # Add noise
    values += rng.normal(0, noise * np.std(values), n_points)

    # Create DataFrame
    df = pd.DataFrame({"date": dates, "value": values})
    df.set_index("date", inplace=True)

    return df


def main():
    """Run basic temporal analysis example."""
    print("=" * 60)
    print("GEO-INFER-TIME: Basic Temporal Analysis Example")
    print("=" * 60)

    # Step 1: Generate sample time series
    print("\n📊 Step 1: Generating sample time series data...")
    df = generate_sample_timeseries(n_points=365, trend=0.05, seasonality=True)
    timeseries = TimeSeries(data=df)
    print(f"   ✅ Created time series with {len(df)} data points")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")

    # Step 2: Temporal analysis
    print("\n🔍 Step 2: Performing temporal analysis...")
    analyzer = TemporalAnalyzer()

    # Detect trend
    trend_result = analyzer.detect_trend(timeseries, method="linear")
    print(f"   ✅ Trend detected: {trend_result['trend_direction']}")
    print(f"   Trend strength: {trend_result['trend_strength']:.4f}")

    # Detect seasonality
    try:
        seasonality_result = analyzer.detect_seasonality(timeseries, max_periods=12)
        if seasonality_result.get("has_seasonality"):
            print(
                f"   ✅ Seasonality detected: period = {seasonality_result.get('period', 'N/A')}"
            )
        else:
            print("   ℹ️  No significant seasonality detected")
    except Exception as e:
        print(f"   ⚠️  Seasonality detection: {e}")

    # Step 3: Forecasting
    print("\n🔮 Step 3: Generating forecasts...")
    forecaster = ForecastingEngine()

    # Linear forecast
    forecast_result = forecaster.forecast_linear(timeseries, horizon=30)
    print(f"   ✅ Generated {forecast_result['horizon']}-step forecast")
    print(f"   Forecast mean: {np.mean(forecast_result['forecast']):.2f}")
    print(f"   Forecast std: {np.std(forecast_result['forecast']):.2f}")

    # Exponential smoothing forecast (if available)
    try:
        es_forecast = forecaster.forecast_exponential_smoothing(
            timeseries, horizon=30, trend="add", seasonal="add", seasonal_periods=12
        )
        print("   ✅ Exponential smoothing forecast generated")
        print(f"   Forecast mean: {np.mean(es_forecast['forecast']):.2f}")
    except Exception as e:
        print(f"   ⚠️  Exponential smoothing: {e}")

    # Step 4: Event detection
    print("\n🚨 Step 4: Detecting events...")
    detector = EventDetector()

    try:
        events = detector.detect_anomalies(timeseries, method="z_score")
        print(f"   ✅ Detected {events['count']} potential events/anomalies")
        if events["anomalies"]:
            first_event = events["anomalies"][0]
            print(f"   First event at: {first_event.get('timestamp', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  Event detection: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Temporal analysis complete!")
    print("=" * 60)
    print("\nKey capabilities demonstrated:")
    print("  • Time series creation and management")
    print("  • Trend detection and analysis")
    print("  • Seasonality analysis")
    print("  • Time series forecasting")
    print("  • Event and anomaly detection")
    print("\nNext steps:")
    print("  • Try with your own time series data")
    print("  • Explore advanced forecasting methods")
    print("  • Integrate with SPACE module for spatio-temporal analysis")


if __name__ == "__main__":
    main()
