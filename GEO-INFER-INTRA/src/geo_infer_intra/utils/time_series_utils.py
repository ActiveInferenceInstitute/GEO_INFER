"""Time series test helper utilities for GEO-INFER-INTRA."""

from typing import Any
from collections.abc import Callable
import datetime
import random
import json
from pathlib import Path
import math


def create_iso8601_timestamp(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    timezone: str = "Z",
) -> str:
    """
    Create an ISO8601 timestamp string.

    Args:
        year: Year
        month: Month
        day: Day
        hour: Hour
        minute: Minute
        second: Second
        microsecond: Microsecond
        timezone: Timezone indicator (default "Z" for UTC)

    Returns:
        ISO8601 timestamp string
    """
    dt = datetime.datetime(year, month, day, hour, minute, second, microsecond)

    if timezone == "Z":
        return (
            dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"
        )  # Truncate microseconds to 3 digits

    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + timezone


def create_timestamp_range(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: datetime.timedelta,
) -> list[str]:
    """
    Create a range of ISO8601 timestamp strings.

    Args:
        start_date: Start date
        end_date: End date
        interval: Time interval between timestamps

    Returns:
        List of ISO8601 timestamp strings
    """
    timestamps = []
    current_date = start_date

    while current_date <= end_date:
        timestamps.append(current_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z")
        current_date += interval

    return timestamps


def create_daily_timestamps(
    start_year: int, start_month: int, start_day: int, num_days: int
) -> list[str]:
    """
    Create a list of daily ISO8601 timestamp strings.

    Args:
        start_year: Starting year
        start_month: Starting month
        start_day: Starting day
        num_days: Number of days

    Returns:
        List of ISO8601 timestamp strings at daily intervals
    """
    start_date = datetime.datetime(start_year, start_month, start_day)
    end_date = start_date + datetime.timedelta(days=num_days - 1)
    return create_timestamp_range(start_date, end_date, datetime.timedelta(days=1))


def create_hourly_timestamps(
    start_year: int, start_month: int, start_day: int, start_hour: int, num_hours: int
) -> list[str]:
    """
    Create a list of hourly ISO8601 timestamp strings.

    Args:
        start_year: Starting year
        start_month: Starting month
        start_day: Starting day
        start_hour: Starting hour
        num_hours: Number of hours

    Returns:
        List of ISO8601 timestamp strings at hourly intervals
    """
    start_date = datetime.datetime(start_year, start_month, start_day, start_hour)
    end_date = start_date + datetime.timedelta(hours=num_hours - 1)
    return create_timestamp_range(start_date, end_date, datetime.timedelta(hours=1))


def create_time_series_data(
    timestamps: list[str], value_generator: Callable[[str, int], float]
) -> dict[str, Any]:
    """
    Create time series data with generated values.

    Args:
        timestamps: List of timestamp strings
        value_generator: Function to generate values based on timestamp and index

    Returns:
        Dictionary with timestamps and values
    """
    values = [value_generator(ts, i) for i, ts in enumerate(timestamps)]

    return {"timestamps": timestamps, "values": values}


def random_walk_generator(
    timestamp: str, index: int, start_value: float = 10.0, step_size: float = 1.0
) -> float:
    """
    Generate a random walk value.

    Args:
        timestamp: Timestamp string (not used, for interface compatibility)
        index: Index in the sequence
        start_value: Starting value
        step_size: Maximum step size

    Returns:
        Random walk value
    """
    if index == 0:
        return start_value

    # Random step between -step_size and +step_size
    step = (random.random() * 2 - 1) * step_size
    return start_value + index * step / 5 + step  # Trend + random


def seasonal_generator(
    timestamp: str,
    index: int,
    base_value: float = 10.0,
    amplitude: float = 2.0,
    period: int = 24,
    noise_level: float = 0.5,
) -> float:
    """
    Generate a seasonal value with optional trend and noise.

    Args:
        timestamp: Timestamp string (not used, for interface compatibility)
        index: Index in the sequence
        base_value: Base value
        amplitude: Amplitude of seasonal component
        period: Period length in same units as index
        noise_level: Magnitude of random noise

    Returns:
        Seasonal value with noise
    """
    # Seasonal component
    seasonal = amplitude * math.sin(2 * math.pi * index / period)

    # Random noise
    noise = (random.random() * 2 - 1) * noise_level

    return base_value + seasonal + noise


def load_time_series_file(file_path: str | Path) -> dict[str, Any]:
    """
    Load time series data from file.

    Args:
        file_path: Path to time series file

    Returns:
        Time series data
    """
    with open(file_path) as f:
        return json.load(f)


def save_time_series_file(
    time_series_data: dict[str, Any], file_path: str | Path
) -> None:
    """
    Save time series data to file.

    Args:
        time_series_data: Time series data to save
        file_path: Path to save to
    """
    with open(file_path, "w") as f:
        json.dump(time_series_data, f, indent=2)
