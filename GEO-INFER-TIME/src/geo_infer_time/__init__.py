"""GEO-INFER-TIME temporal analysis with public components loaded on demand."""

from importlib import import_module
from typing import Any

__version__ = "0.4.0"
__author__ = "GEO-INFER Development Team"

_EXPORTS = {
    "TemporalAnalyzer": ("geo_infer_time.core.analysis", "TemporalAnalyzer"),
    "ForecastingEngine": ("geo_infer_time.core.forecasting", "ForecastingEngine"),
    "StreamProcessor": ("geo_infer_time.core.stream_processing", "StreamProcessor"),
    "StreamIngestAdapter": ("geo_infer_time.core.stream_ingest", "StreamIngestAdapter"),
    "ReplayIngestAdapter": ("geo_infer_time.core.stream_ingest", "ReplayIngestAdapter"),
    "WebSocketIngestAdapter": (
        "geo_infer_time.core.stream_ingest",
        "WebSocketIngestAdapter",
    ),
    "KafkaIngestAdapter": ("geo_infer_time.core.stream_ingest", "KafkaIngestAdapter"),
    "TemporalInterpolator": (
        "geo_infer_time.core.interpolation",
        "TemporalInterpolator",
    ),
    "EventDetector": ("geo_infer_time.core.event_detection", "EventDetector"),
    "TemporalStatistics": ("geo_infer_time.core.statistics", "TemporalStatistics"),
    "TemporalVisualization": (
        "geo_infer_time.core.visualization",
        "TemporalVisualization",
    ),
    "TimeSeries": ("geo_infer_time.models.timeseries", "TimeSeries"),
    "AdvancedForecastingEngine": (
        "geo_infer_time.core.advanced_forecasting",
        "AdvancedForecastingEngine",
    ),
    "normalize_timestamp": ("geo_infer_time.core.timestamps", "normalize_timestamp"),
    "normalize_datetime_index": (
        "geo_infer_time.core.timestamps",
        "normalize_datetime_index",
    ),
    "db": ("geo_infer_time.db", None),
    "io": ("geo_infer_time.io", None),
    "utils": ("geo_infer_time.utils", None),
}

__all__ = [
    "TemporalAnalyzer",
    "ForecastingEngine",
    "StreamProcessor",
    "StreamIngestAdapter",
    "ReplayIngestAdapter",
    "WebSocketIngestAdapter",
    "KafkaIngestAdapter",
    "TemporalInterpolator",
    "EventDetector",
    "TemporalStatistics",
    "TemporalVisualization",
    "TimeSeries",
    "AdvancedForecastingEngine",
    "normalize_timestamp",
    "normalize_datetime_index",
    "db",
    "io",
    "utils",
]


def __getattr__(name: str) -> Any:
    """Resolve a public component when requested; import failures propagate."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
