"""
GEO-INFER-API package.

A standardized API for geospatial interoperability within the GEO-INFER framework.
"""

__version__ = "0.3.0"


from geo_infer_api.core.config import Settings, get_settings
from geo_infer_api.models.geojson import (
    Feature,
    FeatureCollection,
    GeoJSONType,
    Geometry,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    PolygonFeature,
    PolygonFeatureCollection,
)


def __getattr__(name: str):
    """Lazily expose ``main_app`` so importing the package stays side-effect free."""
    if name == "main_app":
        from geo_infer_api.app import main_app

        return main_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "main_app",
    "Settings",
    "get_settings",
    "Feature",
    "FeatureCollection",
    "GeoJSONType",
    "Geometry",
    "LineString",
    "MultiLineString",
    "MultiPoint",
    "MultiPolygon",
    "Point",
    "Polygon",
    "PolygonFeature",
    "PolygonFeatureCollection",
]
