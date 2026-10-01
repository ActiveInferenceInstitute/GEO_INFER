"""
Del Norte data integration utilities.

Provides thin wrappers over specific API clients to expose a consistent
interface expected by Del Norte analyzers (forest, coastal, fire, seismic).

Real data endpoints where available:
- CAL FIRE perimeters via ArcGIS REST
- NOAA Tides and Currents for water levels
- USGS Earthquake Hazards for seismic events

Wrappers inherit shared caching from ``CachedAPIWrapper``.
"""

from __future__ import annotations

import logging
import h3
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, cast

from .caching import CachedAPIWrapper
from ..core.api_clients import CaliforniaAPIManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CAL FIRE wrapper
# ---------------------------------------------------------------------------


class _CALFIREWrapper(CachedAPIWrapper):
    """Wrapper exposing analyzer-facing methods for CAL FIRE data."""

    def __init__(
        self, api_manager: CaliforniaAPIManager, cache_dir: Path | None = None
    ) -> None:
        super().__init__(cache_dir=cache_dir, cache_ttl=timedelta(hours=24))
        self._client = api_manager.calfire

    def get_fire_perimeters(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        start_year: int | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Fetch fire perimeters optionally filtered by year and county via bbox.

        Args:
            bbox: (west, south, east, north).
            start_year: Minimum fire year to include.
            include_metadata: Returned GeoJSON keeps all properties.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        cache_key = self._cache_key(
            "get_fire_perimeters", bbox=bbox, start_year=start_year
        )
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cast(dict[str, Any], cached)

        try:
            geojson = self._client.fetch_perimeters(year=start_year, county="Del Norte")
        except Exception as exc:
            raise RuntimeError("CAL FIRE perimeter acquisition failed") from exc
        if not isinstance(geojson, dict) or "error" in geojson:
            raise RuntimeError(
                f"CAL FIRE perimeter acquisition returned no usable data: {geojson!r}"
            )

        self._write_cache(cache_key, geojson)

        if not bbox:
            return geojson

        # Client-side bbox filter
        west, south, east, north = bbox
        feats: list[dict[str, Any]] = []
        for feat in geojson.get("features", []):
            geom = feat.get("geometry", {})
            coords = []
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [])
            elif geom.get("type") == "MultiPolygon":
                for part in geom.get("coordinates", []):
                    coords.extend(part)
            include = False
            for ring in coords:
                for lon, lat in ring:
                    if west <= lon <= east and south <= lat <= north:
                        include = True
                        break
                if include:
                    break
            if include:
                feats.append(feat)

        return {"type": "FeatureCollection", "features": feats}

    def get_active_incidents(self) -> dict[str, Any]:
        """Fetch active CAL FIRE incidents."""
        cache_key = self._cache_key("get_active_incidents")
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cast(dict[str, Any], cached)

        try:
            incidents = self._client.fetch_incidents()
            del_norte_incidents = []
            if isinstance(incidents, list):
                for incident in incidents:
                    if "Counties" in incident and "Del Norte" in incident.get(
                        "Counties", ""
                    ):
                        del_norte_incidents.append(
                            {
                                "name": incident.get("Name", "Unknown"),
                                "location": incident.get("Location", ""),
                                "acres": incident.get("AcresBurned", 0),
                                "contained": incident.get("PercentContained", 0),
                                "lat": incident.get("Latitude", 0),
                                "lon": incident.get("Longitude", 0),
                                "start_date": incident.get("Started", ""),
                                "status": incident.get("Status", "Unknown"),
                            }
                        )
            result = {"incidents": del_norte_incidents, "success": True}
            self._write_cache(cache_key, result)
            return result
        except Exception as exc:
            logger.warning("Failed to fetch active incidents: %s", exc)
            return {"incidents": [], "success": False, "error": str(exc)}

    def get_timber_operations(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        time_range: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Report that no timber-plan client is configured."""
        raise RuntimeError(
            "CAL FIRE timber harvest plan data requires a configured source client"
        )

    def get_tree_mortality_data(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        time_range: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Report that no mortality-survey client is configured."""
        raise RuntimeError(
            "Tree mortality survey data requires a configured source client"
        )


# ---------------------------------------------------------------------------
# NOAA wrapper
# ---------------------------------------------------------------------------


class _NOAAWrapper(CachedAPIWrapper):
    """Wrapper exposing analyzer-facing methods for NOAA tides and currents."""

    def __init__(
        self, api_manager: CaliforniaAPIManager, cache_dir: Path | None = None
    ) -> None:
        super().__init__(cache_dir=cache_dir, cache_ttl=timedelta(hours=6))
        self._client = api_manager.noaa

    def get_weather_data(self, station_id: str = "KCEC") -> dict[str, Any]:
        """Fetch weather observation data.

        Args:
            station_id: Weather station ID (default KCEC for Crescent City).
        """
        cache_key = self._cache_key("get_weather_data", station_id=station_id)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cast(dict[str, Any], cached)

        try:
            data = self._client.fetch_weather_observations(station_id)
            if isinstance(data, dict):
                properties = data.get("properties", {})
                result = {
                    "temperature": properties.get("temperature", {}).get("value"),
                    "humidity": properties.get("relativeHumidity", {}).get("value"),
                    "wind_speed": properties.get("windSpeed", {}).get("value"),
                    "wind_direction": properties.get("windDirection", {}).get("value"),
                    "pressure": properties.get("barometricPressure", {}).get("value"),
                    "timestamp": properties.get("timestamp"),
                    "success": True,
                    "data_quality": "empirical",
                }
                self._write_cache(cache_key, result)
                return result
        except Exception as exc:
            logger.warning("Failed to fetch weather data: %s", exc)

        return {"success": False, "error": "Weather data fetch failed"}

    def get_tide_gauge_data(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        stations: list[str] | None = None,
        time_range: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Fetch tide gauge water level time series.

        Args:
            bbox: Unused for NOAA direct calls; present for interface parity.
            stations: List of station IDs (default to Crescent City 9419750).
            time_range: (YYYY-MM-DD, YYYY-MM-DD). Defaults to last 7 days.
        """
        if not stations:
            stations = ["9419750"]  # Crescent City

        cache_key = self._cache_key(
            "get_tide_gauge_data", stations=stations, time_range=time_range
        )
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cast(dict[str, Any], cached)

        try:
            if time_range:
                start, end = time_range
                begin_date = datetime.strptime(start, "%Y-%m-%d").strftime("%Y%m%d")
                end_date = datetime.strptime(end, "%Y-%m-%d").strftime("%Y%m%d")
            else:
                end_date = datetime.now(UTC).strftime("%Y%m%d")
                begin_date = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y%m%d")

            series: dict[str, Any] = {}
            for station in stations:
                try:
                    data = self._client.fetch_tide_data(
                        station=station,
                        begin_date=begin_date,
                        end_date=end_date,
                        product="water_level",
                    )
                    if isinstance(data, dict) and "error" not in data:
                        series[station] = data
                    else:
                        raise RuntimeError(
                            f"NOAA returned no usable tide data for station {station}"
                        )
                except Exception as exc:
                    raise RuntimeError(
                        f"NOAA tide acquisition failed for station {station}"
                    ) from exc

            result = {"stations": stations, "series": series}
            self._write_cache(cache_key, result)
            return result

        except Exception as exc:
            raise RuntimeError("NOAA tide gauge acquisition failed") from exc

    def get_current_data(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        time_range: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Fetch ocean current data for the Del Norte coastal area."""
        cache_key = self._cache_key(
            "get_current_data", bbox=bbox, time_range=time_range
        )
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cast(dict[str, Any], cached)

        raise RuntimeError(
            "NOAA current observations require a configured current-data source"
        )


# ---------------------------------------------------------------------------
# USGS Earthquake wrapper
# ---------------------------------------------------------------------------


class _USGSWrapper(CachedAPIWrapper):
    """Wrapper for USGS earthquake data with Cascadia subduction zone focus."""

    def __init__(
        self, api_manager: CaliforniaAPIManager, cache_dir: Path | None = None
    ) -> None:
        super().__init__(cache_dir=cache_dir, cache_ttl=timedelta(minutes=60))
        self._client = api_manager.usgs_eq

    def get_earthquakes(
        self,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Fetch recent earthquakes in the Del Norte / Cascadia area.

        Args:
            bbox: Optional (west, south, east, north) to override default Del Norte bounds.
        """
        cache_key = self._cache_key("get_earthquakes", bbox=bbox)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cast(dict[str, Any], cached)

        if not bbox:
            bbox = (-124.5, 41.4, -123.5, 42.1)  # Del Norte County
        west, south, east, north = bbox

        try:
            data = self._client.fetch_earthquakes()
            local_earthquakes = []
            if isinstance(data, dict):
                for feature in data.get("features", []):
                    coords = feature.get("geometry", {}).get("coordinates", [])
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        if west <= lon <= east and south <= lat <= north:
                            props = feature.get("properties", {})
                            local_earthquakes.append(
                                {
                                    "magnitude": props.get("mag"),
                                    "place": props.get("place"),
                                    "time": props.get("time"),
                                    "lat": lat,
                                    "lon": lon,
                                    "depth": coords[2] if len(coords) > 2 else None,
                                    "h3_cell": h3.latlng_to_cell(lat, lon, 8),
                                }
                            )
            result = {
                "earthquakes": local_earthquakes,
                "success": True,
                "data_quality": "empirical",
            }
            self._write_cache(cache_key, result)
            return result
        except Exception as exc:
            logger.warning("Failed to fetch earthquakes: %s", exc)
            return {"earthquakes": [], "success": False, "error": str(exc)}

    def get_cascadia_seismicity(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Fetch seismicity across the full Cascadia subduction zone.

        Covers the region from Cape Mendocino (40.4N) to Vancouver Island (50.5N),
        extending offshore to include the subduction interface.

        Args:
            days: Number of days to look back (default 30).
        """
        cache_key = self._cache_key("get_cascadia_seismicity", days=days)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cast(dict[str, Any], cached)

        # Full Cascadia subduction zone bounds
        csz_bbox = (-130.0, 40.0, -121.0, 50.5)
        west, south, east, north = csz_bbox

        try:
            # Use the all_month feed for 30-day coverage
            feed = "all_month.geojson" if days > 7 else "all_week.geojson"
            data = self._client.fetch_earthquakes(feed=feed)
            csz_events = []
            if isinstance(data, dict):
                for feature in data.get("features", []):
                    coords = feature.get("geometry", {}).get("coordinates", [])
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        if west <= lon <= east and south <= lat <= north:
                            props = feature.get("properties", {})
                            depth = coords[2] if len(coords) > 2 else None
                            csz_events.append(
                                {
                                    "magnitude": props.get("mag"),
                                    "place": props.get("place"),
                                    "time": props.get("time"),
                                    "lat": lat,
                                    "lon": lon,
                                    "depth": depth,
                                    "is_subduction_depth": depth is not None
                                    and 10 <= depth <= 60,
                                    "h3_cell": h3.latlng_to_cell(lat, lon, 8),
                                }
                            )

            # Classify by depth zones
            shallow = [
                e for e in csz_events if e.get("depth") is not None and e["depth"] < 20
            ]
            intermediate = [
                e
                for e in csz_events
                if e.get("depth") is not None and 20 <= e["depth"] < 70
            ]
            deep = [
                e for e in csz_events if e.get("depth") is not None and e["depth"] >= 70
            ]

            result = {
                "total_events": len(csz_events),
                "events": csz_events,
                "depth_classification": {
                    "shallow_crustal": len(shallow),
                    "intermediate_subduction": len(intermediate),
                    "deep_intraslab": len(deep),
                },
                "bbox": csz_bbox,
                "period_days": days,
                "success": True,
                "data_quality": "empirical",
            }
            self._write_cache(cache_key, result)
            return result
        except Exception as exc:
            logger.warning("Failed to fetch Cascadia seismicity: %s", exc)
            return {
                "total_events": 0,
                "events": [],
                "success": False,
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# Unified integrator
# ---------------------------------------------------------------------------


class DelNorteDataIntegrator:
    """Integrator that aggregates API wrappers for analyzers.

    Attributes:
        calfire_client: Fire perimeters, incidents, timber ops, mortality.
        noaa_client: Tides, weather, ocean currents.
        usgs_client: Earthquakes, Cascadia seismicity.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        api_manager = CaliforniaAPIManager()
        self.calfire_client = _CALFIREWrapper(api_manager, cache_dir)
        self.noaa_client = _NOAAWrapper(api_manager, cache_dir)
        self.usgs_client = _USGSWrapper(api_manager, cache_dir)
