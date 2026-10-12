"""
Specific API Clients for California Data Sources

This module implements API clients for California geospatial data sources,
extending the general BaseAPIManager from GEO-INFER-SPACE with:
- Exponential backoff retry logic
- Response validation
- Data provenance metadata
- Proper error classification (network vs API vs data errors)
"""

import logging
import time
from typing import Any, cast

import requests

from geo_infer_space.core.api_clients import BaseAPIManager

logger = logging.getLogger(__name__)

# Default retry configuration
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds
BACKOFF_MAX = 30.0


def _fetch_with_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    max_retries: int = MAX_RETRIES,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch JSON data with exponential backoff retry.

    Retries on transient network errors and 5xx server errors.
    Does NOT retry on 4xx client errors.

    Args:
        session: Requests session.
        url: Full URL.
        params: Query parameters.
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict, or dict with ``error`` key on failure.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)

            # 4xx: don't retry
            if 400 <= response.status_code < 500:
                logger.warning("Client error %d from %s", response.status_code, url)
                return {
                    "error": {
                        "type": "client_error",
                        "status": response.status_code,
                        "detail": response.text[:200],
                    }
                }

            response.raise_for_status()
            return cast(dict[str, Any], response.json())

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            logger.warning(
                "Timeout on attempt %d/%d for %s", attempt + 1, max_retries + 1, url
            )
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            logger.warning(
                "Connection error on attempt %d/%d for %s",
                attempt + 1,
                max_retries + 1,
                url,
            )
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            status = getattr(exc.response, "status_code", 0)
            if status < 500:
                # Non-retryable HTTP error
                return {
                    "error": {
                        "type": "http_error",
                        "status": status,
                        "detail": str(exc),
                    }
                }
            logger.warning(
                "Server error %d on attempt %d/%d", status, attempt + 1, max_retries + 1
            )
        except (ValueError, requests.exceptions.JSONDecodeError) as exc:
            # Response wasn't JSON
            last_exc = exc
            logger.warning(
                "JSON decode error on attempt %d/%d for %s",
                attempt + 1,
                max_retries + 1,
                url,
            )

        # Exponential backoff (skip sleep on last attempt)
        if attempt < max_retries:
            delay = min(BACKOFF_BASE * (2**attempt), BACKOFF_MAX)
            time.sleep(delay)

    logger.error("All %d retries exhausted for %s: %s", max_retries + 1, url, last_exc)
    return {"error": {"type": "network_error", "detail": str(last_exc)}}


class CALFIREClient(BaseAPIManager):
    """Client for CAL FIRE data access with retry and validation."""

    def __init__(self) -> None:
        super().__init__(
            "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/California_Fire_Perimeters/FeatureServer"
        )
        self.incident_url = (
            "https://www.fire.ca.gov/umbraco/api/IncidentApi/GetIncidents"
        )

    def fetch_incidents(self) -> Any:
        """Fetch active fire incidents from CAL FIRE with retry."""
        data = _fetch_with_retry(self.session, self.incident_url)
        if isinstance(data, dict) and "error" in data:
            logger.warning("Incident fetch returned error: %s", data["error"])
            return []
        return data

    def fetch_perimeters(
        self, year: int | None = None, county: str | None = None
    ) -> dict[str, Any]:
        """Fetch fire perimeters data with retry and validation.

        Args:
            year: Optional year filter.
            county: Optional county filter.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        where = "1=1"
        if year:
            where += f" AND YEAR_ = {year}"
        if county:
            where += f" AND POOCounty = '{county}'"

        params = {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
        }
        url = f"{self.base_url}/0/query"
        data = _fetch_with_retry(self.session, url, params=params)

        # Validate response structure
        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            n = len(data.get("features", []))
            logger.info(
                "Fetched %d fire perimeters (year=%s, county=%s)", n, year, county
            )
            return data

        return data  # May contain error key


class NOAAClient(BaseAPIManager):
    """Client for NOAA Tides and Currents data with retry."""

    def __init__(self) -> None:
        super().__init__("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter")
        self.weather_url = "https://api.weather.gov/stations"

    def fetch_weather_observations(self, station_id: str) -> dict[str, Any]:
        """Fetch latest weather observations with retry.

        Args:
            station_id: Weather station ID (e.g., KCEC).
        """
        url = f"{self.weather_url}/{station_id}/observations/latest"
        return _fetch_with_retry(self.session, url)

    def fetch_tide_data(
        self,
        station: str,
        begin_date: str,
        end_date: str,
        product: str = "water_level",
    ) -> dict[str, Any]:
        """Fetch tide gauge data with retry.

        Args:
            station: Station ID.
            begin_date: Start date (YYYYMMDD).
            end_date: End date (YYYYMMDD).
            product: Data product (default: water_level).
        """
        params = {
            "station": station,
            "begin_date": begin_date,
            "end_date": end_date,
            "product": product,
            "datum": "MLLW",
            "time_zone": "lst",
            "units": "metric",
            "format": "json",
            "application": "GEO-INFER-PLACE",
        }
        return _fetch_with_retry(self.session, self.base_url, params=params)


class USGSClient(BaseAPIManager):
    """Client for USGS water data with retry."""

    def __init__(self) -> None:
        super().__init__("https://waterservices.usgs.gov/nwis/iv")

    def fetch_water_data(
        self,
        sites: str,
        start: str,
        end: str,
        parameter_cd: str = "00060,00065",
    ) -> dict[str, Any]:
        """Fetch water data from USGS with retry.

        Args:
            sites: Comma-separated site IDs.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            parameter_cd: Parameter codes (default: discharge and gage height).
        """
        params = {
            "format": "json",
            "sites": sites,
            "startDT": start,
            "endDT": end,
            "parameterCd": parameter_cd,
        }
        return _fetch_with_retry(self.session, self.base_url, params=params)


class USGSEarthquakeClient(BaseAPIManager):
    """Client for USGS Earthquake data with retry."""

    def __init__(self) -> None:
        super().__init__("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary")

    def fetch_earthquakes(self, feed: str = "all_day.geojson") -> dict[str, Any]:
        """Fetch earthquake data feed with retry.

        Args:
            feed: Feed name (default: all_day.geojson).
        """
        url = f"{self.base_url}/{feed}"
        data = _fetch_with_retry(self.session, url)

        # Validate GeoJSON structure
        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            n = len(data.get("features", []))
            logger.info("Fetched %d earthquake events from %s", n, feed)

        return data


class CDECClient(BaseAPIManager):
    """Client for California Data Exchange Center with retry."""

    def __init__(self) -> None:
        super().__init__("https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet")

    def fetch_sensor_data(
        self,
        stations: str,
        sensor_num: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Fetch sensor data from CDEC with retry.

        Args:
            stations: Comma-separated station IDs.
            sensor_num: Sensor number.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
        """
        params = {
            "Stations": stations,
            "SensorNums": sensor_num,
            "Start": start_date,
            "End": end_date,
        }
        return _fetch_with_retry(self.session, self.base_url, params=params)


class CaliforniaAPIManager:
    """Manager class that aggregates California-specific API clients."""

    def __init__(self) -> None:
        self.calfire = CALFIREClient()
        self.noaa = NOAAClient()
        self.usgs = USGSClient()
        self.usgs_eq = USGSEarthquakeClient()
        self.cdec = CDECClient()

        logger.info("California API Manager initialized with retry-enabled clients")
