#!/usr/bin/env python3
"""
General API Clients Module

This module provides base classes for API management and data retrieval
from various geospatial data sources.
"""

import logging
import requests
import geopandas as gpd
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


class BaseAPIManager:
    """
    Base class for API managers handling data retrieval and caching.
    """

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        """
        Initialize the API manager.

        Args:
            base_url: Base URL of the API
            api_key: Optional API key
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def fetch_data(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch data from API endpoint.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}


class GeneralGeoDataFetcher(BaseAPIManager):
    """
    General fetcher for geospatial data from various sources.
    """

    def __init__(self, base_url: str, cache_dir: Path) -> None:
        super().__init__(base_url)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_geospatial_data(
        self,
        dataset: str,
        params: dict[str, Any] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Get geospatial data for a specific dataset.

        Args:
            dataset: Name of the dataset
            params: Additional parameters

        Returns:
            GeoDataFrame with the data
        """
        cache_path = self.cache_dir / f"{dataset}.geojson"
        if cache_path.exists():
            logger.info(f"Loading cached data for {dataset}")
            return gpd.read_file(cache_path)

        data = self.fetch_data(dataset, params)
        if not data:
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame.from_features(data.get("features", []))
        if not gdf.empty:
            gdf.to_file(cache_path, driver="GeoJSON")
            logger.info(f"Cached data for {dataset} at {cache_path}")

        return gdf


# Example usage - can be extended for specific regions
if __name__ == "__main__":
    fetcher = GeneralGeoDataFetcher("https://example-geo-api.com", Path("data/cache"))
    data = fetcher.get_geospatial_data("boundaries")
    print(data.head())
