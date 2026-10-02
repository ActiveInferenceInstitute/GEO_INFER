"""
Cascadian Groundwater Data Sources

This module is responsible for fetching and processing groundwater data
for the Cascadian bioregion from the USGS National Water Information System (NWIS).
"""

from importlib.resources import files

import logging
import requests
import geopandas as gpd
import json
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import Polygon

# Import H3 utilities from the unified backend's path
from geo_infer_space.utils.h3_utils import (
    cell_to_latlng_boundary,
)


logger = logging.getLogger(__name__)


class CascadianGroundWaterDataSources:
    """
    Manages the acquisition and processing of groundwater data from USGS NWIS.
    """

    def __init__(self):
        config_path = str(
            files("geo_infer_place.locations.cascadia.config").joinpath(
                "data_urls.json"
            )
        )
        try:
            with open(config_path) as f:
                self.config = json.load(f).get("ground_water", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse ground_water config: {e}")
            self.config = {}

        self.usgs_nwis_url = self.config.get(
            "nwis_url", "https://waterservices.usgs.gov/nwis/gwlevels/"
        )
        self.parameter_cd = self.config.get(
            "parameter_cd", "72019"
        )  # Depth to water level, ft
        logger.info("Initialized CascadianGroundWaterDataSources")

    def _calculate_bbox_from_hexagons(
        self, hexagons: list[str]
    ) -> tuple[float, float, float, float]:
        """Calculates a bounding box from a list of H3 hexagons."""
        boundaries = [
            Polygon([(lon, lat) for lat, lon in cell_to_latlng_boundary(h)])
            for h in hexagons
        ]
        min_lon = min(b.bounds[0] for b in boundaries)
        min_lat = min(b.bounds[1] for b in boundaries)
        max_lon = max(b.bounds[2] for b in boundaries)
        max_lat = max(b.bounds[3] for b in boundaries)
        return (min_lon, min_lat, max_lon, max_lat)

    def fetch_groundwater_data(self, hexagons: list[str]) -> gpd.GeoDataFrame:
        """
        Fetches groundwater well data from the USGS NWIS for a list of H3 hexagons,
        breaking the request into smaller chunks to avoid API limits.

        Args:
            hexagons: A list of H3 hexagon identifiers.

        Returns:
            A GeoDataFrame containing groundwater well locations and available data,
            or an empty GeoDataFrame if no data is found or an error occurs.
        """
        if not hexagons:
            logger.warning("No hexagons provided to fetch groundwater data.")
            return gpd.GeoDataFrame()

        all_gdfs = []
        batch_size = 20  # Reduced batch size to avoid API timeouts

        for i in range(0, len(hexagons), batch_size):
            batch_hexagons = hexagons[i : i + batch_size]
            bbox = self._calculate_bbox_from_hexagons(batch_hexagons)

            # Correct bounding box format for USGS NWIS: west,south,east,north
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

            params = {
                "format": "json",
                "bBox": bbox_str,
                "parameterCd": self.parameter_cd,
                "siteStatus": "active",
            }

            headers = {
                "User-Agent": "GEO-INFER Framework/1.0 (https://github.com/geospatial-inference/GEO-INFER)"
            }

            logger.info(
                f"Fetching groundwater level data for batch {i // batch_size + 1}, {len(batch_hexagons)} hexagons..."
            )

            try:
                response = requests.get(
                    self.usgs_nwis_url, params=params, headers=headers, timeout=120
                )
                response.raise_for_status()

                # Check for empty response before trying to parse JSON
                if not response.text or not response.text.strip():
                    logger.info(
                        f"Empty response from USGS NWIS for batch {i // batch_size + 1}."
                    )
                    continue

                data = response.json()

                if not data.get("value", {}).get("timeSeries"):
                    logger.info(
                        f"No groundwater time series data found for batch {i // batch_size + 1}."
                    )
                    continue

                records = []
                for ts in data["value"]["timeSeries"]:
                    site_info = ts.get("sourceInfo", {})
                    site_name = site_info.get("siteName")
                    site_code = site_info.get("siteCode", [{}])[0].get("value")
                    lat = (
                        site_info.get("geoLocation", {})
                        .get("geogLocation", {})
                        .get("latitude")
                    )
                    lon = (
                        site_info.get("geoLocation", {})
                        .get("geogLocation", {})
                        .get("longitude")
                    )

                    if not all([site_code, lat, lon]):
                        continue  # Skip sites with incomplete location data

                    for value_entry in ts.get("values", [{}])[0].get("value", []):
                        records.append(
                            {
                                "site_code": site_code,
                                "site_name": site_name,
                                "latitude": lat,
                                "longitude": lon,
                                "datetime": pd.to_datetime(value_entry.get("dateTime")),
                                "gw_level_ft": pd.to_numeric(
                                    value_entry.get("value"), errors="coerce"
                                ),
                            }
                        )

                if not records:
                    continue

                df = pd.DataFrame(records)
                geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                all_gdfs.append(gdf)

            except requests.exceptions.HTTPError as e:
                # 404 is common if no data in bbox
                if e.response.status_code == 404:
                    logger.info(f"No data found (404) for batch {i // batch_size + 1}")
                else:
                    logger.error(
                        f"HTTP error fetching data from USGS NWIS for batch {i // batch_size + 1}: {e}"
                    )
                continue
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Error fetching data from USGS NWIS for batch {i // batch_size + 1}: {e}"
                )
                continue
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding JSON response from USGS NWIS for batch {i // batch_size + 1}: {e}"
                )
                continue
            except (ValueError, KeyError, IndexError) as e:
                logger.error(
                    f"Error parsing data from USGS NWIS for batch {i // batch_size + 1}: {e}"
                )
                continue

        if not all_gdfs:
            logger.warning("No groundwater data found across all batches.")
            return gpd.GeoDataFrame()

        # Concatenate all results and remove duplicates
        final_gdf = pd.concat(all_gdfs, ignore_index=True)
        final_gdf = final_gdf.drop_duplicates(subset=["site_code", "datetime"])

        logger.info(
            f"Successfully fetched and parsed {len(final_gdf)} unique groundwater level records from {final_gdf['site_code'].nunique()} sites across all batches."
        )
        return final_gdf
