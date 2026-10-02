#!/usr/bin/env python3
"""
Real Data Acquisition Module for Cascadia Agricultural Analysis Framework

This module provides comprehensive real data acquisition capabilities for:
- Zoning data from county/city websites
- Current land use data from agricultural agencies
- Property ownership data from assessor offices
- Building improvements data from permit systems
- Water rights data from state agencies
- Groundwater data from USGS and state agencies
- Surface water data from environmental agencies
- Power source data from utility companies
- Mortgage debt data from financial institutions

Based on web scraping best practices and geospatial data standards.
"""

import requests
import geopandas as gpd
from pathlib import Path
import json
import time
import logging
import zipfile
from urllib.parse import urlparse
import subprocess
import os

# Web scraping libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Enhanced logging
from .enhanced_logging import DataSourceLogger, ProcessingLogger

logger = logging.getLogger(__name__)


class RealDataAcquisition:
    """
    Comprehensive real data acquisition system for Cascadia framework.

    Provides methods to acquire real geospatial data from various sources:
    - Government agencies and open data portals
    - County and city websites
    - State and federal agencies
    - Agricultural and environmental organizations
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the real data acquisition system.

        Args:
            output_dir: Directory to store acquired data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Allow optional insecure downloads to work around SSL issues on some hosts
        # Set env CASCADIA_INSECURE_DOWNLOADS=1 to disable certificate verification for HTTP(S) requests
        self.insecure_downloads = (
            os.environ.get("CASCADIA_INSECURE_DOWNLOADS", "0") == "1"
        )

        # Initialize loggers
        self.data_logger = DataSourceLogger("real_data_acquisition")
        self.processing_logger = ProcessingLogger("real_data_acquisition")

        # Data source configurations
        self.data_sources = {
            "zoning": {
                "name": "Zoning Data",
                "sources": [
                    {
                        "name": "Del Norte County Zoning",
                        "url": "https://www.co.del-norte.ca.us/departments/planning-building",
                        "type": "web_scrape",
                        "description": "Official county zoning data",
                    },
                    {
                        "name": "California State Zoning Database",
                        "url": "https://data.ca.gov/dataset/zoning-data",
                        "type": "api",
                        "description": "Statewide zoning information",
                    },
                ],
            },
            "current_use": {
                "name": "Current Land Use Data",
                "sources": [
                    {
                        "name": "USDA Cropland Data Layer",
                        "url": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php",
                        "type": "download",
                        "description": "High-resolution cropland classification",
                    },
                    {
                        "name": "California Department of Conservation",
                        "url": "https://www.conservation.ca.gov/dlrp/fmml",
                        "type": "web_scrape",
                        "description": "Farmland mapping and monitoring",
                    },
                ],
            },
            "ownership": {
                "name": "Property Ownership Data",
                "sources": [
                    {
                        "name": "Del Norte County Assessor",
                        "url": "https://www.co.del-norte.ca.us/departments/assessor",
                        "type": "web_scrape",
                        "description": "Property ownership and assessment data",
                    },
                    {
                        "name": "California State Board of Equalization",
                        "url": "https://www.boe.ca.gov/proptaxes/proptax.htm",
                        "type": "api",
                        "description": "Statewide property assessment data",
                    },
                ],
            },
            "improvements": {
                "name": "Building Improvements Data",
                "sources": [
                    {
                        "name": "Del Norte County Building Department",
                        "url": "https://www.co.del-norte.ca.us/departments/planning-building",
                        "type": "web_scrape",
                        "description": "Building permits and improvements",
                    },
                    {
                        "name": "California Building Standards Commission",
                        "url": "https://www.dgs.ca.gov/bsc",
                        "type": "api",
                        "description": "Statewide building standards data",
                    },
                ],
            },
            "water_rights": {
                "name": "Water Rights Data",
                "sources": [
                    {
                        "name": "California State Water Resources Control Board",
                        "url": "https://www.waterboards.ca.gov/waterrights/board_info/water_rights_programs/",
                        "type": "api",
                        "description": "Water rights and permits database",
                    },
                    {
                        "name": "USGS National Water Information System",
                        "url": "https://waterdata.usgs.gov/nwis",
                        "type": "download",
                        "description": "National water data and statistics",
                    },
                ],
            },
            "ground_water": {
                "name": "Groundwater Data",
                "sources": [
                    {
                        "name": "USGS Groundwater Data",
                        "url": "https://waterdata.usgs.gov/nwis/gw",
                        "type": "download",
                        "description": "Groundwater levels and quality data",
                    },
                    {
                        "name": "California Department of Water Resources",
                        "url": "https://water.ca.gov/Programs/Groundwater-Management",
                        "type": "web_scrape",
                        "description": "State groundwater management data",
                    },
                ],
            },
            "surface_water": {
                "name": "Surface Water Data",
                "sources": [
                    {
                        "name": "USGS Surface Water Data",
                        "url": "https://waterdata.usgs.gov/nwis/sw",
                        "type": "download",
                        "description": "Streamflow and surface water data",
                    },
                    {
                        "name": "California Department of Water Resources",
                        "url": "https://water.ca.gov/Programs/State-Water-Project",
                        "type": "web_scrape",
                        "description": "State water project data",
                    },
                ],
            },
            "power_source": {
                "name": "Power Source Data",
                "sources": [
                    {
                        "name": "California Energy Commission",
                        "url": "https://www.energy.ca.gov/data-reports",
                        "type": "api",
                        "description": "Energy infrastructure and generation data",
                    },
                    {
                        "name": "Pacific Gas and Electric",
                        "url": "https://www.pge.com/en_US/residential/outages/outage-map.page",
                        "type": "web_scrape",
                        "description": "Utility infrastructure data",
                    },
                ],
            },
            "mortgage_debt": {
                "name": "Mortgage Debt Data",
                "sources": [
                    {
                        "name": "Federal Reserve Economic Data",
                        "url": "https://fred.stlouisfed.org/series/MDOAH",
                        "type": "api",
                        "description": "Mortgage debt outstanding data",
                    },
                    {
                        "name": "California Department of Real Estate",
                        "url": "https://www.dre.ca.gov/",
                        "type": "web_scrape",
                        "description": "Real estate and mortgage data",
                    },
                ],
            },
        }

        # Initialize web driver for scraping
        self._setup_web_driver()

        logger.info("Real Data Acquisition system initialized")

    def _setup_web_driver(self):
        """Set up Selenium web driver for web scraping."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("Web driver initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize web driver: {e}")
            self.driver = None

    def acquire_zoning_data(self) -> Path | None:
        """
        Acquire real zoning data from Del Norte County and state sources.

        Returns:
            Path to acquired zoning data file, or None if failed
        """
        self.processing_logger.log_processing_start(
            "Zoning Data Acquisition",
            {"sources": ["Del Norte County", "California State"]},
        )

        try:
            # Try Del Norte County first
            county_data = self._scrape_del_norte_zoning()
            if county_data is not None:
                self.data_logger.log_real_data_acquisition(
                    source_url="https://www.co.del-norte.ca.us/departments/planning-building",
                    file_path=county_data,
                    data_type="Zoning",
                    row_count=len(gpd.read_file(county_data)),
                    file_size_mb=county_data.stat().st_size / 1024 / 1024,
                    geometry_types=["Polygon"],
                    crs="EPSG:4326",
                )
                return county_data

            # Fallback to state data
            state_data = self._download_california_zoning()
            if state_data is not None:
                self.data_logger.log_real_data_acquisition(
                    source_url="https://data.ca.gov/dataset/zoning-data",
                    file_path=state_data,
                    data_type="Zoning",
                    row_count=len(gpd.read_file(state_data)),
                    file_size_mb=state_data.stat().st_size / 1024 / 1024,
                    geometry_types=["Polygon"],
                    crs="EPSG:4326",
                )
                return state_data

            raise FileNotFoundError(
                "No zoning dataset was returned by configured sources"
            )

        except Exception as e:
            raise RuntimeError("Zoning data acquisition failed") from e

    def _scrape_del_norte_zoning(self) -> Path | None:
        """Scrape zoning data from Del Norte County website."""
        if self.driver is None:
            return None

        try:
            self.driver.get(
                "https://www.co.del-norte.ca.us/departments/planning-building"
            )

            # Look for zoning map or data download links
            links = self.driver.find_elements(By.TAG_NAME, "a")
            zoning_links = [link for link in links if "zoning" in link.text.lower()]

            if zoning_links:
                # Download the first zoning data link found
                zoning_links[0].click()
                time.sleep(5)  # Wait for download

                # Check downloads folder for new files
                download_path = self.output_dir / "zoning_data.geojson"
                if download_path.exists():
                    return download_path

            return None

        except Exception as e:
            logger.error(f"Failed to scrape Del Norte zoning data: {e}")
            return None

    def _download_california_zoning(self) -> Path | None:
        """Download California state zoning data from FMMP."""
        try:
            # Try FMMP data first
            fmmp_url = "https://gis.conservation.ca.gov/server/rest/services/DLRP/Farmland_Mapping_and_Monitoring_Program/FeatureServer/0/query"
            params = {
                "where": "COUNTY=15",  # Del Norte County FIPS code
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "geometryPrecision": 6,
            }

            response = requests.get(fmmp_url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if data.get("features"):
                    # Save the FMMP data
                    output_path = self.output_dir / "fmmp_zoning_data.geojson"
                    with open(output_path, "w") as f:
                        json.dump(data, f)

                    # Log the acquisition
                    self.data_logger.log_real_data_acquisition(
                        source_url=fmmp_url,
                        file_path=output_path,
                        data_type="Zoning (FMMP)",
                        row_count=len(data["features"]),
                        file_size_mb=output_path.stat().st_size / 1024 / 1024,
                        geometry_types=["Polygon"],
                        crs="EPSG:4326",
                    )
                    return output_path

            # Fallback to state data portal
            url = "https://data.ca.gov/api/3/action/package_show?id=zoning-data"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "result" in data and "resources" in data["result"]:
                    for resource in data["result"]["resources"]:
                        if resource.get("format", "").lower() in [
                            "geojson",
                            "shp",
                            "zip",
                        ]:
                            download_url = resource["url"]
                            return self._download_file(
                                download_url, "california_zoning"
                            )

            return None

        except Exception as e:
            logger.error(f"Failed to download California zoning data: {e}")
            return None

    def acquire_current_use_data(self) -> Path | None:
        """
        Acquire real current land use data from USDA and state sources.

        Returns:
            Path to acquired current use data file, or None if failed
        """
        self.processing_logger.log_processing_start(
            "Current Use Data Acquisition",
            {
                "sources": [
                    "USDA Cropland Data Layer",
                    "California Department of Conservation",
                ]
            },
        )

        try:
            # Try USDA Cropland Data Layer
            usda_data = self._download_usda_cropland_data()
            if usda_data is not None:
                self.data_logger.log_real_data_acquisition(
                    source_url="https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php",
                    file_path=usda_data,
                    data_type="Current Land Use",
                    row_count=len(gpd.read_file(usda_data)),
                    file_size_mb=usda_data.stat().st_size / 1024 / 1024,
                    geometry_types=["Polygon"],
                    crs="EPSG:4326",
                )
                return usda_data

            # Fallback to California Department of Conservation
            ca_data = self._scrape_california_farmland()
            if ca_data is not None:
                self.data_logger.log_real_data_acquisition(
                    source_url="https://www.conservation.ca.gov/dlrp/fmml",
                    file_path=ca_data,
                    data_type="Current Land Use",
                    row_count=len(gpd.read_file(ca_data)),
                    file_size_mb=ca_data.stat().st_size / 1024 / 1024,
                    geometry_types=["Polygon"],
                    crs="EPSG:4326",
                )
                return ca_data

            raise FileNotFoundError(
                "No current-use dataset was returned by configured sources"
            )

        except Exception as e:
            raise RuntimeError("Current-use data acquisition failed") from e

    def _download_usda_cropland_data(self) -> Path | None:
        """Download USDA Cropland Data Layer for California."""
        try:
            # Try CDL Web Service API first
            cdl_api_url = (
                "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
            )
            params = {"year": "2023", "state": "CA", "format": "geotiff"}  # California

            response = requests.get(cdl_api_url, params=params, timeout=60)
            if response.status_code == 200:
                # Save the CDL data
                output_path = self.output_dir / "usda_cdl_2023_ca.tif"
                with open(output_path, "wb") as f:
                    f.write(response.content)

                # Log the acquisition
                self.data_logger.log_real_data_acquisition(
                    source_url=cdl_api_url,
                    file_path=output_path,
                    data_type="Cropland Data Layer (CDL)",
                    row_count=0,  # Raster data
                    file_size_mb=output_path.stat().st_size / 1024 / 1024,
                    geometry_types=["Raster"],
                    crs="EPSG:4326",
                )
                return output_path

            # Fallback to direct download
            candidate_urls = [
                # California CDL direct download
                "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2023_30m_cdls.zip",
                # National CDL data
                "https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php",
            ]

            for url in candidate_urls:
                path = self._download_file(url, "usda_cropland_data")
                if path is not None and path.exists():
                    return path

            return None
        except Exception as e:
            logger.error(f"Failed to download USDA cropland data: {e}")
            return None

    def _scrape_california_farmland(self) -> Path | None:
        """Scrape farmland data from California Department of Conservation."""
        if self.driver is None:
            return None

        try:
            self.driver.get("https://www.conservation.ca.gov/dlrp/fmml")

            # Look for farmland mapping data
            links = self.driver.find_elements(By.TAG_NAME, "a")
            farmland_links = [link for link in links if "farmland" in link.text.lower()]

            if farmland_links:
                farmland_links[0].click()
                time.sleep(5)

                download_path = self.output_dir / "california_farmland_data.geojson"
                if download_path.exists():
                    return download_path

            return None

        except Exception as e:
            logger.error(f"Failed to scrape California farmland data: {e}")
            return None

    def _download_file(self, url: str, prefix: str) -> Path | None:
        """Download a file from URL and return the local path with caching and progress bar.

        - If a previously extracted geospatial file exists for the prefix, returns it.
        - If a previous download exists and matches Content-Length, skips re-download.
        - Shows a progress bar during download when Content-Length is available.
        """
        try:
            # Fast path: try mounting remote ZIP via GDAL VSI without downloading (vector data)
            if url.lower().endswith(".zip"):
                mounted_path = self._mount_remote_zip_vector(url, save_prefix=prefix)
                if mounted_path is not None:
                    return mounted_path
            # Prefer previously extracted geospatial artifacts
            for pattern in (f"{prefix}*.geojson", f"{prefix}*.shp"):
                matches = sorted(self.output_dir.rglob(pattern))
                if matches:
                    return matches[0]

            # HEAD request to get size/type for caching decision
            head = requests.head(
                url,
                timeout=15,
                allow_redirects=True,
                verify=not self.insecure_downloads,
            )
            content_type = head.headers.get("content-type", "") if head.ok else ""
            total_bytes = int(head.headers.get("content-length", "0")) if head.ok else 0

            # Determine extension from content-type or URL
            ext = ".dat"
            if "zip" in content_type or url.lower().endswith(".zip"):
                ext = ".zip"
            elif "geojson" in content_type or url.lower().endswith(".geojson"):
                ext = ".geojson"
            elif "shp" in content_type or url.lower().endswith(".shp"):
                ext = ".shp"

            output_path = self.output_dir / f"{prefix}{ext}"

            # If already downloaded and size matches, skip
            try:
                if output_path.exists() and total_bytes > 0:
                    # Allow small discrepancy due to server compression/transfer
                    if abs(output_path.stat().st_size - total_bytes) < max(
                        1024 * 1024, int(0.01 * total_bytes)
                    ):
                        return output_path
            except Exception as exc:
                logger.warning(
                    "Best-effort acquisition step failed; continuing: %s", exc
                )

            # Stream download with progress bar
            with requests.get(
                url, timeout=60, stream=True, verify=not self.insecure_downloads
            ) as response:
                response.raise_for_status()
                chunk_size = 1024 * 512  # 512 KB
                from tqdm import tqdm as _tqdm  # type: ignore

                use_pbar = total_bytes > 0
                pbar = (
                    _tqdm(
                        total=total_bytes,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {prefix}",
                    )
                    if use_pbar
                    else None
                )
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))
                if pbar:
                    pbar.close()

            # Restore urllib3 log level

            # If it's a zip file, extract it and return the first geospatial file found
            if ext == ".zip":
                try:
                    with zipfile.ZipFile(output_path, "r") as zip_ref:
                        zip_ref.extractall(self.output_dir)
                    # Prefer GeoJSON, then shapefile; constrain search to current prefix
                    extracted = None
                    for extracted_file in sorted(
                        self.output_dir.rglob(f"{prefix}*.geojson")
                    ):
                        extracted = extracted_file
                        break
                    if not extracted:
                        for extracted_file in sorted(
                            self.output_dir.rglob(f"{prefix}*.shp")
                        ):
                            extracted = extracted_file
                            break
                    if not extracted:
                        # Fallback: any geojson/shp
                        for extracted_file in self.output_dir.rglob("*.geojson"):
                            extracted = extracted_file
                            break
                        if not extracted:
                            for extracted_file in self.output_dir.rglob("*.shp"):
                                extracted = extracted_file
                                break
                    if extracted:
                        return extracted
                    # No vector formats found; return None to trigger alternate source or fallback
                    return None
                except Exception as ee:
                    logger.error(f"Failed to extract zip {output_path}: {ee}")
                    # Do not return the zip path (not a vector); trigger fallback
                    return None

            return output_path

        except Exception as e:
            logger.error(f"Failed to download file from {url}: {e}")
            return None

    def _mount_remote_zip_vector(
        self, url: str, save_prefix: str, inner_name: str | None = None
    ) -> Path | None:
        """Attempt to mount and read a remote ZIP vector dataset via GDAL VSI without downloading.

        Builds a path like '/vsizip/vsicurl/<url>/<inner_name>.shp' and reads it with GeoPandas.
        If successful, saves a local GeoJSON with the given prefix and returns the path.

        Returns None if mounting or reading fails.
        """
        try:
            # Derive inner shapefile name if not provided
            if inner_name is None:
                stem = Path(urlparse(url).path).stem  # e.g., tl_2023_us_place
                inner_name = f"{stem}.shp"
            # Try VSIZIP/VSICURL first
            vsi_path = f"/vsizip/vsicurl/{url}/{inner_name}"
            gdf = None
            try:
                gdf = gpd.read_file(vsi_path)
            except Exception:
                # Attempt to probe the zip for actual inner names
                shp_names = self._vsi_probe_zip_for_vectors(url)
                for shp in shp_names:
                    try:
                        vsi_alt = f"/vsizip/vsicurl/{url}/{shp}"
                        gdf = gpd.read_file(vsi_alt)
                        if gdf is not None and not gdf.empty:
                            inner_name = shp
                            break
                    except Exception:
                        continue
            # Fallback: try zip+https URI style if GDAL vsicurl is unavailable
            if (gdf is None or gdf.empty) and inner_name:
                try:
                    zip_https_uri = f"zip+https://{urlparse(url).netloc}{urlparse(url).path}!{inner_name}"
                    gdf = gpd.read_file(zip_https_uri)
                except Exception:
                    # Try probed names
                    for shp in self._vsi_probe_zip_for_vectors(url):
                        try:
                            zip_https_uri = f"zip+https://{urlparse(url).netloc}{urlparse(url).path}!{shp}"
                            gdf = gpd.read_file(zip_https_uri)
                            if gdf is not None and not gdf.empty:
                                inner_name = shp
                                break
                        except Exception:
                            continue
            if gdf is None or gdf.empty:
                return None
            # Save to local GeoJSON for caching
            out = self.output_dir / f"{save_prefix}.geojson"
            gdf.to_file(out, driver="GeoJSON")
            self.data_logger.log_real_data_acquisition(
                source_url=url,
                file_path=out,
                data_type="Mounted ZIP Vector",
                row_count=len(gdf),
                file_size_mb=out.stat().st_size / 1024 / 1024,
                geometry_types=(
                    list(gdf.geometry.geom_type.unique())
                    if not gdf.empty
                    else ["Unknown"]
                ),
                crs=str(gdf.crs) if gdf.crs else "Unknown",
            )
            return out
        except Exception as e:
            # Soft fail; return None to proceed with other methods
            logger.debug(f"VSI mount failed for {url}: {e}")
            return None

    def _vsi_probe_zip_for_vectors(self, url: str) -> list[str]:
        """Use gdalinfo to list entries inside a remote ZIP and extract candidate vector filenames.

        Returns a list of inner paths (e.g., 'tl_2023_us_place.shp') likely to be vectors.
        """
        try:
            vsi = f"/vsizip/vsicurl/{url}"
            # Run gdalinfo to list contents; suppress stdout on failure
            proc = subprocess.run(
                ["gdalinfo", vsi], capture_output=True, text=True, timeout=30
            )
            text = proc.stdout or ""
            # Extract .shp and .geojson entries
            candidates: list[str] = []
            for line in text.splitlines():
                line = line.strip()
                if line.lower().endswith(".shp") or line.lower().endswith(".geojson"):
                    # Lines may include full paths within zip; take the last token
                    token = line.split()[-1]
                    name = token.split("/")[-1]
                    if name not in candidates:
                        candidates.append(name)
            return candidates
        except Exception:
            return []

    def __del__(self):
        """Clean up web driver on destruction."""
        if hasattr(self, "driver") and self.driver:
            self.driver.quit()

    # --- Additional acquisition helpers using configured source URLs ---
    def acquire_ownership_data(self) -> Path | None:
        """Attempt to acquire ownership data via configured sources; return path or None."""
        try:
            # Try to download Del Norte County parcel data
            parcel_url = "https://services.countyofdelnorte.us/arcgis/rest/services/Parcels/MapServer/0/query"
            params = {
                "where": "1=1",
                "outFields": "OWNER_NAME,ACRES,LAND_USE,IMPROVEMENT_VALUE,LAND_VALUE,APN",
                "returnGeometry": "true",
                "f": "geojson",
                "geometryPrecision": 6,
                "maxRecordCount": 10000,
            }

            response = requests.get(parcel_url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if data.get("features"):
                    output_path = self.output_dir / "del_norte_parcels.geojson"
                    with open(output_path, "w") as f:
                        json.dump(data, f)

                    # Log the acquisition
                    self.data_logger.log_real_data_acquisition(
                        source_url=parcel_url,
                        file_path=output_path,
                        data_type="Parcel Ownership",
                        row_count=len(data["features"]),
                        file_size_mb=output_path.stat().st_size / 1024 / 1024,
                        geometry_types=["Polygon"],
                        crs="EPSG:4326",
                    )
                    return output_path

            # Fallback to existing empirical file
            existing = (
                self.output_dir.parent.parent
                / "ownership"
                / "data"
                / "empirical"
                / "empirical_ownership_data.geojson"
            )
            return existing if existing.exists() else None
        except Exception as e:
            logger.error(f"Failed to acquire ownership data: {e}")
            # Fallback to existing empirical file
            existing = (
                self.output_dir.parent.parent
                / "ownership"
                / "data"
                / "empirical"
                / "empirical_ownership_data.geojson"
            )
            return existing if existing.exists() else None

    def acquire_improvements_data(self) -> Path | None:
        """Attempt to acquire improvements data via configured sources; return path or None."""
        try:
            # Try to download Del Norte County building/improvement data
            improvements_url = "https://services.countyofdelnorte.us/arcgis/rest/services/BuildingPermits/MapServer/0/query"
            params = {
                "where": "1=1",
                "outFields": "IMPROVEMENT_VALUE,LAND_VALUE,BUILDING_TYPE,YEAR_BUILT,APN",
                "returnGeometry": "true",
                "f": "geojson",
                "geometryPrecision": 6,
                "maxRecordCount": 10000,
            }

            response = requests.get(improvements_url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if data.get("features"):
                    output_path = self.output_dir / "del_norte_improvements.geojson"
                    with open(output_path, "w") as f:
                        json.dump(data, f)

                    # Log the acquisition
                    self.data_logger.log_real_data_acquisition(
                        source_url=improvements_url,
                        file_path=output_path,
                        data_type="Building Improvements",
                        row_count=len(data["features"]),
                        file_size_mb=output_path.stat().st_size / 1024 / 1024,
                        geometry_types=["Polygon"],
                        crs="EPSG:4326",
                    )
                    return output_path

            # Fallback to existing empirical file
            existing = (
                self.output_dir.parent.parent
                / "improvements"
                / "data"
                / "empirical"
                / "empirical_improvements_data.geojson"
            )
            return existing if existing.exists() else None
        except Exception as e:
            logger.error(f"Failed to acquire improvements data: {e}")
            # Fallback to existing empirical file
            existing = (
                self.output_dir.parent.parent
                / "improvements"
                / "data"
                / "empirical"
                / "empirical_improvements_data.geojson"
            )
            return existing if existing.exists() else None


def create_real_data_acquisition(output_dir: Path) -> RealDataAcquisition:
    """
    Create a real data acquisition instance.

    Args:
        output_dir: Directory to store acquired data

    Returns:
        RealDataAcquisition instance
    """
    return RealDataAcquisition(output_dir)
