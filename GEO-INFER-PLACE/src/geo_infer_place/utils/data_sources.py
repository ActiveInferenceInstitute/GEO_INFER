"""
CaliforniaDataSources: Comprehensive data source management for California.

This module provides standardized access to California-specific datasets
including government APIs, research databases, and environmental monitoring
networks. It implements data source discovery, access patterns, and
integration protocols for real California data.
"""

import logging
from datetime import datetime
from typing import Any
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a data source."""

    name: str
    description: str
    base_url: str
    api_key_required: bool
    data_types: list[str]
    spatial_coverage: str
    temporal_coverage: str
    update_frequency: str
    format_types: list[str]
    access_method: str  # 'api', 'download', 'ftp', 'wms'
    documentation_url: str
    license_type: str


class CaliforniaDataSources:
    """
    Comprehensive data source catalog for California geospatial datasets.

    This class provides standardized access to California-specific data sources
    including government agencies, research institutions, and environmental
    monitoring networks. It implements data source discovery, validation,
    and access pattern management.

    Data Categories:
    - Fire and forestry (CAL FIRE, USFS)
    - Coastal and marine (NOAA, Coastal Commission)
    - Climate and weather (DWR, CDEC, NOAA)
    - Water resources (USGS, DWR, SWRCB)
    - Environmental monitoring (CalEPA, Air Districts)
    - Demographics and economics (DOF, EDD, Census)
    - Land use and planning (OPR, HCD, Caltrans)

    Example Usage:
        >>> sources = CaliforniaDataSources()
        >>> fire_sources = sources.get_sources_by_category('fire')
        >>> calfire_api = sources.get_source_config('calfire_fire_perimeters')
        >>> available_data = sources.discover_available_data('del_norte_county')
    """

    def __init__(self) -> None:
        """Initialize California data sources catalog."""
        self.sources = self._initialize_data_sources()
        self.source_categories = self._categorize_sources()

        logger.info(
            f"CaliforniaDataSources initialized with {len(self.sources)} data sources"
        )

    def _initialize_data_sources(self) -> dict[str, DataSource]:
        """Initialize comprehensive catalog of California data sources."""
        sources = {}

        # === FIRE AND FORESTRY DATA SOURCES ===

        sources["calfire_fire_perimeters"] = DataSource(
            name="CAL FIRE Fire Perimeters",
            description="Historical and current fire perimeter data from CAL FIRE",
            base_url="https://calfire-forestry.maps.arcgis.com/home/index.html",
            api_key_required=False,
            data_types=["fire_perimeters", "fire_incidents", "fire_history"],
            spatial_coverage="California statewide",
            temporal_coverage="1950-present",
            update_frequency="daily",
            format_types=["geojson", "shapefile", "kml"],
            access_method="api",
            documentation_url="https://calfire-forestry.maps.arcgis.com/apps/webappviewer/index.html",
            license_type="public_domain",
        )

        sources["calfire_frap"] = DataSource(
            name="CAL FIRE FRAP (Fire and Resource Assessment Program)",
            description="Fire hazard severity zones, forest land ownership, timber harvest plans",
            base_url="https://frap.fire.ca.gov/",
            api_key_required=False,
            data_types=[
                "fire_hazard_zones",
                "forest_ownership",
                "timber_harvest_plans",
            ],
            spatial_coverage="California statewide",
            temporal_coverage="varies by dataset",
            update_frequency="annually",
            format_types=["shapefile", "geojson", "pdf"],
            access_method="download",
            documentation_url="https://frap.fire.ca.gov/mapping/gis-data/",
            license_type="public_domain",
        )

        sources["calfire_weather_stations"] = DataSource(
            name="CAL FIRE Remote Automated Weather Stations (RAWS)",
            description="Real-time fire weather monitoring stations",
            base_url="https://raws.dri.edu/",
            api_key_required=False,
            data_types=["weather_data", "fire_weather_indices", "fuel_moisture"],
            spatial_coverage="California statewide",
            temporal_coverage="1990-present",
            update_frequency="hourly",
            format_types=["csv", "json", "xml"],
            access_method="api",
            documentation_url="https://raws.dri.edu/rtaws/",
            license_type="public_domain",
        )

        # === COASTAL AND MARINE DATA SOURCES ===

        sources["noaa_tide_gauges"] = DataSource(
            name="NOAA Tide Gauges",
            description="Real-time and historical tide gauge data",
            base_url="https://tidesandcurrents.noaa.gov/",
            api_key_required=False,
            data_types=["water_levels", "tidal_predictions", "meteorological_data"],
            spatial_coverage="California coast",
            temporal_coverage="1850-present",
            update_frequency="6-minute intervals",
            format_types=["json", "csv", "xml"],
            access_method="api",
            documentation_url="https://tidesandcurrents.noaa.gov/api/",
            license_type="public_domain",
        )

        sources["noaa_coastal_change"] = DataSource(
            name="NOAA Coastal Change Analysis Program",
            description="Coastal land cover change analysis",
            base_url="https://coast.noaa.gov/ccapatlas/",
            api_key_required=False,
            data_types=["land_cover", "land_cover_change", "impervious_surfaces"],
            spatial_coverage="California coast",
            temporal_coverage="1975-present",
            update_frequency="5-year cycles",
            format_types=["geotiff", "shapefile"],
            access_method="download",
            documentation_url="https://coast.noaa.gov/ccapatlas/",
            license_type="public_domain",
        )

        sources["ca_coastal_commission"] = DataSource(
            name="California Coastal Commission",
            description="Coastal development permits, LCP documents, coastal access",
            base_url="https://www.coastal.ca.gov/",
            api_key_required=False,
            data_types=["coastal_permits", "local_coastal_programs", "public_access"],
            spatial_coverage="California coastal zone",
            temporal_coverage="1976-present",
            update_frequency="monthly",
            format_types=["pdf", "shapefile", "kml"],
            access_method="download",
            documentation_url="https://www.coastal.ca.gov/gis/",
            license_type="public_domain",
        )

        # === CLIMATE AND WEATHER DATA SOURCES ===

        sources["cdec_stations"] = DataSource(
            name="California Data Exchange Center (CDEC)",
            description="Real-time hydrologic and climate data from California",
            base_url="https://cdec.water.ca.gov/",
            api_key_required=False,
            data_types=[
                "streamflow",
                "reservoir_levels",
                "precipitation",
                "temperature",
            ],
            spatial_coverage="California statewide",
            temporal_coverage="1960-present",
            update_frequency="hourly",
            format_types=["csv", "json"],
            access_method="api",
            documentation_url="https://cdec.water.ca.gov/dynamicapp/queryCSV",
            license_type="public_domain",
        )

        sources["cimis_weather"] = DataSource(
            name="California Irrigation Management Information System (CIMIS)",
            description="Weather data for agricultural irrigation management",
            base_url="https://cimis.water.ca.gov/",
            api_key_required=True,
            data_types=["weather_data", "evapotranspiration", "soil_temperature"],
            spatial_coverage="California agricultural areas",
            temporal_coverage="1983-present",
            update_frequency="hourly",
            format_types=["json", "csv"],
            access_method="api",
            documentation_url="https://cimis.water.ca.gov/WSNReportCriteria.aspx",
            license_type="public_domain",
        )

        sources["prism_climate"] = DataSource(
            name="PRISM Climate Group",
            description="High-resolution climate data for the United States",
            base_url="https://prism.oregonstate.edu/",
            api_key_required=False,
            data_types=["temperature", "precipitation", "dewpoint", "vapor_pressure"],
            spatial_coverage="United States (800m resolution)",
            temporal_coverage="1895-present",
            update_frequency="monthly",
            format_types=["bil", "ascii", "netcdf"],
            access_method="download",
            documentation_url="https://prism.oregonstate.edu/documents/",
            license_type="attribution_required",
        )

        # === WATER RESOURCES DATA SOURCES ===

        sources["usgs_water_data"] = DataSource(
            name="USGS National Water Information System",
            description="Surface and groundwater monitoring data",
            base_url="https://waterdata.usgs.gov/nwis",
            api_key_required=False,
            data_types=["streamflow", "groundwater_levels", "water_quality"],
            spatial_coverage="United States",
            temporal_coverage="1900-present",
            update_frequency="15-minute to daily",
            format_types=["json", "csv", "xml"],
            access_method="api",
            documentation_url="https://waterservices.usgs.gov/",
            license_type="public_domain",
        )

        sources["ca_groundwater"] = DataSource(
            name="California Statewide Groundwater Elevation Monitoring",
            description="Groundwater elevation monitoring network",
            base_url="https://sgma.water.ca.gov/",
            api_key_required=False,
            data_types=["groundwater_elevations", "well_information"],
            spatial_coverage="California statewide",
            temporal_coverage="1950-present",
            update_frequency="quarterly",
            format_types=["csv", "shapefile"],
            access_method="download",
            documentation_url="https://sgma.water.ca.gov/casgem/",
            license_type="public_domain",
        )

        # === ENVIRONMENTAL MONITORING ===

        sources["ca_air_quality"] = DataSource(
            name="California Air Resources Board Air Quality Data",
            description="Air quality monitoring station data",
            base_url="https://www.arb.ca.gov/aqmis2/aqmis2.php",
            api_key_required=False,
            data_types=["air_quality_index", "pollutant_concentrations", "meteorology"],
            spatial_coverage="California statewide",
            temporal_coverage="1980-present",
            update_frequency="hourly",
            format_types=["csv", "excel"],
            access_method="download",
            documentation_url="https://www.arb.ca.gov/aqmis2/aqmis2.php",
            license_type="public_domain",
        )

        sources["purple_air"] = DataSource(
            name="PurpleAir Sensor Network",
            description="Real-time air quality monitoring from citizen sensors",
            base_url="https://www.purpleair.com/",
            api_key_required=True,
            data_types=["pm25", "pm10", "temperature", "humidity"],
            spatial_coverage="Global (dense coverage in California)",
            temporal_coverage="2015-present",
            update_frequency="2-minute intervals",
            format_types=["json"],
            access_method="api",
            documentation_url="https://api.purpleair.com/",
            license_type="commercial_license",
        )

        # === DEMOGRAPHICS AND ECONOMICS ===

        sources["ca_dept_finance"] = DataSource(
            name="California Department of Finance Demographics",
            description="Population estimates, projections, and demographics",
            base_url="https://dof.ca.gov/",
            api_key_required=False,
            data_types=[
                "population_estimates",
                "demographic_projections",
                "housing_data",
            ],
            spatial_coverage="California statewide",
            temporal_coverage="1970-present",
            update_frequency="annually",
            format_types=["excel", "csv", "pdf"],
            access_method="download",
            documentation_url="https://dof.ca.gov/Reports/",
            license_type="public_domain",
        )

        sources["us_census_acs"] = DataSource(
            name="US Census American Community Survey",
            description="Detailed demographic and economic data",
            base_url="https://api.census.gov/",
            api_key_required=True,
            data_types=["demographics", "economics", "housing", "transportation"],
            spatial_coverage="United States",
            temporal_coverage="2005-present",
            update_frequency="annually",
            format_types=["json", "csv"],
            access_method="api",
            documentation_url="https://www.census.gov/data/developers/data-sets/acs-1year.html",
            license_type="public_domain",
        )

        # === LAND USE AND GEOSPATIAL ===

        sources["ca_farmland_mapping"] = DataSource(
            name="California Farmland Mapping and Monitoring Program",
            description="Agricultural land use mapping and change analysis",
            base_url="https://www.conservation.ca.gov/dlrp/fmmp",
            api_key_required=False,
            data_types=["farmland_mapping", "land_use_change"],
            spatial_coverage="California statewide",
            temporal_coverage="1984-present",
            update_frequency="biennially",
            format_types=["shapefile", "geotiff"],
            access_method="download",
            documentation_url="https://www.conservation.ca.gov/dlrp/fmmp/Pages/DownloadGISdata.aspx",
            license_type="public_domain",
        )

        sources["ca_protected_areas"] = DataSource(
            name="California Protected Areas Database",
            description="Comprehensive database of protected lands",
            base_url="https://www.calands.org/",
            api_key_required=False,
            data_types=["protected_areas", "land_ownership", "conservation_status"],
            spatial_coverage="California statewide",
            temporal_coverage="current",
            update_frequency="annually",
            format_types=["shapefile", "kml", "geojson"],
            access_method="download",
            documentation_url="https://www.calands.org/data",
            license_type="public_domain",
        )

        return sources

    def _categorize_sources(self) -> dict[str, list[str]]:
        """Organize data sources by category."""
        categories: dict[str, list[str]] = {
            "fire": [],
            "forestry": [],
            "coastal": [],
            "marine": [],
            "climate": [],
            "weather": [],
            "water": [],
            "air_quality": [],
            "demographics": [],
            "economics": [],
            "land_use": [],
            "environmental": [],
            "infrastructure": [],
        }

        # Categorize sources based on data types
        for source_id, source in self.sources.items():
            for data_type in source.data_types:
                if "fire" in data_type:
                    categories["fire"].append(source_id)
                if "forest" in data_type or "timber" in data_type:
                    categories["forestry"].append(source_id)
                if "coastal" in data_type or "tide" in data_type:
                    categories["coastal"].append(source_id)
                if "marine" in data_type or "ocean" in data_type:
                    categories["marine"].append(source_id)
                if (
                    "climate" in data_type
                    or "temperature" in data_type
                    or "precipitation" in data_type
                ):
                    categories["climate"].append(source_id)
                if "weather" in data_type:
                    categories["weather"].append(source_id)
                if (
                    "water" in data_type
                    or "streamflow" in data_type
                    or "groundwater" in data_type
                ):
                    categories["water"].append(source_id)
                if "air_quality" in data_type or "pollutant" in data_type:
                    categories["air_quality"].append(source_id)
                if "population" in data_type or "demographic" in data_type:
                    categories["demographics"].append(source_id)
                if "economic" in data_type or "housing" in data_type:
                    categories["economics"].append(source_id)
                if "land_use" in data_type or "land_cover" in data_type:
                    categories["land_use"].append(source_id)

        return categories

    def get_sources_by_category(self, category: str) -> list[DataSource]:
        """
        Get all data sources in a specific category.

        Args:
            category: Category name

        Returns:
            List of DataSource objects in the category
        """
        source_ids = self.source_categories.get(category, [])
        return [self.sources[source_id] for source_id in source_ids]

    def get_source_config(self, source_id: str) -> DataSource | None:
        """
        Get configuration for a specific data source.

        Args:
            source_id: Identifier for the data source

        Returns:
            DataSource object or None if not found
        """
        source = self.sources.get(source_id)
        if not source:
            logger.warning(f"Data source not found: {source_id}")
            return None

        logger.debug(f"Retrieved configuration for data source: {source_id}")
        return source

    def search_sources(
        self,
        query: str,
        categories: list[str] | None = None,
        data_types: list[str] | None = None,
    ) -> list[DataSource]:
        """
        Search data sources by query and filters.

        Args:
            query: Search query string
            categories: Optional list of categories to filter by
            data_types: Optional list of data types to filter by

        Returns:
            List of matching DataSource objects
        """
        query_lower = query.lower()
        results = []

        for source_id, source in self.sources.items():
            # Check if query matches name or description
            name_match = query_lower in source.name.lower()
            desc_match = query_lower in source.description.lower()

            if not (name_match or desc_match):
                continue

            # Apply category filter
            if categories:
                source_categories = []
                for category, source_ids in self.source_categories.items():
                    if source_id in source_ids:
                        source_categories.append(category)

                if not any(cat in categories for cat in source_categories):
                    continue

            # Apply data type filter
            if data_types:
                if not any(dt in source.data_types for dt in data_types):
                    continue

            results.append(source)

        logger.info(f"Search for '{query}' returned {len(results)} sources")
        return results

    def get_sources_for_location(
        self,
        location_bounds: tuple[float, float, float, float],
        location_name: str | None = None,
    ) -> dict[str, list[DataSource]]:
        """
        Get relevant data sources for a specific location.

        Args:
            location_bounds: Bounding box as (west, south, east, north)
            location_name: Optional location name for context

        Returns:
            Dictionary of categorized data sources relevant to the location
        """
        west, south, east, north = location_bounds

        # Determine if location is in California
        ca_bounds = (-124.482003, 32.528832, -114.131211, 42.009518)
        ca_west, ca_south, ca_east, ca_north = ca_bounds

        in_california = (
            west >= ca_west
            and east <= ca_east
            and south >= ca_south
            and north <= ca_north
        )

        # Determine if location is coastal (within 50km of coast)
        is_coastal = west <= -117.0  # Rough coastal boundary for California

        relevant_sources: dict[str, list[DataSource]] = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
        }

        for source in self.sources.values():
            priority = "low_priority"

            # California-specific sources get high priority for CA locations
            if in_california and "california" in source.name.lower():
                priority = "high_priority"
            elif in_california and any(
                ca_term in source.description.lower()
                for ca_term in ["california", "ca ", "cal "]
            ):
                priority = "high_priority"

            # Coastal sources for coastal locations
            elif is_coastal and any(
                coastal_term in source.description.lower()
                for coastal_term in ["coastal", "tide", "marine", "ocean"]
            ):
                priority = "medium_priority" if priority == "low_priority" else priority

            # NOAA sources are generally relevant for US locations
            elif "noaa" in source.name.lower():
                priority = "medium_priority" if priority == "low_priority" else priority

            # USGS sources are relevant for US locations
            elif "usgs" in source.name.lower():
                priority = "medium_priority" if priority == "low_priority" else priority

            relevant_sources[priority].append(source)

        # Log summary
        high_count = len(relevant_sources["high_priority"])
        medium_count = len(relevant_sources["medium_priority"])
        low_count = len(relevant_sources["low_priority"])

        logger.info(
            f"Found {high_count} high-priority, {medium_count} medium-priority, "
            f"and {low_count} low-priority sources for location {location_name or 'unnamed'}"
        )

        return relevant_sources

    def validate_source_access(
        self, source_id: str, api_key: str | None = None
    ) -> dict[str, Any]:
        """
        Validate access to a data source.

        Args:
            source_id: Identifier for the data source
            api_key: Optional API key for sources that require authentication

        Returns:
            Dictionary with validation results
        """
        source = self.get_source_config(source_id)
        if not source:
            return {
                "success": False,
                "error": f"Data source not found: {source_id}",
                "accessible": False,
            }

        result = {
            "source_id": source_id,
            "source_name": source.name,
            "api_key_required": source.api_key_required,
            "access_method": source.access_method,
            "success": False,
            "accessible": False,
            "error": None,
            "response_time_ms": None,
        }

        try:
            import time

            start_time = time.time()

            # Test basic connectivity
            if source.access_method == "api":
                # Try a simple GET request to the base URL
                response = requests.get(source.base_url, timeout=10)
                result["response_time_ms"] = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    result["success"] = True
                    result["accessible"] = True
                elif (
                    response.status_code == 401
                    and source.api_key_required
                    and not api_key
                ):
                    result["error"] = "API key required but not provided"
                    result["accessible"] = False
                elif response.status_code == 403:
                    result["error"] = "Access forbidden - check API key or permissions"
                    result["accessible"] = False
                else:
                    result["error"] = f"HTTP {response.status_code}: {response.reason}"
                    result["accessible"] = False

            elif source.access_method == "download":
                # For download sources, just check if the URL is reachable
                response = requests.head(source.base_url, timeout=10)
                result["response_time_ms"] = int((time.time() - start_time) * 1000)

                if response.status_code in [200, 301, 302]:
                    result["success"] = True
                    result["accessible"] = True
                else:
                    result["error"] = f"HTTP {response.status_code}: {response.reason}"
                    result["accessible"] = False

            else:
                result["error"] = (
                    f"Access method {source.access_method} not supported for validation"
                )
                result["accessible"] = False

        except requests.exceptions.Timeout:
            result["error"] = "Request timeout - source may be slow or unavailable"
            result["accessible"] = False
        except requests.exceptions.ConnectionError:
            result["error"] = "Connection error - source may be unavailable"
            result["accessible"] = False
        except Exception as e:
            result["error"] = f"Validation error: {str(e)}"
            result["accessible"] = False

        logger.info(
            f"Validation for {source_id}: {'Success' if result['success'] else 'Failed'}"
        )
        return result

    def get_update_schedule(self) -> dict[str, list[str]]:
        """
        Get data sources organized by their update frequency.

        Returns:
            Dictionary with update frequencies as keys and source lists as values
        """
        schedule: dict[str, list[str]] = {}

        for source_id, source in self.sources.items():
            frequency = source.update_frequency
            if frequency not in schedule:
                schedule[frequency] = []
            schedule[frequency].append(source_id)

        # Sort by frequency for better organization
        frequency_order = [
            "2-minute intervals",
            "6-minute intervals",
            "15-minute intervals",
            "hourly",
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "annually",
            "biennially",
            "varies by dataset",
        ]

        ordered_schedule: dict[str, list[str]] = {}
        for freq in frequency_order:
            if freq in schedule:
                ordered_schedule[freq] = schedule[freq]

        # Add any remaining frequencies not in the predefined order
        for freq in schedule:
            if freq not in ordered_schedule:
                ordered_schedule[freq] = schedule[freq]

        return ordered_schedule

    def get_source_summary(self) -> dict[str, Any]:
        """
        Get a summary of all available data sources.

        Returns:
            Dictionary with summary statistics and information
        """
        total_sources = len(self.sources)

        # Count by access method
        access_methods: dict[str, int] = {}
        for source in self.sources.values():
            method = source.access_method
            access_methods[method] = access_methods.get(method, 0) + 1

        # Count API key requirements
        api_key_required = sum(
            1 for source in self.sources.values() if source.api_key_required
        )

        # Count by license type
        license_types: dict[str, int] = {}
        for source in self.sources.values():
            license_type = source.license_type
            license_types[license_type] = license_types.get(license_type, 0) + 1

        # Count by category
        category_counts = {
            cat: len(sources) for cat, sources in self.source_categories.items()
        }

        # Data type coverage
        all_data_types = set()
        for source in self.sources.values():
            all_data_types.update(source.data_types)

        summary = {
            "total_sources": total_sources,
            "access_methods": access_methods,
            "api_key_required": api_key_required,
            "api_key_not_required": total_sources - api_key_required,
            "license_types": license_types,
            "categories": category_counts,
            "unique_data_types": len(all_data_types),
            "data_types": sorted(list(all_data_types)),
            "last_updated": datetime.now().isoformat(),
        }

        logger.info(f"Generated summary for {total_sources} data sources")
        return summary
