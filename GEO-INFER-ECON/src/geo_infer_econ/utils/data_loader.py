"""
Comprehensive Data Loading Utilities for Economic Analysis

This module provides sophisticated data loading, validation, and preprocessing
capabilities for economic datasets, including spatial economic data, time series,
and various data formats used in econometric analysis.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import requests
from typing import Any
from pathlib import Path
import logging
from dataclasses import dataclass


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""

    name: str
    source_type: str  # 'file', 'api', 'database', 'web_service'
    format: str  # 'csv', 'json', 'geojson', 'xlsx', 'api'
    location: str  # file path or URL
    parameters: dict[str, Any] | None = None
    authentication: dict[str, str] | None = None
    cache_settings: dict[str, Any] | None = None


@dataclass
class DataValidationResult:
    """Results of data validation"""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]


class EconomicDataLoader:
    """
    Comprehensive data loading system for economic analysis.

    Supports multiple data sources, formats, and provides validation,
    preprocessing, and caching capabilities.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the Economic Data Loader.

        Args:
            config: Configuration dictionary for data loading settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.data_sources: dict[str, DataSourceConfig] = {}
        self.cache: dict[str, Any] = {}
        self.validation_rules: dict[str, Any] = {}

        # Setup default configurations
        self._setup_default_configs()

    def _setup_default_configs(self) -> None:
        """Setup default configurations for data loading."""
        self.default_cache_dir = Path(self.config.get("cache_dir", "./cache"))
        self.default_cache_dir.mkdir(exist_ok=True)

        # Default validation rules
        self.validation_rules = {
            "economic_indicators": {
                "required_columns": ["region_id", "year", "value"],
                "data_types": {"region_id": "str", "year": "int", "value": "float"},
                "value_ranges": {"year": (1900, 2100), "value": (-1e12, 1e12)},
            },
            "regional_data": {
                "required_columns": ["region_id", "region_name"],
                "geometry_required": True,
                "crs_check": True,
            },
        }

    def register_data_source(self, config: DataSourceConfig) -> None:
        """
        Register a data source for loading.

        Args:
            config: Data source configuration
        """
        self.data_sources[config.name] = config
        self.logger.info(f"Registered data source: {config.name}")

    def load_economic_data(
        self,
        source_name: str,
        filters: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load economic data from a registered source.

        Args:
            source_name: Name of the registered data source
            filters: Optional filters to apply to the data
            validate: Whether to validate the loaded data

        Returns:
            DataFrame with loaded economic data
        """
        if source_name not in self.data_sources:
            raise ValueError(f"Data source '{source_name}' not registered")

        config = self.data_sources[source_name]
        data = self._load_from_source(config)

        if filters:
            data = self._apply_filters(data, filters)

        if validate:
            validation_result = self.validate_economic_data(data, source_name)
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Data validation failed for {source_name}: "
                    f"{validation_result.errors}"
                )

        return data

    def load_regional_data(
        self, source_name: str, geometry_column: str = "geometry"
    ) -> gpd.GeoDataFrame:
        """
        Load regional/spatial economic data.

        Args:
            source_name: Name of the data source
            geometry_column: Name of the geometry column

        Returns:
            GeoDataFrame with regional data
        """
        data = self.load_economic_data(source_name)
        return self._convert_to_geodataframe(data, geometry_column)

    def _load_from_source(self, config: DataSourceConfig) -> pd.DataFrame:
        """Load data from a specific source configuration."""
        cache_key = self._get_cache_key(config)

        # Check cache first
        if self._is_cache_valid(cache_key, config):
            self.logger.info(f"Loading {config.name} from cache")
            return self.cache[cache_key]

        # Load based on source type
        if config.source_type == "file":
            data = self._load_from_file(config)
        elif config.source_type == "api":
            data = self._load_from_api(config)
        elif config.source_type == "web_service":
            data = self._load_from_web_service(config)
        else:
            raise ValueError(f"Unsupported source type: {config.source_type}")

        # Cache the data
        self.cache[cache_key] = data
        self.logger.info(f"Cached data for {config.name}")

        return data

    def _load_from_file(self, config: DataSourceConfig) -> pd.DataFrame:
        """Load data from a file."""
        file_path = Path(config.location)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load based on file format
        if config.format == "csv":
            data = pd.read_csv(file_path, **config.parameters or {})
        elif config.format == "json":
            data = pd.read_json(file_path, **config.parameters or {})
        elif config.format == "xlsx":
            data = pd.read_excel(file_path, **config.parameters or {})
        elif config.format == "geojson":
            gdf = gpd.read_file(file_path, **config.parameters or {})
            return pd.DataFrame(gdf.drop(columns=["geometry"]))
        else:
            raise ValueError(f"Unsupported file format: {config.format}")

        return data

    def _load_from_api(self, config: DataSourceConfig) -> pd.DataFrame:
        """Load data from an API endpoint."""
        headers = config.authentication or {}
        params = config.parameters or {}

        response = requests.get(
            config.location, headers=headers, params=params, timeout=30
        )

        if response.status_code != 200:
            raise ConnectionError(f"API request failed: {response.status_code}")

        if config.format == "json":
            data = pd.DataFrame(response.json())
        else:
            # Assume CSV format if not specified
            data = pd.read_csv(pd.io.common.StringIO(response.text))

        return data

    def _load_from_web_service(self, config: DataSourceConfig) -> pd.DataFrame:
        """Load data from a web service (World Bank, OECD, etc.) via HTTP API."""
        # Delegates to _load_from_api; extend for service-specific response parsing
        return self._load_from_api(config)

    def _apply_filters(
        self, data: pd.DataFrame, filters: dict[str, Any]
    ) -> pd.DataFrame:
        """Apply filters to the loaded data."""
        filtered_data = data.copy()

        for column, filter_value in filters.items():
            if column in filtered_data.columns:
                if isinstance(filter_value, list):
                    filtered_data = filtered_data[
                        filtered_data[column].isin(filter_value)
                    ]
                elif isinstance(filter_value, tuple):
                    # Range filter
                    filtered_data = filtered_data[
                        (filtered_data[column] >= filter_value[0])
                        & (filtered_data[column] <= filter_value[1])
                    ]
                else:
                    filtered_data = filtered_data[filtered_data[column] == filter_value]

        return filtered_data

    def _convert_to_geodataframe(
        self, data: pd.DataFrame, geometry_column: str
    ) -> gpd.GeoDataFrame:
        """Convert DataFrame to GeoDataFrame."""
        if geometry_column not in data.columns:
            raise ValueError(f"Geometry column '{geometry_column}' not found in data")

        # Extract geometry data and convert to GeoDataFrame
        geometry_data = data[geometry_column]
        df_without_geometry = data.drop(columns=[geometry_column])

        # Handle different geometry formats
        if geometry_data.dtype == "object":
            # Assume WKT or GeoJSON format
            try:
                gdf = gpd.GeoDataFrame(
                    df_without_geometry, geometry=gpd.GeoSeries.from_wkt(geometry_data)
                )
            except Exception:
                # Try GeoJSON
                import json

                geometries = []
                for geom_str in geometry_data:
                    if isinstance(geom_str, str):
                        try:
                            geom_dict = json.loads(geom_str)
                            geometries.append(geom_dict)
                        except Exception:
                            geometries.append(None)
                    else:
                        geometries.append(geom_str)

                gdf = gpd.GeoDataFrame(df_without_geometry, geometry=geometries)
        else:
            # Assume shapely geometries
            gdf = gpd.GeoDataFrame(df_without_geometry, geometry=geometry_data)

        return gdf

    def _get_cache_key(self, config: DataSourceConfig) -> str:
        """Generate cache key for a data source."""
        key_parts = [
            config.name,
            config.location,
            str(config.parameters),
            str(config.format),
        ]
        return "_".join(key_parts).replace("/", "_").replace("\\", "_")

    def _is_cache_valid(self, cache_key: str, config: DataSourceConfig) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False

        cache_settings = config.cache_settings or {}
        max_age = cache_settings.get("max_age_hours", 24)

        # Check if cache file exists and is recent enough
        cache_file = self.default_cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            import time

            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            return bool(age_hours < max_age)

        return False

    def validate_economic_data(
        self, data: pd.DataFrame, source_name: str = "unknown"
    ) -> DataValidationResult:
        """
        Validate economic data according to predefined rules.

        Args:
            data: DataFrame to validate
            source_name: Name of the data source for context

        Returns:
            Validation results
        """
        errors = []
        warnings = []
        summary = {}

        # Check for required columns
        if source_name in self.validation_rules:
            rules = self.validation_rules[source_name]

            # Required columns check
            required_cols = rules.get("required_columns", [])
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")

            # Data type checks
            data_types = rules.get("data_types", {})
            for col, expected_type in data_types.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if not self._check_data_type_compatibility(
                        actual_type, expected_type
                    ):
                        warnings.append(
                            f"Column '{col}' has type {actual_type}, expected {expected_type}"
                        )

            # Value range checks
            value_ranges = rules.get("value_ranges", {})
            for col, (min_val, max_val) in value_ranges.items():
                if col in data.columns:
                    out_of_range = data[(data[col] < min_val) | (data[col] > max_val)]
                    if not out_of_range.empty:
                        warnings.append(
                            f"Column '{col}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]"
                        )

        # General data quality checks
        summary["total_rows"] = len(data)
        summary["total_columns"] = len(data.columns)
        summary["missing_values"] = data.isnull().sum().sum()
        summary["duplicate_rows"] = data.duplicated().sum()

        if summary["missing_values"] > 0:
            warnings.append(f"Found {summary['missing_values']} missing values")

        if summary["duplicate_rows"] > 0:
            warnings.append(f"Found {summary['duplicate_rows']} duplicate rows")

        # Check for common economic data issues
        if "gdp" in [str(c).lower() for c in data.columns]:
            negative_gdp = (data["gdp"] < 0).sum()
            if negative_gdp > 0:
                errors.append(f"Found {negative_gdp} negative GDP values")

        return DataValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, summary=summary
        )

    def _check_data_type_compatibility(
        self, actual_type: str, expected_type: str
    ) -> bool:
        """Check if actual data type is compatible with expected type."""
        type_mapping = {
            "int": ["int64", "int32", "int16", "int8"],
            "float": ["float64", "float32"],
            "str": ["object", "string"],
            "datetime": ["datetime64[ns]"],
        }

        if expected_type in type_mapping:
            return actual_type in type_mapping[expected_type]

        return actual_type == expected_type

    def preprocess_economic_data(
        self, data: pd.DataFrame, preprocessing_steps: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Preprocess economic data with common cleaning and transformation steps.

        Args:
            data: DataFrame to preprocess
            preprocessing_steps: List of preprocessing steps to apply

        Returns:
            Preprocessed DataFrame
        """
        if preprocessing_steps is None:
            preprocessing_steps = [
                "remove_nulls",
                "remove_duplicates",
                "standardize_names",
            ]

        processed_data = data.copy()

        for step in preprocessing_steps:
            if step == "remove_nulls":
                processed_data = self._remove_null_values(processed_data)
            elif step == "remove_duplicates":
                processed_data = processed_data.drop_duplicates()
            elif step == "standardize_names":
                processed_data = self._standardize_column_names(processed_data)
            elif step == "handle_outliers":
                processed_data = self._handle_outliers(processed_data)
            elif step == "normalize_values":
                processed_data = self._normalize_values(processed_data)

        return processed_data

    def _remove_null_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove or handle null values in the data."""
        # For now, just drop rows with any null values
        # In practice, this could be more sophisticated
        return data.dropna()

    def _standardize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        data.columns = [
            col.lower().replace(" ", "_").replace("-", "_") for col in data.columns
        ]
        return data

    def _handle_outliers(self, data: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers
                data[col] = np.clip(data[col], lower_bound, upper_bound)

        return data

    def _normalize_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric values to [0, 1] range."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            min_val = data[col].min()
            max_val = data[col].max()

            if max_val != min_val:
                data[col] = (data[col] - min_val) / (max_val - min_val)

        return data

    def merge_economic_datasets(
        self, datasets: list[pd.DataFrame], merge_keys: list[str], how: str = "outer"
    ) -> pd.DataFrame:
        """
        Merge multiple economic datasets.

        Args:
            datasets: List of DataFrames to merge
            merge_keys: Columns to merge on
            how: Type of merge ('inner', 'outer', 'left', 'right')

        Returns:
            Merged DataFrame
        """
        if not datasets:
            raise ValueError("No datasets provided for merging")

        merged_data = datasets[0]

        for dataset in datasets[1:]:
            merged_data = merged_data.merge(dataset, on=merge_keys, how=how)

        return merged_data

    def export_economic_data(
        self, data: pd.DataFrame, file_path: Path, format: str = "csv", **kwargs: Any
    ) -> None:
        """
        Export economic data to various formats.

        Args:
            data: DataFrame to export
            file_path: Path to save the file
            format: Export format ('csv', 'json', 'xlsx', 'geojson')
            **kwargs: Additional format-specific arguments
        """
        file_path = Path(file_path)

        if format == "csv":
            data.to_csv(file_path, index=False, **kwargs)
        elif format == "json":
            data.to_json(file_path, **kwargs)
        elif format == "xlsx":
            data.to_excel(file_path, index=False, **kwargs)
        elif format == "geojson":
            if isinstance(data, gpd.GeoDataFrame):
                data.to_file(file_path, driver="GeoJSON", **kwargs)
            else:
                raise ValueError("GeoJSON export requires GeoDataFrame")
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Exported data to {file_path}")


# Example usage and testing functions
def example_data_loading() -> EconomicDataLoader:
    """
    Example usage of the EconomicDataLoader
    """
    _log = logging.getLogger(__name__)
    _log.info("=== Economic Data Loading Example ===")

    # Initialize loader
    loader = EconomicDataLoader()

    # Register sample data sources
    sources = [
        DataSourceConfig(
            name="regional_gdp",
            source_type="file",
            format="csv",
            location="data/regional_gdp.csv",
        ),
        DataSourceConfig(
            name="employment_data",
            source_type="file",
            format="json",
            location="data/employment.json",
        ),
    ]

    for source in sources:
        loader.register_data_source(source)

    # Load and validate data
    try:
        gdp_data = loader.load_economic_data("regional_gdp")
        _log.info(
            "Loaded GDP data: %d rows, %d columns", len(gdp_data), len(gdp_data.columns)
        )

        employment_data = loader.load_economic_data("employment_data")
        _log.info(
            "Loaded employment data: %d rows, %d columns",
            len(employment_data),
            len(employment_data.columns),
        )

        # Merge datasets
        merged_data = loader.merge_economic_datasets(
            [gdp_data, employment_data], merge_keys=["region_id", "year"]
        )

        _log.info(
            "Merged data: %d rows, %d columns",
            len(merged_data),
            len(merged_data.columns),
        )

    except Exception as e:
        _log.error("Data loading example failed: %s", e)

    return loader


if __name__ == "__main__":
    # Run example
    example_loader = example_data_loading()
