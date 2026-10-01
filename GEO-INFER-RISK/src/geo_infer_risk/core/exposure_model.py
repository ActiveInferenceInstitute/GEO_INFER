"""
Enhanced ExposureModel: Advanced asset exposure modeling with real data integration.

This module provides sophisticated exposure modeling capabilities with:
- Integration with GEO-INFER-SPACE for spatial analysis and H3 indexing
- Integration with GEO-INFER-DATA for comprehensive data management
- Real-time data integration from multiple sources
- Advanced spatial aggregation and analysis
- Time-variant exposure modeling (population movements, business hours)
- Climate change scenario integration
- Economic valuation and depreciation modeling
- Portfolio-level exposure aggregation
"""

from typing import Any
from collections.abc import Sequence
from datetime import datetime
import logging
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# GEO-INFER module imports with error handling
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface

    SPACE_AVAILABLE = True
except ImportError:
    SPACE_AVAILABLE = False
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

try:
    from geo_infer_data.core.data_management import DataManager

    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    DataManager = None

from ..utils.rng import resolve_rng

logger = logging.getLogger(__name__)


class EnhancedExposureModel:
    """
    Enhanced exposure model with advanced data integration and spatial analysis.

    This class provides sophisticated exposure modeling with:
    - Real-time data integration from multiple sources
    - Advanced spatial indexing with H3 and other systems
    - Time-variant exposure modeling (population movements, business cycles)
    - Economic valuation with depreciation and appreciation
    - Climate change scenario integration
    - Portfolio-level aggregation and analysis
    - Real-time exposure monitoring
    - Integration with external data APIs and databases
    """

    # Runtime-integration singleton; typed Optional[Any] so mypy does not
    # collapse it to None (populated at runtime if modules present).
    spatial_interface: Any | None = None

    def __init__(self, exposure_type: str, params: dict[str, Any]):
        """Initialize the enhanced exposure model.

        Args:
            exposure_type: Type of exposure (``property``, ``population``,
                ``infrastructure`` or ``business``).
            params: Model parameters and configuration. A ``random_seed`` key
                (or its alias ``seed``) sets the seed for this model's
                generator; see
                :func:`geo_infer_risk.utils.rng.resolve_rng` for accepted
                forms. Omitting it gives a generator seeded from OS entropy,
                so results are not replayable.
        """
        self.exposure_type = exposure_type
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{exposure_type}")

        # All stochastic draws in this model come from this generator, never
        # from the process-wide numpy.random singleton. Checking for None
        # explicitly keeps random_seed=0 a valid, replayable seed.
        seed = params.get("random_seed")
        if seed is None:
            seed = params.get("seed")
        self._rng: np.random.Generator = resolve_rng(seed)

        # Enhanced parameter handling
        self.data_sources = params.get("data_sources", [])
        self.value_type = params.get("value_type", "replacement_cost")
        self.aggregation_level = params.get("aggregation_level", "building")
        self.include_contents = params.get("include_contents", False)
        self.include_time_variation = params.get("include_time_variation", False)
        self.time_scenarios = params.get("time_scenarios", ["day", "night", "commute"])
        self.current_time_scenario = params.get("current_scenario", "day")
        self.spatial_resolution = params.get("spatial_resolution", 9)  # H3 resolution
        self.update_frequency = params.get("update_frequency", "daily")

        # Initialize interfaces
        self.spatial_interface: Any | None = None
        self.data_manager = None

        if SPACE_AVAILABLE:
            try:
                self.spatial_interface = SpatialIndexingInterface()
                self.logger.info("Spatial interface initialized for exposure model")
            except Exception as e:
                self.logger.warning(f"Failed to initialize spatial interface: {e}")

        if DATA_AVAILABLE:
            try:
                self.data_manager = DataManager()
                self.logger.info("Data manager initialized for exposure model")
            except Exception as e:
                self.logger.warning(f"Failed to initialize data manager: {e}")

        # Model state
        self.is_initialized = False
        self.exposure_data: pd.DataFrame | None = None
        self.spatial_index: str | None = None
        self.spatial_tree: Any | None = None
        self.temporal_profiles: dict[str, Any] = {}
        self.economic_factors: dict[str, Any] = {}
        self.last_update = None

        # Load comprehensive exposure data
        self._initialize_exposure_data()

        # Initialize spatial indexing if available
        if (
            getattr(self, "spatial_interface", None) is not None
            and getattr(self, "exposure_data", None) is not None
        ):
            self._initialize_spatial_indexing()

        # Initialize temporal profiles if time variation is enabled
        if self.include_time_variation:
            self._initialize_temporal_profiles()

        self.is_initialized = True
        self.logger.info(
            f"Enhanced {exposure_type} exposure model initialized successfully"
        )

    def _initialize_exposure_data(self) -> None:
        """Initialize exposure data from multiple sources."""
        try:
            self.exposure_data = None

            # Try to load from each configured data source
            for source in self.data_sources:
                data = self._load_data_from_source(source)
                if data is not None and not data.empty:
                    if self.exposure_data is None:
                        self.exposure_data = data
                    else:
                        # Merge with existing data
                        self.exposure_data = self._merge_exposure_data(
                            self.exposure_data, data
                        )

            # Keep the model explicitly uninitialized until a configured source
            # provides data. Silent fabricated exposure would invalidate risk
            # calculations.
            if self.exposure_data is None or self.exposure_data.empty:
                self.logger.info(
                    "No exposure data loaded from configured sources %s",
                    self.data_sources,
                )
                return

            # Validate and clean data
            self._validate_and_clean_exposure_data()

            # Calculate derived properties
            self._calculate_derived_properties()

            # Initialize economic valuation
            self._initialize_economic_valuation()

            self.logger.info(
                f"Exposure data initialized with {len(self.exposure_data)} records"
            )

        except (OSError, json.JSONDecodeError, pd.errors.ParserError) as e:
            self.logger.error(f"Failed to initialize exposure data from I/O: {e}")
            self.exposure_data = None

    def _load_data_from_source(self, source: str) -> pd.DataFrame | None:
        """Load exposure data from a specific source."""
        try:
            if source.startswith("file://"):
                return self._load_from_file(source)
            elif source.startswith("api://"):
                raise ValueError(
                    "API exposure sources require a configured data connector; "
                    "the model does not synthesize remote data"
                )
            else:
                raise ValueError(f"Unsupported exposure data source: {source}")

        except Exception as e:
            self.logger.error(f"Failed to load data from source {source}: {e}")
            return None

    def _load_from_file(self, file_path: str) -> pd.DataFrame | None:
        """Load exposure data from file."""
        file_path = file_path.replace("file://", "")

        try:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            elif file_path.endswith(".json"):
                with open(file_path) as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            elif file_path.endswith(".parquet"):
                return pd.read_parquet(file_path)
            else:
                self.logger.warning(f"Unsupported file format: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {e}")
            return None

    def _merge_exposure_data(
        self, existing_data: pd.DataFrame, new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge exposure data from multiple sources."""
        try:
            # Use outer join on coordinates and ID fields
            merge_keys = ["longitude", "latitude"]
            if "id" in existing_data.columns and "id" in new_data.columns:
                merge_keys.append("id")

            merged = pd.merge(
                existing_data,
                new_data,
                on=merge_keys,
                how="outer",
                suffixes=("_x", "_y"),
            )

            # Handle duplicate columns
            for col in merged.columns:
                if col.endswith("_y"):
                    base_col = col[:-2]
                    if base_col + "_x" in merged.columns:
                        # Both sources have this column, use the one with more non-null values
                        x_nulls = merged[base_col + "_x"].isnull().sum()
                        y_nulls = merged[col].isnull().sum()

                        if y_nulls < x_nulls:
                            merged[base_col] = merged[col].fillna(
                                merged[base_col + "_x"]
                            )
                        else:
                            merged[base_col] = merged[base_col + "_x"].fillna(
                                merged[col]
                            )

                        merged.drop([base_col + "_x", col], axis=1, inplace=True)
                    else:
                        merged.rename(columns={col: base_col}, inplace=True)

            # Remove helper columns
            for col in merged.columns:
                if col.endswith("_x"):
                    merged.drop(col, axis=1, inplace=True)

            return merged

        except Exception as e:
            # A failed merge must surface, not silently drop the second data
            # source: downgraded exposure would silently invalidate risk
            # calculations downstream.
            raise RuntimeError(
                f"Failed to merge exposure data from multiple sources: {e}"
            ) from e

    def _validate_and_clean_exposure_data(self) -> None:
        """Validate and clean exposure data."""
        if self.exposure_data is None or self.exposure_data.empty:
            return

        # Check for required columns
        required_cols = ["id", "longitude", "latitude"]
        for col in required_cols:
            if col not in self.exposure_data.columns:
                raise ValueError(f"Required column missing: {col}")

        # Validate coordinates
        lat_valid = (-90 <= self.exposure_data["latitude"]) & (
            self.exposure_data["latitude"] <= 90
        )
        lon_valid = (-180 <= self.exposure_data["longitude"]) & (
            self.exposure_data["longitude"] <= 180
        )

        if not lat_valid.all() or not lon_valid.all():
            invalid_count = (~lat_valid | ~lon_valid).sum()
            self.logger.warning(
                f"Removing {invalid_count} records with invalid coordinates"
            )
            self.exposure_data = self.exposure_data[lat_valid & lon_valid]

        # Remove duplicates
        duplicates = self.exposure_data.duplicated(
            subset=["longitude", "latitude"], keep="first"
        )
        if duplicates.any():
            duplicate_count = duplicates.sum()
            self.logger.info(f"Removing {duplicate_count} duplicate records")
            self.exposure_data = self.exposure_data[~duplicates]

        # Handle missing values
        self._handle_missing_values()

        # Validate value columns
        value_cols = ["value", "replacement_cost", "market_value", "population_count"]
        for col in value_cols:
            if col in self.exposure_data.columns:
                self.exposure_data[col] = pd.to_numeric(
                    self.exposure_data[col], errors="coerce"
                )
                self.exposure_data[col] = self.exposure_data[col].fillna(0)

        self.logger.info(
            f"Exposure data validated and cleaned: {len(self.exposure_data)} records remaining"
        )

    def _handle_missing_values(self) -> None:
        """Handle missing values in exposure data."""
        assert self.exposure_data is not None
        # For each column, apply appropriate missing value handling
        for col in self.exposure_data.columns:
            if self.exposure_data[col].isnull().any():
                if col in ["year_built", "building_levels", "stories"]:
                    # Fill with median for structural properties
                    median_val = self.exposure_data[col].median()
                    self.exposure_data[col].fillna(median_val, inplace=True)
                elif col in ["replacement_cost", "market_value", "value"]:
                    # Fill with mean for value properties
                    mean_val = self.exposure_data[col].mean()
                    self.exposure_data[col].fillna(mean_val, inplace=True)
                elif col in ["type", "building_material", "condition"]:
                    # Fill with mode for categorical properties
                    mode_val = (
                        self.exposure_data[col].mode().iloc[0]
                        if not self.exposure_data[col].mode().empty
                        else "unknown"
                    )
                    self.exposure_data[col].fillna(mode_val, inplace=True)

    def _calculate_derived_properties(self) -> None:
        """Calculate derived properties from existing data."""
        if self.exposure_data is None:
            return

        # Calculate area if not present
        if "area" not in self.exposure_data.columns:
            if "building_levels" in self.exposure_data.columns:
                # Estimate area based on building levels
                self.exposure_data["area"] = self.exposure_data[
                    "building_levels"
                ] * self._rng.uniform(100, 500, len(self.exposure_data))
            else:
                self.exposure_data["area"] = self._rng.uniform(
                    50, 200, len(self.exposure_data)
                )

        # Calculate value per square meter if value and area are available
        if (
            "value" in self.exposure_data.columns
            and "area" in self.exposure_data.columns
        ):
            self.exposure_data["value_per_sqm"] = (
                self.exposure_data["value"] / self.exposure_data["area"]
            )

        # Calculate population density for population data
        if (
            self.exposure_type == "population"
            and "population_count" in self.exposure_data.columns
        ):
            if "area" in self.exposure_data.columns:
                self.exposure_data["population_density"] = (
                    self.exposure_data["population_count"] / self.exposure_data["area"]
                )

        # Calculate building age
        if "year_built" in self.exposure_data.columns:
            current_year = datetime.now().year
            self.exposure_data["building_age"] = (
                current_year - self.exposure_data["year_built"]
            )

        self.logger.info("Derived properties calculated")

    def _initialize_economic_valuation(self) -> None:
        """Initialize economic valuation parameters."""
        self.economic_factors = {
            "depreciation_rate": 0.02,  # Annual depreciation rate
            "appreciation_rate": 0.03,  # Annual appreciation rate
            "inflation_rate": 0.025,  # Annual inflation rate
            "discount_rate": 0.05,  # Discount rate for present value calculations
        }

        # Apply economic adjustments to values
        if self.exposure_data is not None and "value" in self.exposure_data.columns:
            self._apply_economic_adjustments()

    def _apply_economic_adjustments(self) -> None:
        """Apply economic adjustments to asset values."""
        assert self.exposure_data is not None
        current_year = datetime.now().year

        if "year_built" in self.exposure_data.columns:
            # Apply depreciation based on age
            age = current_year - self.exposure_data["year_built"]
            depreciation_factor = np.power(
                1 - self.economic_factors["depreciation_rate"], age
            )
            self.exposure_data["depreciated_value"] = (
                self.exposure_data["value"] * depreciation_factor
            )

        # Apply appreciation (market effects)
        if "appreciated_value" not in self.exposure_data.columns:
            appreciation_factor = np.power(
                1 + self.economic_factors["appreciation_rate"],
                age if "year_built" in self.exposure_data.columns else 10,
            )
            self.exposure_data["appreciated_value"] = (
                self.exposure_data["value"] * appreciation_factor
            )

        # Apply inflation adjustment
        inflation_years = current_year - 2020  # Base year
        inflation_factor = np.power(
            1 + self.economic_factors["inflation_rate"], inflation_years
        )
        self.exposure_data["inflated_value"] = (
            self.exposure_data["value"] * inflation_factor
        )

    def _initialize_spatial_indexing(self) -> None:
        """Initialize spatial indexing for efficient queries."""
        if not self.spatial_interface or self.exposure_data is None:
            return

        try:
            # Create spatial index using coordinates
            coords = self.exposure_data[["longitude", "latitude"]].values

            # Use H3 indexing if available
            if hasattr(self.spatial_interface, "latlng_to_cell"):
                h3_cells = []
                for lon, lat in coords:
                    try:
                        cell = self.spatial_interface.latlng_to_cell(
                            lat, lon, self.spatial_resolution
                        )
                        h3_cells.append(cell)
                    except Exception:
                        h3_cells.append(None)

                self.exposure_data["h3_cell"] = h3_cells
                self.spatial_index = "h3"
            else:
                # Fallback to KDTree for spatial queries
                self.spatial_tree = cKDTree(coords)
                self.spatial_index = "kdtree"

            self.logger.info(f"Spatial indexing initialized using {self.spatial_index}")

        except Exception as e:
            self.logger.error(f"Failed to initialize spatial indexing: {e}")
            self.spatial_index = None

    def _initialize_temporal_profiles(self) -> None:
        """Initialize temporal profiles for time-variant analysis."""
        if self.exposure_type == "population":
            self._initialize_population_temporal_profiles()
        elif self.exposure_type == "business":
            self._initialize_business_temporal_profiles()
        else:
            self._initialize_generic_temporal_profiles()

    def _initialize_population_temporal_profiles(self) -> None:
        """Initialize population temporal profiles."""
        assert self.exposure_data is not None
        # Population movement patterns throughout the day
        base_population = (
            self.exposure_data["population_count"].values
            if "population_count" in self.exposure_data.columns
            else np.ones(len(self.exposure_data)) * 100
        )

        # Day scenario: More people at work/school
        self.temporal_profiles["day"] = base_population * 0.8

        # Night scenario: Most people at home
        self.temporal_profiles["night"] = base_population * 1.1

        # Commute scenario: People in transit
        self.temporal_profiles["commute"] = base_population * 0.9

        # Weekend scenario: Different patterns
        self.temporal_profiles["weekend"] = base_population * 1.05

        self.logger.info("Population temporal profiles initialized")

    def _initialize_business_temporal_profiles(self) -> None:
        """Initialize business temporal profiles."""
        assert self.exposure_data is not None
        # Business activity patterns
        base_value = (
            self.exposure_data["value"].values
            if "value" in self.exposure_data.columns
            else np.ones(len(self.exposure_data)) * 100000
        )

        # Business hours: High activity
        self.temporal_profiles["day"] = base_value * 1.2

        # After hours: Low activity
        self.temporal_profiles["night"] = base_value * 0.3

        # Commute times: Medium activity
        self.temporal_profiles["commute"] = base_value * 0.8

        # Weekend: Very low activity
        self.temporal_profiles["weekend"] = base_value * 0.1

        self.logger.info("Business temporal profiles initialized")

    def _initialize_generic_temporal_profiles(self) -> None:
        """Initialize generic temporal profiles."""
        assert self.exposure_data is not None
        base_value = (
            self.exposure_data["value"].values
            if "value" in self.exposure_data.columns
            else np.ones(len(self.exposure_data))
        )

        # Simple day/night variation
        self.temporal_profiles["day"] = base_value * 1.1
        self.temporal_profiles["night"] = base_value * 0.9
        self.temporal_profiles["commute"] = base_value * 1.0

        self.logger.info("Generic temporal profiles initialized")

    def get_exposure_at_location(
        self,
        latitude: float,
        longitude: float,
        radius: float = 1.0,
        time_scenario: str | None = None,
    ) -> dict[str, Any]:
        """
        Get exposure within a radius of a specific location with temporal variation.

        Args:
            latitude: Target latitude
            longitude: Target longitude
            radius: Search radius in kilometers
            time_scenario: Time scenario ('day', 'night', 'commute', etc.)

        Returns:
            Dictionary with exposure summary at the location
        """
        if self.exposure_data is None:
            return {"total_value": 0, "count": 0, "assets": []}

        if time_scenario is None:
            time_scenario = self.current_time_scenario

        # Use spatial indexing for efficient queries
        if self.spatial_index == "h3" and "h3_cell" in self.exposure_data.columns:
            return self._get_exposure_h3(latitude, longitude, radius, time_scenario)
        elif self.spatial_index == "kdtree" and hasattr(self, "spatial_tree"):
            return self._get_exposure_kdtree(latitude, longitude, radius, time_scenario)
        else:
            return self._get_exposure_brute_force(
                latitude, longitude, radius, time_scenario
            )

    def _get_exposure_h3(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        time_scenario: str | None = None,
    ) -> dict[str, Any]:
        """Get exposure using H3 spatial indexing."""
        assert self.spatial_interface is not None
        assert self.exposure_data is not None
        # Convert radius to H3 resolution
        target_cell = self.spatial_interface.latlng_to_cell(
            latitude, longitude, self.spatial_resolution
        )

        # Get neighboring cells within radius
        neighbor_cells = self.spatial_interface.grid_disk(
            target_cell, int(radius * 2)
        )  # Approximate

        # Filter exposure data to cells within radius
        cell_mask = self.exposure_data["h3_cell"].isin(neighbor_cells)
        filtered_data = self.exposure_data[cell_mask]

        return self._calculate_exposure_summary(filtered_data, time_scenario)

    def _get_exposure_kdtree(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        time_scenario: str | None = None,
    ) -> dict[str, Any]:
        """Get exposure using KDTree spatial indexing."""
        assert self.spatial_tree is not None
        assert self.exposure_data is not None
        # Query tree for points within radius
        distances, indices = self.spatial_tree.query(
            [[longitude, latitude]],
            k=len(self.exposure_data),
            distance_upper_bound=radius / 111.0,  # Convert km to degrees
        )

        # Filter valid indices
        valid_indices = indices[0][distances[0] < float("inf")]
        filtered_data = self.exposure_data.iloc[valid_indices]

        return self._calculate_exposure_summary(filtered_data, time_scenario)

    def _get_exposure_brute_force(
        self,
        latitude: float,
        longitude: float,
        radius: float,
        time_scenario: str | None = None,
    ) -> dict[str, Any]:
        """Get exposure using brute force calculation."""
        assert self.exposure_data is not None
        # Calculate distances from target point
        distances = (
            np.sqrt(
                (self.exposure_data["longitude"] - longitude) ** 2
                + (self.exposure_data["latitude"] - latitude) ** 2
            )
            * 111.0
        )  # Convert degrees to km

        # Filter to points within radius
        mask = distances <= radius
        filtered_data = self.exposure_data[mask]

        return self._calculate_exposure_summary(filtered_data, time_scenario)

    def _calculate_exposure_summary(
        self, filtered_data: pd.DataFrame, time_scenario: str | None = None
    ) -> dict[str, Any]:
        """Calculate exposure summary for filtered data."""
        if filtered_data.empty:
            return {"total_value": 0, "count": 0, "assets": []}

        # Apply time scenario adjustment
        if time_scenario and time_scenario in self.temporal_profiles:
            adjustment_factors = self.temporal_profiles[
                time_scenario
            ] / self.temporal_profiles.get(
                self.current_time_scenario, self.temporal_profiles[time_scenario]
            )
            adjusted_values = filtered_data["value"] * adjustment_factors
        else:
            adjusted_values = filtered_data["value"]

        # Calculate summary statistics
        total_value = adjusted_values.sum()
        asset_count = len(filtered_data)

        # Get value by type
        by_type = {}
        for asset_type, group in filtered_data.groupby("type"):
            by_type[asset_type] = {
                "count": len(group),
                "value": (group["value"] * adjustment_factors.loc[group.index]).sum(),
            }

        return {
            "total_value": total_value,
            "count": asset_count,
            "by_type": by_type,
            "assets": filtered_data.to_dict("records"),
            "time_scenario": time_scenario or self.current_time_scenario,
            "spatial_index_used": self.spatial_index,
        }

    def _exposure_value_column(
        self, candidates: Sequence[str] = ("value", "replacement_cost", "market_value")
    ) -> str | None:
        """Return the first present value column, or ``None``."""
        if self.exposure_data is None:
            return None
        for column in candidates:
            if column in self.exposure_data.columns:
                return column
        return None

    def sample_total_exposure(
        self,
        method: str = "bootstrap",
        num_samples: int = 1000,
        random_seed: Any | None = None,
    ) -> dict[str, Any]:
        """Return a distributional estimate of total insurable exposure.

        Parameters
        ----------
        method : str
            ``"bootstrap"`` resamples the asset records with replacement, so the
            exposed total reflects the observed mix of asset values;
            ``"parametric"`` multiplies each asset value by a log-normal factor
            (variance equal to the coefficient of variation of the book, or 10%
            when the book is homogeneous) and sums per sample;
            ``"deterministic"`` returns the exact recorded total.
        num_samples : int
            Number of aggregate draws.
        random_seed : optional
            Seed or generator for the draws; when omitted the model's own
            generator is used (see :mod:`geo_infer_risk.utils.rng`).

        Returns
        -------
        dict
            With ``method``, ``sample_count``, ``deterministic_total``,
            ``total_mean``, ``total_median``, ``percentile_5``,
            ``percentile_95`` and ``samples`` (the per-draw array).

        Raises
        ------
        ValueError
            If the model has no exposure data or no value column, or if
            ``num_samples`` is not positive.
        """
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if method not in {"bootstrap", "parametric", "deterministic"}:
            raise ValueError(
                "method must be 'bootstrap', 'parametric', or 'deterministic'"
            )
        column = self._exposure_value_column()
        if self.exposure_data is None or column is None:
            raise ValueError(
                "exposure data with a value column is required for "
                "distributional exposure sampling"
            )
        values = self.exposure_data[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("exposure value column contains no finite values")
        rng = resolve_rng(random_seed) if random_seed is not None else self._rng

        if method == "deterministic":
            samples = np.array([float(finite.sum())])
        elif method == "bootstrap":
            samples = np.empty(num_samples, dtype=float)
            for i in range(num_samples):
                draw = rng.choice(finite, size=finite.size, replace=True)
                samples[i] = float(draw.sum())
        else:
            if finite.std(ddof=1) <= 0:
                sigma = 0.10
            else:
                sigma = float(finite.std(ddof=1) / finite.mean())
            factors = rng.lognormal(
                mean=-0.5 * sigma * sigma, sigma=sigma, size=num_samples
            )
            samples = finite.sum() * factors

        return {
            "method": method,
            "sample_count": int(samples.size),
            "deterministic_total": float(finite.sum()),
            "total": float(np.mean(samples)),
            "total_median": float(np.median(samples)),
            "percentile_5": float(np.percentile(samples, 5)),
            "percentile_95": float(np.percentile(samples, 95)),
            "samples": samples.tolist(),
        }

    def get_exposure_for_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Get the exposure affected by a hazard event with enhanced features.

        Args:
            event: Hazard event information

        Returns:
            List of exposed assets with their properties and intensities
        """
        # Extract event footprint bounds
        footprint = event.get("footprint", {})
        bounds = footprint.get("bounds", {})

        min_lon = bounds.get("min_lon", -180)
        max_lon = bounds.get("max_lon", 180)
        min_lat = bounds.get("min_lat", -90)
        max_lat = bounds.get("max_lat", 90)

        # Filter exposure data to assets within the event footprint
        if self.exposure_data is not None:
            filtered_exposure = self.exposure_data[
                (self.exposure_data["longitude"] >= min_lon)
                & (self.exposure_data["longitude"] <= max_lon)
                & (self.exposure_data["latitude"] >= min_lat)
                & (self.exposure_data["latitude"] <= max_lat)
            ]

            # For each asset, calculate the hazard intensity at its location
            exposed_assets = []

            for _, asset in filtered_exposure.iterrows():
                # Create a dictionary for the asset
                asset_dict = asset.to_dict()

                # Add the hazard intensity at this asset location
                asset_dict["intensity_at_asset"] = (
                    self._calculate_intensity_at_location(
                        event, asset["latitude"], asset["longitude"]
                    )
                )

                # Add time-variant exposure if enabled
                if self.include_time_variation:
                    time_scenario = event.get(
                        "time_scenario", self.current_time_scenario
                    )
                    asset_dict["time_adjusted_value"] = self._get_time_adjusted_value(
                        asset, time_scenario
                    )

                exposed_assets.append(asset_dict)

            return exposed_assets

        return []

    def _calculate_intensity_at_location(
        self, event: dict[str, Any], latitude: float, longitude: float
    ) -> float:
        """Calculate hazard intensity at a specific location for an event."""
        # Enhanced version with better interpolation
        footprint = event.get("footprint", {})

        # Check if the location is within the footprint bounds
        bounds = footprint.get("bounds", {})
        if bounds.get("min_lon", 0) <= longitude <= bounds.get(
            "max_lon", 0
        ) and bounds.get("min_lat", 0) <= latitude <= bounds.get("max_lat", 0):
            # Enhanced grid-based intensity calculation
            intensity_values = footprint.get("intensity_values", [[0]])
            # Calculate grid indices with higher precision
            lon_range = bounds.get("max_lon", 0) - bounds.get("min_lon", 0)
            lat_range = bounds.get("max_lat", 0) - bounds.get("min_lat", 0)

            if (
                isinstance(intensity_values, list)
                and intensity_values
                and isinstance(intensity_values[0], list)
            ):
                grid_size = len(intensity_values)

                # Convert lat/lon to grid indices
                lon_idx = int(
                    (longitude - bounds.get("min_lon", 0)) / lon_range * (grid_size - 1)
                )
                lat_idx = int(
                    (latitude - bounds.get("min_lat", 0)) / lat_range * (grid_size - 1)
                )

                # Ensure indices are within bounds
                lon_idx = max(0, min(grid_size - 1, lon_idx))
                lat_idx = max(0, min(grid_size - 1, lat_idx))

                # Bilinear interpolation for smoother results
                return self._bilinear_interpolation(
                    intensity_values, lon_idx, lat_idx, longitude, latitude, bounds
                )

            else:
                # Simple case - constant intensity across the footprint
                return float(intensity_values[0]) if intensity_values else 0.0

        return 0.0

    def _bilinear_interpolation(
        self,
        grid: list[list[float]],
        x_idx: int,
        y_idx: int,
        x: float,
        y: float,
        bounds: dict[str, float],
    ) -> float:
        """Perform bilinear interpolation on grid data."""
        grid_size = len(grid)

        # Get corner values
        x1, x2 = x_idx, min(x_idx + 1, grid_size - 1)
        y1, y2 = y_idx, min(y_idx + 1, grid_size - 1)

        # Get bounding box for this grid cell
        lon_range = bounds.get("max_lon", 0) - bounds.get("min_lon", 0)
        lat_range = bounds.get("max_lat", 0) - bounds.get("min_lat", 0)

        cell_width = lon_range / (grid_size - 1)
        cell_height = lat_range / (grid_size - 1)

        x_min = bounds.get("min_lon", 0) + x_idx * cell_width
        y_min = bounds.get("min_lat", 0) + y_idx * cell_height

        # Calculate interpolation weights
        dx = (x - x_min) / cell_width if cell_width > 0 else 0
        dy = (y - y_min) / cell_height if cell_height > 0 else 0

        # Bilinear interpolation
        f11 = grid[y1][x1] if y1 < len(grid) and x1 < len(grid[y1]) else 0
        f12 = grid[y1][x2] if y1 < len(grid) and x2 < len(grid[y1]) else 0
        f21 = grid[y2][x1] if y2 < len(grid) and x1 < len(grid[y2]) else 0
        f22 = grid[y2][x2] if y2 < len(grid) and x2 < len(grid[y2]) else 0

        return (
            f11 * (1 - dx) * (1 - dy)
            + f12 * dx * (1 - dy)
            + f21 * (1 - dx) * dy
            + f22 * dx * dy
        )

    def _get_time_adjusted_value(self, asset: pd.Series, time_scenario: str) -> float:
        """Get time-adjusted asset value for a specific scenario."""
        base_value = asset.get("value", 0)

        if time_scenario in self.temporal_profiles:
            # Get the adjustment factor for this asset
            adjustment_factors = self.temporal_profiles[time_scenario]
            asset_index = asset.name if hasattr(asset, "name") else 0

            if asset_index < len(adjustment_factors):
                return float(base_value * adjustment_factors[asset_index])

        return float(base_value)

    def calculate_total_exposure(
        self,
        bounds: dict[str, float] | None = None,
        time_scenario: str | None = None,
    ) -> dict[str, Any]:
        assert self.exposure_data is not None
        """
        Calculate the total exposure within optional geographic bounds with temporal variation.

        Args:
            bounds: Geographic bounds (min_lon, max_lon, min_lat, max_lat)
            time_scenario: Time scenario for temporal adjustment

        Returns:
            Dictionary with total exposure statistics
        """
        # Apply time scenario adjustment
        if time_scenario and time_scenario in self.temporal_profiles:
            adjustment_factors = self.temporal_profiles[time_scenario]
            adjusted_values = self.exposure_data["value"] * adjustment_factors
        else:
            adjusted_values = self.exposure_data["value"]

        # Filter exposure data to assets within the specified bounds
        if bounds and self.exposure_data is not None:
            min_lon = bounds.get("min_lon", -180)
            max_lon = bounds.get("max_lon", 180)
            min_lat = bounds.get("min_lat", -90)
            max_lat = bounds.get("max_lat", 90)

            filtered_exposure = self.exposure_data[
                (self.exposure_data["longitude"] >= min_lon)
                & (self.exposure_data["longitude"] <= max_lon)
                & (self.exposure_data["latitude"] >= min_lat)
                & (self.exposure_data["latitude"] <= max_lat)
            ]
            adjusted_values = adjusted_values.loc[filtered_exposure.index]
        else:
            filtered_exposure = self.exposure_data

        # If no exposure data is available, return zeros
        if filtered_exposure is None or len(filtered_exposure) == 0:
            return {
                "total_value": 0,
                "count": 0,
                "by_type": {},
                "time_scenario": time_scenario or self.current_time_scenario,
            }

        # Calculate total value
        total_value = adjusted_values.sum()
        count = len(filtered_exposure)

        # Calculate value by type
        by_type = {}
        for asset_type, group in filtered_exposure.groupby("type"):
            group_values = adjusted_values.loc[group.index]
            by_type[asset_type] = {"value": group_values.sum(), "count": len(group)}

        return {
            "total_value": total_value,
            "count": count,
            "by_type": by_type,
            "time_scenario": time_scenario or self.current_time_scenario,
            "spatial_index_used": self.spatial_index,
        }

    def set_time_scenario(self, scenario: str) -> None:
        """
        Set the current time scenario for exposure analysis.

        Args:
            scenario: Scenario name ('day', 'night', 'commute', etc.)
        """
        if scenario in self.time_scenarios:
            self.current_time_scenario = scenario
            self.logger.info(f"Time scenario set to: {scenario}")
        else:
            raise ValueError(
                f"Invalid time scenario: {scenario}. Valid options are {self.time_scenarios}"
            )

    def update_exposure_data(
        self, new_data: pd.DataFrame, merge_strategy: str = "replace"
    ) -> None:
        """
        Update exposure data with new information.

        Args:
            new_data: New exposure data to add
            merge_strategy: How to merge with existing data ('replace', 'append', 'merge')
        """
        if new_data is None or new_data.empty:
            return

        if merge_strategy == "replace":
            self.exposure_data = new_data.copy()
        elif merge_strategy == "append":
            self.exposure_data = pd.concat(
                [self.exposure_data, new_data], ignore_index=True
            )
        elif merge_strategy == "merge":
            self.exposure_data = self._merge_exposure_data(self.exposure_data, new_data)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        # Re-validate and re-initialize after update
        self._validate_and_clean_exposure_data()
        self._calculate_derived_properties()

        if self.spatial_interface:
            self._initialize_spatial_indexing()

        self.logger.info(f"Exposure data updated: {len(self.exposure_data)} records")

    def save_exposure_data(self, output_file: str) -> None:
        """Save exposure data to a CSV file."""
        if self.exposure_data is not None:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save to CSV
            self.exposure_data.to_csv(output_file, index=False)
            self.logger.info(f"Exposure data saved to {output_file}")

    def load_exposure_data(self, input_file: str) -> None:
        """Load exposure data from a CSV file."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Exposure data file not found: {input_file}")

        # Load from CSV
        self.exposure_data = pd.read_csv(input_file)

        # Re-validate and re-initialize
        self._validate_and_clean_exposure_data()
        self._calculate_derived_properties()

        if self.spatial_interface:
            self._initialize_spatial_indexing()

        self.logger.info(
            f"Exposure data loaded from {input_file}: {len(self.exposure_data)} records"
        )

    def get_model_status(self) -> dict[str, Any]:
        """Get comprehensive model status information."""
        return {
            "exposure_type": self.exposure_type,
            "is_initialized": self.is_initialized,
            "data_available": self.exposure_data is not None,
            "record_count": (
                len(self.exposure_data) if self.exposure_data is not None else 0
            ),
            "data_sources": self.data_sources,
            "spatial_indexing": self.spatial_index,
            "time_variation_enabled": self.include_time_variation,
            "current_time_scenario": self.current_time_scenario,
            "integration_status": {
                "spatial_interface": self.spatial_interface is not None,
                "data_manager": self.data_manager is not None,
            },
            "economic_factors": self.economic_factors,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


# Enhanced Specific Exposure Models


class EnhancedPropertyExposureModel(EnhancedExposureModel):
    """Enhanced property exposure model with detailed building characteristics."""

    def __init__(self, params: dict[str, Any]):
        super().__init__("property", params)
        self.include_contents = params.get("include_contents", True)
        self.contents_value_factor = params.get("contents_value_factor", 0.5)

    def _initialize_exposure_data(self) -> None:
        """Initialize property-specific exposure data."""
        super()._initialize_exposure_data()

        # Add contents value if enabled
        if self.include_contents and self.exposure_data is not None:
            if "value" in self.exposure_data.columns:
                self.exposure_data["contents_value"] = (
                    self.exposure_data["value"] * self.contents_value_factor
                )
                self.exposure_data["total_value"] = (
                    self.exposure_data["value"] + self.exposure_data["contents_value"]
                )

    def _calculate_derived_properties(self) -> None:
        """Calculate property-specific derived properties."""
        super()._calculate_derived_properties()

        if self.exposure_data is not None:
            # Calculate building density
            if "area" in self.exposure_data.columns:
                self.exposure_data["building_density"] = (
                    1.0 / self.exposure_data["area"]
                )

            # Calculate value per square foot
            if (
                "value" in self.exposure_data.columns
                and "area" in self.exposure_data.columns
            ):
                self.exposure_data["value_per_sqm"] = (
                    self.exposure_data["value"] / self.exposure_data["area"]
                )

    def get_exposure_for_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Get property exposure affected by a hazard event."""
        exposed_assets = super().get_exposure_for_event(event)

        # Add property-specific calculations
        for asset in exposed_assets:
            # Add insurance considerations
            asset["insurance_coverage"] = (
                asset.get("value", 0) * 0.8
            )  # Assume 80% coverage
            asset["deductible"] = asset.get("value", 0) * 0.02  # 2% deductible

        return exposed_assets


class EnhancedPopulationExposureModel(EnhancedExposureModel):
    """Enhanced population exposure model with demographic analysis."""

    def __init__(self, params: dict[str, Any]):
        super().__init__("population", params)
        self.demographic_factors = params.get(
            "demographic_factors", ["age", "income", "mobility"]
        )
        self.social_vulnerability_index = params.get("social_vulnerability_index", True)

    def _initialize_temporal_profiles(self) -> None:
        """Initialize population-specific temporal profiles."""
        super()._initialize_temporal_profiles()
        assert self.exposure_data is not None

        # Add demographic-specific temporal patterns
        if "median_age" in self.exposure_data.columns:
            # Elderly populations may have different movement patterns
            elderly_mask = self.exposure_data["median_age"] > 65
            self.temporal_profiles["elderly_day"] = self.temporal_profiles["day"].copy()
            self.temporal_profiles["elderly_day"][elderly_mask] *= (
                1.2  # More elderly at home during day
            )

    def _calculate_derived_properties(self) -> None:
        """Calculate population-specific derived properties."""
        super()._calculate_derived_properties()

        if (
            self.exposure_data is not None
            and "population_count" in self.exposure_data.columns
        ):
            # Calculate population density
            if "area" in self.exposure_data.columns:
                self.exposure_data["population_density"] = (
                    self.exposure_data["population_count"] / self.exposure_data["area"]
                )

            # Calculate vulnerability scores
            if self.social_vulnerability_index:
                self._calculate_social_vulnerability_index()

    def _calculate_social_vulnerability_index(self) -> None:
        """Calculate social vulnerability index for population groups."""
        assert self.exposure_data is not None
        # Simplified SVI calculation
        svi_components = []

        if "median_age" in self.exposure_data.columns:
            # Age vulnerability (elderly and children more vulnerable)
            age_vuln = np.where(
                (self.exposure_data["median_age"] > 65)
                | (self.exposure_data["median_age"] < 18),
                1.0,
                0.5,
            )
            svi_components.append(age_vuln)

        if "median_income" in self.exposure_data.columns:
            # Income vulnerability (lower income = higher vulnerability)
            income_vuln = 1.0 - (
                self.exposure_data["median_income"]
                / self.exposure_data["median_income"].max()
            )
            svi_components.append(income_vuln)

        if svi_components:
            self.exposure_data["social_vulnerability"] = np.mean(svi_components, axis=0)

    def get_exposure_for_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Get population exposure affected by a hazard event."""
        exposed_assets = super().get_exposure_for_event(event)

        # Add population-specific calculations
        for asset in exposed_assets:
            population = asset.get("population_count", 0)

            # Estimate vulnerable population
            asset["vulnerable_population"] = population * asset.get(
                "social_vulnerability", 0.5
            )
            asset["evacuation_capacity"] = population * 0.8  # Assume 80% can evacuate
            asset["shelter_requirement"] = population * 0.2  # Assume 20% need shelter

        return exposed_assets


class EnhancedInfrastructureExposureModel(EnhancedExposureModel):
    """Enhanced infrastructure exposure model with network considerations."""

    def __init__(self, params: dict[str, Any]):
        super().__init__("infrastructure", params)
        self.infrastructure_types = params.get(
            "types", ["roads", "bridges", "power_lines", "water_supply"]
        )

    def _initialize_exposure_data(self) -> None:
        """Initialize infrastructure-specific exposure data."""
        super()._initialize_exposure_data()

        # Add infrastructure-specific calculations
        if self.exposure_data is not None:
            self._calculate_infrastructure_derived_properties()

    def _calculate_infrastructure_derived_properties(self) -> None:
        """Calculate infrastructure-specific derived properties."""
        assert self.exposure_data is not None
        if self.exposure_data is None:
            return
        if "criticality" not in self.exposure_data.columns:
            # Assign criticality based on type
            criticality_mapping = {
                "bridge": "high",
                "power_line": "high",
                "water_supply": "high",
                "communication": "medium",
                "road": "medium",
                "transportation": "medium",
            }

            self.exposure_data["criticality"] = (
                self.exposure_data["type"].map(criticality_mapping).fillna("low")
            )

        # Calculate service area
        if "type" in self.exposure_data.columns:
            service_areas = {
                "power_line": 50.0,  # km²
                "water_supply": 25.0,
                "bridge": 100.0,
                "road": 10.0,
                "communication": 30.0,
            }

            self.exposure_data["service_area"] = (
                self.exposure_data["type"].map(service_areas).fillna(5.0)
            )

    def get_exposure_for_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Get infrastructure exposure affected by a hazard event."""
        exposed_assets = super().get_exposure_for_event(event)

        # Add infrastructure-specific calculations
        for asset in exposed_assets:
            infra_type = asset.get("type", "")

            # Calculate network impact
            if infra_type in ["power_line", "bridge"]:
                asset["network_impact"] = "high"
                asset["restoration_priority"] = "critical"
            elif infra_type in ["water_supply", "communication"]:
                asset["network_impact"] = "medium"
                asset["restoration_priority"] = "high"
            else:
                asset["network_impact"] = "low"
                asset["restoration_priority"] = "standard"

            # Estimate outage impact
            asset["affected_users"] = (
                asset.get("service_area", 0) * 100
            )  # Rough estimate

        return exposed_assets


# Factory functions for creating enhanced exposure models
def create_enhanced_property_exposure_model(
    params: dict[str, Any],
) -> EnhancedPropertyExposureModel:
    """Create an enhanced property exposure model."""
    return EnhancedPropertyExposureModel(params)


def create_enhanced_population_exposure_model(
    params: dict[str, Any],
) -> EnhancedPopulationExposureModel:
    """Create an enhanced population exposure model."""
    return EnhancedPopulationExposureModel(params)


def create_enhanced_infrastructure_exposure_model(
    params: dict[str, Any],
) -> EnhancedInfrastructureExposureModel:
    """Create an enhanced infrastructure exposure model."""
    return EnhancedInfrastructureExposureModel(params)
