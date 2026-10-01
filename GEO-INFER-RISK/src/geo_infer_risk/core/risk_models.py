"""
Geospatial risk modeling components for the GEO-INFER-RISK module.

This module provides classes for modeling risk across geographic areas,
including hazard identification, vulnerability assessment, and exposure calculation.
"""

import numpy as np
import geopandas as gpd
import logging
import inspect
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from ..utils.rng import SeedLike, resolve_rng

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Parameters for defining risk model behavior."""

    confidence_level: float = 0.95
    time_horizon: int = 50  # years
    spatial_resolution: float = 1.0  # km
    monte_carlo_iterations: int = 1000
    random_seed: SeedLike = None

    def __post_init__(self) -> None:
        """Validate the numerical contract before a model uses it."""
        if not np.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        if (
            not isinstance(self.time_horizon, (int, np.integer))
            or self.time_horizon < 1
        ):
            raise ValueError("time_horizon must be a positive integer")
        if not np.isfinite(self.spatial_resolution) or self.spatial_resolution <= 0:
            raise ValueError("spatial_resolution must be finite and positive")
        if (
            not isinstance(self.monte_carlo_iterations, (int, np.integer))
            or self.monte_carlo_iterations < 1
        ):
            raise ValueError("monte_carlo_iterations must be a positive integer")
        # resolve_rng owns seed validation; calling it here surfaces a bad
        # seed at construction rather than at the first draw.
        resolve_rng(self.random_seed)


class RiskModel:
    """Base class for all geospatial risk models."""

    def __init__(self, parameters: RiskParameters | None = None):
        """Initialize a risk model with configurable parameters.

        Args:
            parameters: Model configuration parameters
        """
        self.parameters = parameters or RiskParameters()
        self.rng = resolve_rng(self.parameters.random_seed)
        # Components are attached by the set_* methods before assess() runs.
        self.hazard: HazardModel | None = None
        self.vulnerability: VulnerabilityModel | None = None
        self.exposure: ExposureModel | None = None
        logger.info(
            "RiskModel initialized with %d-year horizon", self.parameters.time_horizon
        )

    def set_hazard(self, hazard_model: "HazardModel") -> None:
        """Set the hazard component of the risk model.

        Args:
            hazard_model: A hazard model instance
        """
        self.hazard = hazard_model
        logger.info("Hazard model set: %s", type(hazard_model).__name__)

    def set_vulnerability(self, vulnerability_model: "VulnerabilityModel") -> None:
        """Set the vulnerability component of the risk model.

        Args:
            vulnerability_model: A vulnerability model instance
        """
        self.vulnerability = vulnerability_model
        logger.info("Vulnerability model set: %s", type(vulnerability_model).__name__)

    def set_exposure(self, exposure_model: "ExposureModel") -> None:
        """Set the exposure component of the risk model.

        Args:
            exposure_model: An exposure model instance
        """
        self.exposure = exposure_model
        logger.info("Exposure model set: %s", type(exposure_model).__name__)

    def calculate_risk(
        self, geometry: gpd.GeoDataFrame | gpd.GeoSeries
    ) -> gpd.GeoDataFrame:
        """Calculate risk for the given geographic area.

        Args:
            geometry: Geographic areas to assess risk for

        Returns:
            GeoDataFrame with risk metrics for each area
        """
        hazard_model, vulnerability_model, exposure_model = (
            self.hazard,
            self.vulnerability,
            self.exposure,
        )
        if (
            hazard_model is None
            or vulnerability_model is None
            or exposure_model is None
        ):
            raise ValueError("Hazard, vulnerability, and exposure models must be set")

        logger.info("Calculating risk for %d geometries", len(geometry))

        # Calculate risk components and validate their shared spatial contract.
        hazard_data = hazard_model.calculate(geometry)
        vulnerability_data = vulnerability_model.calculate(geometry)
        exposure_data = exposure_model.calculate(geometry)
        hazard = self._component_values(
            hazard_data, geometry, "hazard_probability", "hazard"
        )
        vulnerability = self._component_values(
            vulnerability_data, geometry, "vulnerability_index", "vulnerability"
        )
        exposure = self._component_values(
            exposure_data, geometry, "exposure_value", "exposure"
        )

        # Combine components to produce risk
        risk_data = hazard_data.copy()
        risk_data["risk_score"] = hazard * vulnerability * exposure

        lower_values = []
        upper_values = []
        has_component_bounds = False
        for frame, column, label, point in (
            (hazard_data, "hazard_probability", "hazard", hazard),
            (vulnerability_data, "vulnerability_index", "vulnerability", vulnerability),
            (exposure_data, "exposure_value", "exposure", exposure),
        ):
            lower, upper, has_bounds = self._component_bounds(
                frame, column, label, point
            )
            lower_values.append(lower)
            upper_values.append(upper)
            has_component_bounds = has_component_bounds or has_bounds

        risk_data["risk_lower_bound"] = np.prod(lower_values, axis=0)
        risk_data["risk_upper_bound"] = np.prod(upper_values, axis=0)
        risk_data["uncertainty_source"] = (
            "component_bounds" if has_component_bounds else "point_estimate"
        )

        logger.info("Risk calculated: mean=%.4f", risk_data["risk_score"].mean())
        return risk_data

    def run_monte_carlo(self, geometry: gpd.GeoDataFrame) -> dict:
        """Run Monte Carlo simulations for risk assessment.

        Args:
            geometry: Geographic areas to assess risk for

        Returns:
            Dictionary with simulation results
        """
        if geometry is None or len(geometry) == 0:
            raise ValueError("geometry must contain at least one area")
        if not all([self.hazard, self.vulnerability, self.exposure]):
            raise ValueError("Hazard, vulnerability, and exposure models must be set")

        # Ensure component sample sizes are aligned with the requested geometry.
        self.calculate_risk(geometry)
        logger.info(
            "Running %d Monte Carlo iterations", self.parameters.monte_carlo_iterations
        )
        results = []
        for i in range(self.parameters.monte_carlo_iterations):
            # Generate random variations in hazard, vulnerability and exposure.
            hazard_variation = self._sample_component(self.hazard, len(geometry))
            vulnerability_variation = self._sample_component(
                self.vulnerability, len(geometry)
            )
            exposure_variation = self._sample_component(self.exposure, len(geometry))

            # Calculate combined risk
            risk = hazard_variation * vulnerability_variation * exposure_variation
            results.append(risk)

        # Process results
        results_array = np.array(results)
        lower_percentile = 100 * (1 - self.parameters.confidence_level) / 2
        upper_percentile = 100 - lower_percentile
        result = {
            "mean": np.mean(results_array, axis=0),
            "median": np.median(results_array, axis=0),
            "std_dev": np.std(results_array, axis=0),
            "lower_bound": np.percentile(results_array, lower_percentile, axis=0),
            "upper_bound": np.percentile(results_array, upper_percentile, axis=0),
            # Preserve the established names while exposing the configured CI.
            "percentile_95": np.percentile(results_array, 95, axis=0),
            "percentile_5": np.percentile(results_array, 5, axis=0),
            "confidence_level": self.parameters.confidence_level,
        }
        logger.info(
            "Monte Carlo complete: mean risk=%.4f", float(np.mean(result["mean"]))
        )
        return result

    @staticmethod
    def _component_values(
        frame: gpd.GeoDataFrame,
        geometry: gpd.GeoDataFrame | gpd.GeoSeries,
        column: str,
        label: str,
    ) -> np.ndarray:
        """Validate and return one aligned, finite component vector."""
        if column not in frame:
            raise ValueError(f"{label} output must contain '{column}'")
        if len(frame) != len(geometry) or not frame.index.equals(geometry.index):
            raise ValueError(f"{label} output must preserve geometry length and index")
        values = np.asarray(frame[column], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{label} values must be finite and non-negative")
        if label == "hazard" and np.any(values > 1):
            raise ValueError("hazard_probability values must be in [0, 1]")
        if label == "vulnerability" and np.any(values > 1):
            raise ValueError("vulnerability_index values must be in [0, 1]")
        return values

    @staticmethod
    def _component_bounds(
        frame: gpd.GeoDataFrame,
        column: str,
        label: str,
        point: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Read optional component bounds without inventing uncertainty."""
        lower_name = f"{column}_lower_bound"
        upper_name = f"{column}_upper_bound"
        present = {lower_name in frame, upper_name in frame}
        if present == {False}:
            return point, point, False
        if present != {True}:
            raise ValueError(
                f"{label} output must provide both {lower_name} and {upper_name}"
            )
        lower = np.asarray(frame[lower_name], dtype=float)
        upper = np.asarray(frame[upper_name], dtype=float)
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower < 0)
            or np.any(upper < lower)
            or lower.shape != point.shape
            or upper.shape != point.shape
            or np.any(point < lower)
            or np.any(point > upper)
        ):
            raise ValueError(
                f"{label} bounds must align with and contain the point estimate"
            )
        if label in {"hazard", "vulnerability"} and np.any(upper > 1):
            raise ValueError(f"{label} upper bounds must be in [0, 1]")
        return lower, upper, True

    def _sample_component(self, component: Any, size: int) -> np.ndarray:
        """Sample a component with the model RNG while retaining old call shapes."""
        sample_method = component.sample
        parameters = inspect.signature(sample_method).parameters
        if "random_state" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            values = sample_method(random_state=self.rng)
        else:
            values = sample_method()
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 1 and size > 1:
            values = np.repeat(values, size)
        if values.size != size or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(
                "component samples must be finite, non-negative, and aligned"
            )
        return values


class HazardModel(ABC):
    """Base class for modeling hazard probability in geographic areas."""

    def __init__(self, hazard_type: str, return_period: int = 100):
        """Initialize a hazard model.

        Args:
            hazard_type: Type of hazard (flood, earthquake, wildfire, etc.)
            return_period: Return period in years for hazard probability
        """
        self.hazard_type = hazard_type
        self.return_period = return_period
        logger.info(
            "HazardModel initialized: type=%s, return_period=%d",
            hazard_type,
            return_period,
        )

    @abstractmethod
    def calculate(self, geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate hazard probability for given areas.

        Args:
            geometry: Geographic areas to assess

        Returns:
            GeoDataFrame with hazard probabilities
        """
        raise RuntimeError(
            "HazardModel.calculate must be implemented by a concrete hazard model"
        )

    @abstractmethod
    def sample(self, random_state: SeedLike = None) -> np.ndarray:
        """Generate a random sample from the hazard model for Monte Carlo simulation.

        Args:
            random_state: Seed or generator for the draw; see
                :func:`geo_infer_risk.utils.rng.resolve_rng`. Concrete models
                must draw only from it, never from the ``numpy.random``
                singleton, so a Monte Carlo run is replayable from its seed.

        Returns:
            Array of sampled hazard values
        """
        raise RuntimeError(
            "HazardModel.sample must be implemented by a concrete hazard model"
        )


class VulnerabilityModel(ABC):
    """Base class for modeling vulnerability of assets or populations."""

    def __init__(self, vulnerability_factors: list[str]):
        """Initialize a vulnerability model.

        Args:
            vulnerability_factors: List of factors that contribute to vulnerability
        """
        self.vulnerability_factors = vulnerability_factors
        logger.info(
            "VulnerabilityModel initialized: %d factors", len(vulnerability_factors)
        )

    @abstractmethod
    def calculate(self, geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate vulnerability indices for given areas.

        Args:
            geometry: Geographic areas to assess

        Returns:
            GeoDataFrame with vulnerability indices
        """
        raise RuntimeError(
            "VulnerabilityModel.calculate must be implemented by a concrete vulnerability model"
        )

    @abstractmethod
    def sample(self, random_state: SeedLike = None) -> np.ndarray:
        """Generate a random sample from the vulnerability model for Monte Carlo simulation.

        Args:
            random_state: Seed or generator for the draw; see
                :func:`geo_infer_risk.utils.rng.resolve_rng`. Concrete models
                must draw only from it, never from the ``numpy.random``
                singleton, so a Monte Carlo run is replayable from its seed.

        Returns:
            Array of sampled vulnerability values
        """
        raise RuntimeError(
            "VulnerabilityModel.sample must be implemented by a concrete vulnerability model"
        )


class ExposureModel(ABC):
    """Base class for modeling exposure (assets, population, etc.)."""

    def __init__(self, exposure_type: str):
        """Initialize an exposure model.

        Args:
            exposure_type: Type of exposure (buildings, population, infrastructure, etc.)
        """
        self.exposure_type = exposure_type
        logger.info("ExposureModel initialized: type=%s", exposure_type)

    @abstractmethod
    def calculate(self, geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate exposure values for given areas.

        Args:
            geometry: Geographic areas to assess

        Returns:
            GeoDataFrame with exposure values
        """
        raise RuntimeError(
            "ExposureModel.calculate must be implemented by a concrete exposure model"
        )

    @abstractmethod
    def sample(self, random_state: SeedLike = None) -> np.ndarray:
        """Generate a random sample from the exposure model for Monte Carlo simulation.

        Args:
            random_state: Seed or generator for the draw; see
                :func:`geo_infer_risk.utils.rng.resolve_rng`. Concrete models
                must draw only from it, never from the ``numpy.random``
                singleton, so a Monte Carlo run is replayable from its seed.

        Returns:
            Array of sampled exposure values
        """
        raise RuntimeError(
            "ExposureModel.sample must be implemented by a concrete exposure model"
        )


# ==================== Concrete Implementations ====================


class FloodHazardModel(HazardModel):
    """Flood hazard model using depth-frequency analysis.

    Estimates flood probability and depth based on return period,
    elevation, and proximity to water bodies. Uses the Gumbel
    extreme value distribution for flood frequency analysis.
    """

    def __init__(self, return_period: int = 100, base_depth_m: float = 2.0):
        """Initialize flood hazard model.

        Args:
            return_period: Return period in years
            base_depth_m: Reference flood depth in meters for the base return period
        """
        super().__init__("flood", return_period)
        self.base_depth_m = base_depth_m
        self._n_samples = 1  # Updated after first calculate call

    def calculate(self, geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate flood hazard probability per geometry.

        Uses annual exceedance probability P = 1 / return_period and
        assigns a depth estimate based on centroid elevation (lower → deeper).

        Args:
            geometry: GeoDataFrame of areas to assess

        Returns:
            GeoDataFrame with hazard_probability & estimated_depth columns
        """
        result = geometry.copy()
        n = len(result)
        self._n_samples = n

        # Annual exceedance probability
        annual_prob = 1.0 / self.return_period

        # Simple elevation proxy: use centroid Y coordinate normalized. GeoPandas
        # intentionally warns when centroid is computed directly in geographic
        # coordinates, so project geographic inputs before the planar operation.
        geometry_series = result.geometry
        if geometry_series.crs is not None and geometry_series.crs.is_geographic:
            projected_crs = geometry_series.estimate_utm_crs()
            if projected_crs is not None:
                geometry_series = geometry_series.to_crs(projected_crs)
        centroids = geometry_series.centroid
        y_vals = centroids.y.values
        y_range = max(y_vals.max() - y_vals.min(), 1e-6)
        # Lower elevation → higher flood probability
        elevation_factor = 1.0 - (y_vals - y_vals.min()) / y_range

        result["hazard_probability"] = annual_prob * (0.5 + 0.5 * elevation_factor)
        result["estimated_depth_m"] = self.base_depth_m * elevation_factor

        logger.info(
            "FloodHazardModel: %d areas, mean P=%.4f",
            n,
            result["hazard_probability"].mean(),
        )
        return result

    def sample(self, random_state: SeedLike = None) -> np.ndarray:
        """Sample flood hazard values from Gumbel distribution."""
        rng = resolve_rng(random_state)
        mu = 1.0 / self.return_period
        beta = mu * 0.3  # Scale parameter
        return rng.gumbel(loc=mu, scale=beta, size=self._n_samples).clip(0, 1)


class BuildingVulnerabilityModel(VulnerabilityModel):
    """Building vulnerability model based on structural factors.

    Models vulnerability as a composite index of building age,
    construction material, stories, and maintenance condition.
    """

    # Default fragility curve parameters by material type
    MATERIAL_FACTORS = {
        "reinforced_concrete": 0.2,
        "steel_frame": 0.25,
        "masonry": 0.5,
        "wood": 0.6,
        "adobe": 0.8,
        "informal": 0.9,
    }

    def __init__(self, factors: list[str] | None = None):
        """Initialize building vulnerability model.

        Args:
            factors: Vulnerability factors to consider
        """
        super().__init__(factors or ["age", "material", "stories", "condition"])
        self._n_samples = 1

    def calculate(self, geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate vulnerability index for each area.

        If the GeoDataFrame contains 'building_material', 'building_age',
        'building_stories', or 'building_condition' columns, those are used.
        Otherwise sensible defaults are applied.

        Returns:
            GeoDataFrame with vulnerability_index (0-1) column
        """
        result = geometry.copy()
        n = len(result)
        self._n_samples = n

        # Material factor
        if "building_material" in result.columns:
            mat_scores = (
                result["building_material"].map(self.MATERIAL_FACTORS).fillna(0.5)
            )
        else:
            mat_scores = 0.5

        # Age factor (older → more vulnerable)
        if "building_age" in result.columns:
            age_scores = result["building_age"].clip(0, 100) / 100.0
        else:
            age_scores = 0.4

        # Condition factor (1 = good, 5 = poor)
        if "building_condition" in result.columns:
            cond_scores = (result["building_condition"].clip(1, 5) - 1) / 4.0
        else:
            cond_scores = 0.3

        result["vulnerability_index"] = (
            0.4 * np.array(mat_scores)
            + 0.3 * np.array(age_scores)
            + 0.3 * np.array(cond_scores)
        ).clip(0, 1)

        logger.info(
            "BuildingVulnerability: %d areas, mean V=%.4f",
            n,
            result["vulnerability_index"].mean(),
        )
        return result

    def sample(self, random_state: SeedLike = None) -> np.ndarray:
        """Sample vulnerability values from Beta distribution."""
        rng = resolve_rng(random_state)
        return rng.beta(2, 5, size=self._n_samples)


class PopulationExposureModel(ExposureModel):
    """Population exposure model based on density and demographics.

    Estimates exposure value as a function of population density,
    per-capita income, and critical infrastructure presence.
    """

    def __init__(self, income_per_capita: float = 30000.0):
        """Initialize population exposure model.

        Args:
            income_per_capita: Regional per-capita income (USD)
        """
        super().__init__("population")
        self.income_per_capita = income_per_capita
        self._n_samples = 1

    def calculate(self, geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate exposure values for each area.

        If the GeoDataFrame has 'population' and/or 'area_km2' columns,
        those are used. Otherwise uniform defaults are applied.

        Returns:
            GeoDataFrame with exposure_value column (monetary estimate)
        """
        result = geometry.copy()
        n = len(result)
        self._n_samples = n

        if "population" in result.columns:
            pop = result["population"].values.astype(float)
        else:
            pop = np.full(n, 1000.0)

        # Exposure is the configured population times regional unit income.
        result["exposure_value"] = pop * self.income_per_capita

        # Normalize to 0-1 for multiplicative risk model
        max_exp = result["exposure_value"].max()
        if max_exp > 0:
            result["exposure_value"] = result["exposure_value"] / max_exp

        logger.info(
            "PopulationExposure: %d areas, mean E=%.4f",
            n,
            result["exposure_value"].mean(),
        )
        return result

    def sample(self, random_state: SeedLike = None) -> np.ndarray:
        """Sample exposure values from log-normal distribution."""
        rng = resolve_rng(random_state)
        return rng.lognormal(mean=0, sigma=0.3, size=self._n_samples).clip(0, 1)
