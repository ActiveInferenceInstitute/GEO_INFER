"""
Ecological Economics for Bioregional Analysis

This module provides ecological economics modeling and analysis capabilities
for the GEO-INFER framework, focusing on the relationship between economic
systems and ecological systems.
"""

import numpy as np
from typing import cast, Any
from dataclasses import dataclass, field
import logging

from ..utils.rng import resolve_rng

logger = logging.getLogger(__name__)


@dataclass
class EcologicalEconomicsConfig:
    """Configuration for ecological economics models."""

    # Economic parameters
    discount_rate: float = 0.05
    time_horizon: int = 50  # years
    currency: str = "USD"

    # Ecological parameters
    ecosystem_services: list[str] = field(
        default_factory=lambda: ["provisioning", "regulating", "cultural", "supporting"]
    )

    # Valuation parameters
    valuation_methods: list[str] = field(
        default_factory=lambda: [
            "market_price",
            "replacement_cost",
            "travel_cost",
            "hedonic_pricing",
            "contingent_valuation",
            "choice_experiment",
        ]
    )

    # Spatial parameters
    spatial_resolution: float = 0.1  # degrees
    analysis_units: str = "hectares"


class BiophysicalEquilibriumModels:
    """
    Models for biophysical equilibrium analysis in ecological economics.

    Provides methods for analyzing the equilibrium between biological
    and physical systems in economic contexts.
    """

    def __init__(
        self,
        config: EcologicalEconomicsConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize biophysical equilibrium models.

        Args:
            config: Configuration parameters
            rng: Optional random generator for spatial dynamics noise. When
                omitted, a fixed-seed generator is used so simulations are
                deterministic by default.
        """
        self.config = config or EcologicalEconomicsConfig()
        self.models: dict[str, Any] = {}
        self._rng = resolve_rng(rng)
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize equilibrium models."""
        self.models = {
            "lotka_volterra": self._lotka_volterra_model,
            "predator_prey": self._predator_prey_model,
            "resource_competition": self._resource_competition_model,
            "ecosystem_services": self._ecosystem_services_model,
        }
        logger.info("Initialized biophysical equilibrium models")

    def analyze_equilibrium(
        self, model_type: str, parameters: dict[str, Any], time_steps: int = 100
    ) -> dict[str, Any]:
        """
        Analyze equilibrium for a specific model type.

        Args:
            model_type: Type of equilibrium model
            parameters: Model parameters
            time_steps: Number of time steps for simulation

        Returns:
            Equilibrium analysis results
        """
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        model_func = self.models[model_type]
        return cast(dict[str, Any], model_func(parameters, time_steps))

    def _lotka_volterra_model(
        self, parameters: dict[str, Any], time_steps: int
    ) -> dict[str, Any]:
        """Lotka-Volterra predator-prey model."""
        # Extract parameters
        r = parameters.get("growth_rate", 0.5)
        a = parameters.get("predation_rate", 0.1)
        b = parameters.get("conversion_efficiency", 0.2)
        m = parameters.get("mortality_rate", 0.3)

        # Initial populations
        prey = parameters.get("initial_prey", 100)
        predator = parameters.get("initial_predator", 20)

        # Time series
        prey_population = [prey]
        predator_population = [predator]

        dt = 0.1  # Time step

        for _ in range(time_steps):
            # Lotka-Volterra equations
            d_prey = r * prey - a * prey * predator
            d_predator = b * a * prey * predator - m * predator

            prey += d_prey * dt
            predator += d_predator * dt

            # Ensure non-negative populations
            prey = max(0, prey)
            predator = max(0, predator)

            prey_population.append(prey)
            predator_population.append(predator)

        return {
            "model_type": "lotka_volterra",
            "equilibrium_reached": self._check_equilibrium(
                prey_population, predator_population
            ),
            "final_prey": prey_population[-1],
            "final_predator": predator_population[-1],
            "prey_population": prey_population,
            "predator_population": predator_population,
            "parameters": parameters,
        }

    def _predator_prey_model(
        self, parameters: dict[str, Any], time_steps: int
    ) -> dict[str, Any]:
        """Enhanced predator-prey model with carrying capacity."""
        # Extract parameters
        r = parameters.get("growth_rate", 0.5)
        K = parameters.get("carrying_capacity", 200)
        a = parameters.get("predation_rate", 0.1)
        b = parameters.get("conversion_efficiency", 0.2)
        m = parameters.get("mortality_rate", 0.3)

        # Initial populations
        prey = parameters.get("initial_prey", 100)
        predator = parameters.get("initial_predator", 20)

        # Time series
        prey_population = [prey]
        predator_population = [predator]

        dt = 0.1  # Time step

        for _ in range(time_steps):
            # Enhanced predator-prey equations with carrying capacity
            d_prey = r * prey * (1 - prey / K) - a * prey * predator
            d_predator = b * a * prey * predator - m * predator

            prey += d_prey * dt
            predator += d_predator * dt

            # Ensure non-negative populations
            prey = max(0, prey)
            predator = max(0, predator)

            prey_population.append(prey)
            predator_population.append(predator)

        return {
            "model_type": "predator_prey",
            "equilibrium_reached": self._check_equilibrium(
                prey_population, predator_population
            ),
            "final_prey": prey_population[-1],
            "final_predator": predator_population[-1],
            "prey_population": prey_population,
            "predator_population": predator_population,
            "parameters": parameters,
        }

    def _resource_competition_model(
        self, parameters: dict[str, Any], time_steps: int
    ) -> dict[str, Any]:
        """Resource competition model for multiple species."""
        # Extract parameters
        n_species = parameters.get("n_species", 2)
        growth_rates = parameters.get("growth_rates", [0.5, 0.4])
        carrying_capacities = parameters.get("carrying_capacities", [100, 80])
        competition_coefficients = parameters.get(
            "competition_coefficients", [[1.0, 0.5], [0.5, 1.0]]
        )

        # Initial populations
        populations = parameters.get("initial_populations", [50, 40])

        # Time series
        population_history = [populations.copy()]

        dt = 0.1  # Time step

        for _ in range(time_steps):
            new_populations = []

            for i in range(n_species):
                # Competition equation
                d_pop = (
                    growth_rates[i]
                    * populations[i]
                    * (1 - populations[i] / carrying_capacities[i])
                )

                # Competition terms
                for j in range(n_species):
                    if i != j:
                        d_pop -= (
                            growth_rates[i]
                            * populations[i]
                            * competition_coefficients[i][j]
                            * populations[j]
                            / carrying_capacities[i]
                        )

                new_pop = populations[i] + d_pop * dt
                new_pop = max(0, new_pop)  # Ensure non-negative
                new_populations.append(new_pop)

            populations = new_populations
            population_history.append(populations.copy())

        return {
            "model_type": "resource_competition",
            "equilibrium_reached": self._check_equilibrium_multi(population_history),
            "final_populations": populations,
            "population_history": population_history,
            "parameters": parameters,
        }

    def _ecosystem_services_model(
        self, parameters: dict[str, Any], time_steps: int
    ) -> dict[str, Any]:
        """Advanced ecosystem services valuation model with spatial dynamics."""
        # Extract parameters
        service_types = parameters.get(
            "service_types", ["provisioning", "regulating", "cultural"]
        )
        initial_values = parameters.get("initial_values", [100, 80, 60])
        growth_rates = parameters.get("growth_rates", [0.1, 0.05, 0.08])
        interaction_matrix = parameters.get(
            "interaction_matrix", [[1.0, 0.2, 0.1], [0.1, 1.0, 0.3], [0.2, 0.1, 1.0]]
        )

        # Spatial parameters
        diffusion_rate = parameters.get("diffusion_rate", 0.1)

        # Initial service values
        service_values = initial_values.copy()

        # Time series
        value_history = [service_values.copy()]
        spatial_distribution = [service_values.copy()]  # Track spatial distribution

        dt = 0.1  # Time step

        for t in range(time_steps):
            new_values = []

            for i in range(len(service_types)):
                # Base growth
                d_value = growth_rates[i] * service_values[i]

                # Interaction effects
                for j in range(len(service_types)):
                    if i != j:
                        d_value += interaction_matrix[i][j] * service_values[j] * 0.01

                # Spatial diffusion (simplified)
                if t > 0:
                    # Simple diffusion between service types
                    diffusion_effect = diffusion_rate * (
                        np.mean(service_values) - service_values[i]
                    )
                    d_value += diffusion_effect

                new_value = service_values[i] + d_value * dt
                new_value = max(0, new_value)  # Ensure non-negative
                new_values.append(new_value)

            service_values = new_values
            value_history.append(service_values.copy())

            # Update spatial distribution (simplified)
            spatial_dist = [v * (1 + self._rng.normal(0, 0.1)) for v in service_values]
            spatial_distribution.append(spatial_dist)

        # Calculate total economic value
        total_value = sum(service_values)

        # Calculate spatial heterogeneity
        spatial_heterogeneity = np.std(spatial_distribution[-1]) / np.mean(
            spatial_distribution[-1]
        )

        return {
            "model_type": "ecosystem_services",
            "equilibrium_reached": self._check_equilibrium_multi(value_history),
            "final_values": service_values,
            "total_economic_value": total_value,
            "value_history": value_history,
            "spatial_distribution": spatial_distribution,
            "spatial_heterogeneity": spatial_heterogeneity,
            "parameters": parameters,
        }

    def _check_equilibrium(
        self,
        population1: list[float],
        population2: list[float],
        tolerance: float = 0.01,
    ) -> bool:
        """Check if populations have reached equilibrium."""
        if len(population1) < 10 or len(population2) < 10:
            return False

        # Check last 10% of time series
        n_check = max(1, len(population1) // 10)

        recent1 = population1[-n_check:]
        recent2 = population2[-n_check:]

        # Calculate coefficient of variation
        cv1 = (
            np.std(recent1) / np.mean(recent1) if np.mean(recent1) > 0 else float("inf")
        )
        cv2 = (
            np.std(recent2) / np.mean(recent2) if np.mean(recent2) > 0 else float("inf")
        )

        return bool(cv1 < tolerance and cv2 < tolerance)

    def _check_equilibrium_multi(
        self, population_history: list[list[float]], tolerance: float = 0.01
    ) -> bool:
        """Check if multiple populations have reached equilibrium."""
        if len(population_history) < 10:
            return False

        n_check = max(1, len(population_history) // 10)
        recent_history = population_history[-n_check:]

        # Check each population
        for i in range(len(recent_history[0])):
            population = [pop[i] for pop in recent_history]
            cv = (
                np.std(population) / np.mean(population)
                if np.mean(population) > 0
                else float("inf")
            )
            if cv >= tolerance:
                return False

        return True

    def calculate_ecosystem_value(
        self, service_values: list[float], valuation_method: str = "market_price"
    ) -> float:
        """
        Calculate total ecosystem value.

        Args:
            service_values: Values of ecosystem services
            valuation_method: Method for valuation

        Returns:
            Total economic value
        """
        if valuation_method == "market_price":
            # Simple market price valuation
            return sum(service_values)
        elif valuation_method == "replacement_cost":
            # Replacement cost valuation (typically higher)
            return sum(service_values) * 1.5
        elif valuation_method == "contingent_valuation":
            # Contingent valuation (willingness to pay)
            return sum(service_values) * 2.0
        else:
            return sum(service_values)


# Export the main class
__all__ = [
    "BiophysicalEquilibriumModels",
    "EcologicalEconomicsConfig",
    "EcologicalEconomicsEngine",
    "ThermoeconomicModels",
    "EcologicalFootprintAnalysis",
    "CarryingCapacityModels",
]


class ThermoeconomicModels:
    """Energy-flow economics using emergy analysis and exergy efficiency.

    Models the thermodynamic basis of economic value, tracking energy
    transformations across ecological-economic systems.
    """

    def __init__(self, config: EcologicalEconomicsConfig | None = None):
        self.config = config or EcologicalEconomicsConfig()
        # Solar emjoules per joule for common energy forms
        self._transformities: dict[str, float] = {
            "sunlight": 1.0,
            "wind": 1_496.0,
            "rain_chemical": 18_199.0,
            "wood": 34_500.0,
            "coal": 40_000.0,
            "electricity": 174_000.0,
            "human_labor": 6_800_000.0,
        }
        logger.info("ThermoeconomicModels initialized")

    def emergy_analysis(self, flows: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute total emergy (solar emjoules) for a set of energy flows.

        Args:
            flows: List of dicts with ``energy_type``, ``joules``.

        Returns:
            Dict with per-flow emergy, total emergy, and emergy yield ratio.
        """
        results = []
        total_emergy = 0.0
        renewable_emergy = 0.0

        for flow in flows:
            energy_type = flow.get("energy_type", "sunlight")
            joules = float(flow.get("joules", 0))
            transformity = self._transformities.get(energy_type, 1.0)
            emergy = joules * transformity
            total_emergy += emergy
            if energy_type in {"sunlight", "wind", "rain_chemical"}:
                renewable_emergy += emergy
            results.append(
                {
                    "energy_type": energy_type,
                    "joules": joules,
                    "transformity": transformity,
                    "emergy_sej": round(emergy, 2),
                }
            )

        # Emergy Yield Ratio = total emergy / purchased emergy
        purchased_emergy = total_emergy - renewable_emergy
        eyr = (
            round(total_emergy / purchased_emergy, 4)
            if purchased_emergy > 0
            else float("inf")
        )

        return {
            "flows": results,
            "total_emergy_sej": round(total_emergy, 2),
            "renewable_fraction": round(renewable_emergy / max(total_emergy, 1e-10), 4),
            "emergy_yield_ratio": eyr,
        }

    def exergy_efficiency(
        self,
        energy_input: float,
        useful_work: float,
        waste_heat: float,
    ) -> dict[str, Any]:
        """Compute exergy (second-law) efficiency.

        Args:
            energy_input: Total energy input (joules).
            useful_work: Useful work extracted (joules).
            waste_heat: Waste heat dissipated (joules).

        Returns:
            Dict with first-law and second-law efficiencies.
        """
        first_law = useful_work / max(energy_input, 1e-10)
        exergy_available = energy_input - waste_heat
        second_law = useful_work / max(exergy_available, 1e-10)

        return {
            "energy_input": energy_input,
            "useful_work": useful_work,
            "waste_heat": waste_heat,
            "first_law_efficiency": round(first_law, 4),
            "second_law_efficiency": round(min(second_law, 1.0), 4),
        }


class EcologicalFootprintAnalysis:
    """Ecological footprint vs biocapacity accounting.

    Computes per-capita and aggregate footprints across six land-use types
    following the Global Footprint Network methodology.
    """

    LAND_USE_TYPES = [
        "cropland",
        "grazing",
        "forest",
        "fishing",
        "built_up",
        "carbon",
    ]

    # Default global-average yield factors (gha / ha)
    _DEFAULT_YIELD_FACTORS: dict[str, float] = {
        "cropland": 2.51,
        "grazing": 0.46,
        "forest": 1.26,
        "fishing": 0.37,
        "built_up": 2.51,
        "carbon": 1.26,
    }

    # Equivalence factors (global hectares per bioproductive hectare)
    _DEFAULT_EQ_FACTORS: dict[str, float] = {
        "cropland": 2.51,
        "grazing": 0.46,
        "forest": 1.26,
        "fishing": 0.37,
        "built_up": 2.51,
        "carbon": 1.26,
    }

    def __init__(self, config: EcologicalEconomicsConfig | None = None):
        self.config = config or EcologicalEconomicsConfig()
        logger.info("EcologicalFootprintAnalysis initialized")

    def compute_footprint(
        self,
        consumption: dict[str, float],
        population: float = 1.0,
        yield_factors: dict[str, float] | None = None,
        equivalence_factors: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compute ecological footprint in global hectares.

        Args:
            consumption: Dict mapping land-use type → hectares consumed.
            population: Population for per-capita calculation.
            yield_factors: Optional custom yield factors.
            equivalence_factors: Optional custom equivalence factors.

        Returns:
            Dict with per-type footprint, total, and per-capita values.
        """
        yf = yield_factors or self._DEFAULT_YIELD_FACTORS
        ef = equivalence_factors or self._DEFAULT_EQ_FACTORS

        breakdown: dict[str, float] = {}
        for lut in self.LAND_USE_TYPES:
            area = float(consumption.get(lut, 0))
            gha = area * yf.get(lut, 1.0) * ef.get(lut, 1.0)
            breakdown[lut] = round(gha, 4)

        total_gha = sum(breakdown.values())
        per_capita = total_gha / max(population, 1)

        return {
            "footprint_by_type_gha": breakdown,
            "total_footprint_gha": round(total_gha, 4),
            "per_capita_gha": round(per_capita, 4),
            "population": population,
        }

    def compute_biocapacity(
        self,
        areas: dict[str, float],
        yield_factors: dict[str, float] | None = None,
        equivalence_factors: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compute biocapacity in global hectares.

        Args:
            areas: Dict mapping land-use type → bioproductive hectares available.
            yield_factors: Optional custom yield factors.
            equivalence_factors: Optional custom equivalence factors.

        Returns:
            Dict with per-type biocapacity and total.
        """
        yf = yield_factors or self._DEFAULT_YIELD_FACTORS
        ef = equivalence_factors or self._DEFAULT_EQ_FACTORS

        breakdown: dict[str, float] = {}
        for lut in self.LAND_USE_TYPES:
            area = float(areas.get(lut, 0))
            gha = area * yf.get(lut, 1.0) * ef.get(lut, 1.0)
            breakdown[lut] = round(gha, 4)

        return {
            "biocapacity_by_type_gha": breakdown,
            "total_biocapacity_gha": round(sum(breakdown.values()), 4),
        }

    def overshoot_analysis(
        self,
        footprint_gha: float,
        biocapacity_gha: float,
    ) -> dict[str, Any]:
        """Determine ecological overshoot or reserve.

        Args:
            footprint_gha: Total ecological footprint.
            biocapacity_gha: Total biocapacity.

        Returns:
            Dict with deficit/reserve, number of earths, and overshoot day.
        """
        balance = biocapacity_gha - footprint_gha
        n_earths = footprint_gha / max(biocapacity_gha, 1e-10)
        overshoot_day = int(365 / max(n_earths, 1e-10)) if n_earths > 1 else None

        return {
            "ecological_balance_gha": round(balance, 4),
            "status": "reserve" if balance >= 0 else "deficit",
            "number_of_earths": round(n_earths, 4),
            "overshoot_day": overshoot_day,
        }


class CarryingCapacityModels:
    """Population and resource carrying capacity estimation.

    Uses logistic growth with resource constraints to estimate the maximum
    sustainable population or throughput for a bioregion.
    """

    def __init__(self, config: EcologicalEconomicsConfig | None = None):
        self.config = config or EcologicalEconomicsConfig()
        logger.info("CarryingCapacityModels initialized")

    def estimate_carrying_capacity(
        self,
        resources: list[dict[str, Any]],
        current_population: float,
        safety_margin: float = 0.2,
    ) -> dict[str, Any]:
        """Estimate carrying capacity from multiple resource constraints.

        Args:
            resources: List of dicts with ``name``, ``available``,
                       ``per_capita_requirement``.
            current_population: Current population count.
            safety_margin: Fraction reserved as buffer (0-1).

        Returns:
            Dict with per-resource capacity, binding constraint, and status.
        """
        capacities: dict[str, float] = {}
        for r in resources:
            name = r.get("name", "unknown")
            available = float(r.get("available", 0))
            per_cap = float(r.get("per_capita_requirement", 1))
            usable = available * (1 - safety_margin)
            cap = usable / max(per_cap, 1e-10)
            capacities[name] = round(cap, 2)

        binding_resource = min(capacities, key=lambda x: capacities[x])
        carrying_cap = capacities[binding_resource]
        utilisation = current_population / max(carrying_cap, 1e-10)

        return {
            "per_resource_capacity": capacities,
            "carrying_capacity": round(carrying_cap, 2),
            "binding_resource": binding_resource,
            "current_population": current_population,
            "utilisation_ratio": round(utilisation, 4),
            "status": "within_capacity" if utilisation <= 1.0 else "over_capacity",
            "safety_margin": safety_margin,
        }

    def simulate_logistic_growth(
        self,
        initial_population: float,
        carrying_capacity: float,
        growth_rate: float = 0.03,
        time_horizon: int | None = None,
    ) -> dict[str, Any]:
        """Simulate logistic population growth toward carrying capacity.

        Args:
            initial_population: Starting population.
            carrying_capacity: Maximum sustainable population K.
            growth_rate: Intrinsic growth rate r.
            time_horizon: Years to simulate.

        Returns:
            Dict with time-series, half-saturation year, and equilibrium flag.
        """
        T = time_horizon or self.config.time_horizon
        pop = float(initial_population)
        K = float(carrying_capacity)
        series = [round(pop, 2)]

        half_k_year: int | None = None

        for year in range(1, T + 1):
            dpop = growth_rate * pop * (1 - pop / K)
            pop = max(0.0, pop + dpop)
            series.append(round(pop, 2))
            if half_k_year is None and pop >= K / 2:
                half_k_year = year

        equilibrium = abs(series[-1] - K) / K < 0.01

        return {
            "population_series": series,
            "final_population": series[-1],
            "carrying_capacity": K,
            "growth_rate": growth_rate,
            "half_saturation_year": half_k_year,
            "equilibrium_reached": equilibrium,
            "years": T,
        }


class EcologicalEconomicsEngine:
    """Unified orchestrator for ecological economics analyses.

    Composes BiophysicalEquilibrium, ThermoeconomicModels,
    EcologicalFootprintAnalysis, and CarryingCapacityModels into a
    single entry-point.
    """

    def __init__(
        self,
        config: EcologicalEconomicsConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize the ecological economics engine.

        Args:
            config: Configuration parameters.
            rng: Optional random generator threaded into stochastic
                sub-models. When omitted, a fixed-seed generator is used so
                analyses are deterministic by default.
        """
        self.config = config or EcologicalEconomicsConfig()
        self.biophysical = BiophysicalEquilibriumModels(self.config, rng=rng)
        self.thermodynamics = ThermoeconomicModels(self.config)
        self.footprint = EcologicalFootprintAnalysis(self.config)
        self.carrying_capacity = CarryingCapacityModels(self.config)
        logger.info("EcologicalEconomicsEngine initialized")

    def run_analysis(self, analysis_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """Run an ecological economics analysis by type.

        Args:
            analysis_type: One of ``'equilibrium'``, ``'emergy'``,
                ``'footprint'``, ``'carrying_capacity'``.
            data: Parameters for the chosen analysis.

        Returns:
            Analysis results dict.

        Raises:
            ValueError: If *analysis_type* is unknown.
        """
        dispatch: dict[str, Any] = {
            "equilibrium": self._run_equilibrium,
            "emergy": self._run_emergy,
            "footprint": self._run_footprint,
            "carrying_capacity": self._run_carrying_capacity,
        }
        if analysis_type not in dispatch:
            raise ValueError(
                f"Unknown analysis type '{analysis_type}'. "
                f"Choose from: {list(dispatch.keys())}"
            )
        logger.info("Running ecological economics analysis: %s", analysis_type)
        return cast(dict[str, Any], dispatch[analysis_type](data))

    # ------------------------------------------------------------------
    # Private dispatch targets
    # ------------------------------------------------------------------

    def _run_equilibrium(self, data: dict[str, Any]) -> dict[str, Any]:
        model_type = data.pop("model_type", "lotka_volterra")
        time_steps = data.pop("time_steps", 100)
        return self.biophysical.analyze_equilibrium(model_type, data, time_steps)

    def _run_emergy(self, data: dict[str, Any]) -> dict[str, Any]:
        flows = data.get("flows", [])
        return self.thermodynamics.emergy_analysis(flows)

    def _run_footprint(self, data: dict[str, Any]) -> dict[str, Any]:
        consumption = data.get("consumption", {})
        population = data.get("population", 1.0)
        fp = self.footprint.compute_footprint(consumption, population)

        areas = data.get("biocapacity_areas")
        if areas:
            bc = self.footprint.compute_biocapacity(areas)
            overshoot = self.footprint.overshoot_analysis(
                fp["total_footprint_gha"], bc["total_biocapacity_gha"]
            )
            fp.update(bc)
            fp.update(overshoot)

        return fp

    def _run_carrying_capacity(self, data: dict[str, Any]) -> dict[str, Any]:
        resources = data.get("resources", [])
        current_pop = data.get("current_population", 0)
        safety = data.get("safety_margin", 0.2)
        return self.carrying_capacity.estimate_carrying_capacity(
            resources, current_pop, safety
        )
