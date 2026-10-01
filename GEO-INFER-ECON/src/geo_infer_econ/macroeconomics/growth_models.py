"""
Macroeconomic Growth Models Module

Implements comprehensive growth models including:
- Solow growth model with spatial extensions
- Endogenous growth models (AK, R&D-based)
- Regional convergence analysis
- Technology diffusion models
- Spatial growth spillovers
"""

import numpy as np
import pandas as pd
from typing import Any, cast
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class RegionProfile:
    """Profile of a region for macroeconomic analysis"""

    region_id: str
    initial_capital: float
    initial_output: float
    population: float
    technology_level: float
    location: tuple[float, float]  # (lat, lon)
    institutions: dict[str, float]  # quality indices
    natural_resources: dict[str, float]
    connectivity: dict[str, float]  # infrastructure connectivity measures


class SolowGrowthModel:
    """
    Implementation of the Solow growth model with spatial extensions
    """

    def __init__(self, parameters: dict[str, float] | None = None):
        """
        Initialize Solow model with parameters

        Args:
            parameters: Dictionary with model parameters
                - alpha: Capital share (0 < alpha < 1)
                - s: Savings rate (0 < s < 1)
                - n: Population growth rate
                - delta: Depreciation rate
                - g: Technology growth rate
        """
        self.params = parameters or {
            "alpha": 0.33,
            "s": 0.2,
            "n": 0.02,
            "delta": 0.05,
            "g": 0.02,
        }

        self.steady_state: dict[str, float] = {}
        self.convergence_rate: float | None = None

    def production_function(self, K: float, L: float, A: float = 1.0) -> float:
        """
        Cobb-Douglas production function: Y = A * K^α * L^(1-α)

        Args:
            K: Capital stock
            L: Labor force
            A: Technology level

        Returns:
            Output level
        """
        return cast(
            float, A * (K ** self.params["alpha"]) * (L ** (1 - self.params["alpha"]))
        )

    def capital_dynamics(self, K: float, L: float, A: float = 1.0) -> float:
        """
        Capital accumulation equation: dK/dt = s*Y - (n + δ + g)*K

        Args:
            K: Capital stock
            L: Labor force
            A: Technology level

        Returns:
            Change in capital stock
        """
        Y = self.production_function(K, L, A)
        return (
            self.params["s"] * Y
            - (self.params["n"] + self.params["delta"] + self.params["g"]) * K
        )

    def steady_state_values(self) -> dict[str, float]:
        """
        Calculate steady-state values

        Returns:
            Dictionary with steady-state capital, output, consumption
        """
        # Steady-state capital per effective worker
        k_star = (
            self.params["s"]
            / (self.params["n"] + self.params["delta"] + self.params["g"])
        ) ** (1 / (1 - self.params["alpha"]))

        # Steady-state output per effective worker
        y_star = k_star ** self.params["alpha"]

        # Steady-state consumption per effective worker
        c_star = (1 - self.params["s"]) * y_star

        self.steady_state = {
            "capital_per_worker": k_star,
            "output_per_worker": y_star,
            "consumption_per_worker": c_star,
        }

        return self.steady_state

    def convergence_analysis(self, initial_capital_ratio: float) -> dict[str, Any]:
        """
        Analyze convergence to steady state

        Args:
            initial_capital_ratio: Initial capital as ratio of steady state

        Returns:
            Dictionary with convergence analysis results
        """
        # Calculate convergence rate (lambda)
        self.convergence_rate = (1 - self.params["alpha"]) * (
            self.params["n"] + self.params["delta"] + self.params["g"]
        )

        # Half-life of convergence
        half_life = np.log(2) / self.convergence_rate

        return {
            "convergence_rate": self.convergence_rate,
            "half_life_years": half_life,
            "initial_capital_ratio": initial_capital_ratio,
            "time_to_90_percent": np.log(10) / self.convergence_rate,
        }

    def simulate_growth_path(
        self, initial_conditions: dict[str, float], time_horizon: int = 50
    ) -> pd.DataFrame:
        """
        Simulate growth path over time

        Args:
            initial_conditions: Dictionary with initial K, L, A
            time_horizon: Number of years to simulate

        Returns:
            DataFrame with time series of economic variables
        """

        def system_dynamics(t: float, y: Any) -> list[float]:
            K, L, A = y
            dK_dt = self.capital_dynamics(K, L, A)
            dL_dt = self.params["n"] * L
            dA_dt = self.params["g"] * A
            return [dK_dt, dL_dt, dA_dt]

        # Time points
        t_span = (0, time_horizon)
        t_eval = np.linspace(0, time_horizon, time_horizon + 1)

        # Initial conditions
        y0 = [initial_conditions["K"], initial_conditions["L"], initial_conditions["A"]]

        # Solve system
        solution = solve_ivp(system_dynamics, t_span, y0, t_eval=t_eval)

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "year": t_eval,
                "capital": solution.y[0],
                "labor": solution.y[1],
                "technology": solution.y[2],
            }
        )

        # Calculate derived variables
        results["output"] = self.production_function(
            results["capital"], results["labor"], results["technology"]
        )
        results["output_per_worker"] = results["output"] / results["labor"]
        results["capital_per_worker"] = results["capital"] / results["labor"]
        results["consumption"] = (1 - self.params["s"]) * results["output"]

        return results


class SpatialGrowthModels:
    """
    Spatial extensions of growth models incorporating geographic factors
    """

    def __init__(self, regions: list[RegionProfile]):
        self.regions = regions
        self.spatial_weights: Any = {}
        self.spillover_effects: Any = {}

    def calculate_spatial_weights(self, decay_parameter: float = 0.1) -> np.ndarray:
        """
        Calculate spatial weight matrix based on distances

        Args:
            decay_parameter: Distance decay parameter

        Returns:
            Spatial weight matrix
        """
        n_regions = len(self.regions)
        weights = np.zeros((n_regions, n_regions))

        for i, region_i in enumerate(self.regions):
            for j, region_j in enumerate(self.regions):
                if i != j:
                    # Calculate distance
                    lat1, lon1 = region_i.location
                    lat2, lon2 = region_j.location
                    distance = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

                    # Distance decay weight
                    weights[i, j] = np.exp(-decay_parameter * distance)

        # Row standardize
        row_sums = weights.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums[:, np.newaxis]

        self.spatial_weights = weights
        return cast(np.ndarray, weights)

    def spatial_solow_model(self, spillover_strength: float = 0.1) -> dict[str, Any]:
        """
        Multi-region Solow model with technology spillovers

        Args:
            spillover_strength: Strength of technology spillovers

        Returns:
            Dictionary with regional growth dynamics
        """

        def spatial_dynamics(t: float, y: Any) -> np.ndarray:
            n_regions = len(self.regions)
            K = y[:n_regions]
            A = y[n_regions:]

            dK_dt = np.zeros(n_regions)
            dA_dt = np.zeros(n_regions)

            for i, region in enumerate(self.regions):
                # Capital dynamics for region i
                Y_i = (
                    region.technology_level * (K[i] ** 0.33) * (region.population**0.67)
                )
                dK_dt[i] = 0.2 * Y_i - (0.02 + 0.05) * K[i]

                # Technology dynamics with spatial spillovers
                spillover = (
                    spillover_strength * np.sum(self.spatial_weights[i, :] * A)
                    if hasattr(self, "spatial_weights")
                    else 0
                )
                dA_dt[i] = 0.02 * A[i] + spillover

            return np.concatenate([dK_dt, dA_dt])

        # Initial conditions
        n_regions = len(self.regions)
        K0 = [region.initial_capital for region in self.regions]
        A0 = [region.technology_level for region in self.regions]
        y0 = K0 + A0

        # Solve system
        t_span = (0, 50)
        t_eval = np.linspace(0, 50, 51)

        solution = solve_ivp(spatial_dynamics, t_span, y0, t_eval=t_eval)

        return {
            "time": t_eval,
            "capital_paths": solution.y[:n_regions],
            "technology_paths": solution.y[n_regions:],
            "regions": [region.region_id for region in self.regions],
        }


class EndogenousGrowthModels:
    """
    Implementation of endogenous growth models with sophisticated algorithms
    """

    def __init__(self, model_type: str = "ak"):
        self.model_type = model_type
        self.parameters: dict[str, Any] = {}
        self.solution_cache: dict[str, Any] = {}

    def ak_model(self, A: float, s: float, delta: float) -> dict[str, Any]:
        """
        AK model: Y = AK, where A is constant returns to capital
        Growth rate = sA - δ

        Args:
            A: Productivity of capital
            s: Savings rate
            delta: Depreciation rate

        Returns:
            Dictionary with growth rate and other metrics
        """
        growth_rate = s * A - delta

        return {
            "growth_rate": growth_rate,
            "productivity": A,
            "savings_rate": s,
            "depreciation_rate": delta,
            "model_type": "AK",
        }

    def romer_model(self, parameters: dict[str, float]) -> dict[str, Any]:
        """
        Romer (1990) R&D-based growth model with sophisticated implementation

        Args:
            parameters: Model parameters including:
                - alpha: Capital share
                - beta: R&D share
                - gamma: Innovation parameter
                - L: Total labor force
                - s_r: Share of labor in R&D

        Returns:
            Dictionary with growth dynamics and analysis
        """
        alpha = parameters.get("alpha", 0.33)
        beta = parameters.get("beta", 0.3)
        gamma = parameters.get("gamma", 0.1)
        L = parameters.get("L", 1000)
        s_r = parameters.get("s_r", 0.05)

        # Growth rate of technology
        g_A = gamma * s_r * L

        # Balanced growth rate
        g = g_A / (1 - alpha)

        # Knowledge stock dynamics
        A0 = parameters.get("A0", 1.0)
        time_steps = int(parameters.get("time_steps", 100))
        A_path = [A0]

        for _ in range(1, time_steps):
            A_t = A_path[-1] + gamma * s_r * L * A_path[-1] ** beta
            A_path.append(A_t)

        # Economic variables along balanced growth path
        bgp_analysis = self._analyze_balanced_growth_path(g, alpha, s_r, L, parameters)

        return {
            "technology_growth": g_A,
            "balanced_growth_rate": g,
            "rd_labor_share": s_r,
            "innovation_parameter": gamma,
            "model_type": "Romer_1990",
            "technology_path": A_path,
            "balanced_growth_analysis": bgp_analysis,
        }

    def _analyze_balanced_growth_path(
        self, g: float, alpha: float, s_r: float, L: float, parameters: dict[str, float]
    ) -> dict[str, Any]:
        """Analyze properties of balanced growth path"""
        # Capital-output ratio
        k_y_ratio = (g + parameters.get("delta", 0.05)) / (
            parameters.get("s", 0.2) * (1 - alpha)
        )

        # R&D intensity
        rd_intensity = s_r * L / (L * (1 - s_r))  # R&D workers / production workers

        # Innovation rate per worker
        innovation_rate = parameters.get("gamma", 0.1) * s_r * L

        return {
            "capital_output_ratio": k_y_ratio,
            "rd_intensity": rd_intensity,
            "innovation_rate_per_worker": innovation_rate,
            "growth_sustainability": self._assess_growth_sustainability(g, parameters),
        }

    def _assess_growth_sustainability(
        self, g: float, parameters: dict[str, float]
    ) -> dict[str, Any]:
        """Assess long-term sustainability of growth"""
        # Simple sustainability analysis
        # In practice, would include environmental constraints, resource limits, etc.

        sustainability_score = 1.0

        # Reduce score for very high growth rates (resource constraints)
        if g > 0.05:
            sustainability_score *= 0.8

        # Reduce score for low R&D investment (technological stagnation)
        if parameters.get("s_r", 0.05) < 0.02:
            sustainability_score *= 0.7

        return {
            "sustainability_score": sustainability_score,
            "sustainability_level": "high"
            if sustainability_score > 0.8
            else "medium"
            if sustainability_score > 0.5
            else "low",
            "limiting_factors": self._identify_limiting_factors(g, parameters),
        }

    def _identify_limiting_factors(
        self, g: float, parameters: dict[str, float]
    ) -> list[str]:
        """Identify potential limiting factors for sustained growth"""
        factors = []

        if g > 0.05:
            factors.append("resource_constraints")
        if parameters.get("s_r", 0.05) < 0.02:
            factors.append("insufficient_rd_investment")
        if parameters.get("alpha", 0.33) > 0.5:
            factors.append("capital_dominated_growth")

        return factors

    def schumpeterian_model(self, parameters: dict[str, float]) -> dict[str, Any]:
        """
        Schumpeterian creative destruction model

        Args:
            parameters: Model parameters

        Returns:
            Dictionary with growth and innovation dynamics
        """
        # Baseline for Schumpeterian model implementation
        return {
            "model_type": "Schumpeterian",
            "innovation_rate": parameters.get("innovation_rate", 0.05),
            "creative_destruction": parameters.get("destruction_rate", 0.03),
        }


class RegionalConvergenceAnalysis:
    """
    Analysis of regional economic convergence patterns
    """

    def __init__(self, regions_data: pd.DataFrame):
        """
        Initialize with regional economic data

        Args:
            regions_data: DataFrame with columns for region_id, gdp_per_capita,
                         population, geographic coordinates, time period
        """
        self.data = regions_data
        self.convergence_results: dict[str, Any] = {}

    def beta_convergence_analysis(
        self, initial_year: int, final_year: int
    ) -> dict[str, Any]:
        """
        Analyze beta convergence (catch-up effect)

        Args:
            initial_year: Starting year for analysis
            final_year: Ending year for analysis

        Returns:
            Dictionary with convergence analysis results
        """
        from sklearn.linear_model import LinearRegression

        # Get initial and final year data
        initial_data = self.data[self.data["year"] == initial_year]
        final_data = self.data[self.data["year"] == final_year]

        # Merge data
        merged = initial_data.merge(
            final_data, on="region_id", suffixes=("_init", "_final")
        )

        # Calculate growth rate
        years_diff = final_year - initial_year
        merged["growth_rate"] = (
            np.log(merged["gdp_per_capita_final"])
            - np.log(merged["gdp_per_capita_init"])
        ) / years_diff

        # Run beta convergence regression: growth = alpha - beta * log(initial_gdp)
        X = np.log(merged["gdp_per_capita_init"]).values.reshape(-1, 1)
        y = merged["growth_rate"].values

        model = LinearRegression()
        model.fit(X, y)

        beta = -model.coef_[0]  # Convergence coefficient
        alpha = model.intercept_
        r_squared = model.score(X, y)

        # Calculate convergence rate and half-life
        convergence_rate = beta
        half_life = np.log(2) / beta if beta > 0 else np.inf

        self.convergence_results["beta"] = {
            "beta_coefficient": beta,
            "alpha_coefficient": alpha,
            "r_squared": r_squared,
            "convergence_rate": convergence_rate,
            "half_life_years": half_life,
            "converging": beta > 0,
        }

        return cast(dict[str, Any], self.convergence_results["beta"])

    def sigma_convergence_analysis(self) -> dict[str, Any]:
        """
        Analyze sigma convergence (reduction in dispersion)

        Returns:
            Dictionary with sigma convergence results
        """
        # Calculate coefficient of variation over time
        sigma_results = []

        for year in sorted(self.data["year"].unique()):
            year_data = self.data[self.data["year"] == year]
            mean_gdp = year_data["gdp_per_capita"].mean()
            std_gdp = year_data["gdp_per_capita"].std()
            cv = std_gdp / mean_gdp

            sigma_results.append(
                {
                    "year": year,
                    "coefficient_of_variation": cv,
                    "mean_gdp_per_capita": mean_gdp,
                    "std_gdp_per_capita": std_gdp,
                }
            )

        sigma_df = pd.DataFrame(sigma_results)

        # Test for sigma convergence (declining CV over time)
        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(
            sigma_df["year"], sigma_df["coefficient_of_variation"]
        )

        self.convergence_results["sigma"] = {
            "trend_slope": slope,
            "r_squared": r_value**2,
            "p_value": p_value,
            "converging": slope < 0 and p_value < 0.05,
            "time_series": sigma_df,
        }

        return cast(dict[str, Any], self.convergence_results["sigma"])

    def spatial_convergence_analysis(
        self, spatial_weights: np.ndarray
    ) -> dict[str, Any]:
        """
        Analyze spatial convergence patterns

        Args:
            spatial_weights: Spatial weight matrix

        Returns:
            Dictionary with spatial convergence analysis
        """
        # Calculate spatial lag of GDP per capita
        spatial_results = []

        for year in sorted(self.data["year"].unique()):
            year_data = self.data[self.data["year"] == year].sort_values("region_id")
            gdp_values = year_data["gdp_per_capita"].values

            # Spatial lag
            spatial_lag = spatial_weights @ gdp_values

            # Moran's I for spatial autocorrelation
            from scipy.stats import pearsonr

            moran_i, p_value = pearsonr(gdp_values, spatial_lag)

            spatial_results.append(
                {
                    "year": year,
                    "morans_i": moran_i,
                    "p_value": p_value,
                    "spatial_correlation": "positive" if moran_i > 0 else "negative",
                }
            )

        spatial_df = pd.DataFrame(spatial_results)

        self.convergence_results["spatial"] = {
            "time_series": spatial_df,
            "average_morans_i": spatial_df["morans_i"].mean(),
            "trend_in_spatial_correlation": spatial_df["morans_i"].iloc[-1]
            - spatial_df["morans_i"].iloc[0],
        }

        return cast(dict[str, Any], self.convergence_results["spatial"])


class TechnologyDiffusionModels:
    """
    Models of technology diffusion across space
    """

    def __init__(self) -> None:
        self.diffusion_parameters: dict[str, Any] = {}

    def bass_diffusion_spatial(
        self,
        regions: list[RegionProfile],
        innovation_params: dict[str, float],
        spatial_weights: np.ndarray,
    ) -> dict[str, Any]:
        """
        Spatial Bass diffusion model for technology adoption

        Args:
            regions: List of region profiles
            innovation_params: Dictionary with p (innovation coefficient) and q (imitation coefficient)
            spatial_weights: Spatial weight matrix for diffusion

        Returns:
            Dictionary with diffusion dynamics
        """

        def spatial_bass_dynamics(t: float, y: Any) -> np.ndarray:
            """
            Spatial Bass diffusion dynamics
            y[i] = cumulative adopters in region i
            """
            n_regions = len(regions)
            adoption_rates = np.zeros(n_regions)

            for i in range(n_regions):
                # Market potential
                m_i = regions[i].population

                # Current adoption level
                F_i = y[i] / m_i if m_i > 0 else 0

                # Spatial influence (weighted average of neighboring adoption rates)
                spatial_influence = (
                    np.sum(spatial_weights[i, :] * y) / m_i if m_i > 0 else 0
                )

                # Bass diffusion equation with spatial component
                p = innovation_params.get("p", 0.03)  # Innovation coefficient
                q = innovation_params.get("q", 0.38)  # Imitation coefficient
                spatial_q = innovation_params.get("spatial_q", 0.1)  # Spatial imitation

                adoption_rates[i] = (
                    m_i * (1 - F_i) * (p + q * F_i + spatial_q * spatial_influence)
                )

            return adoption_rates

        # Initial conditions (small initial adoption in each region)
        y0 = [region.population * 0.01 for region in regions]

        # Solve system
        t_span = (0, 20)  # 20 time periods
        t_eval = np.linspace(0, 20, 41)

        solution = solve_ivp(spatial_bass_dynamics, t_span, y0, t_eval=t_eval)

        return {
            "time": t_eval,
            "adoption_paths": solution.y,
            "regions": [region.region_id for region in regions],
            "parameters": innovation_params,
        }

    def knowledge_spillover_model(
        self, regions: list[RegionProfile], rd_data: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Model knowledge spillovers and productivity growth

        Args:
            regions: List of region profiles
            rd_data: DataFrame with R&D expenditure by region and time

        Returns:
            Dictionary with spillover analysis results
        """
        # Baseline for knowledge spillover implementation
        # This would involve modeling how R&D in one region affects productivity in neighboring regions

        results: dict[str, Any] = {
            "spillover_elasticities": {},
            "productivity_effects": {},
            "spatial_knowledge_networks": {},
        }

        return results


# Example usage and testing functions
def example_growth_analysis() -> pd.DataFrame:
    """
    Example usage of growth models
    """
    print("=== Solow Growth Model Example ===")

    # Initialize Solow model
    solow = SolowGrowthModel()

    # Calculate steady state
    steady_state = solow.steady_state_values()
    print(f"Steady-state capital per worker: {steady_state['capital_per_worker']:.2f}")
    print(f"Steady-state output per worker: {steady_state['output_per_worker']:.2f}")

    # Convergence analysis
    convergence = solow.convergence_analysis(0.5)  # Start at 50% of steady state
    print(f"Convergence rate: {convergence['convergence_rate']:.3f}")
    print(f"Half-life: {convergence['half_life_years']:.1f} years")

    # Simulate growth path
    initial_conditions: dict[str, float] = {"K": 100.0, "L": 100.0, "A": 1.0}
    growth_path = solow.simulate_growth_path(initial_conditions, 50)

    print(f"Final output per worker: {growth_path['output_per_worker'].iloc[-1]:.2f}")

    return growth_path


if __name__ == "__main__":
    # Run example
    example_result = example_growth_analysis()
