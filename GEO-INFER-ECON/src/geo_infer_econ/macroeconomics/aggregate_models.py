"""Aggregate, business-cycle, monetary, fiscal, and trade models.

Contains the general macroeconomic model classes for GEO-INFER-ECON:
``AggregateGrowthModels``, ``BusinessCycleModels``, ``MonetaryPolicyModels``,
``FiscalPolicyModels``, and ``TradeModels``. Growth (Solow / endogenous /
spatial) models live in :mod:`geo_infer_econ.macroeconomics.growth_models`.
"""

import logging
import numpy as np
from typing import cast, Any

logger = logging.getLogger(__name__)


class AggregateGrowthModels:
    """
    Aggregate growth modeling using augmented Solow framework.

    Supports multi-sector growth decomposition, total factor productivity
    analysis, and growth accounting.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize aggregate growth models.

        Args:
            config: Optional dict with alpha (capital share), depreciation_rate,
                    technology_growth_rate.
        """
        self.config = config or {}
        self.alpha = self.config.get("alpha", 0.33)
        self.delta = self.config.get("depreciation_rate", 0.05)
        self.g = self.config.get("technology_growth_rate", 0.02)
        logger.info("AggregateGrowthModels initialized (alpha=%.2f)", self.alpha)

    def model_growth(self, data: dict[str, Any]) -> dict[str, Any]:
        """Model aggregate growth via growth accounting.

        Args:
            data: Dict with:
                - gdp_series: list of annual GDP values
                - capital_series: list of annual capital stock values
                - labor_series: list of annual labor force values
                - years: list of corresponding years

        Returns:
            Dict with growth rates, TFP residual, factor contributions.
        """
        gdp = np.array(data.get("gdp_series", [100, 105]), dtype=float)
        capital = np.array(data.get("capital_series", [300, 310]), dtype=float)
        labor = np.array(data.get("labor_series", [50, 51]), dtype=float)
        years = data.get("years", list(range(len(gdp))))

        logger.info("Growth accounting over %d periods", len(gdp) - 1)

        # Growth rates (log differences)
        gdp_growth = np.diff(np.log(gdp))
        cap_growth = np.diff(np.log(capital))
        lab_growth = np.diff(np.log(labor))

        # Solow residual: TFP growth = GDP growth - alpha*K growth - (1-alpha)*L growth
        tfp_growth = (
            gdp_growth - self.alpha * cap_growth - (1 - self.alpha) * lab_growth
        )

        # Factor contributions
        capital_contrib = self.alpha * cap_growth
        labor_contrib = (1 - self.alpha) * lab_growth

        return {
            "years": years[1:],
            "gdp_growth_rate": [round(g, 6) for g in gdp_growth],
            "capital_contribution": [round(c, 6) for c in capital_contrib],
            "labor_contribution": [round(c, 6) for c in labor_contrib],
            "tfp_growth": [round(t, 6) for t in tfp_growth],
            "average_gdp_growth": round(float(np.mean(gdp_growth)), 6),
            "average_tfp_growth": round(float(np.mean(tfp_growth)), 6),
            "capital_share": self.alpha,
            "tfp_share_of_growth": round(
                float(np.mean(tfp_growth) / max(np.mean(gdp_growth), 1e-10)), 4
            ),
        }


class BusinessCycleModels:
    """
    Business cycle modeling using HP filter decomposition and
    spectral analysis.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize business cycle models.

        Args:
            config: Optional dict with hp_lambda (smoothing parameter).
        """
        self.config = config or {}
        self.hp_lambda = self.config.get("hp_lambda", 1600)  # Quarterly default
        logger.info("BusinessCycleModels initialized (lambda=%d)", self.hp_lambda)

    def model_cycles(self, data: dict[str, Any]) -> dict[str, Any]:
        """Decompose economic series into trend and cyclical components.

        Args:
            data: Dict with:
                - series: list of GDP or output values
                - frequency: 'quarterly' or 'annual'
                - series_name: label for the series

        Returns:
            Dict with trend, cycle, amplitude, duration, and turning points.
        """
        series = np.array(data.get("series", []), dtype=float)
        freq = data.get("frequency", "quarterly")
        name = data.get("series_name", "output")

        if len(series) < 4:
            logger.warning("Series too short for cycle analysis (%d obs)", len(series))
            return {"error": "Series must have at least 4 observations"}

        lam = self.hp_lambda if freq == "quarterly" else 6.25

        logger.info("Decomposing %s (%d obs, freq=%s)", name, len(series), freq)

        # HP filter: minimise sum((y-tau)^2) + lambda * sum((tau_{t+1}-2*tau_t+tau_{t-1})^2)
        log_series = np.log(series)
        trend = self._hp_filter(log_series, lam)
        cycle = log_series - trend

        # Identify turning points
        peaks = []
        troughs = []
        for i in range(1, len(cycle) - 1):
            if cycle[i] > cycle[i - 1] and cycle[i] > cycle[i + 1]:
                peaks.append(i)
            elif cycle[i] < cycle[i - 1] and cycle[i] < cycle[i + 1]:
                troughs.append(i)

        # Cycle statistics
        cycle_amplitude = float(np.std(cycle))
        durations = []
        for i in range(1, len(peaks)):
            durations.append(peaks[i] - peaks[i - 1])

        return {
            "series_name": name,
            "n_observations": len(series),
            "trend": [round(float(np.exp(t)), 2) for t in trend],
            "cycle": [round(float(c), 6) for c in cycle],
            "cycle_std": round(cycle_amplitude, 6),
            "peak_indices": peaks,
            "trough_indices": troughs,
            "n_cycles": len(peaks),
            "avg_cycle_duration": round(float(np.mean(durations)), 1)
            if durations
            else None,
            "current_phase": "expansion"
            if len(cycle) > 0 and cycle[-1] > 0
            else "contraction",
        }

    @staticmethod
    def _hp_filter(y: np.ndarray, lam: float) -> np.ndarray:
        """Hodrick-Prescott filter via pentadiagonal system."""
        n = len(y)
        # Build second-difference penalty matrix
        diag_main = np.ones(n)
        diag_main[0] = 1 + lam
        diag_main[1] = 1 + 5 * lam
        diag_main[-2] = 1 + 5 * lam
        diag_main[-1] = 1 + lam
        for i in range(2, n - 2):
            diag_main[i] = 1 + 6 * lam

        diag_1 = np.zeros(n - 1)
        diag_1[0] = -2 * lam
        diag_1[-1] = -2 * lam
        for i in range(1, n - 2):
            diag_1[i] = -4 * lam

        diag_2 = np.full(n - 2, lam)

        # Solve via Thomas-like approach for pentadiagonal
        from scipy.linalg import solve_banded

        ab = np.zeros((5, n))
        for i in range(n - 2):
            ab[0, i + 2] = diag_2[i]
        for i in range(n - 1):
            ab[1, i + 1] = diag_1[i]
        ab[2, :] = diag_main
        for i in range(n - 1):
            ab[3, i] = diag_1[i]
        for i in range(n - 2):
            ab[4, i] = diag_2[i]

        return cast(np.ndarray, solve_banded((2, 2), ab, y))


class MonetaryPolicyModels:
    """
    Monetary policy modeling using Taylor rule and interest rate transmission.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize monetary policy models.

        Args:
            config: Optional dict with inflation_target, output_gap_weight,
                    inflation_weight, neutral_rate.
        """
        self.config = config or {}
        self.inflation_target = self.config.get("inflation_target", 2.0)
        self.phi_pi = self.config.get("inflation_weight", 1.5)  # Taylor coefficient
        self.phi_y = self.config.get("output_gap_weight", 0.5)  # Taylor coefficient
        self.r_star = self.config.get("neutral_rate", 2.0)
        logger.info(
            "MonetaryPolicyModels initialized (target=%.1f%%, r*=%.1f%%)",
            self.inflation_target,
            self.r_star,
        )

    def model_policy(self, policy_data: dict[str, Any]) -> dict[str, Any]:
        """Model monetary policy using Taylor rule.

        Args:
            policy_data: Dict with:
                - inflation_rate: current inflation (%)
                - output_gap: current output gap (%)
                - current_rate: current policy rate (%)
                - inflation_forecast: list of expected future inflation rates
                - gdp_growth: current GDP growth rate (%)

        Returns:
            Dict with recommended rate, Taylor rule decomposition, stance.
        """
        inflation = float(policy_data.get("inflation_rate", 3.0))
        output_gap = float(policy_data.get("output_gap", 0.0))
        current_rate = float(policy_data.get("current_rate", 5.0))
        forecast = policy_data.get("inflation_forecast", [inflation])

        logger.info(
            "Modeling monetary policy: inflation=%.1f%%, output_gap=%.1f%%",
            inflation,
            output_gap,
        )

        # Taylor rule: i = r* + π + φ_π(π - π*) + φ_y(y)
        inflation_gap = inflation - self.inflation_target
        taylor_rate = (
            self.r_star
            + inflation
            + self.phi_pi * inflation_gap
            + self.phi_y * output_gap
        )

        # Rate change recommendation
        rate_change = taylor_rate - current_rate

        # Determine monetary stance
        if rate_change > 0.5:
            stance = "tightening"
            action = "raise rates"
        elif rate_change < -0.5:
            stance = "easing"
            action = "cut rates"
        else:
            stance = "neutral"
            action = "hold rates"

        # Real interest rate
        real_rate = current_rate - inflation

        # Forward-looking component
        avg_forecast = float(np.mean(forecast)) if forecast else inflation
        forward_taylor = (
            self.r_star
            + avg_forecast
            + self.phi_pi * (avg_forecast - self.inflation_target)
            + self.phi_y * output_gap
        )

        return {
            "taylor_rule_rate": round(taylor_rate, 2),
            "forward_looking_rate": round(forward_taylor, 2),
            "current_rate": current_rate,
            "recommended_change_bps": round(rate_change * 100, 0),
            "monetary_stance": stance,
            "recommended_action": action,
            "decomposition": {
                "neutral_rate": self.r_star,
                "inflation_component": round(inflation, 2),
                "inflation_gap_component": round(self.phi_pi * inflation_gap, 2),
                "output_gap_component": round(self.phi_y * output_gap, 2),
            },
            "real_interest_rate": round(real_rate, 2),
            "inflation_gap": round(inflation_gap, 2),
        }


class FiscalPolicyModels:
    """
    Fiscal policy modeling with multiplier analysis and debt sustainability.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize fiscal policy models.

        Args:
            config: Optional dict with marginal_propensity_consume, tax_rate,
                    import_propensity.
        """
        self.config = config or {}
        self.mpc = self.config.get("marginal_propensity_consume", 0.75)
        self.tax_rate = self.config.get("tax_rate", 0.25)
        self.mpi = self.config.get("import_propensity", 0.15)
        logger.info("FiscalPolicyModels initialized (MPC=%.2f)", self.mpc)

    def model_fiscal_policy(self, policy_data: dict[str, Any]) -> dict[str, Any]:
        """Model fiscal policy with multiplier effects.

        Args:
            policy_data: Dict with:
                - gdp: current GDP
                - government_spending: government expenditure
                - tax_revenue: tax revenue
                - public_debt: outstanding public debt
                - spending_change: proposed change in spending
                - interest_rate: government borrowing rate (%)

        Returns:
            Dict with fiscal multiplier, GDP impact, debt sustainability.
        """
        gdp = float(policy_data.get("gdp", 1000))
        gov_spending = float(policy_data.get("government_spending", 200))
        tax_rev = float(policy_data.get("tax_revenue", 250))
        debt = float(policy_data.get("public_debt", 800))
        dg = float(policy_data.get("spending_change", 10))
        rate = float(policy_data.get("interest_rate", 3.0)) / 100

        logger.info("Modeling fiscal policy: GDP=%.0f, ΔG=%.0f", gdp, dg)

        # Keynesian multiplier: k = 1 / (1 - MPC(1-t) + MPI)
        multiplier = 1 / (1 - self.mpc * (1 - self.tax_rate) + self.mpi)

        # GDP impact
        gdp_impact = multiplier * dg
        new_gdp = gdp + gdp_impact

        # Fiscal balance
        budget_balance = tax_rev - gov_spending
        new_balance = (tax_rev + self.tax_rate * gdp_impact) - (gov_spending + dg)

        # Debt sustainability (Domar condition: g > r)
        gdp_growth_from_stimulus = gdp_impact / gdp
        debt_to_gdp = debt / gdp
        new_debt_to_gdp = (debt + abs(min(new_balance, 0))) / new_gdp
        debt_sustainable = gdp_growth_from_stimulus > rate

        # Interest burden
        interest_burden = debt * rate
        interest_to_gdp = interest_burden / gdp

        return {
            "fiscal_multiplier": round(multiplier, 3),
            "spending_change": dg,
            "gdp_impact": round(gdp_impact, 2),
            "gdp_growth_effect": round(gdp_growth_from_stimulus * 100, 2),
            "new_gdp": round(new_gdp, 2),
            "budget_balance": {
                "current": round(budget_balance, 2),
                "projected": round(new_balance, 2),
                "status": "surplus" if new_balance > 0 else "deficit",
            },
            "debt_analysis": {
                "debt_to_gdp_current": round(debt_to_gdp * 100, 1),
                "debt_to_gdp_projected": round(new_debt_to_gdp * 100, 1),
                "interest_burden": round(interest_burden, 2),
                "interest_to_gdp": round(interest_to_gdp * 100, 2),
                "sustainable": debt_sustainable,
            },
            "multiplier_components": {
                "mpc": self.mpc,
                "tax_rate": self.tax_rate,
                "import_propensity": self.mpi,
            },
        }


class TradeModels:
    """
    International trade modeling using gravity model approach.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize trade models.

        Args:
            config: Optional dict with distance_elasticity, gdp_elasticity.
        """
        self.config = config or {}
        self.dist_elasticity = self.config.get("distance_elasticity", -1.1)
        self.gdp_elasticity = self.config.get("gdp_elasticity", 0.8)
        logger.info("TradeModels initialized")

    def model_trade(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        """Model bilateral trade flows using gravity model.

        Args:
            trade_data: Dict with:
                - countries: list of {id, gdp, lat, lon, population}
                - trade_costs: optional dict of bilateral trade cost multipliers
                - sectors: optional list of sectors to model

        Returns:
            Dict with bilateral trade matrix, trade openness, concentration indices.
        """
        countries = trade_data.get("countries", [])
        n = len(countries)

        if n < 2:
            return {"error": "Need at least 2 countries"}

        logger.info("Modeling trade flows for %d countries", n)

        # Build gravity model trade matrix
        # T_ij = G * (GDP_i^β * GDP_j^β) / D_ij^δ
        trade_matrix = np.zeros((n, n))
        country_ids = [c.get("id", f"c{i}") for i, c in enumerate(countries)]

        for i, ci in enumerate(countries):
            gdp_i = float(ci.get("gdp", 100))
            lat_i, lon_i = float(ci.get("lat", 0)), float(ci.get("lon", 0))

            for j, cj in enumerate(countries):
                if i == j:
                    continue
                gdp_j = float(cj.get("gdp", 100))
                lat_j, lon_j = float(cj.get("lat", 0)), float(cj.get("lon", 0))

                dist = self._haversine(lat_i, lon_i, lat_j, lon_j)
                dist = max(dist, 1.0)  # minimum 1 km

                # Gravity equation
                trade_flow = (
                    (gdp_i**self.gdp_elasticity)
                    * (gdp_j**self.gdp_elasticity)
                    * (dist**self.dist_elasticity)
                )
                trade_matrix[i, j] = trade_flow

        # Normalize to reasonable scale
        if trade_matrix.max() > 0:
            scale = (
                np.mean([float(c.get("gdp", 100)) for c in countries])
                * 0.3
                / trade_matrix.max()
            )
            trade_matrix *= scale

        # Calculate trade statistics per country
        country_stats = []
        for i, c in enumerate(countries):
            exports = float(np.sum(trade_matrix[i, :]))
            imports = float(np.sum(trade_matrix[:, i]))
            gdp_val = float(c.get("gdp", 100))
            openness = (exports + imports) / max(gdp_val, 1e-6)

            # HHI concentration
            if exports > 0:
                shares = trade_matrix[i, :] / exports
                hhi = float(np.sum(shares**2))
            else:
                hhi = 0

            country_stats.append(
                {
                    "country_id": country_ids[i],
                    "exports": round(exports, 2),
                    "imports": round(imports, 2),
                    "trade_balance": round(exports - imports, 2),
                    "trade_openness": round(openness, 4),
                    "export_concentration_hhi": round(hhi, 4),
                }
            )

        # Build bilateral flows list
        bilateral = []
        for i in range(n):
            for j in range(n):
                if i != j and trade_matrix[i, j] > 0.01:
                    bilateral.append(
                        {
                            "exporter": country_ids[i],
                            "importer": country_ids[j],
                            "trade_value": round(float(trade_matrix[i, j]), 2),
                        }
                    )

        return {
            "n_countries": n,
            "country_statistics": country_stats,
            "bilateral_flows": bilateral,
            "total_world_trade": round(float(np.sum(trade_matrix)), 2),
            "model_parameters": {
                "distance_elasticity": self.dist_elasticity,
                "gdp_elasticity": self.gdp_elasticity,
            },
        }

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon / 2) ** 2
        )
        return float(R * 2 * np.arcsin(np.sqrt(a)))
