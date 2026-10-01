"""Wind energy assessment with Weibull distribution and power curve modeling.

Implements wind resource characterization using the Weibull probability
distribution, wind turbine power curve calculation, and annual energy
production estimation.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class WindAnalyzer:
    """Analyze wind energy potential using statistical and engineering methods.

    Implements Weibull distribution fitting, wind shear correction,
    turbine power curve modeling, and energy yield estimation.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize wind analyzer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.air_density: float = self.config.get("air_density", 1.225)  # kg/m^3

    def fit_weibull(
        self,
        wind_speeds: np.ndarray,
    ) -> dict[str, float]:
        """Fit Weibull distribution to wind speed data.

        Uses the empirical method (mean and standard deviation):
        k = (sigma / mean)^(-1.086)
        c = mean / Gamma(1 + 1/k)

        Args:
            wind_speeds: Array of wind speed observations (m/s).

        Returns:
            Dictionary with shape parameter k and scale parameter c.
        """
        valid = wind_speeds[wind_speeds > 0]
        if len(valid) < 2:
            return {
                "shape_k": 2.0,
                "scale_c": 0.0,
                "mean_speed": 0.0,
                "std_speed": 0.0,
            }

        mean_v = float(np.mean(valid))
        std_v = float(np.std(valid))

        if std_v == 0 or mean_v == 0:
            return {
                "shape_k": 2.0,
                "scale_c": mean_v,
                "mean_speed": mean_v,
                "std_speed": float(std_v),
            }

        k = (std_v / mean_v) ** (-1.086)
        k = max(1.0, min(10.0, k))

        from math import gamma as gamma_fn

        c = mean_v / gamma_fn(1.0 + 1.0 / k)

        return {
            "shape_k": float(k),
            "scale_c": float(c),
            "mean_speed": float(mean_v),
            "std_speed": float(std_v),
        }

    def weibull_pdf(
        self,
        wind_speeds: np.ndarray,
        k: float,
        c: float,
    ) -> np.ndarray:
        """Calculate Weibull probability density function.

        f(v) = (k/c) * (v/c)^(k-1) * exp(-(v/c)^k)

        Args:
            wind_speeds: Wind speed values (m/s).
            k: Shape parameter.
            c: Scale parameter.

        Returns:
            Probability density at each wind speed.
        """
        v = np.asarray(wind_speeds, dtype=float)
        result = np.zeros_like(v)
        mask = v > 0
        v_safe = v[mask]
        result[mask] = (k / c) * (v_safe / c) ** (k - 1) * np.exp(-((v_safe / c) ** k))
        return result

    def wind_power_density(
        self,
        wind_speed: np.ndarray,
        air_density: float | None = None,
    ) -> np.ndarray:
        """Calculate wind power density.

        P/A = 0.5 * rho * v^3

        Args:
            wind_speed: Wind speed (m/s).
            air_density: Air density (kg/m^3). Defaults to 1.225.

        Returns:
            Wind power density (W/m^2).
        """
        rho = air_density if air_density is not None else self.air_density
        return 0.5 * rho * np.asarray(wind_speed, dtype=float) ** 3

    def extrapolate_wind_speed(
        self,
        speed_ref: float,
        height_ref: float,
        height_target: float,
        roughness_length: float = 0.03,
    ) -> float:
        """Extrapolate wind speed to different height using log wind profile.

        v(z) = v_ref * ln(z/z0) / ln(z_ref/z0)

        Args:
            speed_ref: Wind speed at reference height (m/s).
            height_ref: Reference measurement height (meters).
            height_target: Target height (meters).
            roughness_length: Surface roughness length (meters, default 0.03 for open terrain).

        Returns:
            Wind speed at target height (m/s).
        """
        if roughness_length <= 0:
            roughness_length = 0.001
        if height_ref <= roughness_length or height_target <= roughness_length:
            return speed_ref

        log_ratio = np.log(height_target / roughness_length) / np.log(
            height_ref / roughness_length
        )
        return float(speed_ref * log_ratio)

    def turbine_power_curve(
        self,
        wind_speed: np.ndarray,
        rated_power_kw: float = 2000.0,
        cut_in_speed: float = 3.0,
        rated_speed: float = 12.0,
        cut_out_speed: float = 25.0,
    ) -> np.ndarray:
        """Calculate turbine power output using idealized power curve.

        Below cut-in: 0
        Cut-in to rated: cubic interpolation P = P_rated * ((v - v_in)/(v_rated - v_in))^3
        Rated to cut-out: P_rated
        Above cut-out: 0

        Args:
            wind_speed: Wind speeds (m/s).
            rated_power_kw: Rated power output (kW).
            cut_in_speed: Cut-in wind speed (m/s).
            rated_speed: Rated wind speed (m/s).
            cut_out_speed: Cut-out wind speed (m/s).

        Returns:
            Power output at each wind speed (kW).
        """
        v = np.asarray(wind_speed, dtype=float)
        power = np.zeros_like(v)

        ramp = (v >= cut_in_speed) & (v < rated_speed)
        full = (v >= rated_speed) & (v <= cut_out_speed)

        power[ramp] = (
            rated_power_kw
            * ((v[ramp] - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3
        )
        power[full] = rated_power_kw

        return power

    def annual_energy_production(
        self,
        k: float,
        c: float,
        rated_power_kw: float = 2000.0,
        cut_in_speed: float = 3.0,
        rated_speed: float = 12.0,
        cut_out_speed: float = 25.0,
        availability: float = 0.95,
    ) -> dict[str, float]:
        """Estimate Annual Energy Production using Weibull distribution.

        Integrates turbine power curve weighted by Weibull probability.

        Args:
            k: Weibull shape parameter.
            c: Weibull scale parameter.
            rated_power_kw: Rated turbine power (kW).
            cut_in_speed: Cut-in speed (m/s).
            rated_speed: Rated speed (m/s).
            cut_out_speed: Cut-out speed (m/s).
            availability: Turbine availability factor (0-1).

        Returns:
            Dictionary with AEP and capacity factor.
        """
        speeds = np.linspace(0, 30, 601)
        pdf = self.weibull_pdf(speeds, k, c)
        power = self.turbine_power_curve(
            speeds, rated_power_kw, cut_in_speed, rated_speed, cut_out_speed
        )

        aep_kwh = float(np.trapz(power * pdf, speeds) * 8760.0 * availability)
        capacity_factor = (
            aep_kwh / (rated_power_kw * 8760.0) if rated_power_kw > 0 else 0.0
        )

        return {
            "aep_kwh": aep_kwh,
            "aep_mwh": aep_kwh / 1000.0,
            "capacity_factor": float(capacity_factor),
            "rated_power_kw": rated_power_kw,
            "weibull_k": k,
            "weibull_c": c,
            "availability": availability,
        }
