"""Water quality assessment module."""

import logging
from typing import Any, cast
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np
import xarray as xr
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class WaterBodyType(Enum):
    """Types of water bodies."""

    RIVER = "river"
    LAKE = "lake"
    RESERVOIR = "reservoir"
    GROUNDWATER = "groundwater"
    ESTUARY = "estuary"
    COASTAL = "coastal"
    WETLAND = "wetland"


class PollutantType(Enum):
    """Types of water pollutants."""

    NUTRIENT = "nutrient"
    ORGANIC = "organic"
    PATHOGEN = "pathogen"
    HEAVY_METAL = "heavy_metal"
    SEDIMENT = "sediment"
    THERMAL = "thermal"
    PLASTIC = "plastic"


@dataclass
class WaterSample:
    """Water quality sample data."""

    sample_id: str
    location: tuple[float, float]  # (lon, lat)
    timestamp: str
    ph: float
    dissolved_oxygen: float  # mg/L
    turbidity: float  # NTU
    temperature: float  # Celsius
    conductivity: float | None = None  # µS/cm
    nitrate: float | None = None  # mg/L
    phosphate: float | None = None  # mg/L
    ammonia: float | None = None  # mg/L
    e_coli: float | None = None  # CFU/100mL
    total_dissolved_solids: float | None = None  # mg/L


class WaterQualityAssessor:
    """
    Comprehensive water quality assessment system.

    Provides water quality analysis including:
    - Multi-parameter quality assessment
    - Water Quality Index (WQI) calculation
    - Pollution source tracking
    - Trend analysis
    - Risk assessment
    - Regulatory compliance checking
    """

    def __init__(self, config: dict | None = None):
        """Initialize water quality assessor."""
        self.config = config or {}

        # Water quality standards (EPA/WHO default values)
        self.standards: dict[str, dict[str, float]] = {
            "ph": {"min": 6.5, "max": 8.5, "optimal": 7.0},
            "dissolved_oxygen": {"min": 5.0, "optimal": 8.0},  # mg/L
            "turbidity": {"max": 1.0, "optimal": 0.5},  # NTU
            "nitrate": {"max": 10.0, "optimal": 1.0},  # mg/L
            "phosphate": {"max": 0.1, "optimal": 0.02},  # mg/L
            "ammonia": {"max": 0.5, "optimal": 0.05},  # mg/L
            "e_coli": {"max": 126.0, "optimal": 0.0},  # CFU/100mL
            "temperature": {"max": 25.0, "optimal": 20.0},  # Celsius
            "conductivity": {"max": 1000.0, "optimal": 500.0},  # µS/cm
            "total_dissolved_solids": {"max": 500.0, "optimal": 250.0},  # mg/L
        }

        # WQI parameter weights (NSF WQI method)
        self.wqi_weights = {
            "dissolved_oxygen": 0.17,
            "e_coli": 0.16,
            "ph": 0.11,
            "temperature_change": 0.10,
            "phosphate": 0.10,
            "nitrate": 0.10,
            "turbidity": 0.08,
            "total_dissolved_solids": 0.07,
        }

        # Sample history for trend analysis
        self.sample_history: list[WaterSample] = []

    def assess_water_quality(
        self,
        ph: xr.DataArray,
        dissolved_oxygen: xr.DataArray | None = None,
        turbidity: xr.DataArray | None = None,
        nitrate: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Assess water quality against standards.

        Args:
            ph: pH values
            dissolved_oxygen: Optional dissolved oxygen (mg/L)
            turbidity: Optional turbidity (NTU)
            nitrate: Optional nitrate concentration (mg/L)

        Returns:
            Water quality assessment
        """
        results: dict[str, Any] = {}

        # pH assessment
        ph_standard = self.standards["ph"]
        ph_compliant = (ph >= ph_standard["min"]) & (ph <= ph_standard["max"])
        results["ph_compliant"] = ph_compliant
        results["ph"] = ph

        # Dissolved oxygen
        if dissolved_oxygen is not None:
            do_standard = self.standards["dissolved_oxygen"]
            do_compliant = dissolved_oxygen >= do_standard["min"]
            results["do_compliant"] = do_compliant
            results["dissolved_oxygen"] = dissolved_oxygen

        # Turbidity
        if turbidity is not None:
            turb_standard = self.standards["turbidity"]
            turb_compliant = turbidity <= turb_standard["max"]
            results["turb_compliant"] = turb_compliant
            results["turbidity"] = turbidity

        # Nitrate
        if nitrate is not None:
            nit_standard = self.standards["nitrate"]
            nit_compliant = nitrate <= nit_standard["max"]
            results["nit_compliant"] = nit_compliant
            results["nitrate"] = nitrate

        # Overall quality index
        compliance_scores = [v for k, v in results.items() if k.endswith("_compliant")]
        if compliance_scores:
            quality_index = sum(compliance_scores) / len(compliance_scores)
            results["quality_index"] = quality_index

        return xr.Dataset(results)

    def calculate_wqi(
        self, sample: WaterSample, reference_temperature: float = 20.0
    ) -> dict[str, Any]:
        """
        Calculate Water Quality Index using NSF WQI method.

        Args:
            sample: Water sample data
            reference_temperature: Reference temperature for comparison

        Returns:
            WQI score and component analysis
        """
        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0

        # DO sub-index (percent saturation based)
        if sample.dissolved_oxygen is not None:
            do_saturation = (
                sample.dissolved_oxygen / 9.0 * 100
            )  # DO saturation modelled against the 9.0 mg/L reference value
            do_score = min(100, max(0, do_saturation))
            scores["dissolved_oxygen"] = do_score
            weighted_sum += do_score * self.wqi_weights["dissolved_oxygen"]
            total_weight += self.wqi_weights["dissolved_oxygen"]

        # pH sub-index
        if sample.ph is not None:
            if 6.5 <= sample.ph <= 8.5:
                ph_score = 100 - abs(sample.ph - 7.0) * 20
            else:
                ph_score = max(0, 50 - abs(sample.ph - 7.0) * 15)
            scores["ph"] = ph_score
            weighted_sum += ph_score * self.wqi_weights["ph"]
            total_weight += self.wqi_weights["ph"]

        # Temperature change sub-index
        temp_change = abs(sample.temperature - reference_temperature)
        temp_score = max(0, 100 - temp_change * 4)
        scores["temperature"] = temp_score
        weighted_sum += temp_score * self.wqi_weights.get("temperature_change", 0.1)
        total_weight += self.wqi_weights.get("temperature_change", 0.1)

        # Turbidity sub-index
        turb_score = max(0, 100 - sample.turbidity * 10)
        scores["turbidity"] = turb_score
        weighted_sum += turb_score * self.wqi_weights["turbidity"]
        total_weight += self.wqi_weights["turbidity"]

        # Nitrate sub-index
        if sample.nitrate is not None:
            nit_score = max(0, 100 - sample.nitrate * 5)
            scores["nitrate"] = nit_score
            weighted_sum += nit_score * self.wqi_weights["nitrate"]
            total_weight += self.wqi_weights["nitrate"]

        # E. coli sub-index
        if sample.e_coli is not None:
            if sample.e_coli <= 1:
                ecoli_score = 100
            else:
                ecoli_score = max(0, 100 - np.log10(sample.e_coli) * 25)
            scores["e_coli"] = ecoli_score
            weighted_sum += ecoli_score * self.wqi_weights["e_coli"]
            total_weight += self.wqi_weights["e_coli"]

        # Calculate final WQI
        wqi = weighted_sum / total_weight if total_weight > 0 else 0

        # Classify water quality
        if wqi >= 90:
            classification = "Excellent"
        elif wqi >= 70:
            classification = "Good"
        elif wqi >= 50:
            classification = "Medium"
        elif wqi >= 25:
            classification = "Bad"
        else:
            classification = "Very Bad"

        return {
            "wqi": float(wqi),
            "classification": classification,
            "sub_indices": scores,
            "sample_id": sample.sample_id,
            "location": sample.location,
        }

    def identify_pollution_sources(
        self,
        pollutant_concentration: xr.DataArray,
        flow_direction: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Identify potential pollution sources.

        Args:
            pollutant_concentration: Pollutant concentration map
            flow_direction: Optional D8 flow-direction field. When
                supplied, hotspots are refined to the most upstream cells
                (those whose downstream neighbour is not itself a hotspot),
                isolating likely source locations from downstream transport.

        Returns:
            Dataset with ``pollution_hotspots`` (all cells above the 95th
            percentile), ``potential_sources`` (refined via flow direction
            when supplied, otherwise equal to ``pollution_hotspots``), and
            ``concentration``.
        """
        threshold = pollutant_concentration.quantile(0.95)
        hotspots = pollutant_concentration >= threshold

        # When a D8 flow-direction field is supplied, refine the hotspots to
        # the most upstream cells: a hotspot is a candidate source only if
        # its immediate downstream neighbour is not also a hotspot (i.e. the
        # hotspot sits at the upstream end of a polluted reach rather than
        # being fed by an upstream source).
        if flow_direction is not None:
            d8_offsets = {
                1: (0, 1),
                2: (1, 1),
                4: (1, 0),
                8: (1, -1),
                16: (0, -1),
                32: (-1, -1),
                64: (-1, 0),
                128: (-1, 1),
            }
            hotspots_vals = hotspots.values
            fd_vals = flow_direction.values
            # Lat/lon (or y/x) are the last two dims.
            spatial_dims = list(hotspots.dims)[-2:]
            ny, nx = hotspots_vals.shape
            sources = np.zeros_like(hotspots_vals, dtype=bool)
            for i in range(ny):
                for j in range(nx):
                    if not hotspots_vals[i, j]:
                        continue
                    code = int(fd_vals[i, j]) if np.isfinite(fd_vals[i, j]) else 0
                    dy, dx = d8_offsets.get(code, (0, 0))
                    ni, nj = i + dy, j + dx
                    # A hotspot with no downstream hotspot (off-grid or
                    # non-hotspot downstream) is a candidate source.
                    if not (0 <= ni < ny and 0 <= nj < nx) or not hotspots_vals[ni, nj]:
                        sources[i, j] = True
            potential_sources = xr.DataArray(sources, dims=spatial_dims)
        else:
            potential_sources = hotspots

        return xr.Dataset(
            {
                "pollution_hotspots": hotspots,
                "potential_sources": potential_sources,
                "concentration": pollutant_concentration,
            }
        )

    def track_pollution_plume(
        self,
        initial_location: tuple[float, float],
        pollutant_type: PollutantType,
        flow_velocity: tuple[float, float],  # (vx, vy) in m/s
        diffusion_coefficient: float,
        time_hours: float,
        grid_resolution: float = 100.0,
    ) -> dict[str, Any]:
        """
        Model pollution plume dispersion using advection-diffusion.

        Args:
            initial_location: Source (lon, lat)
            pollutant_type: Type of pollutant
            flow_velocity: Water flow velocity (vx, vy)
            diffusion_coefficient: Diffusion coefficient (m²/s)
            time_hours: Simulation time in hours
            grid_resolution: Grid cell size in meters

        Returns:
            Plume dispersion model results
        """
        time_seconds = time_hours * 3600

        # Create grid (2D Gaussian plume); grid_resolution sets the cell size.
        grid_size = 50
        half_extent = (grid_size / 2) * grid_resolution
        x = np.linspace(-half_extent, half_extent, grid_size)
        y = np.linspace(-half_extent, half_extent, grid_size)
        X, Y = np.meshgrid(x, y)
        cell_size = float(x[1] - x[0])

        # Advection displacement
        x_displacement = flow_velocity[0] * time_seconds
        y_displacement = flow_velocity[1] * time_seconds

        # Gaussian dispersion
        sigma = np.sqrt(max(0.0, 2 * diffusion_coefficient * time_seconds))

        # Calculate concentration field
        X_shifted = X - x_displacement
        Y_shifted = Y - y_displacement

        if sigma <= 0.0:
            # Zero dispersion or zero time: concentration stays at the source.
            concentration = np.zeros_like(X)
            concentration[grid_size // 2, grid_size // 2] = 1.0
        else:
            concentration = np.exp(-(X_shifted**2 + Y_shifted**2) / (2 * sigma**2))
            max_concentration = float(concentration.max())
            if max_concentration > 0 and np.isfinite(max_concentration):
                concentration = concentration / max_concentration  # Normalize

        # Calculate plume extent using the actual grid cell size, not the
        # nominal grid_resolution (they differ by the endpoint convention).
        threshold = 0.01  # 1% of max concentration
        plume_mask = concentration > threshold
        plume_area = np.sum(plume_mask) * (cell_size**2) / 1e6  # km²

        return {
            "initial_location": initial_location,
            "pollutant_type": pollutant_type.value,
            "time_hours": time_hours,
            "plume_center": (
                initial_location[0] + x_displacement / 111000,  # Approximate lon offset
                initial_location[1] + y_displacement / 111000,  # Approximate lat offset
            ),
            "dispersion_sigma_m": float(sigma),
            "plume_area_km2": float(plume_area),
            "max_extent_km": float(3 * sigma / 1000),  # 3-sigma extent
            "concentration_field": concentration.tolist(),
            "grid_x": x.tolist(),
            "grid_y": y.tolist(),
        }

    def analyze_trends(
        self, samples: list[WaterSample], parameter: str, time_window_days: int = 365
    ) -> dict[str, Any]:
        """
        Analyze water quality trends over time.

        Samples are sorted chronologically by ``timestamp`` before fitting
        a linear regression with :func:`scipy.stats.linregress`, so out-of-
        order samples no longer corrupt the trend. Significance is judged
        from the regression p-value at alpha = 0.05.

        Args:
            samples: List of water samples
            parameter: Parameter to analyze (e.g., 'ph', 'dissolved_oxygen')
            time_window_days: Analysis time window (informational)

        Returns:
            Trend analysis results
        """
        if not samples:
            return {"error": "No samples provided"}

        # Sort samples chronologically by timestamp so the regression is
        # over real time, not sample-list order.
        def _parse_ts(ts: str) -> datetime:
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                # Fall back to a best-effort parse; otherwise treat as
                # unorderable (datetime.min keeps it in place).
                try:
                    return datetime.strptime(str(ts), "%Y-%m-%d")
                except ValueError:
                    return datetime.min

        ordered = sorted(samples, key=lambda s: _parse_ts(s.timestamp))

        # Extract values for the requested parameter in time order.
        pairs: list[tuple[float, float]] = []
        for s in ordered:
            value = getattr(s, parameter, None)
            if value is None:
                continue
            t = _parse_ts(s.timestamp)
            # Ordinal day as the regression x-axis (float days since epoch).
            pairs.append((float(t.toordinal()), float(value)))

        if len(pairs) < 3:
            return {"error": "Insufficient data for trend analysis"}

        time_points = np.array([p[0] for p in pairs], dtype=float)
        values_arr = np.array([p[1] for p in pairs], dtype=float)

        # Linear regression with a proper p-value (scipy.stats.linregress).
        result = scipy_stats.linregress(time_points, values_arr)
        slope = float(result.slope)
        intercept = float(result.intercept)
        p_value = float(result.pvalue)
        r_squared = float(result.rvalue**2)
        trend_line = slope * time_points + intercept

        # Trend significance from the regression p-value at alpha=0.05.
        trend_significant = bool(p_value < 0.05)
        if trend_significant:
            trend_direction = "increasing" if slope > 0 else "decreasing"
        else:
            trend_direction = "stable"

        # Statistics
        mean_val = float(np.mean(values_arr))
        std_val = float(np.std(values_arr))
        min_val = float(np.min(values_arr))
        max_val = float(np.max(values_arr))

        # Check against standards
        standard = self.standards.get(parameter, {})
        if "max" in standard:
            exceedance_count = int(np.sum(values_arr > standard["max"]))
        elif "min" in standard:
            exceedance_count = int(np.sum(values_arr < standard["min"]))
        else:
            exceedance_count = 0

        return {
            "parameter": parameter,
            "sample_count": len(pairs),
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "trend_slope": slope,
            "trend_intercept": intercept,
            "trend_p_value": p_value,
            "trend_r_squared": r_squared,
            "trend_direction": trend_direction,
            "trend_significant": trend_significant,
            "exceedance_count": exceedance_count,
            "exceedance_rate": float(exceedance_count / len(pairs)),
            "trend_values": trend_line.tolist(),
        }

    def assess_risk(
        self,
        samples: list[WaterSample],
        water_body_type: WaterBodyType,
        usage_type: str = "drinking",
    ) -> dict[str, Any]:
        """
        Assess water quality risk for specific usage.

        Args:
            samples: List of water samples
            water_body_type: Type of water body
            usage_type: Intended water use ('drinking', 'recreation', 'irrigation', 'aquatic_life')

        Returns:
            Risk assessment results
        """
        if not samples:
            return {"error": "No samples provided"}

        # Usage-specific thresholds
        usage_thresholds: dict[str, dict[str, dict[str, float]]] = {
            "drinking": {
                "ph": {"min": 6.5, "max": 8.5},
                "turbidity": {"max": 1.0},
                "nitrate": {"max": 10.0},
                "e_coli": {"max": 0.0},
            },
            "recreation": {
                "ph": {"min": 6.0, "max": 9.0},
                "turbidity": {"max": 5.0},
                "e_coli": {"max": 200.0},
            },
            "irrigation": {
                "ph": {"min": 6.0, "max": 8.5},
                "conductivity": {"max": 2000.0},
                "nitrate": {"max": 30.0},
            },
            "aquatic_life": {
                "ph": {"min": 6.5, "max": 9.0},
                "dissolved_oxygen": {"min": 5.0},
                "temperature": {"max": 30.0},
            },
        }

        thresholds = usage_thresholds.get(usage_type, usage_thresholds["drinking"])

        violations = []

        for sample in samples:
            for param, limits in thresholds.items():
                value = getattr(sample, param, None)
                if value is None:
                    continue

                # Check violations
                if "max" in limits and value > limits["max"]:
                    violations.append(
                        {
                            "sample_id": sample.sample_id,
                            "parameter": param,
                            "value": value,
                            "limit": limits["max"],
                            "exceedance": value - limits["max"],
                        }
                    )
                if "min" in limits and value < limits["min"]:
                    violations.append(
                        {
                            "sample_id": sample.sample_id,
                            "parameter": param,
                            "value": value,
                            "limit": limits["min"],
                            "deficit": limits["min"] - value,
                        }
                    )

        # Calculate overall risk score
        violation_rate = len(violations) / (len(samples) * len(thresholds))

        if violation_rate == 0:
            risk_level = "Low"
            risk_score = 0.1
        elif violation_rate < 0.1:
            risk_level = "Moderate"
            risk_score = 0.4
        elif violation_rate < 0.3:
            risk_level = "High"
            risk_score = 0.7
        else:
            risk_level = "Critical"
            risk_score = 0.95

        return {
            "water_body_type": water_body_type.value,
            "usage_type": usage_type,
            "sample_count": len(samples),
            "risk_level": risk_level,
            "risk_score": float(risk_score),
            "violation_count": len(violations),
            "violation_rate": float(violation_rate),
            "violations": violations[:10],  # Top 10 violations
            "thresholds_applied": thresholds,
            "recommendation": self._get_risk_recommendation(risk_level, usage_type),
        }

    def _get_risk_recommendation(self, risk_level: str, usage_type: str) -> str:
        """Generate risk-based recommendation."""
        if risk_level == "Low":
            return f"Water quality is suitable for {usage_type}."
        elif risk_level == "Moderate":
            return (
                f"Water may require treatment before {usage_type}. Monitor regularly."
            )
        elif risk_level == "High":
            return f"Water requires treatment before {usage_type}. Investigate pollution sources."
        else:
            return f"Water is NOT suitable for {usage_type}. Immediate action required."

    def check_regulatory_compliance(
        self, samples: list[WaterSample], regulations: str = "EPA"
    ) -> dict[str, Any]:
        """
        Check compliance with regulatory standards.

        Args:
            samples: List of water samples
            regulations: Regulatory framework ('EPA', 'WHO', 'EU')

        Returns:
            Compliance report
        """
        # Regulatory limit thresholds
        reg_limits: dict[str, dict[str, dict[str, float]]] = {
            "EPA": {
                "ph": {"min": 6.5, "max": 8.5},
                "nitrate": {"max": 10.0},
                "turbidity": {"max": 1.0},
                "e_coli": {"max": 0.0},
            },
            "WHO": {
                "ph": {"min": 6.5, "max": 8.5},
                "nitrate": {"max": 50.0},
                "turbidity": {"max": 4.0},
                "e_coli": {"max": 0.0},
            },
            "EU": {
                "ph": {"min": 6.5, "max": 9.5},
                "nitrate": {"max": 50.0},
                "turbidity": {"max": 1.0},
                "e_coli": {"max": 0.0},
            },
        }

        limits = reg_limits.get(regulations, reg_limits["EPA"])

        # Check each parameter
        compliance_results: dict[str, Any] = {}
        overall_compliant = True

        for param, lim in limits.items():
            values = [getattr(s, param, None) for s in samples]
            values = [v for v in values if v is not None]
            float_values = cast("list[float]", values)

            if not values:
                compliance_results[param] = {"status": "No data"}
                continue

            violations = 0
            for v in float_values:
                if "max" in lim and v > lim["max"]:
                    violations += 1
                if "min" in lim and v < lim["min"]:
                    violations += 1

            compliant = violations == 0
            if not compliant:
                overall_compliant = False

            compliance_results[param] = {
                "compliant": compliant,
                "samples_tested": len(float_values),
                "violations": violations,
                "limits": lim,
                "mean_value": float(np.mean(float_values)),
                "max_value": float(np.max(float_values)),
                "min_value": float(np.min(float_values)),
            }

        return {
            "regulations": regulations,
            "overall_compliant": overall_compliant,
            "parameters_tested": len(compliance_results),
            "results": compliance_results,
            "sample_count": len(samples),
            "compliance_rate": sum(
                1 for r in compliance_results.values() if r.get("compliant", False)
            )
            / len(compliance_results),
        }

    def calculate_pollutant_load(
        self,
        concentration_mg_l: float,
        flow_rate_m3_s: float,
        time_period_hours: float = 24.0,
    ) -> dict[str, float]:
        """
        Calculate pollutant load from concentration and flow.

        Args:
            concentration_mg_l: Pollutant concentration in mg/L
            flow_rate_m3_s: Water flow rate in m³/s
            time_period_hours: Time period for load calculation

        Returns:
            Pollutant load in various units
        """
        time_seconds = time_period_hours * 3600
        volume_liters = flow_rate_m3_s * time_seconds * 1000

        load_mg = concentration_mg_l * volume_liters
        load_kg = load_mg / 1e6
        load_tonnes = load_kg / 1000

        return {
            "concentration_mg_l": concentration_mg_l,
            "flow_rate_m3_s": flow_rate_m3_s,
            "time_period_hours": time_period_hours,
            "volume_liters": volume_liters,
            "load_mg": load_mg,
            "load_kg": load_kg,
            "load_tonnes": load_tonnes,
            "load_kg_per_day": load_kg * (24 / time_period_hours),
        }
