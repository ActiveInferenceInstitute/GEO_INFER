"""Hydrological modeling module."""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class HydrologicalModeler:
    """Model hydrological processes."""

    def __init__(self, config: dict | None = None):
        """Initialize hydrological modeler."""
        self.config = config or {}

    def rainfall_runoff_model(
        self,
        precipitation: xr.DataArray,
        soil_moisture: xr.DataArray | None = None,
        infiltration_rate: float = 0.5,
    ) -> xr.Dataset:
        """
        Simple rainfall-runoff model.

        Precipitation is split into infiltration and runoff. When soil
        moisture is supplied (0-1 fraction of saturation) the effective
        infiltration rate is reduced and the split is renormalized so that
        ``runoff + infiltration == precipitation`` (mass is conserved).

        Args:
            precipitation: Precipitation data
            soil_moisture: Optional soil moisture data (0-1 saturation)
            infiltration_rate: Infiltration rate (0-1)

        Returns:
            Runoff and infiltration results
        """
        # Split precipitation into infiltration and runoff.
        infiltration = precipitation * infiltration_rate
        runoff = precipitation - infiltration

        # Adjust for soil moisture if available. Soil moisture is assumed
        # to be on a 0-1 scale (fraction of saturation): wetter soil reduces
        # infiltration capacity and shifts more water to runoff. The split
        # is renormalized so that runoff + infiltration == precipitation
        # (mass is conserved) for every saturation level.
        if soil_moisture is not None:
            saturation_factor = xr.where(soil_moisture > 1.0, 1.0, soil_moisture)
            saturation_factor = xr.where(
                saturation_factor < 0.0, 0.0, saturation_factor
            )
            # Effective infiltration fraction shrinks as soil saturates.
            effective_infiltration_rate = infiltration_rate * (
                1.0 - saturation_factor * 0.5
            )
            effective_infiltration_rate = xr.where(
                effective_infiltration_rate < 0.0, 0.0, effective_infiltration_rate
            )
            infiltration = precipitation * effective_infiltration_rate
            runoff = precipitation - infiltration

        return xr.Dataset(
            {
                "runoff": runoff,
                "infiltration": infiltration,
                "precipitation": precipitation,
            }
        )

    def estimate_groundwater_recharge(
        self,
        infiltration: xr.DataArray,
        evapotranspiration: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """
        Estimate groundwater recharge.

        Args:
            infiltration: Infiltration data
            evapotranspiration: Optional ET data

        Returns:
            Groundwater recharge
        """
        recharge = infiltration.copy()

        if evapotranspiration is not None:
            # ET reduces recharge
            recharge = recharge - evapotranspiration * 0.3
            recharge = xr.where(recharge < 0, 0, recharge)

        return recharge

    def calculate_water_balance(
        self,
        precipitation: xr.DataArray,
        evapotranspiration: xr.DataArray,
        runoff: xr.DataArray,
    ) -> xr.Dataset:
        """Calculate water balance.

        Delegates to :meth:`WaterBalanceModeler.water_balance_closure`,
        the canonical water-balance owner in this module, to avoid a
        second implementation of the same residual calculation.

        Args:
            precipitation: Precipitation
            evapotranspiration: Evapotranspiration
            runoff: Runoff

        Returns:
            Water balance components
        """
        from .water_balance import WaterBalanceModeler

        return WaterBalanceModeler(self.config).water_balance_closure(
            precipitation, evapotranspiration, runoff
        )

    def green_ampt_infiltration(
        self,
        precipitation: xr.DataArray,
        ks: float,
        suction_head: float = 50.0,
        delta_theta: float = 0.34,
        dt: float = 1.0,
    ) -> xr.Dataset:
        """Green-Ampt infiltration with ponding-time handling (WATER-01).

        Physically based partition of event rainfall into infiltration and
        runoff.  The infiltration capacity after ``F`` mm has entered the
        soil is ``f = Ks * (1 + S / F)`` with the sorptivity term
        ``S = suction_head * delta_theta``.  While the step rainfall depth
        is at or below the step capacity the whole depth infiltrates
        (pre-ponding); once it exceeds the capacity the step is ponded and
        the cumulative infiltration is advanced with the implicit
        Green-Ampt relation

            F' = F + Ks*dt + S * ln((F' + S) / (F + S))

        solved by Newton iteration, so the update is unconditionally
        stable and the partition is mass exact:
        ``runoff + infiltration == precipitation``.

        Documented approximation: the ponding decision and the capacity
        are evaluated at the start of each step, so ponding that begins
        mid-step is resolved at step granularity.

        Args:
            precipitation: Rain depth per time step (mm). The leading
                dimension is time; trailing dimensions are preserved
                (e.g. ``(time, y, x)``).
            ks: Saturated hydraulic conductivity (mm/hr).
            suction_head: Wetting-front suction head psi (mm).
            delta_theta: Moisture deficit, porosity minus initial water
                content (dimensionless).
            dt: Time-step length (hours).

        Returns:
            Dataset with ``infiltration`` (mm per step),
            ``infiltration_rate`` (mm/hr), ``runoff`` (mm per step),
            ``cumulative_infiltration`` (mm, end of step), and ``ponded``
            (bool per step).

        Raises:
            ValueError: when ``ks <= 0``, ``suction_head < 0``,
                ``delta_theta < 0``, ``dt <= 0``, or precipitation holds
                negative or non-finite depths.
        """
        if ks <= 0:
            raise ValueError("ks must be positive (mm/hr)")
        if suction_head < 0:
            raise ValueError("suction_head must be non-negative (mm)")
        if delta_theta < 0:
            raise ValueError("delta_theta must be non-negative")
        if dt <= 0:
            raise ValueError("dt must be positive (hours)")

        p = np.asarray(precipitation.values, dtype=float)
        if p.shape[0] == 0:
            raise ValueError("precipitation must hold at least one time step")
        if not np.all(np.isfinite(p)) or np.any(p < 0.0):
            raise ValueError("precipitation must be finite and non-negative (mm)")

        sorptivity = suction_head * delta_theta

        def _implicit_advance(f_old: np.ndarray) -> np.ndarray:
            """Solve F' = F + Ks*dt + S*ln((F'+S)/(F+S)) for F' (Newton)."""
            target = f_old + ks * dt
            f_new = np.array(target, dtype=float)
            for _ in range(100):
                ratio = (f_new + sorptivity) / (f_old + sorptivity)
                residual = f_new - target - sorptivity * np.log(ratio)
                slope = 1.0 - sorptivity / (f_new + sorptivity)
                f_new = f_new - residual / np.where(slope > 1e-12, slope, 1e-12)
                if np.all(np.abs(residual) < 1e-10 * np.maximum(1.0, target)):
                    break
            return f_new

        infiltration = np.zeros_like(p)
        runoff = np.zeros_like(p)
        ponded = np.zeros(p.shape, dtype=bool)
        cumulative = np.zeros_like(p)
        f_state = np.zeros(p.shape[1:], dtype=float)

        for k in range(p.shape[0]):
            rain = p[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                capacity = np.where(
                    f_state > 0.0,
                    ks * (1.0 + sorptivity / f_state),
                    np.inf,
                )
            is_ponded = rain > capacity * dt
            d_f = np.where(is_ponded, 0.0, rain)

            if np.any(is_ponded):
                f_advanced = _implicit_advance(f_state[is_ponded])
                d_f[is_ponded] = f_advanced - f_state[is_ponded]
                ponded[k] = True

            runoff[k] = rain - d_f
            f_state = f_state + d_f
            cumulative[k] = f_state
            infiltration[k] = d_f

        dims = precipitation.dims
        return xr.Dataset(
            {
                "infiltration": xr.DataArray(
                    infiltration, dims=dims, coords=precipitation.coords
                ),
                "infiltration_rate": xr.DataArray(
                    infiltration / dt, dims=dims, coords=precipitation.coords
                ),
                "runoff": xr.DataArray(runoff, dims=dims, coords=precipitation.coords),
                "cumulative_infiltration": xr.DataArray(
                    cumulative, dims=dims, coords=precipitation.coords
                ),
                "ponded": xr.DataArray(ponded, dims=dims, coords=precipitation.coords),
            }
        )
