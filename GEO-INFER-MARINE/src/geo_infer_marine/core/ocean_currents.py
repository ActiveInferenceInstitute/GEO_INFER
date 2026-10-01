"""Ocean current modeling including Ekman transport and geostrophic flow.

Implements physical oceanography calculations for wind-driven and
pressure-gradient-driven ocean currents.
"""

import logging
from typing import cast

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

EARTH_ROTATION_RATE: float = 7.2921e-5  # rad/s
WATER_DENSITY: float = 1025.0  # kg/m^3
GRAVITY: float = 9.81  # m/s^2


class OceanCurrentModeler:
    """Model ocean currents from wind and pressure fields.

    Implements Ekman spiral/transport and geostrophic current calculations
    from wind stress and sea surface height data.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize ocean current modeler.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}

    @staticmethod
    def coriolis_parameter(latitude_deg: float) -> float:
        """Calculate Coriolis parameter f = 2 * Omega * sin(lat).

        Args:
            latitude_deg: Latitude in degrees.

        Returns:
            Coriolis parameter (s^-1). Returns minimum magnitude of 1e-5
            to avoid division by zero near equator.
        """
        f = 2.0 * EARTH_ROTATION_RATE * np.sin(np.radians(latitude_deg))
        # Same rule as _equator_safe_coriolis: |f| >= 1e-5, sign preserved.
        if abs(f) < 1e-5:
            f = -1e-5 if f < 0 else 1e-5
        return float(f)

    @staticmethod
    def _equator_safe_coriolis(latitude: xr.DataArray) -> xr.DataArray:
        """Compute the Coriolis parameter with an equator guard.

        Single implementation of the guard used by every division by
        ``f``: values with |f| < 1e-5 s^-1 are clamped to that
        magnitude while preserving their sign (exact zero becomes
        positive).

        Args:
            latitude: Latitude values (degrees).

        Returns:
            Guarded Coriolis parameter (s^-1).
        """
        f = 2.0 * EARTH_ROTATION_RATE * np.sin(np.radians(latitude))
        magnitude = xr.where(np.abs(f) < 1e-5, 1e-5, np.abs(f))
        return cast(xr.DataArray, xr.where(f < 0, -magnitude, magnitude))

    def calculate_ekman_transport(
        self,
        wind_stress_x: xr.DataArray,
        wind_stress_y: xr.DataArray,
        latitude: xr.DataArray,
    ) -> xr.Dataset:
        """Calculate Ekman transport from wind stress.

        Ekman transport is 90 degrees to the right (left) of the wind
        in the Northern (Southern) hemisphere:
        M_x = tau_y / f
        M_y = -tau_x / f

        Args:
            wind_stress_x: Zonal wind stress (N/m^2).
            wind_stress_y: Meridional wind stress (N/m^2).
            latitude: Latitude values (degrees).

        Returns:
            Dataset with Ekman transport components (kg/m/s).
        """
        f = self._equator_safe_coriolis(latitude)

        transport_x = wind_stress_y / f
        transport_y = -wind_stress_x / f

        magnitude = np.sqrt(transport_x**2 + transport_y**2)

        return xr.Dataset(
            {
                "ekman_transport_x": transport_x,
                "ekman_transport_y": transport_y,
                "ekman_transport_magnitude": magnitude,
            }
        )

    def calculate_ekman_pumping(
        self,
        wind_stress_x: xr.DataArray,
        wind_stress_y: xr.DataArray,
        latitude: xr.DataArray,
        dx: float = 111000.0,
        dy: float = 111000.0,
    ) -> xr.DataArray:
        """Calculate Ekman pumping velocity from wind stress curl.

        w_E = (1 / (rho * f)) * curl(tau)

        Positive = upwelling, Negative = downwelling.

        Args:
            wind_stress_x: Zonal wind stress (N/m^2).
            wind_stress_y: Meridional wind stress (N/m^2).
            latitude: Latitude (degrees).
            dx: Grid spacing in x (meters).
            dy: Grid spacing in y (meters).

        Returns:
            Ekman pumping velocity (m/s).
        """
        f = self._equator_safe_coriolis(latitude)

        dtau_y_dx = (
            wind_stress_y.diff("lon") / dx
            if "lon" in wind_stress_y.dims
            else xr.zeros_like(wind_stress_y)
        )
        dtau_x_dy = (
            wind_stress_x.diff("lat") / dy
            if "lat" in wind_stress_x.dims
            else xr.zeros_like(wind_stress_x)
        )

        min_shape = {
            dim: min(
                dtau_y_dx.sizes.get(dim, 999),
                dtau_x_dy.sizes.get(dim, 999),
                f.sizes.get(dim, 999),
            )
            for dim in set(list(dtau_y_dx.dims) + list(dtau_x_dy.dims))
        }

        curl_tau = dtau_y_dx.isel(
            {d: slice(0, s) for d, s in min_shape.items() if d in dtau_y_dx.dims}
        ) - dtau_x_dy.isel(
            {d: slice(0, s) for d, s in min_shape.items() if d in dtau_x_dy.dims}
        )
        f_trimmed = f.isel(
            {d: slice(0, s) for d, s in min_shape.items() if d in f.dims}
        )

        pumping = curl_tau / (WATER_DENSITY * f_trimmed)
        pumping.name = "ekman_pumping"
        return cast(xr.DataArray, pumping)

    def calculate_geostrophic_current(
        self,
        sea_surface_height: xr.DataArray,
        latitude: xr.DataArray,
        dx: float = 111000.0,
        dy: float = 111000.0,
    ) -> xr.Dataset:
        """Calculate geostrophic currents from sea surface height.

        u_g = -(g/f) * d(SSH)/dy
        v_g = (g/f) * d(SSH)/dx

        Args:
            sea_surface_height: Sea surface height anomaly (meters).
            latitude: Latitude (degrees).
            dx: Zonal grid spacing (meters).
            dy: Meridional grid spacing (meters).

        Returns:
            Dataset with geostrophic velocity components (m/s).
        """
        f = self._equator_safe_coriolis(latitude)

        if "lat" in sea_surface_height.dims:
            dssh_dy = sea_surface_height.diff("lat") / dy
        else:
            dssh_dy = xr.zeros_like(sea_surface_height)

        if "lon" in sea_surface_height.dims:
            dssh_dx = sea_surface_height.diff("lon") / dx
        else:
            dssh_dx = xr.zeros_like(sea_surface_height)

        min_lat = min(
            dssh_dy.sizes.get("lat", 999),
            f.sizes.get("lat", 999),
            dssh_dx.sizes.get("lat", 999),
        )
        min_lon = min(
            dssh_dy.sizes.get("lon", 999),
            f.sizes.get("lon", 999),
            dssh_dx.sizes.get("lon", 999),
        )

        sel = {}
        if "lat" in f.dims:
            sel["lat"] = slice(0, min_lat)
        if "lon" in f.dims:
            sel["lon"] = slice(0, min_lon)

        f_s = f.isel(sel) if sel else f
        dssh_dy_s = dssh_dy.isel({k: v for k, v in sel.items() if k in dssh_dy.dims})
        dssh_dx_s = dssh_dx.isel({k: v for k, v in sel.items() if k in dssh_dx.dims})

        u_geo = -(GRAVITY / f_s) * dssh_dy_s
        v_geo = (GRAVITY / f_s) * dssh_dx_s

        speed = np.sqrt(u_geo**2 + v_geo**2)

        return xr.Dataset(
            {
                "u_geostrophic": u_geo,
                "v_geostrophic": v_geo,
                "geostrophic_speed": speed,
            }
        )

    def calculate_mixed_layer_depth(
        self,
        temperature_profile: xr.DataArray,
        depth: xr.DataArray,
        delta_t: float = 0.2,
    ) -> xr.DataArray:
        """Estimate mixed layer depth from temperature profile.

        Uses the temperature threshold criterion: MLD is the depth
        at which temperature differs from the surface by delta_t.

        Args:
            temperature_profile: Temperature with depth dimension (Celsius).
            depth: Depth values (meters, positive downward).
            delta_t: Temperature difference threshold (default 0.2C).

        Returns:
            Mixed layer depth (meters).
        """
        if "depth" not in temperature_profile.dims:
            raise ValueError("temperature_profile must have a 'depth' dimension")

        sst = temperature_profile.isel(depth=0)
        diff = np.abs(temperature_profile - sst)

        exceeds = diff > delta_t

        mld = xr.full_like(sst, fill_value=float(depth.max()))

        for i in range(len(depth)):
            d_val = float(depth[i])
            mask = exceeds.isel(depth=i) if hasattr(exceeds, "isel") else exceeds[i]
            mld = xr.where(mask & (mld == float(depth.max())), d_val, mld)

        mld.name = "mixed_layer_depth"
        return mld
