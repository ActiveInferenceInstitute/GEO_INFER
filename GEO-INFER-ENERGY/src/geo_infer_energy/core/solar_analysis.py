"""Solar energy analysis with irradiance modeling and panel optimization.

Implements solar position calculation, clear-sky irradiance estimation,
and optimal panel tilt/azimuth determination.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SolarAnalyzer:
    """Analyze solar energy potential with physics-based irradiance models.

    Implements solar geometry, clear-sky radiation models, and
    panel orientation optimization for photovoltaic site assessment.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize solar analyzer.

        Args:
            config: Configuration dictionary. Reserved contract: stored for
                API stability but currently ignored (only ``WindAnalyzer``
                consumes config keys).
        """
        self.config = config or {}
        self.solar_constant: float = 1361.0  # W/m^2

    def solar_declination(self, day_of_year: int) -> float:
        """Calculate solar declination angle.

        Uses the Spencer (1971) equation:
        delta = 23.45 * sin(360/365 * (284 + n))

        Args:
            day_of_year: Day of year (1-365).

        Returns:
            Solar declination in degrees.
        """
        return float(23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + day_of_year))))

    def hour_angle(self, solar_time_hours: float) -> float:
        """Calculate hour angle from solar time.

        Args:
            solar_time_hours: Solar time in hours (12.0 = solar noon).

        Returns:
            Hour angle in degrees (negative before noon, positive after).
        """
        return 15.0 * (solar_time_hours - 12.0)

    def solar_elevation(
        self,
        latitude_deg: float,
        day_of_year: int,
        solar_time_hours: float,
    ) -> float:
        """Calculate solar elevation angle.

        sin(alpha) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(h)

        Args:
            latitude_deg: Latitude in degrees.
            day_of_year: Day of year.
            solar_time_hours: Solar time in hours.

        Returns:
            Solar elevation angle in degrees.
        """
        dec = np.radians(self.solar_declination(day_of_year))
        lat = np.radians(latitude_deg)
        h = np.radians(self.hour_angle(solar_time_hours))

        sin_elev = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h)
        sin_elev = max(-1.0, min(1.0, sin_elev))
        return float(np.degrees(np.arcsin(sin_elev)))

    def extraterrestrial_irradiance(self, day_of_year: int) -> float:
        """Calculate extraterrestrial irradiance on a horizontal surface.

        Accounts for Earth-Sun distance variation:
        I0 = S * (1 + 0.033 * cos(360*n/365))

        Args:
            day_of_year: Day of year.

        Returns:
            Extraterrestrial irradiance (W/m^2).
        """
        return float(
            self.solar_constant
            * (1.0 + 0.033 * np.cos(np.radians(360.0 * day_of_year / 365.0)))
        )

    def clear_sky_ghi(
        self,
        latitude_deg: float,
        day_of_year: int,
        solar_time_hours: float,
        altitude_m: float = 0.0,
    ) -> float:
        """Estimate clear-sky Global Horizontal Irradiance.

        Uses the Hottel (1976) clear-sky model with altitude correction.

        Documented contract: standard-atmosphere clear-sky estimate only —
        beam transmittance via the Hottel a0/a1/k altitude terms and a fixed
        diffuse fraction (0.3); no cloud cover, aerosol variability or
        site-specific atmospheric data are modeled.

        Args:
            latitude_deg: Latitude (degrees).
            day_of_year: Day of year.
            solar_time_hours: Solar time in hours.
            altitude_m: Altitude above sea level (meters).

        Returns:
            Clear-sky GHI (W/m^2). Returns 0 if sun is below horizon.
        """
        elev = self.solar_elevation(latitude_deg, day_of_year, solar_time_hours)
        if elev <= 0:
            return 0.0

        i0 = self.extraterrestrial_irradiance(day_of_year)

        altitude_km = altitude_m / 1000.0
        a0 = 0.4237 - 0.00821 * (6.0 - altitude_km) ** 2
        a1 = 0.5055 + 0.00595 * (6.5 - altitude_km) ** 2
        k = 0.2711 + 0.01858 * (2.5 - altitude_km) ** 2

        sin_elev = np.sin(np.radians(elev))
        tau = a0 + a1 * np.exp(-k / sin_elev)

        beam = i0 * tau * sin_elev
        diffuse = 0.3 * (1.0 - tau) * i0 * sin_elev

        ghi = beam + diffuse
        return float(max(0.0, ghi))

    def daily_insolation(
        self,
        latitude_deg: float,
        day_of_year: int,
        altitude_m: float = 0.0,
    ) -> float:
        """Calculate daily clear-sky insolation by integrating hourly GHI.

        Args:
            latitude_deg: Latitude (degrees).
            day_of_year: Day of year.
            altitude_m: Altitude (meters).

        Returns:
            Daily insolation (kWh/m^2/day).
        """
        total_wh = 0.0
        for hour in range(24):
            for quarter in [0.0, 0.25, 0.5, 0.75]:
                t = hour + quarter
                ghi = self.clear_sky_ghi(latitude_deg, day_of_year, t, altitude_m)
                total_wh += ghi * 0.25  # quarter-hour integration

        return float(total_wh / 1000.0)

    def optimal_tilt_angle(
        self,
        latitude_deg: float,
    ) -> float:
        """Calculate optimal fixed panel tilt angle.

        Rule of thumb: tilt = latitude * 0.76 + 3.1 degrees
        (Jacobson and Jadhav, 2018 approximation for annual optimization).

        Args:
            latitude_deg: Site latitude (degrees).

        Returns:
            Optimal tilt angle (degrees from horizontal).
        """
        return float(abs(latitude_deg) * 0.76 + 3.1)

    def tilted_irradiance_factor(
        self,
        tilt_deg: float,
        azimuth_deg: float,
        solar_elevation_deg: float,
        solar_azimuth_deg: float,
    ) -> float:
        """Calculate irradiance factor on a tilted surface.

        Uses the geometric relationship between sun position
        and panel orientation.

        Args:
            tilt_deg: Panel tilt from horizontal (degrees).
            azimuth_deg: Panel azimuth (degrees from north, 180=south).
            solar_elevation_deg: Solar elevation (degrees).
            solar_azimuth_deg: Solar azimuth (degrees from north).

        Returns:
            Ratio of tilted to horizontal irradiance. Clamped to [0, 3].
        """
        if solar_elevation_deg <= 0:
            return 0.0

        tilt = np.radians(tilt_deg)
        azm = np.radians(azimuth_deg)
        elev = np.radians(solar_elevation_deg)
        saz = np.radians(solar_azimuth_deg)

        cos_incidence = np.sin(elev) * np.cos(tilt) + np.cos(elev) * np.sin(
            tilt
        ) * np.cos(saz - azm)

        cos_zenith = np.sin(elev)

        if cos_zenith <= 0:
            return 0.0

        factor = cos_incidence / cos_zenith
        return float(max(0.0, min(3.0, factor)))

    def estimate_pv_output(
        self,
        ghi_kwh_m2_day: float,
        panel_area_m2: float,
        efficiency: float = 0.20,
        performance_ratio: float = 0.80,
    ) -> dict[str, float]:
        """Estimate PV system energy output.

        Args:
            ghi_kwh_m2_day: Global horizontal irradiance (kWh/m^2/day).
            panel_area_m2: Total panel area (m^2).
            efficiency: Panel efficiency (default 0.20 for modern silicon).
            performance_ratio: System performance ratio (default 0.80).

        Returns:
            Dictionary with daily and annual energy estimates.
        """
        daily_kwh = ghi_kwh_m2_day * panel_area_m2 * efficiency * performance_ratio
        annual_kwh = daily_kwh * 365.0
        peak_kw = panel_area_m2 * efficiency

        capacity_factor = daily_kwh / (peak_kw * 24.0) if peak_kw > 0 else 0.0

        return {
            "daily_kwh": float(daily_kwh),
            "annual_kwh": float(annual_kwh),
            "annual_mwh": float(annual_kwh / 1000.0),
            "peak_capacity_kw": float(peak_kw),
            "capacity_factor": float(capacity_factor),
            "panel_area_m2": float(panel_area_m2),
            "efficiency": float(efficiency),
        }
