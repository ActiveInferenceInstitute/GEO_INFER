"""Green-Ampt infiltration modeling.

Implements the Green-Ampt (1911) infiltration equation with explicit,
mass-conservative time stepping under rainfall supply limits. The
wetting front is treated as a sharp interface: at cumulative
infiltration ``F`` the potential rate is ``Ks * (1 + psi * dtheta / F)``
where ``Ks`` is the saturated hydraulic conductivity, ``psi`` the
wetting-front suction head, and ``dtheta = theta_s - theta_i`` the
volumetric water-content change across the front. Each step infiltrates
the smaller of the available rainfall and the per-step capacity, so
runoff plus infiltration equals rainfall at every step.
"""

import math

import numpy as np


class InfiltrationModeler:
    """Green-Ampt infiltration modeler.

    Point-scale infiltration with explicit, mass-conservative time
    stepping under rainfall supply limits. All physical parameters are
    supplied explicitly to each call; ``config`` is stored for caller
    convenience.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize infiltration modeler.

        Args:
            config: Optional configuration mapping.
        """
        self.config = config or {}

    def green_ampt_infiltration(
        self,
        rainfall_mm: np.ndarray,
        hydraulic_conductivity_mm_hr: float = 10.0,
        suction_head_mm: float = 100.0,
        saturated_water_content: float = 0.45,
        initial_water_content: float = 0.15,
        time_step_hr: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Simulate infiltration with the Green-Ampt method.

        Time-stepped Green-Ampt infiltration under rainfall supply
        limits. At cumulative infiltration ``F`` the potential rate is
        ``Ks * (1 + psi * dtheta / F)``; before the first infiltration
        (``F = 0``) the capacity is unbounded, so the first step absorbs
        any finite rainfall. Each step infiltrates
        ``min(rain, capacity)`` and the remainder runs off.

        Args:
            rainfall_mm: Rainfall depth per time step (mm); array-like,
                values must be finite and non-negative.
            hydraulic_conductivity_mm_hr: Saturated hydraulic
                conductivity ``Ks`` (mm/hr); strictly positive.
            suction_head_mm: Wetting-front suction head ``psi``
                (mm of water); non-negative.
            saturated_water_content: Saturated volumetric water content
                ``theta_s``; must satisfy ``0 < theta_s <= 1``.
            initial_water_content: Initial volumetric water content
                ``theta_i``; must satisfy ``0 <= theta_i < theta_s``.
            time_step_hr: Time step length (hr); strictly positive.

        Returns:
            Dictionary with ``rainfall_mm``, ``infiltration_mm``,
            ``runoff_mm``, ``cumulative_infiltration_mm``,
            ``cumulative_runoff_mm``, ``infiltration_rate_mm_hr``
            (realized rate per step), and ``wetting_front_depth_mm``
            (cumulative infiltration divided by ``dtheta``). Infiltration
            and runoff sum to rainfall at every step.

        Raises:
            ValueError: If a physical parameter or rainfall value is
                outside its valid range.
        """
        rain = np.atleast_1d(np.asarray(rainfall_mm, dtype=float))
        if hydraulic_conductivity_mm_hr <= 0.0:
            raise ValueError("hydraulic_conductivity_mm_hr must be > 0")
        if suction_head_mm < 0.0:
            raise ValueError("suction_head_mm must be >= 0")
        if not 0.0 < saturated_water_content <= 1.0:
            raise ValueError("saturated_water_content must be in (0, 1]")
        if not 0.0 <= initial_water_content < saturated_water_content:
            raise ValueError(
                "initial_water_content must satisfy "
                "0 <= initial_water_content < saturated_water_content"
            )
        if time_step_hr <= 0.0:
            raise ValueError("time_step_hr must be > 0")
        if not np.all(np.isfinite(rain)):
            raise ValueError("rainfall_mm values must be finite")
        if np.any(rain < 0.0):
            raise ValueError("rainfall_mm values must be non-negative")

        delta_theta = saturated_water_content - initial_water_content
        capillary_drive_mm = suction_head_mm * delta_theta

        n_steps = rain.size
        infiltration = np.zeros(n_steps)
        cumulative_infiltration = np.zeros(n_steps)

        f_cum = 0.0
        for i in range(n_steps):
            if f_cum <= 0.0:
                # Before the first infiltration the Green-Ampt capacity
                # is unbounded (F -> 0 implies rate -> infinity), so any
                # finite rainfall depth is absorbed.
                capacity_mm = math.inf
            else:
                capacity_mm = (
                    hydraulic_conductivity_mm_hr
                    * time_step_hr
                    * (1.0 + capillary_drive_mm / f_cum)
                )
            infiltration[i] = min(rain[i], capacity_mm)
            f_cum += infiltration[i]
            cumulative_infiltration[i] = f_cum

        runoff = rain - infiltration
        return {
            "rainfall_mm": rain,
            "infiltration_mm": infiltration,
            "runoff_mm": runoff,
            "cumulative_infiltration_mm": cumulative_infiltration,
            "cumulative_runoff_mm": np.cumsum(runoff),
            "infiltration_rate_mm_hr": infiltration / time_step_hr,
            "wetting_front_depth_mm": cumulative_infiltration / delta_theta,
        }
