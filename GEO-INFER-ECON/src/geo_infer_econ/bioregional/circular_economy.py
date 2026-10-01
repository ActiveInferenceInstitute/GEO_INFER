"""
Circular Economy Models Module

Provides circular economy modeling, material flow analysis, industrial ecology,
waste-to-resource system design, and regenerative design capabilities.
"""

from typing import Any
import numpy as np
import pandas as pd
import logging
from datetime import datetime, UTC

logger = logging.getLogger(__name__)


class CircularEconomyModels:
    """
    Circular economy flow modeling using input-output analysis.

    Models material and energy flows through economic systems, quantifying
    circularity via the Material Circularity Indicator (MCI) framework
    aligned with the Ellen MacArthur Foundation methodology.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize circular economy models."""
        self.config = config or {}
        self.include_energy = self.config.get("include_energy", True)
        logger.info("CircularEconomyModels initialized")

    def model_circular_flows(self, flow_data: dict[str, Any]) -> pd.DataFrame:
        """Model circular economy flows.

        Args:
            flow_data: Dict with:
                - materials: list of {name, mass_kg, virgin_fraction, recycled_fraction,
                  reused_fraction, waste_fraction, energy_mj}
                - processes: list of {name, inputs: [{material, mass_kg}],
                  outputs: [{material, mass_kg}], waste_kg}
                - region_id: optional region identifier

        Returns:
            DataFrame with material flow analysis including circularity metrics.
        """
        materials = flow_data.get("materials", [])
        processes = flow_data.get("processes", [])
        region_id = flow_data.get("region_id", "unknown")

        logger.info(
            "Modeling circular flows: %d materials, %d processes",
            len(materials),
            len(processes),
        )

        rows = []
        for mat in materials:
            name = mat.get("name", "unnamed")
            mass = float(mat.get("mass_kg", 0))
            virgin = float(mat.get("virgin_fraction", 1.0))
            recycled = float(mat.get("recycled_fraction", 0.0))
            reused = float(mat.get("reused_fraction", 0.0))
            waste = float(mat.get("waste_fraction", 0.0))
            energy = float(mat.get("energy_mj", 0))

            # Material Circularity Indicator (MCI) = 1 - LFI * F(utility)
            # LFI = Linear Flow Index = (virgin_input + waste_output) / (2 * mass)
            virgin_input = mass * virgin
            waste_output = mass * waste
            lfi = (virgin_input + waste_output) / max(2 * mass, 1e-6)
            mci = max(0, 1 - lfi)

            rows.append(
                {
                    "region_id": region_id,
                    "material": name,
                    "total_mass_kg": mass,
                    "virgin_input_kg": round(virgin_input, 2),
                    "recycled_input_kg": round(mass * recycled, 2),
                    "reused_input_kg": round(mass * reused, 2),
                    "waste_output_kg": round(waste_output, 2),
                    "energy_mj": energy,
                    "linear_flow_index": round(lfi, 4),
                    "circularity_index": round(mci, 4),
                    "circular_category": self._categorize_circularity(mci),
                }
            )

        # Add process-level flows
        for proc in processes:
            total_input = sum(
                float(i.get("mass_kg", 0)) for i in proc.get("inputs", [])
            )
            total_output = sum(
                float(o.get("mass_kg", 0)) for o in proc.get("outputs", [])
            )
            proc_waste = float(proc.get("waste_kg", 0))
            efficiency = total_output / max(total_input, 1e-6)

            rows.append(
                {
                    "region_id": region_id,
                    "material": f"process:{proc.get('name', 'unnamed')}",
                    "total_mass_kg": total_input,
                    "virgin_input_kg": 0,
                    "recycled_input_kg": 0,
                    "reused_input_kg": 0,
                    "waste_output_kg": proc_waste,
                    "energy_mj": 0,
                    "linear_flow_index": round(1 - efficiency, 4),
                    "circularity_index": round(efficiency, 4),
                    "circular_category": self._categorize_circularity(efficiency),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            avg_mci = df["circularity_index"].mean()
            logger.info(
                "Average circularity index: %.4f (%s)",
                avg_mci,
                self._categorize_circularity(avg_mci),
            )

        return df

    @staticmethod
    def _categorize_circularity(mci: float) -> str:
        if mci >= 0.8:
            return "highly_circular"
        elif mci >= 0.5:
            return "moderately_circular"
        elif mci >= 0.2:
            return "low_circularity"
        return "linear"


class MaterialFlowAnalysis:
    """Material flow analysis (MFA) for tracking substance flows through systems."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("MaterialFlowAnalysis initialized")

    def analyze_flows(self, material_data: dict[str, Any]) -> pd.DataFrame:
        """Analyze material flows through a system.

        Args:
            material_data: Dict with:
                - system_boundary: name of the system
                - inflows: list of {source, material, mass_kg, type}
                - outflows: list of {destination, material, mass_kg, type}
                - stocks: list of {location, material, mass_kg}
                - time_period: analysis period (e.g., "2024")

        Returns:
            DataFrame with mass balance, flow intensities, and efficiency metrics.
        """
        system = material_data.get("system_boundary", "unnamed_system")
        inflows = material_data.get("inflows", [])
        outflows = material_data.get("outflows", [])
        stocks = material_data.get("stocks", [])
        period = material_data.get("time_period", "annual")

        logger.info("Analyzing material flows for %s (period: %s)", system, period)

        # Aggregate inflows and outflows by material
        inflow_totals: dict[str, float] = {}
        for f in inflows:
            mat = f.get("material", "unknown")
            inflow_totals[mat] = inflow_totals.get(mat, 0) + float(f.get("mass_kg", 0))

        outflow_totals: dict[str, float] = {}
        for f in outflows:
            mat = f.get("material", "unknown")
            outflow_totals[mat] = outflow_totals.get(mat, 0) + float(
                f.get("mass_kg", 0)
            )

        stock_totals: dict[str, float] = {}
        for s in stocks:
            mat = s.get("material", "unknown")
            stock_totals[mat] = stock_totals.get(mat, 0) + float(s.get("mass_kg", 0))

        # Build mass balance table
        all_materials = set(
            list(inflow_totals.keys())
            + list(outflow_totals.keys())
            + list(stock_totals.keys())
        )
        rows = []
        for mat in sorted(all_materials):
            inflow = inflow_totals.get(mat, 0)
            outflow = outflow_totals.get(mat, 0)
            stock = stock_totals.get(mat, 0)
            balance = inflow - outflow  # positive = accumulation
            efficiency = outflow / max(inflow, 1e-6)

            rows.append(
                {
                    "system": system,
                    "material": mat,
                    "period": period,
                    "inflow_kg": round(inflow, 2),
                    "outflow_kg": round(outflow, 2),
                    "stock_kg": round(stock, 2),
                    "balance_kg": round(balance, 2),
                    "throughput_efficiency": round(efficiency, 4),
                    "accumulation_rate": (
                        round(balance / max(stock, 1e-6), 4) if stock > 0 else 0
                    ),
                    "mass_balanced": abs(balance - (stock * 0.01))
                    < max(inflow * 0.05, 1),
                }
            )

        df = pd.DataFrame(rows)
        total_in = sum(inflow_totals.values())
        total_out = sum(outflow_totals.values())
        logger.info(
            "MFA complete: total inflow=%.1f kg, outflow=%.1f kg, balance=%.1f kg",
            total_in,
            total_out,
            total_in - total_out,
        )
        return df


class IndustrialEcologyModels:
    """Industrial ecology models for industrial symbiosis and eco-industrial parks."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("IndustrialEcologyModels initialized")

    def model_industrial_ecology(self, data: dict[str, Any]) -> dict[str, Any]:
        """Model industrial ecology / symbiosis networks.

        Args:
            data: Dict with:
                - facilities: list of {id, name, sector,
                  outputs: [{material, mass_kg, is_waste}],
                  inputs: [{material, mass_kg, is_virgin}]}
                - proximity_km: max exchange distance

        Returns:
            Dict with symbiosis opportunities, network metrics, savings.
        """
        facilities = data.get("facilities", [])

        logger.info("Modeling industrial ecology: %d facilities", len(facilities))

        # Identify symbiosis opportunities
        # Match waste outputs from one facility to input needs of another
        waste_outputs = []
        input_needs = []

        for fac in facilities:
            fac_id = fac.get("id", "unknown")
            for out in fac.get("outputs", []):
                if out.get("is_waste", False):
                    waste_outputs.append(
                        {
                            "facility_id": fac_id,
                            "material": out["material"],
                            "mass_kg": float(out.get("mass_kg", 0)),
                        }
                    )
            for inp in fac.get("inputs", []):
                if inp.get("is_virgin", True):
                    input_needs.append(
                        {
                            "facility_id": fac_id,
                            "material": inp["material"],
                            "mass_kg": float(inp.get("mass_kg", 0)),
                        }
                    )

        # Match waste outputs to input needs
        opportunities = []
        total_diverted = 0.0
        for waste in waste_outputs:
            for need in input_needs:
                if (
                    waste["material"] == need["material"]
                    and waste["facility_id"] != need["facility_id"]
                ):
                    exchange_mass = min(waste["mass_kg"], need["mass_kg"])
                    savings_estimate = (
                        exchange_mass * 0.5
                    )  # Assume 50% cost saving vs virgin
                    opportunities.append(
                        {
                            "supplier": waste["facility_id"],
                            "receiver": need["facility_id"],
                            "material": waste["material"],
                            "exchange_mass_kg": round(exchange_mass, 2),
                            "estimated_savings_usd": round(savings_estimate, 2),
                        }
                    )
                    total_diverted += exchange_mass

        # Network metrics
        total_waste = sum(w["mass_kg"] for w in waste_outputs)
        diversion_rate = total_diverted / max(total_waste, 1e-6)
        connected_facilities = len(
            set(
                [o["supplier"] for o in opportunities]
                + [o["receiver"] for o in opportunities]
            )
        )

        result = {
            "total_facilities": len(facilities),
            "symbiosis_opportunities": opportunities,
            "opportunity_count": len(opportunities),
            "total_waste_kg": round(total_waste, 2),
            "total_diverted_kg": round(total_diverted, 2),
            "diversion_rate": round(diversion_rate, 4),
            "connected_facilities": connected_facilities,
            "network_density": round(connected_facilities / max(len(facilities), 1), 4),
            "total_estimated_savings_usd": round(
                sum(o["estimated_savings_usd"] for o in opportunities), 2
            ),
        }

        logger.info(
            "Found %d symbiosis opportunities, %.1f%% waste diversion",
            len(opportunities),
            diversion_rate * 100,
        )
        return result


class WasteToResourceSystems:
    """Waste-to-resource system design using circular cascading principles."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("WasteToResourceSystems initialized")

    # Waste hierarchy priorities (highest to lowest value retention)
    WASTE_HIERARCHY = ["prevent", "reuse", "recycle", "recover_energy", "dispose"]

    # Recovery rates by waste type and method
    DEFAULT_RECOVERY_RATES = {
        "organic": {
            "composting": 0.85,
            "anaerobic_digestion": 0.90,
            "incineration": 0.25,
        },
        "plastic": {
            "mechanical_recycling": 0.60,
            "chemical_recycling": 0.75,
            "incineration": 0.35,
        },
        "metal": {"recycling": 0.95, "smelting": 0.90},
        "glass": {"recycling": 0.90, "cullet_reuse": 0.95},
        "paper": {"recycling": 0.70, "composting": 0.50},
        "construction": {"crushing": 0.80, "sorting": 0.65},
        "electronic": {"disassembly": 0.50, "smelting": 0.70},
    }

    def design_system(self, waste_data: dict[str, Any]) -> dict[str, Any]:
        """Design a waste-to-resource system.

        Args:
            waste_data: Dict with:
                - waste_streams: list of {type, mass_kg_annual, contamination_rate}
                - available_technologies: list of technology names
                - budget_usd: available capital budget
                - space_m2: available space

        Returns:
            Dict with system design, expected recovery, costs and timeline.
        """
        streams = waste_data.get("waste_streams", [])
        technologies = waste_data.get("available_technologies", [])
        budget = float(waste_data.get("budget_usd", 0))
        space = float(waste_data.get("space_m2", 0))

        logger.info("Designing waste-to-resource system: %d streams", len(streams))

        # Design optimal processing path for each stream
        stream_designs = []
        total_input = 0.0
        total_recovered = 0.0

        for stream in streams:
            waste_type = stream.get("type", "mixed")
            mass = float(stream.get("mass_kg_annual", 0))
            contamination = float(stream.get("contamination_rate", 0.1))
            total_input += mass

            # Find best technology for this waste type
            available_methods = self.DEFAULT_RECOVERY_RATES.get(waste_type, {})
            best_method = None
            best_rate = 0.0

            for method, rate in available_methods.items():
                if not technologies or method in technologies:
                    adjusted_rate = rate * (1 - contamination)
                    if adjusted_rate > best_rate:
                        best_rate = adjusted_rate
                        best_method = method

            if best_method is None:
                best_method = "landfill"
                best_rate = 0.0

            recovered = mass * best_rate
            total_recovered += recovered

            stream_designs.append(
                {
                    "waste_type": waste_type,
                    "input_mass_kg": mass,
                    "processing_method": best_method,
                    "recovery_rate": round(best_rate, 4),
                    "recovered_mass_kg": round(recovered, 2),
                    "residual_mass_kg": round(mass - recovered, 2),
                    "hierarchy_level": self._hierarchy_level(best_method),
                }
            )

        overall_recovery = total_recovered / max(total_input, 1e-6)

        return {
            "stream_designs": stream_designs,
            "summary": {
                "total_input_kg": round(total_input, 2),
                "total_recovered_kg": round(total_recovered, 2),
                "total_residual_kg": round(total_input - total_recovered, 2),
                "overall_recovery_rate": round(overall_recovery, 4),
                "zero_waste_score": round(overall_recovery * 100, 1),
            },
            "budget_usd": budget,
            "space_m2": space,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _hierarchy_level(method: str) -> str:
        reuse_methods = {"cullet_reuse", "disassembly"}
        recycle_methods = {
            "recycling",
            "mechanical_recycling",
            "chemical_recycling",
            "composting",
            "crushing",
            "sorting",
        }
        energy_methods = {"incineration", "anaerobic_digestion", "smelting"}
        if method in reuse_methods:
            return "reuse"
        elif method in recycle_methods:
            return "recycle"
        elif method in energy_methods:
            return "recover_energy"
        return "dispose"


class RegenerativeDesign:
    """Regenerative design principles for built and natural environments."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        logger.info("RegenerativeDesign initialized")

    # Regenerative design principles (Bill Reed / LENSES framework)
    PRINCIPLES = [
        "whole_systems_thinking",
        "living_systems_alignment",
        "place_based_potential",
        "co_evolution",
        "nested_systems",
        "mutually_beneficial_relationships",
    ]

    def design_regenerative_system(
        self, design_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Design a regenerative system.

        Args:
            design_params: Dict with:
                - site: {area_ha, latitude, longitude, climate_zone, soil_type}
                - current_state: {vegetation_cover, soil_health, water_retention, biodiversity}
                - goals: list of regenerative goals
                - budget_usd: available budget
                - timeline_years: implementation timeline

        Returns:
            Dict with regenerative design plan, metrics trajectory, interventions.
        """
        site = design_params.get("site", {})
        current = design_params.get("current_state", {})
        goals = design_params.get(
            "goals", ["soil_restoration", "biodiversity", "water_cycling"]
        )
        budget = float(design_params.get("budget_usd", 0))
        timeline = int(design_params.get("timeline_years", 10))

        area_ha = float(site.get("area_ha", 1))
        logger.info(
            "Designing regenerative system: %.1f ha, %d-year timeline",
            area_ha,
            timeline,
        )

        # Assess current state
        current_scores = {
            "vegetation_cover": float(current.get("vegetation_cover", 0.3)),
            "soil_health": float(current.get("soil_health", 0.3)),
            "water_retention": float(current.get("water_retention", 0.3)),
            "biodiversity": float(current.get("biodiversity", 0.3)),
        }
        current_index = float(np.mean(list(current_scores.values())))

        # Project trajectory with interventions (logistic growth curve)
        target_scores = {k: min(v + 0.4, 0.95) for k, v in current_scores.items()}
        trajectory = []
        for year in range(timeline + 1):
            year_scores = {}
            for metric, current_val in current_scores.items():
                target_val = target_scores[metric]
                # Logistic growth: rapid early gains, plateauing
                progress = 1 / (1 + np.exp(-0.5 * (year - timeline / 2)))
                year_scores[metric] = round(
                    current_val + (target_val - current_val) * progress, 4
                )
            year_scores["year"] = year
            year_scores["composite"] = round(
                float(np.mean([v for k, v in year_scores.items() if k != "year"])), 4
            )
            trajectory.append(year_scores)

        # Design interventions
        interventions = []
        per_ha_budget = budget / max(area_ha, 1)

        if "soil_restoration" in goals:
            interventions.append(
                {
                    "intervention": "Cover cropping and composting",
                    "target_metric": "soil_health",
                    "estimated_cost_per_ha": min(per_ha_budget * 0.3, 2000),
                    "timeline_years": 3,
                    "expected_improvement": 0.3,
                }
            )
        if "biodiversity" in goals:
            interventions.append(
                {
                    "intervention": "Native species corridor planting",
                    "target_metric": "biodiversity",
                    "estimated_cost_per_ha": min(per_ha_budget * 0.25, 3000),
                    "timeline_years": 5,
                    "expected_improvement": 0.25,
                }
            )
        if "water_cycling" in goals:
            interventions.append(
                {
                    "intervention": "Swales, retention ponds, and riparian buffers",
                    "target_metric": "water_retention",
                    "estimated_cost_per_ha": min(per_ha_budget * 0.35, 4000),
                    "timeline_years": 2,
                    "expected_improvement": 0.35,
                }
            )

        # Evaluate against regenerative principles
        principle_alignment = {}
        for p in self.PRINCIPLES:
            principle_alignment[p] = len(goals) >= 2  # Multi-goal = systems thinking

        return {
            "site_summary": {
                "area_ha": area_ha,
                "climate_zone": site.get("climate_zone", "temperate"),
                "soil_type": site.get("soil_type", "unknown"),
            },
            "current_state": current_scores,
            "current_regenerative_index": round(current_index, 4),
            "target_state": target_scores,
            "projected_trajectory": trajectory,
            "interventions": interventions,
            "budget_allocation": {
                "total_usd": budget,
                "per_ha_usd": round(per_ha_budget, 2),
            },
            "principle_alignment": principle_alignment,
            "estimated_roi_years": max(3, timeline // 2),
        }
