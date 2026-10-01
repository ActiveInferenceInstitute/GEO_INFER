"""Data-backed analyzers for the Del Norte dashboard."""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping


def _required_mapping(data: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    """Validate that an analyzer received a configured observation mapping."""
    if not isinstance(data, Mapping) or not data:
        raise RuntimeError(f"A configured {name} data source is required")
    return data


class ClimateAnalyzer:
    """Analyze provider-backed climate observations and projections."""

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any):
        self.data = data

    def generate_climate_projections(self) -> dict[str, Any]:
        """Return configured historical and projected climate observations."""
        data = _required_mapping(self.data, "climate")
        if "historical" not in data and "projections" not in data:
            raise ValueError(
                "Climate data must contain historical or projections observations"
            )
        return dict(data)

    def calculate_climate_risks(self) -> dict[str, float]:
        """Return risk indicators calculated by the configured climate provider."""
        data = _required_mapping(self.data, "climate")
        risks = data.get("risk_indicators")
        if not isinstance(risks, Mapping) or not risks:
            raise ValueError("Climate data must contain risk_indicators")
        return {str(key): float(value) for key, value in risks.items()}

    def run_analysis(self) -> dict[str, Any]:
        """Return the configured climate analysis."""
        result = dict(self.generate_climate_projections())
        result["risk_indicators"] = self.calculate_climate_risks()
        return result


class ZoningAnalyzer:
    """Analyze provider-backed zoning and land-use observations."""

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any):
        self.data = data

    def generate_zoning_analysis(self) -> dict[str, Any]:
        """Return configured zoning observations with percentages."""
        data = _required_mapping(self.data, "zoning")
        zones = data.get("zoning_breakdown", data.get("zones"))
        if not isinstance(zones, Mapping) or not zones:
            raise ValueError("Zoning data must contain zoning_breakdown")
        total_acres = sum(float(zone.get("acres", 0)) for zone in zones.values())
        if total_acres <= 0:
            raise ValueError("Zoning observations must contain positive acreage")
        breakdown = {}
        for name, zone in zones.items():
            if not isinstance(zone, Mapping) or "acres" not in zone:
                raise ValueError(f"Zoning record {name!r} must contain acres")
            entry = dict(zone)
            entry["percentage"] = round(float(entry["acres"]) / total_acres * 100, 2)
            breakdown[str(name)] = entry
        result = dict(data)
        result["total_area_acres"] = total_acres
        result["zoning_breakdown"] = breakdown
        result["zone_breakdown"] = breakdown
        return result

    def run_analysis(self) -> dict[str, Any]:
        """Return the configured zoning analysis."""
        return self.generate_zoning_analysis()


class AgroEconomicAnalyzer:
    """Analyze provider-backed agricultural and economic observations."""

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any):
        self.data = data

    def generate_economic_analysis(self) -> dict[str, Any]:
        """Return configured economic sector observations and derived shares."""
        data = _required_mapping(self.data, "economic")
        sectors = data.get("sector_analysis", data.get("sectors"))
        if not isinstance(sectors, Mapping) or not sectors:
            raise ValueError("Economic data must contain sector_analysis")
        total_employment = sum(
            float(value.get("employment", 0)) for value in sectors.values()
        )
        if total_employment <= 0:
            raise ValueError("Economic observations must contain positive employment")
        result = dict(data)
        result["total_employment"] = total_employment
        result["sector_analysis"] = {
            str(name): {
                **dict(value),
                "employment_share": round(
                    float(value.get("employment", 0)) / total_employment * 100, 2
                ),
            }
            for name, value in sectors.items()
        }
        return result

    def run_analysis(self) -> dict[str, Any]:
        """Return the configured economic analysis."""
        return self.generate_economic_analysis()
