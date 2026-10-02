"""
GEO-INFER-RISK: Geospatial Risk Analysis and Catastrophe Modeling Framework

A framework for modeling, analyzing, and visualizing geospatial risk
across multiple hazards, vulnerabilities, and exposure types.
"""

__version__ = "0.4.0"
__author__ = "GEO-INFER Team"
__license__ = "CC-BY-NC-SA-4.0"

from typing import Any

from geo_infer_risk.civic_intel import (
    CRESCENT_CITY_GEO_INTEL_SCHEMA,
    CivicHazardDomain,
    CrescentCityAnchor,
    CrescentCityBounds,
    CrescentCityHazardIntel,
    MunicipalCodeSection,
    crescent_city_hazard_weights,
    load_crescent_city_hazard,
    parse_crescent_city_hazard,
)

# Core components. All hard runtime dependencies are declared in
# pyproject.toml, so these imports must succeed; a failure is a real
# packaging bug and propagates instead of silently nulling the public API.
# geo-infer-bayes is deliberately NOT a hard dependency: it ships under the
# optional "integrations" extra, and its import sites (civic_intel,
# core/risk_engine) guard it and degrade gracefully when the extra is not
# installed.
from geo_infer_risk.core import (
    CatastropheConfig,
    EnhancedCatastropheModel,
    EnhancedExposureModel,
    EnhancedHazardModel,
    EnhancedRiskEngine,
    EnhancedVulnerabilityModel,
    ExposureModel,
    HazardModel,
    MultiHazardInteractionMatrix,
    RiskModel,
    VulnerabilityModel,
    calculate_compound_exceedance_probability,
)
from geo_infer_risk.utils import config_loader, risk_metrics, validation

# Define module level constants
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RETURN_PERIODS = [10, 25, 50, 100, 250, 500, 1000]


def create_risk_analysis(config_path: str | None = None, **kwargs: Any) -> Any:
    """
    Create a new risk analysis engine with the specified configuration.

    Args:
        config_path: Path to configuration file. If not provided,
                     default configuration will be used.
        **kwargs: Additional configuration parameters that override file settings.

    Returns:
        EnhancedRiskEngine: Configured risk analysis engine instance.
    """
    from geo_infer_risk.utils.config_loader import (
        load_config,
        load_config_with_defaults,
    )

    if config_path:
        config = load_config(config_path)
    else:
        config = load_config_with_defaults()

    for key, value in kwargs.items():
        config[key] = value

    return EnhancedRiskEngine(config)


__all__ = [
    "CRESCENT_CITY_GEO_INTEL_SCHEMA",
    "CivicHazardDomain",
    "CrescentCityAnchor",
    "CrescentCityBounds",
    "CrescentCityHazardIntel",
    "MunicipalCodeSection",
    "EnhancedRiskEngine",
    "RiskModel",
    "HazardModel",
    "VulnerabilityModel",
    "ExposureModel",
    "EnhancedHazardModel",
    "EnhancedVulnerabilityModel",
    "EnhancedExposureModel",
    "EnhancedCatastropheModel",
    "CatastropheConfig",
    "MultiHazardInteractionMatrix",
    "calculate_compound_exceedance_probability",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_RETURN_PERIODS",
    "crescent_city_hazard_weights",
    "load_crescent_city_hazard",
    "parse_crescent_city_hazard",
    "create_risk_analysis",
    "config_loader",
    "risk_metrics",
    "validation",
]
