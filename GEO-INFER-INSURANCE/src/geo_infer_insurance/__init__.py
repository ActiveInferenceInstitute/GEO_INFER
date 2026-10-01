"""
GEO-INFER-INSURANCE: Underwriting, Policy, Claims, and Pricing Operations

A framework for insurance operations on geospatial risk data: underwriting
decisions, policy lifecycle management, claims processing, premium pricing,
portfolio management, and regulatory compliance.
"""

__version__ = "0.3.0"
__author__ = "GEO-INFER Development Team"
__license__ = "CC-BY-NC-SA-4.0"

from typing import Any, Optional

from geo_infer_insurance.underwriting import (
    UnderwritingEngine,
    UnderwritingConfig,
    RiskAssessmentEngine,
    PolicyManager,
    ClaimsProcessor,
    PortfolioManager,
    UnderwritingRulesEngine,
    PricingEngine,
    UnderwritingDecisionEngine,
    Policy,
    Claim,
    UnderwritingCase,
    Decision,
    create_underwriting_engine,
    underwrite_policy,
    process_claim,
    assess_risk,
    calculate_premium,
)

__all__ = [
    "UnderwritingEngine",
    "UnderwritingConfig",
    "RiskAssessmentEngine",
    "PolicyManager",
    "ClaimsProcessor",
    "PortfolioManager",
    "UnderwritingRulesEngine",
    "PricingEngine",
    "UnderwritingDecisionEngine",
    "Policy",
    "Claim",
    "UnderwritingCase",
    "Decision",
    "create_underwriting_engine",
    "create_underwriting_system",
    "underwrite_policy",
    "underwrite_insurance_policy",
    "process_claim",
    "process_insurance_claim",
    "assess_risk",
    "calculate_premium",
]


def create_underwriting_system(config: Any | None = None) -> Any:
    """
    Create an underwriting system.

    Args:
        config: Underwriting configuration. If None, uses defaults.

    Returns:
        UnderwritingEngine: Configured underwriting engine.
    """
    return create_underwriting_engine(config)


def underwrite_insurance_policy(
    application_data: Any, config: Any | None = None
) -> Any:
    """
    Underwrite an insurance policy application.

    Args:
        application_data: Policy application data
        config: Underwriting configuration

    Returns:
        UnderwritingCase: Completed underwriting case
    """
    return underwrite_policy(application_data, config)


def process_insurance_claim(claim_data: Any, config: Any | None = None) -> Any:
    """
    Process an insurance claim.

    Args:
        claim_data: Claim information
        config: Claims processing configuration

    Returns:
        Claim: Processed claim
    """
    return process_claim(claim_data, config)
