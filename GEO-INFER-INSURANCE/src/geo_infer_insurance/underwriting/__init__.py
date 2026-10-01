"""
GEO-INFER-INSURANCE Underwriting Module

Comprehensive underwriting system for risk assessment, policy management,
and insurance operations within the GEO-INFER framework.

This module provides enterprise-grade underwriting capabilities including:
- Risk assessment and pricing algorithms
- Policy management and lifecycle operations
- Claims processing and settlement
- Portfolio management and optimization
- Underwriting guidelines and compliance
- Real-time underwriting decisions
- Scaffold for external data-source integration (no live upstream: the
  built-in default endpoints are placeholders, fetches fail closed to None)
"""

__version__ = "0.3.0"
__author__ = "GEO-INFER Development Team"

from typing import Any

# Import main underwriting components
from .core.underwriting_engine import (
    UnderwritingEngine,
    UnderwritingConfig,
    UnderwritingMetrics,
)
from .core.risk_assessment import (
    RiskAssessmentEngine,
    RiskAssessmentConfig,
    RiskMetrics,
)
from .core.policy_management import (
    PolicyManager,
    PolicyLifecycle,
    Policy,
    Coverage,
    Endorsement,
)
from .core.claims_processing import (
    ClaimsProcessor,
    ClaimsEngine,
    Claim,
    ClaimStatus,
    ClaimsProcessingConfig,
    Payment,
    Reserve,
)
from .core.portfolio_management import PortfolioManager, PortfolioOptimizer
from .core.underwriting_rules import (
    UnderwritingRulesEngine,
    RuleEvaluator,
    UnderwritingRule,
    RuleCondition,
    RuleType,
)
from .core.pricing_engine import (
    PricingEngine,
    PremiumCalculator,
    PremiumCalculation,
    PricingMethod,
)
from .core.underwriting_decisions import (
    UnderwritingDecisionEngine,
    DecisionFramework,
    DecisionCriteria,
)

# Import utility modules
from .utils.validation import UnderwritingValidator, PolicyValidator
from .utils.data_integration import DataIntegrationManager, ExternalDataSource
from .utils.compliance import ComplianceEngine, RegulatoryFramework, ComplianceStatus
from .utils.reporting import UnderwritingReporter, ReportingEngine

# Import models and data structures
from .models.policy_models import Exclusion
from .models.risk_models import RiskProfile, ExposureProfile, VulnerabilityProfile
from .models.underwriting_models import UnderwritingCase, Decision, Guideline

# Import enums and types
from .core.underwriting_engine import UnderwritingStatus
from .models.policy_models import CoverageType
from .models.claim_models import ClaimType, PaymentType
from .models.risk_models import RiskLevel, RiskCategory
from .core.underwriting_decisions import DecisionType
from .utils.compliance import ComplianceFramework


# Convenience functions
def underwrite_policy(
    application_data: dict[str, Any], config: UnderwritingConfig | None = None
) -> UnderwritingCase:
    """Convenience function to underwrite a policy."""
    engine = create_underwriting_engine(config)
    return engine.underwrite_policy(application_data)


def process_claim(
    claim_data: dict[str, Any], config: dict[str, Any] | None = None
) -> Claim:
    """Convenience function to process a claim.

    ``config`` maps onto ``ClaimsProcessingConfig`` fields, e.g.
    ``{"processing_mode": "manual"}`` keeps claims under review.
    """
    processor = create_claims_processor(config)
    return processor.process_claim(claim_data)


def assess_risk(
    entity_data: dict[str, Any], assessment_type: str = "comprehensive"
) -> dict[str, Any]:
    """Convenience function to assess risk."""
    engine = create_risk_assessment()
    return engine.assess_risk(entity_data, assessment_type)


def calculate_premium(
    policy_data: dict[str, Any],
    risk_assessment: dict[str, Any],
    rule_evaluation: dict[str, Any],
) -> PremiumCalculation:
    """Convenience function to calculate premium."""
    engine = create_pricing_engine()
    return engine.calculate_premium(policy_data, risk_assessment, rule_evaluation)


def create_pricing_engine(config: dict[str, Any] | None = None) -> PricingEngine:
    """Create a pricing engine for premium calculations."""
    from .core.pricing_engine import PricingEngine

    return PricingEngine(config)


def create_underwriting_engine(
    config: UnderwritingConfig | None = None,
) -> UnderwritingEngine:
    """Create a new underwriting engine."""
    from .core.underwriting_engine import UnderwritingEngine

    return UnderwritingEngine(config)


def create_risk_assessment(
    config: RiskAssessmentConfig | None = None,
) -> RiskAssessmentEngine:
    """Create a risk assessment engine."""
    from .core.risk_assessment import RiskAssessmentEngine

    return RiskAssessmentEngine(config)


def create_policy_manager(config: dict[str, Any] | None = None) -> PolicyManager:
    """Create a policy manager."""
    from .core.policy_management import PolicyManager

    return PolicyManager(config)


def create_claims_processor(
    config: dict[str, Any] | ClaimsProcessingConfig | None = None,
) -> ClaimsProcessor:
    """Create a claims processor.

    Args:
        config: A ``ClaimsProcessingConfig`` instance or a dict whose keys are
            ``ClaimsProcessingConfig`` field names, e.g.
            ``{"processing_mode": "manual"}``. Unknown fields are rejected.
    """
    if config is None or isinstance(config, ClaimsProcessingConfig):
        return ClaimsProcessor(config)
    if isinstance(config, dict):
        claims_config = ClaimsProcessingConfig()
        for key, value in config.items():
            if not hasattr(claims_config, key):
                raise ValueError(
                    f"create_claims_processor: unknown claims config field {key!r}"
                )
            setattr(claims_config, key, value)
        return ClaimsProcessor(claims_config)
    raise TypeError(
        "create_claims_processor: config must be a ClaimsProcessingConfig or "
        f"dict, got {type(config).__name__}"
    )


# Package exports
__all__ = [
    # Core Engines and Config
    "UnderwritingEngine",
    "UnderwritingConfig",
    "UnderwritingMetrics",
    "RiskAssessmentEngine",
    "RiskAssessmentConfig",
    "RiskMetrics",
    "PolicyManager",
    "PolicyLifecycle",
    "ClaimsProcessor",
    "ClaimsEngine",
    "PortfolioManager",
    "PortfolioOptimizer",
    "UnderwritingRulesEngine",
    "RuleEvaluator",
    "UnderwritingRule",
    "RuleCondition",
    "PricingEngine",
    "PremiumCalculator",
    "UnderwritingDecisionEngine",
    "DecisionFramework",
    # Models
    "Policy",
    "Coverage",
    "Endorsement",
    "Exclusion",
    "Claim",
    "ClaimStatus",
    "Payment",
    "Reserve",
    "RiskProfile",
    "ExposureProfile",
    "VulnerabilityProfile",
    "UnderwritingCase",
    "Decision",
    "Guideline",
    # Utilities
    "UnderwritingValidator",
    "PolicyValidator",
    "DataIntegrationManager",
    "ExternalDataSource",
    "ComplianceEngine",
    "RegulatoryFramework",
    "UnderwritingReporter",
    "ReportingEngine",
    # Enums and Types
    "UnderwritingStatus",
    "CoverageType",
    "ClaimType",
    "PaymentType",
    "RiskLevel",
    "RiskCategory",
    "DecisionType",
    "DecisionCriteria",
    "RuleType",
    "ComplianceFramework",
    "ComplianceStatus",
    "PricingMethod",
    # Convenience functions
    "create_underwriting_engine",
    "create_risk_assessment",
    "create_policy_manager",
    "create_claims_processor",
    "create_pricing_engine",
    "underwrite_policy",
    "process_claim",
    "assess_risk",
    "calculate_premium",
]
