---
name: geo-infer-insurance
description: Underwriting, policy, claims, and pricing operations for geospatial insurance workflows in the GEO-INFER framework. Use for underwriting decisions, premium calculation, claims lifecycle, policy management, portfolio analysis, and regulatory compliance checks.
---

# GEO-INFER-INSURANCE

Insurance operations layer for GEO-INFER, packaging its underwriting subsystem
as an independently installable workspace module.

## Instructions

### Core Capabilities

- **Underwriting engine** (`underwriting.core.underwriting_engine`): orchestrates
  underwriting cases end to end via `UnderwritingEngine` and
  `create_underwriting_engine`.
- **Risk assessment** (`underwriting.core.risk_assessment`): scores applications;
  uses `geo_infer_risk.core.risk_engine.EnhancedRiskEngine` when the
  geo-infer-risk package is installed (guarded import, degrades gracefully).
- **Decision engine** (`underwriting.core.underwriting_decisions`):
  approve/decline/refer decisions from rules and scored risk profiles.
- **Rules engine** (`underwriting.core.underwriting_rules`): configurable
  guidelines evaluated by `UnderwritingRulesEngine` / `RuleEvaluator`.
- **Pricing** (`underwriting.core.pricing_engine`): premium calculation via
  `PricingEngine` and `PremiumCalculator`; `calculate_premium` convenience
  function.
- **Policy management** (`underwriting.core.policy_management`): policy
  lifecycle, coverage, endorsements via `PolicyManager`.
- **Claims** (`underwriting.core.claims_processing`): claim intake, reserves,
  and payments via `ClaimsProcessor`; `process_claim` convenience function.
- **Portfolio** (`underwriting.core.portfolio_management`): policy/claim
  aggregation and capacity optimization via `PortfolioManager` /
  `PortfolioOptimizer`.
- **Compliance** (`underwriting.utils.compliance`): `ComplianceEngine` against
  declared regulatory frameworks.
- **Validation, data integration, reporting** (`underwriting.utils.*`):
  `UnderwritingValidator`, `DataIntegrationManager`, `UnderwritingReporter`.

```python
from geo_infer_insurance import (
    UnderwritingEngine,
    PricingEngine,
    PolicyManager,
    ClaimsProcessor,
    PortfolioManager,
    assess_risk,
    process_claim,
    underwrite_policy,
    create_underwriting_engine,
)
```

### Dependencies

- Core: numpy, pandas.
- Optional (`integrations` extra): geo-infer-risk (risk engine), geo-infer-space
  (spatial indexing/analytics). All cross-module imports are guarded and the
  module functions without them.

### Integrations

- `geo_infer_risk.core.risk_engine.EnhancedRiskEngine` supplies hazard scoring
  when the optional geo-infer-risk dependency is installed.
- `geo_infer_space` spatial indexing/analytics interfaces support
  location-aware rating when the optional geo-infer-space dependency is
  installed.

## Examples

```python
from geo_infer_insurance import (
    assess_risk,
    create_underwriting_engine,
    PricingEngine,
    process_claim,
    underwrite_policy,
)

# Risk scoring for an insured entity
risk = assess_risk({"entity_type": "property", "value": 500_000})

# End-to-end underwriting of an application
case = underwrite_policy({"property_value": 500_000.0, "coverage": "fire"})

# Engine-level use
engine = create_underwriting_engine()
pricing = PricingEngine()
```

## Guidelines

- Control queue-test timestamps and assert exact known durations and running
  means, including legitimate zero waits; avoid sleeps and assumptions that
  immediate operations must advance the wall clock.
- Run the module suite with the shared workspace venv:
  `uv run python GEO-INFER-TEST/run_unified_tests.py --module INSURANCE --timeout=300`.
- Cross-module scoring imports are guarded: the module works without the
  optional integrations and says so in results rather than guessing.
- Keep pyproject.toml as the only dependency ledger; the root uv.lock pins
  resolution.
