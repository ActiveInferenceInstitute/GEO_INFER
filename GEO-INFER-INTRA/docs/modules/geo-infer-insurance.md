# GEO-INFER-INSURANCE: Insurance Operations

GEO-INFER-INSURANCE provides Python components for underwriting cases, risk
assessment, premium calculations, policies, claims, and portfolio operations.
It is an independently packaged workspace module. Its outputs reflect the
configured software rules and available inputs; this page does not establish
actuarial calibration, current regulatory compliance, or production acceptance.

## Find the implemented interface

- [Module README](../../../GEO-INFER-INSURANCE/README.md) gives installation and
  validation commands.
- [Package exports](../../../GEO-INFER-INSURANCE/src/geo_infer_insurance/__init__.py)
  define the top-level names.
- [Underwriting implementation](../../../GEO-INFER-INSURANCE/src/geo_infer_insurance/underwriting/core/underwriting_engine.py)
  defines the orchestrator and its configuration.
- [Integration tests](../../../GEO-INFER-INSURANCE/tests/integration/test_integration.py)
  exercise the actual local lifecycle.

```bash
uv run --frozen geo-infer catalog INSURANCE --query UnderwritingEngine --limit 20
uv run --frozen geo-infer describe geo_infer_insurance:UnderwritingEngine
uv run --frozen geo-infer describe geo_infer_insurance:process_claim
```

Runtime discovery requires the package's actual dependencies. Agent access uses
the shared [script, CLI, and MCP contract](../integration/agent_interfaces.md),
including native object handles and explicit resource cleanup.

## Main public components

| Surface | Implemented role |
| --- | --- |
| `UnderwritingEngine`, `UnderwritingConfig` | Coordinate validation, risk, rules, premium, and decision processing |
| `RiskAssessmentEngine`, `assess_risk` | Return an assessment with its selected method and diagnostic fields |
| `PricingEngine`, `calculate_premium` | Calculate a premium from policy, assessment, and rule inputs |
| `PolicyManager`, `Policy` | Manage the module's policy objects and lifecycle |
| `ClaimsProcessor`, `Claim`, `process_claim` | Validate and process claims using selected processing configuration |
| `PortfolioManager` | Aggregate the policies and claims supplied to its own state |
| `UnderwritingRulesEngine`, `UnderwritingDecisionEngine` | Apply configured rule and decision logic |
| `UnderwritingCase`, `Decision` | Carry case state, rationale, and results |

`create_underwriting_system`, `underwrite_insurance_policy`, and
`process_insurance_claim` are compatibility wrappers around the corresponding
underwriting factory and convenience functions. Configuration shapes differ:
`underwrite_policy` takes an `UnderwritingConfig`, while `process_claim` accepts
a mapping onto `ClaimsProcessingConfig` fields. Use the actual signature rather
than sharing an unvalidated configuration dictionary between them.

## Executable validation example

This example deliberately submits an empty application and checks the software's
rejection path. It uses no real applicant, policy, payment, or underwriting
recommendation:

```python
from geo_infer_insurance import UnderwritingConfig, create_underwriting_engine

engine = create_underwriting_engine(UnderwritingConfig(
    risk_assessment_method="basic",
    external_data_sources=[],
))
try:
    case = engine.underwrite_policy({})
    assert case.status == "declined"
    assert case.decision is not None
    assert case.decision.reason == "Application validation failed"
    assert case.decision.conditions
finally:
    engine.shutdown()
```

The engine owns a thread pool; call `shutdown()` when finished. An agent handle
release alone does not call that method. Reuse an explicitly owned engine when
its state and cleanup lifecycle matter.

## Inputs, integrations, and result limits

Application validation requires `property`, `applicant`, and `coverage_requests`;
individual fields and coverage entries have additional checks in the
[validator](../../../GEO-INFER-INSURANCE/src/geo_infer_insurance/underwriting/utils/validation.py).
Claims require their own policy, loss-date, amount, and description fields.
Validation success alone does not prove that coverage exists or a payment was
made outside the process.

Optional RISK and SPACE imports and initialization have guarded paths. Inspect
the assessment method, diagnostics, errors, and actual integration state rather
than assuming an installed package means an integrated backend succeeded.
Local fallback scores and reported confidence fields are not independent
empirical validation. External data sources need explicitly configured,
authorized, and tested integrations.

A `processing_mode="batch"` configuration value is not a measured throughput
claim or proof of a public multi-application batch method. Inspect the actual
API before designing a batch runner. Compliance helpers evaluate their declared
software requirements; their names do not certify current legal compliance.

## Validate a change

```bash
uv run --frozen python GEO-INFER-TEST/run_unified_tests.py --module INSURANCE
```

Run integration tests with the actual optional backend profile required by your
application. Preserve input provenance, configuration, selected method, numerical
outputs, and failures when evaluating a model or operational workflow.
