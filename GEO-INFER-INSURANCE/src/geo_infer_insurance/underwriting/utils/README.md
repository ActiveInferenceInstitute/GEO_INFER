# GEO-INFER-INSURANCE/src/geo_infer_insurance/underwriting/utils

Utils workspace within `GEO-INFER-INSURANCE`.

## Contents

- `__init__.py`
- `compliance.py`
- `data_integration.py`
- `reporting.py`
- `validation.py`

## Public Interface

- `compliance.py:ComplianceFramework` (class)
- `compliance.py:ComplianceStatus` (class)
- `compliance.py:RegulatoryRequirement` (class)
- `compliance.py:ComplianceCheck` (class)
- `compliance.py:ComplianceEngine` (class)
- `compliance.py:RegulatoryFramework` (class)
- `compliance.py:create_compliance_engine` (function)
- `compliance.py:check_policy_compliance` (function)
- `data_integration.py:ExternalDataSource` (class)
- `data_integration.py:DataIntegrationManager` (class)
- `data_integration.py:create_data_integration_manager` (function)
- `data_integration.py:get_credit_score` (function)
- `data_integration.py:get_property_history` (function)
- `reporting.py:ReportConfig` (class)
- `reporting.py:UnderwritingReporter` (class)
- `reporting.py:ReportingEngine` (class)
- `reporting.py:create_underwriting_reporter` (function)
- `reporting.py:generate_underwriting_summary` (function)
- `reporting.py:generate_portfolio_report` (function)
- `reporting.py:generate_claims_report` (function)

## Module Metadata

- Module: `GEO-INFER-INSURANCE`
- Package: `geo_infer_insurance`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-insurance`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INSURANCE`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INSURANCE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
