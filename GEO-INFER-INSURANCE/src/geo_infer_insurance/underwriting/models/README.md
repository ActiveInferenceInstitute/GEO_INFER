# GEO-INFER-INSURANCE/src/geo_infer_insurance/underwriting/models

Models workspace within `GEO-INFER-INSURANCE`.

## Contents

- `__init__.py`
- `claim_models.py`
- `policy_models.py`
- `risk_models.py`
- `underwriting_models.py`

## Public Interface

- `claim_models.py:ClaimStatus` (class)
- `claim_models.py:ClaimType` (class)
- `claim_models.py:PaymentType` (class)
- `claim_models.py:Reserve` (class)
- `claim_models.py:Payment` (class)
- `claim_models.py:Claim` (class)
- `policy_models.py:PolicyStatus` (class)
- `policy_models.py:CoverageType` (class)
- `policy_models.py:Coverage` (class)
- `policy_models.py:Endorsement` (class)
- `policy_models.py:Exclusion` (class)
- `policy_models.py:Policy` (class)
- `risk_models.py:RiskLevel` (class)
- `risk_models.py:RiskCategory` (class)
- `risk_models.py:RiskProfile` (class)
- `risk_models.py:ExposureProfile` (class)
- `risk_models.py:VulnerabilityProfile` (class)
- `underwriting_models.py:DecisionStatus` (class)
- `underwriting_models.py:GuidelineType` (class)
- `underwriting_models.py:Decision` (class)

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
