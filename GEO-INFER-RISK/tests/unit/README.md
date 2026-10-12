# GEO-INFER-RISK/tests/unit

Unit workspace within `GEO-INFER-RISK`.

## Contents

- `test_aal_exposure_years.py`
- `test_aep_pml_curve.py`
- `test_catastrophe_compound_paths.py`
- `test_civic_intel.py`
- `test_compound_risk_matrix.py`
- `test_config_loader.py`
- `test_core.py`
- `test_distributional_uncertainty.py`
- `test_exposure_model.py`
- `test_generator_isolation.py`
- `test_hazard_model.py`
- `test_insurance_models.py`
- `test_intensity_distribution_fit.py`
- `test_package_exports.py`
- `test_reproducibility.py`
- `test_risk_engine.py`
- `test_risk_metrics.py`
- `test_risk_models.py`
- `test_rng.py`
- `test_validation.py`
- `test_validation_packaged_config.py`
- `test_vulnerability_model.py`
- `test_wheel_safety.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-RISK`
- Package: `geo_infer_risk`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-risk`
- Tests: `uv run python -m pytest GEO-INFER-RISK/tests/unit`

## Dependencies

- `jsonschema>=4.17.0`
- `pyyaml>=6.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `geopandas>=0.13.0`
- `shapely>=2.0.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-RISK/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
