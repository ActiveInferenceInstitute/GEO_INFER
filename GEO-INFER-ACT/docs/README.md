# GEO-INFER-ACT/docs

Docs workspace within `GEO-INFER-ACT`.

## Contents

- `active_inference_overview.md`
- `api_schema.yaml`
- `categorical_inference.md`
- `fep_lean_notation_bridge.md`
- `free_energy_principle.md`
- `geospatial_applications.md`
- `gnn_factored_interchange.md`
- `gnn_interchange.md`
- `mathematical_framework.md`
- `method_inventory.md`
- `references.md`
- `world_systems_modeling.md`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-ACT`
- Package: `geo_infer_act`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-act`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT`

## Dependencies

- `matplotlib>=3.4.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyyaml>=6.0`
- `requests>=2.25.0`
- `scipy>=1.7.0`
- `seaborn>=0.11.0`
- `inferactively-pymdp==1.0.3`
- `h3>=4.5.0,<5`
- `geo-infer-bayes>=0.4.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
