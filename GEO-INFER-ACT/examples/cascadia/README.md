# GEO-INFER-ACT/examples/cascadia

Cascadia workspace within `GEO-INFER-ACT`.

## Contents

- `data/`
- `cascadia_homing.py`
- `cascadia_homing.yaml`

## Public Interface

- `cascadia_homing.py:sha256_bytes` (function)
- `cascadia_homing.py:sha256_file` (function)
- `cascadia_homing.py:load_config` (function)
- `cascadia_homing.py:load_flowlines` (function)
- `cascadia_homing.py:build_model` (function)
- `cascadia_homing.py:render_sources` (function)
- `cascadia_homing.py:export_with_gnn` (function)
- `cascadia_homing.py:fixture_problems` (function)
- `cascadia_homing.py:exact_first_update` (function)
- `cascadia_homing.py:analyze` (function)
- `cascadia_homing.py:save_context_map` (function)
- `cascadia_homing.py:write_figures` (function)
- `cascadia_homing.py:write_json` (function)
- `cascadia_homing.py:run` (function)
- `cascadia_homing.py:parse_args` (function)
- `cascadia_homing.py:main` (function)

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
- `inferactively-pymdp==1.0.4`
- `h3>=4.5.0,<5`
- `geo-infer-bayes>=0.4.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
