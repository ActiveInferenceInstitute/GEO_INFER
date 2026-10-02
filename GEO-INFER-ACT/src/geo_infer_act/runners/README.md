# GEO-INFER-ACT/src/geo_infer_act/runners

Runners workspace within `GEO-INFER-ACT`.

## Contents

- `__init__.py`
- `cli.py`
- `contracts.py`
- `gallery.py`
- `h3.py`
- `io.py`
- `scenarios.py`
- `wrapper.py`

## Public Interface

- `cli.py:build_parser` (function)
- `cli.py:config_from_args` (function)
- `cli.py:main` (function)
- `cli.py:run_all_main` (function)
- `contracts.py:RunConfig` (class)
- `contracts.py:ScenarioRunResult` (class)
- `contracts.py:SuiteRunResult` (class)
- `contracts.py:normalize_scenario_name` (function)
- `contracts.py:normalize_scenario_list` (function)
- `gallery.py:run_spatial_active_inference_gallery` (function)
- `h3.py:setup_san_francisco_boundary` (function)
- `h3.py:h3_cells_for_config` (function)
- `h3.py:generate_realistic_environmental_observations` (function)
- `h3.py:observation_dict_to_vector` (function)
- `h3.py:create_h3_model` (function)
- `h3.py:run_h3_active_inference` (function)
- `io.py:to_jsonable` (function)
- `io.py:package_version` (function)
- `io.py:utc_now` (function)
- `io.py:ensure_output_tree` (function)

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
