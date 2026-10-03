# GEO-INFER-ACT/src/geo_infer_act/utils

Utils workspace within `GEO-INFER-ACT`.

## Contents

- `__init__.py`
- `analysis.py`
- `bayeux_backend.py`
- `config.py`
- `h3_adapter.py`
- `integration.py`
- `math.py`
- `pymdp_adapter.py`
- `spatial_diagnostics.py`
- `spatial_research.py`
- `visualization.py`

## Public Interface

- `analysis.py:ActiveInferenceAnalyzer` (class)
- `analysis.py:create_shared_visualizations` (function)
- `analysis.py:create_belief_heatmap` (function)
- `analysis.py:create_free_energy_plots` (function)
- `analysis.py:create_policy_plots` (function)
- `analysis.py:create_correlation_analysis` (function)
- `config.py:load_config` (function)
- `config.py:save_config` (function)
- `config.py:merge_configs` (function)
- `config.py:get_config_value` (function)
- `h3_adapter.py:H3Adapter` (class)
- `h3_adapter.py:get_h3_adapter` (function)
- `h3_adapter.py:get_nested_h3_grid_class` (function)
- `h3_adapter.py:normalize_belief_vector` (function)
- `h3_adapter.py:edge_count_from_graph` (function)
- `integration.py:initialize_logger` (function)
- `integration.py:ModernToolsIntegration` (class)
- `integration.py:integrate_rxinfer` (function)
- `integration.py:integrate_bayeux` (function)
- `integration.py:integrate_pymdp` (function)

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
