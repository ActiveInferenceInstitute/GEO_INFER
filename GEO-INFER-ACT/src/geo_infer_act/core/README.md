# GEO-INFER-ACT/src/geo_infer_act/core

Core workspace within `GEO-INFER-ACT`.

## Contents

- `__init__.py`
- `active_inference.py`
- `belief_updating.py`
- `civic_intel.py`
- `dynamic_causal_model.py`
- `efe_decomposition.py`
- `free_energy.py`
- `generative_model.py`
- `gnn_contract.py`
- `gnn_factored_contract.py`
- `gnn_gaussian_contract.py`
- `gnn_spec.py`
- `h3_flow_network.py`
- `markov_decision_process.py`
- `policy_selection.py`
- `spatial_agent.py`
- `types.py`
- `variational_inference.py`

## Public Interface

- `active_inference.py:ActiveInferenceModel` (class)
- `belief_updating.py:BayesianBeliefUpdate` (class)
- `civic_intel.py:CivicIntelBounds` (class)
- `civic_intel.py:GeoIntelSection` (class)
- `civic_intel.py:GeoIntelTopic` (class)
- `civic_intel.py:HazardDomain` (class)
- `civic_intel.py:CrescentCityIntel` (class)
- `civic_intel.py:parse_crescent_city_intel` (function)
- `civic_intel.py:hazard_policy_prior` (function)
- `dynamic_causal_model.py:DynamicCausalModel` (class)
- `efe_decomposition.py:log_partition` (function)
- `efe_decomposition.py:preference_law` (function)
- `efe_decomposition.py:entropy` (function)
- `efe_decomposition.py:PolicyTerms` (class)
- `efe_decomposition.py:one_step_policy_terms` (function)
- `efe_decomposition.py:support` (function)
- `efe_decomposition.py:light_cone_violations` (function)
- `free_energy.py:validate_spd_precision` (function)
- `free_energy.py:compute_policy_expected_free_energy` (function)
- `free_energy.py:FreeEnergyCalculator` (class)

## Module Metadata

- Module: `GEO-INFER-ACT`
- Package: `geo_infer_act`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-ACT`
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
- `geo-infer-bayes>=0.3.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
