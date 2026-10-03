# Agent Instructions: GEO-INFER-ACT/src/geo_infer_act/core

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Core workspace within `GEO-INFER-ACT`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_act` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `__init__.py`
- `active_inference.py`
- `belief_updating.py`
- `civic_intel.py`
- `dynamic_causal_model.py`
- `factored_runtime.py`
- `free_energy.py`
- `generative_model.py`
- `gnn_contract.py`
- `gnn_factored_contract.py`
- `gnn_gaussian_contract.py`
- `markov_decision_process.py`
- `policy_selection.py`
- `spatial_agent.py`
- `types.py`
- `variational_inference.py`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
