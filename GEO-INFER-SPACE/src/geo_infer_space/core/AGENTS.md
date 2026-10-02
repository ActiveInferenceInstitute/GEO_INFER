# Agent Instructions: GEO-INFER-SPACE/src/geo_infer_space/core

## Scope

- Owning module: `GEO-INFER-SPACE`
- Python package: `geo_infer_space`
- Directory role: Core workspace within `GEO-INFER-SPACE`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_space` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `__init__.py`
- `algorithm_registry.py`
- `analytics.py`
- `api_clients.py`
- `base_module.py`
- `data_integrator.py`
- `dispatcher.py`
- `geolibre_projects.py`
- `geometric_operations.py`
- `h3_policy.py`
- `interfaces.py`
- `place_analyzer.py`
- `sparse_transition.py`
- `spatial_indexing.py`
- `spatial_methods.py`
- `spatial_processor.py`
- `spatiotemporal.py`
- `state_space.py`
- `statistics.py`
- `unified_backend.py`
- `visualization_engine.py`
- `visualization_receipt.py`
- `whitebox_bridge.py`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
