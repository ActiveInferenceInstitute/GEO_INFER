# Agent Instructions: GEO-INFER-GIT/tests/unit

## Scope

- Owning module: `GEO-INFER-GIT`
- Python package: `geo_infer_git`
- Directory role: Unit workspace within `GEO-INFER-GIT`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_git` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_advanced_git.py`
- `test_cli.py`
- `test_config_loader.py`
- `test_config_loader_packaged_config.py`
- `test_distributed_coordinator.py`
- `test_error_handler.py`
- `test_error_recovery_strategies.py`
- `test_github_api.py`
- `test_intelligent_cache_prefetch.py`
- `test_main.py`
- `test_main_entrypoint.py`
- `test_multi_platform_api.py`
- `test_repo_analyzer.py`
- `test_repo_cloner.py`
- `test_repo_manager.py`
- `test_rest_api.py`
- `test_rest_api_error_handling.py`
- `test_validation.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-GIT/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
